# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import json
import os
import re
import wandb
import yaml
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.modeling_utils import no_init_weights
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from training import (
    DynamicDepthConfig,
    enable_dynamic_depth,
    set_training_stage,
)
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


UNDERSTANDING_DELTA_PARAM_KEYWORDS = (
    "understanding_adapters",
    "understanding_attention_lora",
    "understanding_mlp_lora",
    "understanding_depth_fusion",
    "understanding_extra_",
    "understanding_named_",
)


def is_understanding_delta_parameter(name: str) -> bool:
    return any(keyword in name for keyword in UNDERSTANDING_DELTA_PARAM_KEYWORDS)


def is_tafe_parameter(name: str) -> bool:
    """Return whether a parameter belongs to an FFN or attention TAFE route."""

    return (
        "tafe_gate_" in name
        or "tafe_attention_gate_" in name
        or ".specialized_banks." in name
    )


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def reset_understanding_delta_modules(
    model: torch.nn.Module,
    logger,
) -> None:
    """Reset zero-init understanding deltas after loading a base checkpoint.

    When ``skip_pretrained_init`` is used, modules that are absent from the
    resume checkpoint can be constructed without their normal initializers.
    The understanding adapters/LoRA/fusion modules are intended to start as
    exact no-ops, so reset only those modules explicitly.
    """

    reset_names = []
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in {
            "UnderstandingResidualAdapter",
            "UnderstandingDepthFusion",
        }:
            module.reset_parameters()
            reset_names.append(name)
        elif cls_name in {
            "UnderstandingLoRALinear",
            "UnderstandingMLPLoRALinear",
        }:
            module.reset_lora_parameters()
            if hasattr(module, "reset_extra_lora_parameters"):
                module.reset_extra_lora_parameters()
            if hasattr(module, "reset_named_lora_parameters"):
                module.reset_named_lora_parameters()
            reset_names.append(name)
    logger.info(
        "Reinitialized %d understanding delta modules: %s",
        len(reset_names),
        ", ".join(reset_names[:16]) + (" ..." if len(reset_names) > 16 else ""),
    )


ROUTER_DATASET_TASKS = {
    "vlm_sft_target_family": "mmbench",
    "vlm_sft_mmmu_family": "mmmu",
    "vlm_sft_math": "mathvista",
    "vlm_sft_reasoning": "mmstar",
    "vlm_sft_ocr": "mme",
    "vlm_sft_knowledge": "mmmu",
    "vlm_sft_semantic_choice": "mmstar",
    "vlm_sft_semantic_choice_mmstar": "mmstar",
    "vlm_sft_semantic_choice_mathvista": "mathvista",
    "vlm_sft_transfer_mmstar": "mmstar",
    "vlm_sft_transfer_mathvista": "mathvista",
    "vlm_sft": "understanding",
    "mmbench_dev_v11_bilingual": "mmbench",
    "mmmu_dev_train": "mmmu",
    "mathv360k_reasoning_train": "mathvista",
    "aokvqa_train": "mathvista",
    "visual_logic_26k_train": "mmstar",
    "textvqa_train": "mme",
    "scienceqa_train": "mmstar",
    "ai2d_train": "mmstar",
    "mmlu_auxiliary_train": "mmmu",
    "semantic_choice_augmented": "mmstar",
    "train_all_instructions": "mmbench",
    "instructpix2pix_flow": "generation",
}


_SAFE_ROUTER_TASK_RE = re.compile(r"[^a-z0-9]+")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?answer\s+is\s*[:：]?\s*",
    re.IGNORECASE,
)
_INTEGER_RE = re.compile(r"^[+-]?\d+(?:,\d{3})*$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$")

_MATHVISTA_SINGLE_TYPE_SOURCE = {
    "A-OKVQA": "text",
    "AI2D": "text",
    "CLEVR-Math": "integer",
    "DocVQA": "text",
    "DVQA": "integer",
    "FigureQA": "text",
    "FunctionQA": "text",
    "GeoQA+": "text",
    "Geometry3K": "text",
    "IconQA": "text",
    "IQTest": "text",
    "MapQA": "text",
    "PMC-VQA": "text",
    "PaperQA": "text",
    "PlotQA": "integer",
    "ScienceQA": "text",
    "Super-CLEVR": "integer",
    "TQA": "text",
    "TextVQA": "integer",
    "UniGeo": "text",
    "VQA-AS": "text",
    "VQA-RAD": "text",
    "VizWiz": "integer",
}

_MMSTAR_SOURCE_L2_PROXY = {
    "ai2d_train": "diagram reasoning",
    "scienceqa_train": "biology & chemistry & physics",
}


def _safe_router_task(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.lower().replace("&", "and")
    text = _SAFE_ROUTER_TASK_RE.sub("_", text).strip("_")
    return text or "unknown"


def _metadata_value(item: dict, *keys: str):
    for key in keys:
        if key in item:
            return item[key]
        normalized = key.replace(".", "_")
        if normalized in item:
            return item[normalized]
        current = item
        found = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found:
            return current
    return None


def _normalize_mmbench_language(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"en", "eng", "english"}:
        return "en"
    if text in {"cn", "zh", "zho", "chi", "chinese", "zh-cn"}:
        return "cn"
    return None


def _normalize_mathvista_source(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "MathV360K:" in text:
        text = text.rsplit("MathV360K:", 1)[-1]
    elif ":" in text:
        text = text.rsplit(":", 1)[-1]
    aliases = {
        "AOKVQA": "A-OKVQA",
        "A-OKVQA": "A-OKVQA",
        "AI2D": "AI2D",
        "CLEVR-Math": "CLEVR-Math",
        "DVQA": "DVQA",
        "FigureQA": "FigureQA",
        "FunctionQA": "FunctionQA",
        "GEOS": "GEOS",
        "GeoQA+": "GeoQA+",
        "Geometry3K": "Geometry3K",
        "PlotQA": "PlotQA",
        "ScienceQA": "ScienceQA",
        "TabMWP": "TabMWP",
        "UniGeo": "UniGeo",
    }
    return aliases.get(text, text)


def _infer_answer_type(value) -> str | None:
    if value is None:
        return None
    text = _ANSWER_PREFIX_RE.sub("", str(value)).strip()
    text = text.strip().strip(".。").strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        return "list"
    normalized = text.replace(",", "")
    if _INTEGER_RE.match(text) or normalized.isdigit():
        return "integer"
    if _FLOAT_RE.match(normalized):
        return "float"
    return "text"


def router_metadata_task_from_index(
    item: dict,
    *,
    default_task: str = "understanding",
) -> str:
    explicit_task = _metadata_value(item, "metadata_router_task", "router_task")
    if explicit_task:
        return str(explicit_task)

    dataset_name = str(item.get("source_dataset_name") or item.get("dataset_name") or "")
    group_name = str(item.get("dataset_group_name") or item.get("dataset_name") or "")
    broad_task = ROUTER_DATASET_TASKS.get(
        dataset_name,
        ROUTER_DATASET_TASKS.get(group_name, default_task),
    )

    if dataset_name == "mmbench_dev_v11_bilingual" or broad_task == "mmbench":
        language = _normalize_mmbench_language(
            _metadata_value(item, "language", "lang", "metadata.language")
        )
        if language:
            return f"mmbench_{language}"

    if dataset_name == "mmmu_dev_train" or broad_task == "mmmu":
        subject = _metadata_value(item, "subject", "metadata.subject")
        if subject:
            return f"mmmu_subject_{_safe_router_task(subject)}"

    if dataset_name == "mathv360k_reasoning_train" or broad_task == "mathvista":
        source = _normalize_mathvista_source(
            _metadata_value(item, "metadata.source", "metadata_source", "source")
        )
        answer_type = _metadata_value(
            item,
            "answer_type",
            "metadata.answer_type",
            "metadata_answer_type",
        )
        if not answer_type:
            answer_type = _infer_answer_type(
                _metadata_value(item, "gpt_answer", "answer")
            )
        if source in _MATHVISTA_SINGLE_TYPE_SOURCE:
            answer_type = _MATHVISTA_SINGLE_TYPE_SOURCE[source]
        if source and answer_type:
            return f"mathvista_{_safe_router_task([source, answer_type])}"

    if dataset_name in {
        "scienceqa_train",
        "ai2d_train",
        "semantic_choice_augmented",
    } or broad_task == "mmstar":
        l2_category = _metadata_value(
            item,
            "l2_category",
            "metadata.l2_category",
            "metadata_l2_category",
        )
        if l2_category:
            return f"mmstar_l2_{_safe_router_task(l2_category)}"
        if dataset_name in _MMSTAR_SOURCE_L2_PROXY:
            return f"mmstar_l2_{_safe_router_task(_MMSTAR_SOURCE_L2_PROXY[dataset_name])}"

    return broad_task


def router_sample_tasks_from_indexes(
    data_indexes: list[dict] | None,
    *,
    target_len: int | None = None,
    default_task: str = "understanding",
) -> list[str] | None:
    """Map packed training samples to benchmark-family router tasks."""

    if not data_indexes:
        if target_len is not None and target_len > 0:
            return [default_task] * target_len
        return None
    tasks = []
    for item in data_indexes:
        tasks.append(
            router_metadata_task_from_index(item, default_task=default_task)
        )
    if target_len is not None:
        if len(tasks) < target_len:
            tasks.extend([default_task] * (target_len - len(tasks)))
        elif len(tasks) > target_len:
            tasks = tasks[:target_len]
    return tasks


def checkpoint_has_residual_gates(
    resume_from: str | None,
    *,
    resume_from_ema: bool,
) -> bool:
    """Inspect a resume checkpoint before constructing the decoder layers.

    Residual gates are optional parameters in ``Qwen2MoTDecoderLayer``.  They
    therefore need to be enabled in the LLM config before the model exists;
    otherwise a gated checkpoint would silently report them as unexpected
    keys and generation-only adaptation would train a different model.
    """

    if not resume_from:
        return False
    checkpoint = Path(resume_from)
    if checkpoint.is_dir():
        filename = "ema.safetensors" if resume_from_ema else "model.safetensors"
        checkpoint = checkpoint / filename
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        return False
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        return any(
            key.endswith((".gen_attn_gate", ".gen_mlp_gate"))
            for key in handle.keys()
        )


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="/path/to/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    dynamic_depth_config: Optional[str] = field(
        default="configs/training.json",
        metadata={
            "help": (
                "Training dynamic-depth JSON. Pass an empty string to retain "
                "the original full-depth backbone."
            )
        },
    )
    model_init_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "Default parameter dtype used while constructing BAGEL."
        },
    )
    skip_pretrained_init: bool = field(
        default=False,
        metadata={
            "help": (
                "When finetuning from a complete BAGEL checkpoint, construct "
                "the pretrained Qwen/SigLIP modules without random weight "
                "initialization. Newly added Training modules still initialize "
                "normally after construction."
            )
        },
    )
    llm_path: str = field(
        default="/path/to/language-model",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="/path/to/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="/path/to/vision-model",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/training.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )
    dynamic_depth_task: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional named Training route, such as vqa or visual_reasoning."
        },
    )
    training_stage: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Current Training staged trainability policy: understanding, "
                "generation, or joint. Leave unset for legacy policies."
            )
        },
    )
    train_generation_base_ffn: bool = field(
        default=False,
        metadata={
            "help": (
                "When using the generation Training stage, also train the "
                "generation branch's base FFN parameters."
            )
        },
    )
    reinitialize_depth_router: bool = field(
        default=False,
        metadata={
            "help": (
                "Reset only the small depth router after loading a checkpoint; "
                "useful for a separate prompt-only calibration stage."
            )
        },
    )
    reinitialize_generation_depth_router: bool = field(
        default=False,
        metadata={
            "help": (
                "Reset only the generation depth router after loading a "
                "checkpoint; useful when training generation exit selection "
                "after an understanding-router stage."
            )
        },
    )
    allow_mismatched_resume_shapes: bool = field(
        default=False,
        metadata={
            "help": (
                "Drop resume tensors whose shapes differ from the current "
                "model, retaining the current initialization for those keys."
            )
        },
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    router_lr: float = field(
        default=1e-4,
        metadata={
            "help": "Peak learning rate for newly initialized depth-router parameters."
        },
    )
    tafe_lr: float = field(
        default=1e-4,
        metadata={
            "help": "Peak learning rate for Training TAFE gates and specialized FFN banks."
        },
    )
    fusion_lr: float = field(
        default=1e-4,
        metadata={
            "help": "Peak learning rate for Training fusion-sealing modules."
        },
    )
    adapter_lr: float = field(
        default=1e-4,
        metadata={
            "help": (
                "Peak learning rate for mode-isolated understanding adapters."
            )
        },
    )
    adapter_weight_decay: float = field(
        default=0.0,
        metadata={
            "help": (
                "Decoupled AdamW weight decay applied only to understanding "
                "adapter/LoRA/fusion parameters."
            )
        },
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    use_ema: bool = field(
        default=True,
        metadata={"help": "Maintain and save a separate FSDP EMA model."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    cross_modal_recon_weight: float = field(
        default=0.0,
        metadata={"help": "Scaling factor for cross-modal latent reconstruction loss."}
    )
    cross_modal_recon_vit_mask_prob: float = field(
        default=0.2,
        metadata={"help": "Probability of masking a ViT token sequence entry for reconstruction target."}
    )
    cross_modal_recon_vae_mask_prob: float = field(
        default=0.2,
        metadata={"help": "Probability of masking a VAE token sequence entry for reconstruction target."}
    )
    cross_modal_recon: bool = field(
        default=False,
        metadata={"help": "Enable cross-modal reconstruction between ViT and VAE token pathways."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    multi_exit_ce_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for CE averaged across the router's candidate exits."
        },
    )
    exit_hidden_distill_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for cosine distillation from the full-depth "
                "representation into earlier candidate exits."
            )
        },
    )
    router_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for learned halting quality/compute objective."
        },
    )
    tafe_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the differentiable TAFE FFN-routing objective."
        },
    )
    fusion_sealing_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": "Weight for the Training continue-versus-seal objective."
        },
    )
    understanding_delta_l2_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for an L2 anchor on zero-init understanding "
                "adapter/LoRA/fusion parameters. This keeps adapter-only "
                "tuning close to the loaded checkpoint behavior."
            )
        },
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )
    use_orig_params: bool = field(
        default=False,
        metadata={
            "help": "Use FSDP original parameters for mixed frozen/trainable modules."
        },
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    trainable_param_keywords: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated substrings. When set, only matching "
                "BAGEL parameters remain trainable."
            )
        },
    )
    trainable_param_exclude_keywords: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated substrings excluded after the positive "
                "trainable_param_keywords filter."
            )
        },
    )
    use_residual_gates: bool = field(
        default=False,
        metadata={"help": "Instantiate task-specific scalar residual gates."},
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    reinitialize_understanding_deltas: bool = field(
        default=False,
        metadata={
            "help": (
                "Reset understanding adapter/LoRA/fusion modules after loading "
                "the resume checkpoint. Use when adding zero-init deltas on top "
                "of a checkpoint while skip_pretrained_init is enabled."
            )
        },
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )


def main():
    assert torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = local_rank
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier(device_ids=[device])
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Construct the 7B-MoT checkpoint directly in BF16 by default. This avoids
    # a large transient FP32 allocation on every distributed rank.
    original_default_dtype = torch.get_default_dtype()
    try:
        init_dtype = getattr(torch, model_args.model_init_dtype)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported model_init_dtype={model_args.model_init_dtype!r}"
        ) from exc
    if init_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("model_init_dtype must be float32, float16, or bfloat16")
    torch.set_default_dtype(init_dtype)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    llm_config.use_residual_gates = training_args.use_residual_gates or checkpoint_has_residual_gates(
        resume_from,
        resume_from_ema=finetune_from_ema,
    )
    if llm_config.use_residual_gates:
        logger.info("Enabled residual gates found in the resume checkpoint.")
    skip_pretrained_init = (
        bool(model_args.skip_pretrained_init)
        and training_args.finetune_from_hf
        and resume_from is not None
        and os.path.exists(resume_from)
    )
    if skip_pretrained_init:
        logger.info(
            "Skipping random initialization for pretrained Qwen/SigLIP "
            "modules before loading the resume checkpoint."
        )
    def pretrained_init_context():
        return (
            no_init_weights(_enable=True)
            if skip_pretrained_init
            else nullcontext()
        )

    if training_args.finetune_from_hf:
        with pretrained_init_context():
            language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            with pretrained_init_context():
                vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        cross_modal_recon=training_args.cross_modal_recon,
        cross_modal_recon_vit_mask_prob=training_args.cross_modal_recon_vit_mask_prob,
        cross_modal_recon_vae_mask_prob=training_args.cross_modal_recon_vae_mask_prob,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )
    dynamic_depth_enabled = bool(model_args.dynamic_depth_config)
    if dynamic_depth_enabled:
        depth_config_path = Path(model_args.dynamic_depth_config)
        if not depth_config_path.is_absolute():
            depth_config_path = Path(__file__).resolve().parents[1] / depth_config_path
        depth_config = DynamicDepthConfig.from_json(depth_config_path)
        model = enable_dynamic_depth(model, depth_config)
        logger.info(f"Enabled Training dynamic depth from {depth_config_path}")

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    torch.set_default_dtype(original_default_dtype)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if training_args.training_stage:
        staged_trainable = set_training_stage(
            model,
            training_args.training_stage,
            train_generation_base_ffn=training_args.train_generation_base_ffn,
        )
        logger.info(
            "Applied Training training stage %s: %d trainable tensors",
            training_args.training_stage,
            len(staged_trainable),
        )

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    if training_args.trainable_param_keywords:
        trainable_keywords = tuple(
            keyword.strip()
            for keyword in training_args.trainable_param_keywords.split(",")
            if keyword.strip()
        )
        if not trainable_keywords:
            raise ValueError("trainable_param_keywords did not contain a keyword")
        exclude_keywords = tuple(
            keyword.strip()
            for keyword in (training_args.trainable_param_exclude_keywords or "").split(",")
            if keyword.strip()
        )
        for name, param in model.named_parameters():
            param.requires_grad = param.requires_grad and any(
                keyword in name for keyword in trainable_keywords
            ) and not any(keyword in name for keyword in exclude_keywords)
        logger.info(
            "Restricted training to parameter names containing: "
            f"{trainable_keywords}; excluded: {exclude_keywords}"
        )
        raw_tafe_parameters = [
            (name, int(parameter.numel()))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and is_tafe_parameter(name)
        ]
        logger.info(
            "Raw TAFE parameter audit: tensors=%d numel=%d sample=%s",
            len(raw_tafe_parameters),
            sum(numel for _, numel in raw_tafe_parameters),
            [name for name, _ in raw_tafe_parameters[:4]],
        )

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
        use_orig_params=training_args.use_orig_params,
    )
    ema_model = deepcopy(model) if training_args.use_ema else None
    drop_ckpt_prefixes = []
    if training_args.reinitialize_depth_router:
        drop_ckpt_prefixes.append("language_model.model.depth_router.")
    if training_args.reinitialize_generation_depth_router:
        drop_ckpt_prefixes.append(
            "language_model.model.generation_depth_router."
        )
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from,
        logger,
        model,
        ema_model,
        resume_from_ema=finetune_from_ema,
        drop_prefixes=tuple(drop_ckpt_prefixes),
        drop_mismatched_shapes=training_args.allow_mismatched_resume_shapes,
    )
    if training_args.reinitialize_understanding_deltas:
        reset_understanding_delta_modules(model, logger)
        if ema_model is not None:
            reset_understanding_delta_modules(ema_model, logger)
    if training_args.reinitialize_depth_router:
        routers = [getattr(model.language_model.model, "depth_router", None)]
        if ema_model is not None:
            routers.append(
                getattr(ema_model.language_model.model, "depth_router", None)
            )
        if any(router is None for router in routers):
            raise ValueError(
                "reinitialize_depth_router requires a router-enabled depth config"
            )
        for router in routers:
            router.reset_parameters()
        logger.info("Reinitialized Training depth-router parameters.")
    if training_args.reinitialize_generation_depth_router:
        routers = [
            getattr(
                model.language_model.model,
                "generation_depth_router",
                None,
            )
        ]
        if ema_model is not None:
            routers.append(
                getattr(
                    ema_model.language_model.model,
                    "generation_depth_router",
                    None,
                )
            )
        if any(router is None for router in routers):
            raise ValueError(
                "reinitialize_generation_depth_router requires a config with "
                "router_candidates_generation"
            )
        for router in routers:
            router.reset_parameters()
        logger.info("Reinitialized Training generation-router parameters.")
    if ema_model is not None:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    fsdp_tafe_parameters = [
        (name, int(parameter.numel()))
        for name, parameter in fsdp_model.named_parameters()
        if parameter.requires_grad
        and is_tafe_parameter(name)
    ]
    logger.info(
        "FSDP TAFE parameter audit: tensors=%d numel=%d sample=%s",
        len(fsdp_tafe_parameters),
        sum(numel for _, numel in fsdp_tafe_parameters),
        [name for name, _ in fsdp_tafe_parameters[:4]],
    )
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    trainable_param_count = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    logger.info(
        f"Trainable parameter count: {trainable_param_count / 1e9:.2f}B "
        f"({100 * trainable_param_count / total_param_count:.1f}% of BAGEL)"
    )

    # Setup optimizer and scheduler
    router_parameters = []
    tafe_parameters = []
    fusion_parameters = []
    adapter_parameters = []
    backbone_parameters = []
    for name, parameter in fsdp_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            "fusion_sealing_" in name
            or "fusion_release_controller" in name
        ):
            target = fusion_parameters
        elif is_tafe_parameter(name):
            target = tafe_parameters
        elif "depth_router" in name:
            target = router_parameters
        elif is_understanding_delta_parameter(name):
            target = adapter_parameters
        else:
            target = backbone_parameters
        target.append(parameter)
    if training_args.understanding_delta_l2_weight > 0 and not adapter_parameters:
        raise ValueError(
            "understanding_delta_l2_weight was set, but no trainable "
            "understanding delta parameters were found."
        )
    optimizer_groups = []
    if backbone_parameters:
        optimizer_groups.append(
            {"params": backbone_parameters, "lr": training_args.lr, "weight_decay": 0.0}
        )
    if adapter_parameters:
        optimizer_groups.append(
            {
                "params": adapter_parameters,
                "lr": training_args.adapter_lr,
                "weight_decay": training_args.adapter_weight_decay,
            }
        )
    if router_parameters:
        optimizer_groups.append(
            {"params": router_parameters, "lr": training_args.router_lr, "weight_decay": 0.0}
        )
    if tafe_parameters:
        optimizer_groups.append(
            {"params": tafe_parameters, "lr": training_args.tafe_lr, "weight_decay": 0.0}
        )
    if fusion_parameters:
        optimizer_groups.append(
            {"params": fusion_parameters, "lr": training_args.fusion_lr, "weight_decay": 0.0}
        )
    logger.info(
        f"Optimizer parameter groups: backbone={len(backbone_parameters)} "
        f"(lr={training_args.lr:g}), adapters={len(adapter_parameters)} "
        f"(lr={training_args.adapter_lr:g}, wd={training_args.adapter_weight_decay:g}), "
        f"router={len(router_parameters)} "
        f"(lr={training_args.router_lr:g})"
        f", tafe={len(tafe_parameters)} "
        f"(lr={training_args.tafe_lr:g})"
        f", fusion={len(fusion_parameters)} "
        f"(lr={training_args.fusion_lr:g})"
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        # Modal's mounted volume filesystem does not support the Unix socket
        # used by multiprocessing's resource sharer.  With no DataLoader
        # workers, PyTorch requires prefetch_factor to be None.
        prefetch_factor=(
            data_args.prefetch_factor if data_args.num_workers > 0 else None
        ),
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        if curr_step == train_step and dist.get_rank() == 0:
            logger.info(
                "Initial batch index audit: und=%d gen=%d ce=%d mse=%d",
                len(data.get("packed_und_token_indexes", [])),
                len(data.get("packed_gen_token_indexes", [])),
                len(data.get("ce_loss_indexes", [])),
                len(data.get("mse_loss_indexes", [])),
            )
            logger.info(
                "Initial modality index audit: text=%d vit=%d vae=%d mse_true=%d",
                len(data.get("packed_text_indexes", [])),
                len(data.get("packed_vit_token_indexes", [])),
                len(data.get("packed_vae_token_indexes", [])),
                int(data.get("mse_loss_indexes", torch.zeros(0, dtype=torch.bool)).sum()),
            )
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen and 'padded_images' in data:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                dynamic_kwargs = {}
                if dynamic_depth_enabled:
                    if training_args.dynamic_depth_task:
                        dynamic_kwargs["depth_task"] = training_args.dynamic_depth_task
                    sample_tasks = router_sample_tasks_from_indexes(
                        data_indexes,
                        target_len=len(data.get("sample_lens") or []),
                        default_task=training_args.dynamic_depth_task
                        or "understanding",
                    )
                    if sample_tasks is not None:
                        dynamic_kwargs["depth_sample_tasks"] = sample_tasks
                        if curr_step == train_step and dist.get_rank() == 0:
                            logger.info(
                                "Initial router task audit: %s",
                                dict(Counter(sample_tasks).most_common(20)),
                            )
                loss_dict = fsdp_model(**data, **dynamic_kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        multi_exit_ce = loss_dict.pop("multi_exit_ce", None)
        if multi_exit_ce is not None:
            if training_args.ce_loss_reweighting:
                multi_exit_ce = multi_exit_ce * ce_loss_weights
                multi_exit_ce = (
                    multi_exit_ce.sum()
                    * dist.get_world_size()
                    / total_ce_loss_weights
                )
            else:
                multi_exit_ce = (
                    multi_exit_ce.sum()
                    * dist.get_world_size()
                    / total_ce_tokens
                )
            loss = (
                loss
                + multi_exit_ce * training_args.multi_exit_ce_weight
            )
            loss_dict["multi_exit_ce"] = multi_exit_ce.detach()

        exit_hidden_distill = loss_dict.pop("exit_hidden_distill", None)
        if exit_hidden_distill is not None:
            loss = (
                loss
                + exit_hidden_distill
                * training_args.exit_hidden_distill_weight
            )
            loss_dict["exit_hidden_distill"] = exit_hidden_distill.detach()

        router_loss = loss_dict.pop("router_loss", None)
        if router_loss is not None:
            loss = loss + router_loss * training_args.router_loss_weight
            loss_dict["router_loss"] = router_loss.detach()

        tafe_loss = loss_dict.pop("tafe_loss", None)
        if tafe_loss is not None:
            loss = loss + tafe_loss * training_args.tafe_loss_weight
            loss_dict["tafe_loss"] = tafe_loss.detach()

        fusion_sealing_loss = loss_dict.pop("fusion_sealing_loss", None)
        if fusion_sealing_loss is not None:
            loss = (
                loss
                + fusion_sealing_loss
                * training_args.fusion_sealing_loss_weight
            )
            loss_dict["fusion_sealing_loss"] = fusion_sealing_loss.detach()

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            if mse is not None:
                total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
                dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
                mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
                loss_dict["mse"] = mse.detach()
                loss = loss + mse * training_args.mse_weight
            else:
                loss_dict["mse"] = torch.tensor(0, device=device)
                total_mse_tokens = torch.tensor(0, device=device)
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        cross_modal_recon = loss_dict.pop("cross_modal_recon", None)
        if cross_modal_recon is not None:
            if cross_modal_recon.numel() > 0:
                total_cross_modal_recon_tokens = torch.tensor(len(cross_modal_recon), device=device)
                dist.all_reduce(total_cross_modal_recon_tokens, op=dist.ReduceOp.SUM)
                cross_modal_recon_sum = cross_modal_recon.sum()
                dist.all_reduce(cross_modal_recon_sum, op=dist.ReduceOp.SUM)
                cross_modal_recon = cross_modal_recon_sum / total_cross_modal_recon_tokens
                loss_dict["cross_modal_recon"] = cross_modal_recon.detach()
                loss = loss + cross_modal_recon * training_args.cross_modal_recon_weight
            else:
                loss_dict["cross_modal_recon"] = torch.tensor(0, device=device)
                total_cross_modal_recon_tokens = torch.tensor(0, device=device)
        else:
            loss_dict["cross_modal_recon"] = torch.tensor(0, device=device)
            total_cross_modal_recon_tokens = torch.tensor(0, device=device)

        if training_args.understanding_delta_l2_weight > 0:
            delta_l2_sum = None
            delta_l2_count = 0
            for parameter in adapter_parameters:
                if not parameter.requires_grad:
                    continue
                current_sum = parameter.float().square().sum()
                delta_l2_sum = (
                    current_sum
                    if delta_l2_sum is None
                    else delta_l2_sum + current_sum
                )
                delta_l2_count += parameter.numel()
            if delta_l2_sum is None or delta_l2_count == 0:
                delta_l2 = torch.tensor(0.0, device=device)
            else:
                delta_l2 = delta_l2_sum / delta_l2_count
            loss = loss + delta_l2 * training_args.understanding_delta_l2_weight
            loss_dict["understanding_delta_l2"] = delta_l2.detach()

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if micro_step == 0:
            gate_grad_norm = 0.0
            gate_grad_count = 0
            tafe_grad_norm = 0.0
            tafe_grad_count = 0
            tafe_trainable_count = 0
            for name, parameter in fsdp_model.named_parameters():
                if is_tafe_parameter(name):
                    if parameter.requires_grad:
                        tafe_trainable_count += int(parameter.numel())
                    if parameter.grad is not None:
                        tafe_grad_norm += float(
                            parameter.grad.detach().float().abs().sum().item()
                        )
                        tafe_grad_count += int(parameter.grad.numel())
                if "_gate" in name and parameter.grad is not None:
                    gate_grad_norm += float(parameter.grad.detach().float().abs().sum().item())
                    gate_grad_count += int(parameter.grad.numel())
            audit = torch.tensor(
                [
                    tafe_trainable_count,
                    tafe_grad_count,
                    tafe_grad_norm,
                    gate_grad_count,
                    gate_grad_norm,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(audit, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                logger.info(
                    "Initial residual-gate gradient audit: tensors=%d abs_grad_sum=%.6g",
                    int(audit[3].item()),
                    audit[4].item(),
                )
                logger.info(
                    "Initial TAFE gradient audit: trainable_numel=%d "
                    "grad_numel=%d abs_grad_sum=%.6g",
                    int(audit[0].item()),
                    int(audit[1].item()),
                    audit[2].item(),
                )

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            if os.environ.get("TRAINING_DEBUG_FINITE", "0") == "1":
                bad_gradients = []
                for name, parameter in fsdp_model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            bad_gradients.append(name)
                if bad_gradients:
                    print(
                        f"[finite-debug rank={dist.get_rank()}] "
                        f"non-finite gradients: {bad_gradients[:50]}",
                        flush=True,
                    )
                    logger.error(
                        "Non-finite gradients before optimizer step: %s",
                        bad_gradients[:20],
                    )
                    raise FloatingPointError(
                        "non-finite gradients before optimizer step"
                    )
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            if (
                os.environ.get("TRAINING_DEBUG_FINITE", "0") == "1"
                and not torch.isfinite(total_norm)
            ):
                raise FloatingPointError(
                    f"non-finite clipped gradient norm: {total_norm.item()}"
                )
            optimizer.step()
            if os.environ.get("TRAINING_DEBUG_FINITE", "0") == "1":
                bad_parameters = []
                for name, parameter in fsdp_model.named_parameters():
                    if parameter.requires_grad and not torch.isfinite(parameter).all():
                        bad_parameters.append(name)
                if bad_parameters:
                    logger.error(
                        "Non-finite parameters after optimizer step: %s",
                        bad_parameters[:20],
                    )
                    raise FloatingPointError(
                        "non-finite parameters after optimizer step"
                    )
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            if len(optimizer.param_groups) > 1:
                wandb_log['router_lr'] = optimizer.param_groups[1]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_cross_modal_recon_tokens'] = total_cross_modal_recon_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list
        )
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
