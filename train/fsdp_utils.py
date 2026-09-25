# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
        use_orig_params=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.use_orig_params = use_orig_params


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=fsdp_config.use_orig_params,
    )


def adapt_tafe_task_embedding_state(
    state_dict,
    expected_state,
    *,
    task_names: Sequence[str] | None = None,
    logger=None,
):
    """Expand a TAFE task embedding while preserving named family rows.

    Older lightweight checkpoints use a compact vocabulary (for example,
    ``unknown, understanding, mmbench, mmmu, mathvista, mmstar, vqa``), while
    granular runs insert language and subject labels into the destination
    vocabulary.  Positional copying would silently assign the old MMMU row to
    MMBench-CN, etc.  Map the legacy rows by name and initialize every new
    fine-grained label from its longest registered benchmark-family prefix.
    """

    legacy_by_size = {
        7: (
            "unknown",
            "understanding",
            "mmbench",
            "mmmu",
            "mathvista",
            "mmstar",
            "vqa",
        ),
        9: (
            "unknown",
            "understanding",
            "mmbench",
            "mmbench_en",
            "mmbench_cn",
            "mmmu",
            "mathvista",
            "mmstar",
            "vqa",
        ),
    }

    def source_index(name: str, source_names: tuple[str, ...]) -> int:
        normalized = str(name).strip().lower().replace("-", "_")
        exact = {value: index for index, value in enumerate(source_names)}
        if normalized in exact:
            return exact[normalized]
        candidates = [
            (value, index)
            for value, index in exact.items()
            if value != "unknown"
            and (
                normalized.startswith(f"{value}_")
                or normalized.startswith(f"{value}:")
            )
        ]
        if candidates:
            return max(candidates, key=lambda item: len(item[0]))[1]
        return 0

    for key in list(state_dict):
        if not key.endswith(
            (
                "tafe_gate_understanding.task_embedding.weight",
                "tafe_attention_gate_understanding.task_embedding.weight",
            )
        ):
            continue
        expected = expected_state.get(key)
        value = state_dict[key]
        if (
            expected is None
            or not hasattr(value, "shape")
            or value.ndim != 2
            or expected.ndim != 2
            or value.shape[1] != expected.shape[1]
            or value.shape == expected.shape
        ):
            continue
        adapted = expected.detach().clone()
        source_names = legacy_by_size.get(value.shape[0])
        if source_names is None or task_names is None:
            rows = min(value.shape[0], expected.shape[0])
            adapted[:rows].copy_(value[:rows].to(dtype=adapted.dtype))
            mapping = f"prefix-{rows}"
        else:
            mapping_parts = []
            for destination_index, name in enumerate(task_names):
                if destination_index >= expected.shape[0]:
                    break
                source_row = source_index(name, source_names)
                if source_row < value.shape[0]:
                    adapted[destination_index].copy_(
                        value[source_row].to(dtype=adapted.dtype)
                    )
                    mapping_parts.append(f"{name}->{source_names[source_row]}")
            mapping = ", ".join(mapping_parts[:8])
        state_dict[key] = adapted
        if logger is not None:
            logger.info(
                "Expanded TAFE task embedding %s from %s to %s using named "
                "family initialization (%s).",
                key,
                tuple(value.shape),
                tuple(expected.shape),
                mapping,
            )
    return state_dict


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(
        resume_from,
        logger,
        model,
        ema_model=None,
        resume_from_ema=False,
        drop_prefixes=(),
        drop_mismatched_shapes=False,
    ):
        def drop_keys(state_dict):
            if not drop_prefixes:
                return
            for key in list(state_dict):
                if any(key.startswith(prefix) for prefix in drop_prefixes):
                    state_dict.pop(key)

        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            resume_path = os.fspath(resume_from)
            if resume_path.endswith(".safetensors"):
                # Post-training and materialized Training checkpoints are often
                # distributed as one standalone safetensors file rather than
                # an FSDP training directory.
                model_state_dict_path = resume_path
            elif resume_from_ema:
                model_state_dict_path = os.path.join(resume_path, "ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_path, "model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed', None)
            model_state_dict.pop('vit_pos_embed.pos_embed', None)
            drop_keys(model_state_dict)
            # A granular TAFE vocabulary can add benchmark/subject labels to
            # an existing lightweight checkpoint.  Preserve the rows that
            # already existed (unknown, understanding, and the broad family
            # labels) while retaining the current initialization for newly
            # added task rows.  This makes a TAFE vocabulary expansion a real
            # continuation instead of silently reinitializing the learned
            # controller embedding.
            model_state = model.state_dict()
            tafe_gate = getattr(
                getattr(getattr(model, "language_model", None), "model", None),
                "tafe_gate_understanding",
                None,
            )
            if tafe_gate is None:
                tafe_gate = getattr(
                    getattr(
                        getattr(model, "language_model", None),
                        "model",
                        None,
                    ),
                    "tafe_attention_gate_understanding",
                    None,
                )
            adapt_tafe_task_embedding_state(
                model_state_dict,
                model_state,
                task_names=getattr(tafe_gate, "task_names", None),
                logger=logger,
            )
            if drop_mismatched_shapes:
                mismatched = []
                for key in list(model_state_dict):
                    expected = model_state.get(key)
                    value = model_state_dict[key]
                    if (
                        expected is not None
                        and hasattr(value, "shape")
                        and tuple(value.shape) != tuple(expected.shape)
                    ):
                        mismatched.append(
                            (key, tuple(value.shape), tuple(expected.shape))
                        )
                        model_state_dict.pop(key)
                for key, loaded_shape, expected_shape in mismatched:
                    logger.warning(
                        "Dropping mismatched resume tensor %s: checkpoint=%s, "
                        "model=%s; the model initialization will be retained.",
                        key,
                        loaded_shape,
                        expected_shape,
                    )
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_path, "ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed', None)
                ema_state_dict.pop('vit_pos_embed.pos_embed', None)
                drop_keys(ema_state_dict)
                if drop_mismatched_shapes:
                    ema_state = ema_model.state_dict()
                    for key in list(ema_state_dict):
                        expected = ema_state.get(key)
                        value = ema_state_dict[key]
                        if (
                            expected is not None
                            and hasattr(value, "shape")
                            and tuple(value.shape) != tuple(expected.shape)
                        ):
                            ema_state_dict.pop(key)
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info("Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
