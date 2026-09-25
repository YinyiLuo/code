#!/usr/bin/env bash
set -euo pipefail

CODEBASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${CODEBASE_DIR}"

export BAGEL_DATA_ROOT="${BAGEL_DATA_ROOT:-/path/to/dataset}"
MODEL_PATH="${MODEL_PATH:-/path/to/BAGEL-7B-MoT}"
NUM_GPUS="${NUM_GPUS:-1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

mkdir -p results/training/logs results/training/checkpoints
export PYTHONPATH="${CODEBASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  train/pretrain_unified_navit.py \
  --model_path "${MODEL_PATH}" \
  --finetune_from_hf True \
  --resume_from "${MODEL_PATH}" \
  --resume_model_only True \
  --finetune_from_ema False \
  --copy_init_moe False \
  --dynamic_depth_config configs/training.json \
  --dataset_config_file data/configs/training.yaml \
  --training_stage joint \
  --visual_und True \
  --visual_gen True \
  --freeze_vit "${FREEZE_VIT:-True}" \
  --freeze_vae True \
  --model_init_dtype bfloat16 \
  --skip_pretrained_init True \
  --use_orig_params True \
  --sharding_strategy FULL_SHARD \
  --num_shard "${NUM_GPUS}" \
  --num_replicate 1 \
  --max_latent_size "${MAX_LATENT_SIZE:-64}" \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS:-4096}" \
  --max_num_tokens "${MAX_NUM_TOKENS:-6144}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE:-4096}" \
  --num_workers "${NUM_WORKERS:-1}" \
  --prefetch_factor "${PREFETCH_FACTOR:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --lr "${LR:-1e-5}" \
  --router_lr "${ROUTER_LR:-1e-4}" \
  --tafe_lr "${TAFE_LR:-1e-4}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --lr_scheduler "${LR_SCHEDULER:-constant}" \
  --ce_weight "${CE_WEIGHT:-1.0}" \
  --multi_exit_ce_weight "${MULTI_EXIT_CE_WEIGHT:-1.0}" \
  --exit_hidden_distill_weight "${EXIT_HIDDEN_DISTILL_WEIGHT:-0.5}" \
  --router_loss_weight "${ROUTER_LOSS_WEIGHT:-1.0}" \
  --tafe_loss_weight "${TAFE_LOSS_WEIGHT:-1.0}" \
  --mse_weight "${MSE_WEIGHT:-1.0}" \
  --total_steps "${TOTAL_STEPS:-1000}" \
  --save_every "${SAVE_EVERY:-200}" \
  --log_every "${LOG_EVERY:-1}" \
  --results_dir results/training/logs \
  --checkpoint_dir results/training/checkpoints \
  --wandb_project "${WANDB_PROJECT:-training}" \
  --wandb_name "${WANDB_NAME:-training}" \
  --wandb_offline "${WANDB_OFFLINE:-True}" \
  "$@"
