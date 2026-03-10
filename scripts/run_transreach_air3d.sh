#!/usr/bin/env bash
set -euo pipefail

# TransReach experiment suite v2 for an Ubuntu GPU machine.
# Sweep:
# 1) DeepReach-SIREN baseline
# 2) TransReach-v2 k=5, dt=0.02
# 3) TransReach-v2 k=5, dt=0.05
# 4) TransReach-v2 k=5, dt=0.10
# 5) TransReach-v2 k=3, dt=0.05
# 6) TransReach-v2 k=7, dt=0.05
#
# Usage:
#   chmod +x scripts/run_transreach_air3d_ubuntu.sh
#   ./scripts/run_transreach_air3d_ubuntu.sh
#
# Optional overrides:
#   DEVICE=cuda:0 EPOCHS=100000 RUN_ROOT=./runs ./scripts/run_transreach_air3d_ubuntu.sh

DEVICE="${DEVICE:-cuda:0}"
RUN_ROOT="${RUN_ROOT:-./runs}"
EPOCHS="${EPOCHS:-100000}"
NUMPOINTS="${NUMPOINTS:-2000}"
SEED="${SEED:-0}"

# Air3D parameters from DeepReach paper defaults
COLLISION_R="${COLLISION_R:-0.25}"
VELOCITY="${VELOCITY:-0.75}"
OMEGA_MAX="${OMEGA_MAX:-3.0}"
ANGLE_ALPHA="${ANGLE_ALPHA:-1.2}"

WANDB_ENTITY="${WANDB_ENTITY:-oxcarxierra-seoul-national-university}"
WANDB_GROUP="air3d_transreach_v2"

COMMON_BASE=(
  --mode train
  --experiments_dir "${RUN_ROOT}"
  --experiment_class DeepReach
  --dynamics_class Air3D
  --minWith target
  --collisionR "${COLLISION_R}"
  --velocity "${VELOCITY}"
  --omega_max "${OMEGA_MAX}"
  --angle_alpha_factor "${ANGLE_ALPHA}"
  --numpoints "${NUMPOINTS}"
  --num_src_samples 1000
  --seed "${SEED}"
  --device "${DEVICE}"
  --deepreach_model exact
  --batch_size 1
  --steps_til_summary 100
  --epochs_til_ckpt 2000
  --lr 2e-5
)

WANDB_ARGS=(
  --use_wandb
  --wandb_project deepreach
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_group "${WANDB_GROUP}"
)

# echo "[1/6] Baseline DeepReach-SIREN"
# python run_experiment.py \
#   "${COMMON_BASE[@]}" \
#   "${WANDB_ARGS[@]}" \
#   --experiment_name air3d_baseline_siren \
#   --wandb_name air3d_baseline_siren \
#   --model sine \
#   --num_nl 512 \
#   --num_hl 3 \
#   --num_epochs "${EPOCHS}"

echo "[2/6] TransReach-v2 k=5, dt=0.02"
python run_experiment_transreach.py \
  "${COMMON_BASE[@]}" \
  "${WANDB_ARGS[@]}" \
  --experiment_name air3d_transreach_v2_k5_dt002 \
  --wandb_name air3d_transreach_v2_k5_dt002 \
  --num_nl 256 \
  --num_heads 2 \
  --num_encoder_layers 1 \
  --num_decoder_layers 1 \
  --pseudo_steps 5 \
  --pseudo_dt 0.02 \
  --num_epochs "${EPOCHS}"

echo "[3/6] TransReach-v2 k=5, dt=0.05"
python run_experiment_transreach.py \
  "${COMMON_BASE[@]}" \
  "${WANDB_ARGS[@]}" \
  --experiment_name air3d_transreach_v2_k5_dt005 \
  --wandb_name air3d_transreach_v2_k5_dt005 \
  --num_nl 256 \
  --num_heads 2 \
  --num_encoder_layers 1 \
  --num_decoder_layers 1 \
  --pseudo_steps 5 \
  --pseudo_dt 0.05 \
  --num_epochs "${EPOCHS}"

echo "[4/6] TransReach-v2 k=5, dt=0.10"
python run_experiment_transreach.py \
  "${COMMON_BASE[@]}" \
  "${WANDB_ARGS[@]}" \
  --experiment_name air3d_transreach_v2_k5_dt010 \
  --wandb_name air3d_transreach_v2_k5_dt010 \
  --num_nl 256 \
  --num_heads 2 \
  --num_encoder_layers 1 \
  --num_decoder_layers 1 \
  --pseudo_steps 5 \
  --pseudo_dt 0.10 \
  --num_epochs "${EPOCHS}"

echo "[5/6] TransReach-v2 k=3, dt=0.05"
python run_experiment_transreach.py \
  "${COMMON_BASE[@]}" \
  "${WANDB_ARGS[@]}" \
  --experiment_name air3d_transreach_v2_k3_dt005 \
  --wandb_name air3d_transreach_v2_k3_dt005 \
  --num_nl 256 \
  --num_heads 2 \
  --num_encoder_layers 1 \
  --num_decoder_layers 1 \
  --pseudo_steps 3 \
  --pseudo_dt 0.05 \
  --num_epochs "${EPOCHS}"

echo "[6/6] TransReach-v2 k=7, dt=0.05"
python run_experiment_transreach.py \
  "${COMMON_BASE[@]}" \
  "${WANDB_ARGS[@]}" \
  --experiment_name air3d_transreach_v2_k7_dt005 \
  --wandb_name air3d_transreach_v2_k7_dt005 \
  --num_nl 256 \
  --num_heads 2 \
  --num_encoder_layers 1 \
  --num_decoder_layers 1 \
  --pseudo_steps 7 \
  --pseudo_dt 0.05 \
  --num_epochs "${EPOCHS}"

echo "Finished Air3D TransReach-v2 sweep."
