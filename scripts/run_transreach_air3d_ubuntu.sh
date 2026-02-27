#!/usr/bin/env bash
set -euo pipefail

# TransReach experiment suite for an Ubuntu GPU machine.
# This script executes a 1-9 sweep:
# 1) DeepReach-SIREN baseline
# 2) TransReach (k=1, no decoder; near-pointwise ablation)
# 3) TransReach (k=3, dt=1e-3)
# 4) TransReach (k=5, dt=1e-3)
# 5) TransReach (k=7, dt=1e-3)
# 6) TransReach (k=5, dt=1e-2)
# 7) TransReach (k=5, dt=1e-3)
# 8) TransReach (k=5, dt=1e-4)
# 9) TransReach final run (k=5, dt=1e-3, longer training)
#
# Usage:
#   chmod +x scripts/run_transreach_air3d_ubuntu.sh
#   ./scripts/run_transreach_air3d_ubuntu.sh
#
# Optional overrides:
#   DEVICE=cuda:0 EPOCHS=30000 FINAL_EPOCHS=120000 RUN_ROOT=./runs ./scripts/run_transreach_air3d_ubuntu.sh

DEVICE="${DEVICE:-cuda:0}"
RUN_ROOT="${RUN_ROOT:-./runs}"
EPOCHS="${EPOCHS:-100000}"
FINAL_EPOCHS="${FINAL_EPOCHS:-200000}"
NUMPOINTS="${NUMPOINTS:-2000}"
SEED="${SEED:-0}"

# Air3D parameters from DeepReach paper defaults
COLLISION_R="${COLLISION_R:-0.25}"
VELOCITY="${VELOCITY:-0.75}"
OMEGA_MAX="${OMEGA_MAX:-3.0}"
ANGLE_ALPHA="${ANGLE_ALPHA:-1.2}"

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

# echo "[1/9] Baseline DeepReach-SIREN"
# python run_experiment.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_baseline_siren \
#   --model sine \
#   --num_nl 512 \
#   --num_hl 3 \
#   --num_epochs "${EPOCHS}"

# echo "[2/9] TransReach ablation k=1 (near pointwise)"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_k1_nodecoder \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 0 \
#   --pseudo_steps 1 \
#   --pseudo_dt 1e-3 \
#   --num_epochs "${EPOCHS}" \
#   --use_wandb \
#   --wandb_project deepreach \
#   --wandb_entity oxcarxierra-seoul-national-university \
#   --wandb_group air3d_transreach_run \
#   --wandb_name air3d_transreach_k1_nodecoder

echo "[3/9] TransReach k=3, dt=1e-3"
python run_experiment_transreach.py \
  "${COMMON_BASE[@]}" \
  --experiment_name air3d_transreach_k3_dt1e3 \
  --num_nl 256 \
  --num_heads 2 \
  --num_encoder_layers 1 \
  --num_decoder_layers 1 \
  --pseudo_steps 3 \
  --pseudo_dt 1e-3 \
  --num_epochs "${EPOCHS}" \
  --use_wandb \
  --wandb_project deepreach \
  --wandb_entity oxcarxierra-seoul-national-university \
  --wandb_group air3d_transreach_run \
  --wandb_name air3d_transreach_k3_dt1e3

# echo "[4/9] TransReach k=5, dt=1e-3"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_k5_dt1e3 \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 1 \
#   --pseudo_steps 5 \
#   --pseudo_dt 1e-3 \
#   --num_epochs "${EPOCHS}" \
  # --use_wandb \
  # --wandb_project deepreach \
  # --wandb_entity oxcarxierra-seoul-national-university \
  # --wandb_group air3d_transreach_run \
  # --wandb_name air3d_transreach_k5_dt1e3

# echo "[5/9] TransReach k=7, dt=1e-3"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_k7_dt1e3 \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 1 \
#   --pseudo_steps 7 \
#   --pseudo_dt 1e-3 \
#   --num_epochs "${EPOCHS}" \
#   --use_wandb \
#   --wandb_project deepreach \
#   --wandb_entity oxcarxierra-seoul-national-university \
#   --wandb_group air3d_transreach_run \
#   --wandb_name air3d_transreach_k7_dt1e3

# echo "[6/9] TransReach k=5, dt=1e-2"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_k5_dt1e2 \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 1 \
#   --pseudo_steps 5 \
#   --pseudo_dt 1e-2 \
#   --num_epochs "${EPOCHS}" \
#   --use_wandb \
#   --wandb_project deepreach \
#   --wandb_entity oxcarxierra-seoul-national-university \
#   --wandb_group air3d_transreach_run \
#   --wandb_name air3d_transreach_k5_dt1e2

# echo "[7/9] TransReach k=5, dt=1e-3 (repeat seed for stability)"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_k5_dt1e3_repeat \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 1 \
#   --pseudo_steps 5 \
#   --pseudo_dt 1e-3 \
#   --num_epochs "${EPOCHS}" \
#   --use_wandb \
#   --wandb_project deepreach \
#   --wandb_entity oxcarxierra-seoul-national-university \
#   --wandb_group air3d_transreach_run \
#   --wandb_name air3d_transreach_k5_dt1e3_repeat

# echo "[8/9] TransReach k=5, dt=1e-4"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_k5_dt1e4 \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 1 \
#   --pseudo_steps 5 \
#   --pseudo_dt 1e-4 \
#   --num_epochs "${EPOCHS}" \
#   --use_wandb \
#   --wandb_project deepreach \
#   --wandb_entity oxcarxierra-seoul-national-university \
#   --wandb_group air3d_transreach_run \
#   --wandb_name air3d_transreach_k5_dt1e4

# echo "[9/9] Final TransReach long run (k=5, dt=1e-3)"
# python run_experiment_transreach.py \
#   "${COMMON_BASE[@]}" \
#   --experiment_name air3d_transreach_final \
#   --num_nl 256 \
#   --num_heads 2 \
#   --num_encoder_layers 1 \
#   --num_decoder_layers 1 \
#   --pseudo_steps 5 \
#   --pseudo_dt 1e-3 \
#   --num_epochs "${FINAL_EPOCHS}" \
#   --use_wandb \
#   --wandb_project deepreach \
#   --wandb_entity oxcarxierra-seoul-national-university \
#   --wandb_group air3d_transreach_run \
#   --wandb_name air3d_transreach_final

# echo "Finished 1-9 Air3D run suite."
