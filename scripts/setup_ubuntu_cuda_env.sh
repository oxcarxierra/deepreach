#!/usr/bin/env bash
set -euo pipefail

# Creates a fresh venv and installs DeepReach + CUDA PyTorch dependencies.
# Usage:
#   bash scripts/setup_ubuntu_cuda_env.sh
# Optional:
#   PYTHON=python3.10 CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_ubuntu_cuda_env.sh

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url "${CUDA_INDEX_URL}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
