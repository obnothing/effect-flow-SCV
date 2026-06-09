#!/usr/bin/env bash
set -e

CONFIG_PATH=${1:-configs/train_mlsmote_codebert_server.yaml}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MODEL_DIR="./models/codebert-base"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] config: ${CONFIG_PATH}"
echo "[INFO] nproc_per_node: ${NPROC_PER_NODE}"
echo "[INFO] checking GPU..."
nvidia-smi || true

if [ ! -d "${MODEL_DIR}" ]; then
  echo "[ERROR] local CodeBERT model directory not found: ${MODEL_DIR}"
  echo "[ERROR] copy/download the model to ${MODEL_DIR} before training."
  exit 1
fi

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  src/train.py \
  --config "${CONFIG_PATH}"
