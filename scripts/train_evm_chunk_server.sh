#!/usr/bin/env bash
set -e

CONFIG_PATH=${1:-configs/train_evm_chunk_server.yaml}

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] config: ${CONFIG_PATH}"
echo "[INFO] checking GPU..."
nvidia-smi || true

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python src/train.py --config "${CONFIG_PATH}"
