set -euo pipefail
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  --rdzv_id=local_run \
  train_gpt.py
