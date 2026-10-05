#!/usr/bin/env bash
# Run only inside the user-authorized train_TTS Slurm allocation.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != smoke && "$1" != train ) ]]; then
  echo 'usage: run_books_sasrec_gpu.sh smoke|train' >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Refusing to run outside a Slurm step' >&2
  exit 2
fi
if [[ "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" != 'hoangnv242 RUNNING senvoice-pro-opt' ]]; then
  echo 'Unexpected reserved allocation' >&2; exit 2
fi

MEMREC_ROOT=/mnt/data/users/hoangnv242/memrec-hnv
REPO="$MEMREC_ROOT/repo/MemRec-hnv"
RUN_ID=sasrec-books-dev-v3-hnv
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
LOG_DIR="$MEMREC_ROOT/logs"
PYTHON="$MEMREC_ROOT/envs/sasrec-hnv/bin/python"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/$RUN_ID-$1.log") 2>&1
cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Refusing GPU run with dirty tracked worktree' >&2
  exit 2
fi

SNAPSHOT_BEFORE="$LOG_DIR/$RUN_ID-$1-gpu-before.csv"
SNAPSHOT_AFTER="$LOG_DIR/$RUN_ID-$1-gpu-after.csv"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$SNAPSHOT_BEFORE"
if [[ "$(wc -l < "$SNAPSHOT_BEFORE")" -lt 2 ]]; then
  echo 'Reserved GPU1 must be present' >&2
  exit 2
fi

GPU_INDEX=1
GPU_ROW=$(grep "^$GPU_INDEX," "$SNAPSHOT_BEFORE")
GPU_UTIL=$(printf '%s\n' "$GPU_ROW" | cut -d, -f2 | tr -d ' ')
GPU_MEMORY_MB=$(printf '%s\n' "$GPU_ROW" | cut -d, -f3 | tr -d ' ')
if [[ ! "$GPU_UTIL" =~ ^[0-9]+$ || ! "$GPU_MEMORY_MB" =~ ^[0-9]+$ ]]; then
  echo 'Malformed selected-GPU utilization/memory snapshot' >&2
  exit 2
fi
if (( GPU_UTIL >= 20 || GPU_MEMORY_MB >= 512 )); then
  echo 'No sufficiently idle H100 for a non-intrusive MemRec run' >&2
  exit 2
fi
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$LOG_DIR/$RUN_ID-$1-apps-before.csv"
GPU_UUID=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
if grep -Fq "$GPU_UUID" "$LOG_DIR/$RUN_ID-$1-apps-before.csv"; then
  echo 'GPU1 still has a compute process; require a verified handoff first' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TRITON_CACHE_DIR="$MEMREC_ROOT/cache/triton"

cleanup() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$SNAPSHOT_AFTER" || true
}
trap cleanup EXIT

echo "run_id=$RUN_ID action=$1 job=$SLURM_JOB_ID gpu=$GPU_INDEX commit=$(git rev-parse HEAD)"
SECONDS=0
"$PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("torch", torch.__version__, "cuda", torch.version.cuda, "visible_gpus", torch.cuda.device_count())'

if [[ "$1" == smoke ]]; then
  "$PYTHON" scripts/train_books_sasrec.py --gpu-smoke --output-dir "$RUN_DIR"
else
  "$PYTHON" scripts/train_books_sasrec.py --train-dev --output-dir "$RUN_DIR"
fi
echo "elapsed_seconds=$SECONDS"
