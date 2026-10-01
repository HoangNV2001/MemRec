#!/usr/bin/env bash
# Invoke through the authorized train_TTS Slurm step. This stage holds no GPU.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Preparation requires an authorized Slurm step' >&2
  exit 2
fi
if [[ "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" != 'anhntc2 RUNNING train_TTS' ]]; then
  echo 'Unexpected allocation' >&2
  exit 2
fi
MEMREC_ROOT=/mnt/data/users/anhnct/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Tracked source must be clean before preparation' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$MEMREC_ROOT/cache/uv"
export PIP_CACHE_DIR="$MEMREC_ROOT/cache/pip"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TMPDIR="$MEMREC_ROOT/cache/tmp"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$MEMREC_ROOT/logs"
exec 9>"$MEMREC_ROOT/cache/qwen35-prep-hnv.lock"
flock -n 9 || { echo 'Another MemRec Qwen3.5 preparation is active' >&2; exit 2; }
MEMREC_ENV="$MEMREC_ROOT/envs/cmirank-qwen35-t513-hnv"
MEMREC_BASE_PYTHON="$MEMREC_ROOT/envs/llm-hnv/bin/python"
if [[ ! -x "$MEMREC_ENV/bin/python" ]]; then
  "$MEMREC_BASE_PYTHON" -m venv "$MEMREC_ENV"
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$MEMREC_ENV/bin/python" --torch-backend=cu128 'torch==2.8.0'
  uv pip install --python "$MEMREC_ENV/bin/python" -r configs/cmirank/qwen35_smoke_requirements.txt
else
  "$MEMREC_ENV/bin/python" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
  "$MEMREC_ENV/bin/python" -m pip install -r configs/cmirank/qwen35_smoke_requirements.txt
fi
"$MEMREC_ENV/bin/python" -m pip check
"$MEMREC_ENV/bin/python" -c 'import torch,transformers; from transformers import Qwen3_5ForConditionalGeneration; print("torch",torch.__version__,"transformers",transformers.__version__); assert not torch.cuda.is_available()'
"$MEMREC_ENV/bin/python" -m pip freeze > "$MEMREC_ENV/environment-hnv.txt"
"$MEMREC_ENV/bin/python" scripts/cmirank/05_download_qwen35.py
