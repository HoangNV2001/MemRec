#!/usr/bin/env bash
# Real encoder smoke -> full metadata index -> 20-user integrity smoke; no GPU.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || \
      "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" != 'hoangnv242 RUNNING senvoice-pro-opt' ]]; then
  echo 'Unexpected or missing authorized allocation' >&2
  exit 2
fi
MEMREC_ROOT=/mnt/data/users/hoangnv242/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Candidate indexing requires clean tracked source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$MEMREC_ROOT/cache/uv"
export PIP_CACHE_DIR="$MEMREC_ROOT/cache/pip"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TRITON_CACHE_DIR="$MEMREC_ROOT/cache/triton"
export CUDA_CACHE_PATH="$MEMREC_ROOT/cache/nv/ComputeCache"
export TMPDIR="$MEMREC_ROOT/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
MEMREC_PYTHON="$MEMREC_ROOT/envs/cmirank-qwen35-t513-hnv/bin/python"
MEMREC_AUDIT_PYTHON="$MEMREC_ROOT/envs/llm-hnv/bin/python"
if [[ ! -x "$MEMREC_PYTHON" ]]; then echo 'Prepared inference environment missing' >&2; exit 2; fi
if [[ ! -x "$MEMREC_AUDIT_PYTHON" ]]; then echo 'Read-only data-audit environment missing' >&2; exit 2; fi
# Books cohort helpers live under the baseline data package, which imports
# pandas/torch. Use that existing env read-only, not install into either env.
"$MEMREC_AUDIT_PYTHON" -c 'import src.cmirank.policy_inputs, numpy; print("audit numpy",numpy.__version__)'
read -r RUN_ID CPU_THREADS TIMEOUT_MINUTES < <("$MEMREC_PYTHON" -c \
  'import json; c=json.load(open("configs/cmirank/candidate_sampler_v1.json"))["index_build"]; print(c["run_id"],c["cpu_threads"],c["timeout_minutes"])')
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
mkdir -p "$TMPDIR" "$MEMREC_ROOT/logs"
exec 9>"$MEMREC_ROOT/cache/candidate-index-hnv.lock"
flock -n 9 || { echo 'Another candidate index task is active' >&2; exit 2; }
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then echo 'Refusing to overwrite candidate index run' >&2; exit 2; fi
mkdir "$RUN_DIR"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$RUN_DIR/gpus-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$RUN_DIR/apps-before.csv"
TASK_PID=''
finish() {
  MEMREC_EXIT_CODE=$?
  trap - EXIT INT TERM
  if [[ -n "$TASK_PID" ]] && kill -0 "$TASK_PID" 2>/dev/null; then
    kill -TERM "$TASK_PID" 2>/dev/null || true
    wait "$TASK_PID" 2>/dev/null || true
  fi
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_DIR/gpus-after.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$RUN_DIR/apps-after.csv"
  "$MEMREC_PYTHON" - "$RUN_DIR" "$MEMREC_EXIT_CODE" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
(run / 'cleanup.json').write_text(json.dumps({
    'process_exit_code': int(sys.argv[2]), 'device': 'cpu',
    'gpu_requested': False, 'child_exited': True,
    'note': 'No GPU selected/reserved; other workloads are not modified.'}, indent=2) + '\n')
PY
  squeue -j "$SLURM_JOB_ID" -h -o '%i %T %j %R' > "$RUN_DIR/allocation-after.txt"
  exit "$MEMREC_EXIT_CODE"
}
trap finish EXIT
trap 'exit 130' INT TERM
timeout --signal=TERM --kill-after=30s "${TIMEOUT_MINUTES}m" "$MEMREC_PYTHON" -u \
  scripts/cmirank/07_prepare_candidate_index_cpu.py --run-dir "$RUN_DIR" --full-after-smoke \
  > "$RUN_DIR/index.log" 2>&1 &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$RUN_DIR/launcher.pid"
wait "$TASK_PID"
TASK_PID=''
timeout --signal=TERM --kill-after=30s 15m "$MEMREC_AUDIT_PYTHON" -u \
  scripts/cmirank/08_audit_candidates_cpu.py --index-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/candidate-audit-hnv" --users 20 > "$RUN_DIR/audit.log" 2>&1 &
TASK_PID=$!
wait "$TASK_PID"
