#!/usr/bin/env bash
# Approved episode recipe: 20-user boundary smoke, then full only on a pass.
set -euo pipefail
MEMREC_RECIPE_VERSION=${1:-1}
case "$MEMREC_RECIPE_VERSION" in
  1) MEMREC_EXPECTED_ALLOCATION_NAME=train_TTS ;;
  2) : "${MEMREC_EXPECTED_ALLOCATION_NAME:?Pass the privately approved allocation name}" ;;
  *) echo 'Unsupported recipe version' >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || \
      "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" != "anhntc2 RUNNING $MEMREC_EXPECTED_ALLOCATION_NAME" ]]; then
  echo 'Unexpected or missing authorized allocation' >&2
  exit 2
fi
MEMREC_ROOT=/mnt/data/users/anhnct/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Candidate preparation requires clean tracked source' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TMPDIR="$MEMREC_ROOT/cache/tmp"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
if (( ${SLURM_CPUS_PER_TASK:-0} < 8 )); then echo 'Eight CPUs required' >&2; exit 2; fi
MEMREC_PYTHON="$MEMREC_ROOT/envs/llm-hnv/bin/python"
RUN_DIR="$MEMREC_ROOT/runs/cmirank-policy-candidates-v1-20261002-hnv"
MEMREC_CANDIDATE_OPTIONS=()
if [[ "$MEMREC_RECIPE_VERSION" == 2 ]]; then
  RUN_DIR="$MEMREC_ROOT/runs/cmirank-policy-candidates-v2-20261005-hnv"
  MEMREC_CANDIDATE_OPTIONS=(--recipe-version 2 --warmup-run-dir
    "$MEMREC_ROOT/runs/cmirank-policy-candidates-v1-20261002-hnv")
fi
INDEX_DIR="$MEMREC_ROOT/runs/cmirank-minilm-candidate-index-v2-20261001-hnv"
mkdir -p "$TMPDIR"
exec 9>"$MEMREC_ROOT/cache/policy-candidates-hnv.lock"
flock -n 9 || { echo 'Another policy candidate preparation is active' >&2; exit 2; }
if [[ -e "$RUN_DIR" ]]; then echo 'Refusing to overwrite candidate run' >&2; exit 2; fi
mkdir "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$RUN_DIR/gpus-before.csv"
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
  "$MEMREC_PYTHON" - "$RUN_DIR" "$MEMREC_EXIT_CODE" <<'PY'
import json
from pathlib import Path
import sys
run=Path(sys.argv[1])
(run/'cleanup.json').write_text(json.dumps({'process_exit_code':int(sys.argv[2]),
    'device':'cpu','gpu_requested':False,'child_exited':True},indent=2)+'\n')
PY
  squeue -j "$SLURM_JOB_ID" -h -o '%i %T %j %R' > "$RUN_DIR/allocation-after.txt"
  exit "$MEMREC_EXIT_CODE"
}
trap finish EXIT
trap 'exit 130' INT TERM
timeout --signal=TERM --kill-after=30s 5m "$MEMREC_PYTHON" -u \
  scripts/cmirank/08_audit_candidates_cpu.py --index-dir "$INDEX_DIR" \
  --output-dir "$RUN_DIR/smoke-hnv" --users 20 "${MEMREC_CANDIDATE_OPTIONS[@]}" \
  > "$RUN_DIR/smoke.log" 2>&1 &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$RUN_DIR/launcher.pid"
wait "$TASK_PID"
TASK_PID=''
timeout --signal=TERM --kill-after=30s 60m "$MEMREC_PYTHON" -u \
  scripts/cmirank/08_audit_candidates_cpu.py --index-dir "$INDEX_DIR" \
  --output-dir "$RUN_DIR/full-hnv" --users all --smoke-dir "$RUN_DIR/smoke-hnv" \
  "${MEMREC_CANDIDATE_OPTIONS[@]}" \
  > "$RUN_DIR/full.log" 2>&1 &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$RUN_DIR/launcher.pid"
wait "$TASK_PID"
