#!/usr/bin/env bash
# One-card smoke only; allocation name is supplied from the private runbook.
set -euo pipefail
: "${MEMREC_EXPECTED_ALLOCATION_NAME:?Pass the privately approved allocation name}"
: "${MEMREC_EXPECTED_COMMIT:?Pass the exact tested source commit}"
if [[ "$MEMREC_EXPECTED_ALLOCATION_NAME" != senvoice-pro-opt ]]; then
  echo 'Only the new reserved allocation is authorized' >&2; exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || \
      "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" != "hoangnv242 RUNNING $MEMREC_EXPECTED_ALLOCATION_NAME" || \
      "${SLURM_CPUS_PER_TASK:-0}" -lt 8 ]]; then
  echo 'Unexpected allocation or insufficient CPUs' >&2; exit 2
fi
export MEMREC_ROOT=/mnt/data/users/hoangnv242/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
if [[ "$(git rev-parse HEAD)" != "$MEMREC_EXPECTED_COMMIT" || \
      -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Exact clean deployed source required' >&2; exit 2
fi
MEMREC_PYTHON="$MEMREC_ROOT/envs/llm-hnv/bin/python"
MEMREC_CONTRACT_VERSION=${1:-4}
case "$MEMREC_CONTRACT_VERSION" in 1|2|3|4|5) ;; *) echo 'Unsupported smoke version' >&2; exit 2;; esac
RUN_ID=$("$MEMREC_PYTHON" - "$MEMREC_CONTRACT_VERSION" <<'PY'
from pathlib import Path
import sys
from src.cmirank.memory_smoke import load_memory_contract
print(load_memory_contract(Path.cwd(), int(sys.argv[1]))[0]['run_id'])
PY
)
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then echo 'Refusing to overwrite smoke' >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$MEMREC_ROOT/cache/uv"
export PIP_CACHE_DIR="$MEMREC_ROOT/cache/pip"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TRITON_CACHE_DIR="$MEMREC_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$MEMREC_ROOT/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$MEMREC_ROOT/cache/torchinductor"
export CUDA_CACHE_PATH="$MEMREC_ROOT/cache/nv/ComputeCache"
export TMPDIR="$MEMREC_ROOT/cache/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$MEMREC_ROOT/logs"
exec 9>"$MEMREC_ROOT/cache/real-memory-gpu-hnv.lock"
flock -n 9 || { echo 'Another MemRec real-memory smoke is active' >&2; exit 2; }
TASK_PID=''
finish() {
  MEMREC_EXIT_CODE=$?
  trap - EXIT INT TERM
  if [[ -n "$TASK_PID" ]] && kill -0 "$TASK_PID" 2>/dev/null; then
    kill -TERM "$TASK_PID" 2>/dev/null || true
    wait "$TASK_PID" 2>/dev/null || true
  fi
  if [[ ! -d "$RUN_DIR" ]]; then exit "$MEMREC_EXIT_CODE"; fi
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_DIR/gpus-after.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$RUN_DIR/apps-after.csv"
  squeue -j "$SLURM_JOB_ID" -h -o '%i %T %j %R' > "$RUN_DIR/allocation-after.txt"
  if ! "$MEMREC_PYTHON" - "$RUN_DIR" "$MEMREC_EXIT_CODE" <<'PY'
import json
from pathlib import Path
import sys
from src.cmirank.gpu_resources import parse_gpu_snapshot, compute_gpu_processes, gpu_release_verified
run = Path(sys.argv[1])
child_path = run / 'child-cleanup.json'
child = json.loads(child_path.read_text()) if child_path.exists() else {}
uuid = child.get('gpu_uuid')
cards = parse_gpu_snapshot((run / 'gpus-after.csv').read_text())
apps = compute_gpu_processes((run / 'apps-after.csv').read_text())
released = None
after_mem = None
if uuid:
    card = next(c for c in cards if c.uuid == uuid)
    before = next(c for c in parse_gpu_snapshot((run / 'gpus-load-time.csv').read_text()) if c.uuid == uuid)
    before_apps = compute_gpu_processes((run / 'apps-load-time.csv').read_text()).get(uuid, set())
    released = gpu_release_verified(card, before, before_apps, apps.get(uuid, set()))
    after_mem = card.used_mib
allocation_running = ' RUNNING ' in (run / 'allocation-after.txt').read_text()
record = {'process_exit_code': int(sys.argv[2]), 'child_exited': True,
          'owned_server_exited': child.get('owned_server_exited', False),
          'gpu_uuid': uuid, 'gpu_released': released, 'gpu_used_mib_after': after_mem,
          'allocation_still_running': allocation_running, 'training_ready': False}
(run / 'cleanup.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
if int(sys.argv[2]) == 0 and (not released or not record['owned_server_exited'] or not allocation_running):
    raise SystemExit(4)
PY
  then
    if (( MEMREC_EXIT_CODE == 0 )); then MEMREC_EXIT_CODE=4; fi
  fi
  exit "$MEMREC_EXIT_CODE"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SMOKE_TIMEOUT_MINUTES=$("$MEMREC_PYTHON" -c 'import json; print(json.load(open("configs/cmirank/real_memory_smoke_v1.json"))["timeout_minutes"])')
HANDOFF_ARGS=()
if [[ -n "${MEMREC_OWNER_CONFIRMED_GPU1_STEP:-}" ]]; then
  if [[ ( "$MEMREC_CONTRACT_VERSION" != 4 && "$MEMREC_CONTRACT_VERSION" != 5 ) || \
        ! "$MEMREC_OWNER_CONFIRMED_GPU1_STEP" =~ ^${SLURM_JOB_ID}\.[1-9][0-9]*$ ]]; then
    echo 'Handoff must name the owner-confirmed numeric step in this allocation and current smoke scope' >&2
    exit 2
  fi
  HANDOFF_ARGS=(--owner-confirmed-generator-step "$MEMREC_OWNER_CONFIRMED_GPU1_STEP")
fi
timeout --signal=TERM --kill-after=60s "${SMOKE_TIMEOUT_MINUTES}m" "$MEMREC_PYTHON" -u \
  scripts/cmirank/10_smoke_real_memory.py --run-dir "$RUN_DIR" \
  --index-dir "$MEMREC_ROOT/runs/cmirank-minilm-candidate-index-v2-20261001-hnv" \
  --candidate-run-dir "$MEMREC_ROOT/runs/cmirank-policy-candidates-v2-20261005-hnv" \
  --audit-dir "$MEMREC_ROOT/runs/cmirank-shortcut-audit-v2-20261005-hnv" \
  --contract-version "$MEMREC_CONTRACT_VERSION" \
  "${HANDOFF_ARGS[@]}" \
  > "$MEMREC_ROOT/logs/$RUN_ID.log" 2>&1 &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$MEMREC_ROOT/logs/$RUN_ID-launcher.pid"
wait "$TASK_PID"
