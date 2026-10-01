#!/usr/bin/env bash
# Authorized train_TTS step; default idle-only, with one explicitly approved smoke exception.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || \
      "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" != 'anhntc2 RUNNING train_TTS' ]]; then
  echo 'Unexpected or missing authorized allocation' >&2
  exit 2
fi
MEMREC_ROOT=/mnt/data/users/anhnct/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
mkdir -p "$MEMREC_ROOT/cache"
exec 9>"$MEMREC_ROOT/cache/qwen35-gpu-hnv.lock"
flock -n 9 || { echo 'Another MemRec Qwen3.5 GPU smoke is active' >&2; exit 2; }
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'GPU smoke requires clean tracked source' >&2
  exit 2
fi
MEMREC_PYTHON="$MEMREC_ROOT/envs/cmirank-qwen35-t513-hnv/bin/python"
if [[ ! -x "$MEMREC_PYTHON" ]]; then
  echo 'Qwen3.5 smoke environment is not prepared' >&2
  exit 2
fi
RUN_ID=$("$MEMREC_PYTHON" -c 'import json; print(json.load(open("configs/cmirank/qwen35_gpu_smoke.json"))["run_id"])')
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite run $RUN_ID" >&2
  exit 2
fi
mkdir -p "$MEMREC_ROOT/logs"
PREFLIGHT="$MEMREC_ROOT/logs/$RUN_ID-gpus-before.csv"
PREFLIGHT_APPS="$MEMREC_ROOT/logs/$RUN_ID-apps-before.csv"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$PREFLIGHT"
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$PREFLIGHT_APPS"
GPU_UUID=$("$MEMREC_PYTHON" - "$PREFLIGHT" "$PREFLIGHT_APPS" <<'PY'
import json
from pathlib import Path
import sys
from src.cmirank.gpu_resources import parse_gpu_snapshot, compute_gpu_uuids, select_qwen_smoke_gpu
config = json.loads(Path("configs/cmirank/qwen35_gpu_smoke.json").read_text())
card = select_qwen_smoke_gpu(parse_gpu_snapshot(Path(sys.argv[1]).read_text()),
                             compute_gpu_uuids(Path(sys.argv[2]).read_text()), config)
print(card.uuid)
PY
)
mkdir "$RUN_DIR"
cp "$PREFLIGHT" "$RUN_DIR/gpus-before.csv"
cp "$PREFLIGHT_APPS" "$RUN_DIR/apps-before.csv"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TRITON_CACHE_DIR="$MEMREC_ROOT/cache/triton"
export CUDA_CACHE_PATH="$MEMREC_ROOT/cache/nv/ComputeCache"
export TMPDIR="$MEMREC_ROOT/cache/tmp"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
TASK_PID=''
finish() {
  MEMREC_EXIT_CODE=$?
  trap - EXIT INT TERM
  if [[ -n "$TASK_PID" ]] && kill -0 "$TASK_PID" 2>/dev/null; then
    # This PID was created by this launcher; timeout forwards TERM to its own child.
    kill -TERM "$TASK_PID" 2>/dev/null || true
    wait "$TASK_PID" 2>/dev/null || true
  fi
  sleep 2
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$RUN_DIR/gpus-after.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$RUN_DIR/apps-after.csv"
  if ! "$MEMREC_PYTHON" - "$RUN_DIR" "$GPU_UUID" "$MEMREC_EXIT_CODE" <<'PY'
import json
from pathlib import Path
import sys
from src.cmirank.gpu_resources import parse_gpu_snapshot, compute_gpu_processes, gpu_release_verified
run = Path(sys.argv[1])
cards = parse_gpu_snapshot((run / "gpus-after.csv").read_text())
before_cards = parse_gpu_snapshot((run / "gpus-before.csv").read_text())
before = compute_gpu_processes((run / "apps-before.csv").read_text()).get(sys.argv[2], set())
after = compute_gpu_processes((run / "apps-after.csv").read_text()).get(sys.argv[2], set())
card = next(value for value in cards if value.uuid == sys.argv[2])
baseline = next(value for value in before_cards if value.uuid == sys.argv[2])
config = json.loads(Path("configs/cmirank/qwen35_gpu_smoke.json").read_text())
shared = config.get("gpu_policy") == "shared_gpu1_single_smoke_20261001"
# The launcher has already waited for its own child. Never require another
# workload to disappear, and never kill baseline or newly observed processes.
released = gpu_release_verified(card, baseline, before, after, shared=shared)
record = {"process_exit_code": int(sys.argv[3]), "gpu_uuid": card.uuid,
          "gpu_used_mib_after": card.used_mib, "gpu_released": released,
          "release_scope": "memrec_workload_only" if shared else "whole_card_idle",
          "baseline_used_mib": baseline.used_mib,
          "baseline_gpu_pids": sorted(before), "gpu_pids_after": sorted(after),
          "baseline_processes_still_present": before <= after}
(run / "cleanup.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
if not released:
    raise SystemExit(4)
PY
  then
    if (( MEMREC_EXIT_CODE == 0 )); then MEMREC_EXIT_CODE=4; fi
  fi
  squeue -j "$SLURM_JOB_ID" -h -o '%i %T %j %R' > "$RUN_DIR/allocation-after.txt"
  exit "$MEMREC_EXIT_CODE"
}
trap finish EXIT
trap 'exit 130' INT TERM
SMOKE_TIMEOUT_MINUTES=$("$MEMREC_PYTHON" -c 'import json; print(json.load(open("configs/cmirank/qwen35_gpu_smoke.json"))["timeout_minutes"])')
timeout --signal=TERM --kill-after=30s "${SMOKE_TIMEOUT_MINUTES}m" "$MEMREC_PYTHON" -u \
  scripts/cmirank/06_smoke_qwen35_gpu.py --run-dir "$RUN_DIR" \
  > "$RUN_DIR/smoke.log" 2>&1 &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$RUN_DIR/launcher.pid"
wait "$TASK_PID"
