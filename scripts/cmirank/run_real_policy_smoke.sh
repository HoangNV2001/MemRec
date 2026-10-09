#!/usr/bin/env bash
# Worker-only functional smoke. Login controller restores the persistent keeper.
set -euo pipefail
: "${MEMREC_EXPECTED_COMMIT:?Exact tested source required}"
: "${MEMREC_OWNER_CONFIRMED_GPU1_STEP:?Fresh current generator confirmation required}"
MEMREC_ROOT=/mnt/data/users/hoangnv242/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
[[ -n "${SLURM_JOB_ID:-}" && "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" == 'hoangnv242 RUNNING senvoice-pro-opt' ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$MEMREC_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
[[ "$MEMREC_OWNER_CONFIRMED_GPU1_STEP" =~ ^${SLURM_JOB_ID}\.[1-9][0-9]*$ ]] || exit 2
export MEMREC_ROOT CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$MEMREC_ROOT/cache" HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" TORCH_HOME="$MEMREC_ROOT/cache/torch" TMPDIR="$MEMREC_ROOT/cache/tmp"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
MEMREC_PYTHON="$MEMREC_ROOT/envs/cmirank-qwen35-t513-hnv/bin/python"
RUN_ID=$($MEMREC_PYTHON -c 'import json;print(json.load(open("configs/cmirank/real_policy_smoke_v1.json"))["run_id"])')
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || exit 2
export TRITON_CACHE_DIR="$MEMREC_ROOT/cache/runtime-hnv/$RUN_ID/triton"
export TORCHINDUCTOR_CACHE_DIR="$MEMREC_ROOT/cache/runtime-hnv/$RUN_ID/torchinductor"
export CUDA_CACHE_PATH="$MEMREC_ROOT/cache/runtime-hnv/$RUN_ID/cuda"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MEMREC_ROOT/logs"
exec 9>"$MEMREC_ROOT/cache/real-memory-gpu-hnv.lock"
flock -n 9 || exit 2
TASK_PID=''
finish() {
  MEMREC_EXIT_CODE=$?
  trap - EXIT INT TERM
  if [[ -n "$TASK_PID" ]] && kill -0 "$TASK_PID" 2>/dev/null; then
    kill -TERM "$TASK_PID" 2>/dev/null || true
    wait "$TASK_PID" 2>/dev/null || true
  fi
  [[ -d "$RUN_DIR" ]] || exit "$MEMREC_EXIT_CODE"
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$RUN_DIR/gpus-after.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits > "$RUN_DIR/apps-after.csv"
  squeue -j "$SLURM_JOB_ID" -h -o '%i %T %j %R' > "$RUN_DIR/allocation-after.txt"
  "$MEMREC_PYTHON" - "$RUN_DIR" "$MEMREC_EXIT_CODE" <<'PY'
import json,sys
from pathlib import Path
from src.cmirank.gpu_resources import parse_gpu_snapshot,compute_gpu_processes
run=Path(sys.argv[1]); child=run/'child-cleanup.json'; d=json.loads(child.read_text()) if child.exists() else {}
handoff=run/'gpu1-handoff.json'; h=json.loads(handoff.read_text()) if handoff.exists() else {}
uuid=d.get('gpu_uuid') or h.get('gpu1_uuid')
cards=parse_gpu_snapshot((run/'gpus-after.csv').read_text()); apps=compute_gpu_processes((run/'apps-after.csv').read_text())
card=next((c for c in cards if c.uuid==uuid),None)
released=card is not None and card.used_mib<512 and not apps.get(uuid)
record={'process_exit_code':int(sys.argv[2]),'child_exited':True,'gpu_uuid':uuid,
 'gpu_released':released,'gpu_used_mib_after':card.used_mib if card else None,
 'allocation_still_running':' RUNNING ' in (run/'allocation-after.txt').read_text(),'training_ready':False}
(run/'cleanup.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n');print(json.dumps(record),flush=True)
PY
  exit "$MEMREC_EXIT_CODE"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout --signal=TERM --kill-after=60s 30m "$MEMREC_PYTHON" -u scripts/cmirank/13_smoke_real_policy_gpu.py \
  --run-dir "$RUN_DIR" --owner-confirmed-generator-step "$MEMREC_OWNER_CONFIRMED_GPU1_STEP" &
TASK_PID=$!
wait "$TASK_PID"
TASK_PID=''
