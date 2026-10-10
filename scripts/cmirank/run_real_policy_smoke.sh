#!/usr/bin/env bash
# Worker-only policy/format task. Login controller restores the persistent keeper.
set -euo pipefail
: "${MEMREC_EXPECTED_COMMIT:?Exact tested source required}"
: "${MEMREC_OWNER_CONFIRMED_GENERATOR_STEP:?Fresh current generator confirmation required}"
MEMREC_POLICY_CONTRACT_VERSION=${MEMREC_POLICY_CONTRACT_VERSION:-2}
[[ "$MEMREC_POLICY_CONTRACT_VERSION" == 1 || "$MEMREC_POLICY_CONTRACT_VERSION" == 2 ]] || exit 2
MEMREC_TASK_KIND=${MEMREC_TASK_KIND:-real_policy}
MEMREC_SFT_PHASE=${MEMREC_SFT_PHASE:-smoke}
[[ "$MEMREC_TASK_KIND" == real_policy || "$MEMREC_TASK_KIND" == format_sft || "$MEMREC_TASK_KIND" == ppo_compat ]] || exit 2
[[ "$MEMREC_SFT_PHASE" == smoke || "$MEMREC_SFT_PHASE" == full ]] || exit 2
MEMREC_ROOT=/mnt/data/users/hoangnv242/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
[[ -n "${SLURM_JOB_ID:-}" && "$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')" == 'hoangnv242 RUNNING senvoice-pro-opt' ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$MEMREC_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
[[ "$MEMREC_OWNER_CONFIRMED_GENERATOR_STEP" =~ ^${SLURM_JOB_ID}\.[1-9][0-9]*$ ]] || exit 2
export MEMREC_ROOT CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="$MEMREC_ROOT/cache" HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" TORCH_HOME="$MEMREC_ROOT/cache/torch" TMPDIR="$MEMREC_ROOT/cache/tmp"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
MEMREC_PYTHON="$MEMREC_ROOT/envs/cmirank-qwen35-t513-hnv/bin/python"
if [[ "$MEMREC_TASK_KIND" == format_sft ]]; then
  read -r RUN_ID MEMREC_GPU_INDEX MEMREC_TIMEOUT < <("$MEMREC_PYTHON" -c 'import sys;from pathlib import Path;from src.cmirank.format_sft import load_format_config;c,_=load_format_config(Path.cwd());print(c[sys.argv[1]+"_run_id"],c["gpu_index"],c["timeout_minutes"])' "$MEMREC_SFT_PHASE")
  MEMREC_TASK_SCRIPT=scripts/cmirank/16_train_format_sft_gpu.py
  MEMREC_TASK_ARGS=(--phase "$MEMREC_SFT_PHASE")
elif [[ "$MEMREC_TASK_KIND" == ppo_compat ]]; then
  read -r MEMREC_ENV_NAME MEMREC_OVERLAY < <("$MEMREC_PYTHON" -c 'from pathlib import Path;from src.cmirank.ppo_runtime import load_runtime_config;from src.cmirank.ppo_compat import load_compat_config;r,_=load_runtime_config(Path.cwd());c,_=load_compat_config(Path.cwd());print(r["environment_name"],c["overlay_name"])')
  MEMREC_PYTHON="$MEMREC_ROOT/envs/$MEMREC_ENV_NAME/bin/python"
  export PYTHONPATH="$MEMREC_ROOT/overlays/$MEMREC_OVERLAY"
  read -r RUN_ID MEMREC_GPU_INDEX MEMREC_TIMEOUT < <("$MEMREC_PYTHON" -c 'from pathlib import Path;from src.cmirank.ppo_compat import load_compat_config;c,_=load_compat_config(Path.cwd());print(c["run_id"],c["gpu_index"],c["timeout_minutes"])')
  MEMREC_TASK_SCRIPT=scripts/cmirank/21_smoke_ppo_full_roles_gpu.py
  MEMREC_TASK_ARGS=()
else
  read -r RUN_ID MEMREC_GPU_INDEX MEMREC_TIMEOUT < <("$MEMREC_PYTHON" -c 'import sys;from pathlib import Path;from src.cmirank.policy_smoke import load_policy_smoke_contract;c,_=load_policy_smoke_contract(Path.cwd(),int(sys.argv[1]));print(c["run_id"],c.get("gpu_index",1),c["timeout_minutes"])' "$MEMREC_POLICY_CONTRACT_VERSION")
  MEMREC_TASK_SCRIPT=scripts/cmirank/13_smoke_real_policy_gpu.py
  MEMREC_TASK_ARGS=(--contract-version "$MEMREC_POLICY_CONTRACT_VERSION")
fi
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
  "$MEMREC_PYTHON" - "$RUN_DIR" "$MEMREC_EXIT_CODE" "$MEMREC_GPU_INDEX" <<'PY'
import json,sys
from pathlib import Path
from src.cmirank.gpu_resources import parse_gpu_snapshot,compute_gpu_processes
run=Path(sys.argv[1]); child=run/'child-cleanup.json'; d=json.loads(child.read_text()) if child.exists() else {}
index=int(sys.argv[3]); handoff=run/f'gpu{index}-handoff.json'; h=json.loads(handoff.read_text()) if handoff.exists() else {}
uuid=d.get('gpu_uuid') or h.get(f'gpu{index}_uuid')
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
timeout --signal=TERM --kill-after=60s "${MEMREC_TIMEOUT}m" "$MEMREC_PYTHON" -u "$MEMREC_TASK_SCRIPT" \
  --run-dir "$RUN_DIR" "${MEMREC_TASK_ARGS[@]}" \
  --owner-confirmed-generator-step "$MEMREC_OWNER_CONFIRMED_GENERATOR_STEP" &
TASK_PID=$!
wait "$TASK_PID"
TASK_PID=''
