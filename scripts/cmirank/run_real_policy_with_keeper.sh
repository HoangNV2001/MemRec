#!/usr/bin/env bash
# Login-node orchestration only: all heavy work in the existing reservation.
set -euo pipefail
: "${MEMREC_EXPECTED_COMMIT:?Exact tested source required}"
: "${MEMREC_OWNER_CONFIRMED_GPU1_STEP:?Current exclusive generator confirmation required}"
MEMREC_ROOT=/mnt/data/users/hoangnv242/memrec-hnv
cd "$MEMREC_ROOT/repo/MemRec-hnv"
[[ "$(id -un)" == hoangnv242 && "$(git rev-parse HEAD)" == "$MEMREC_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]] || exit 2
MEMREC_ALLOC_JOB=$(squeue -u hoangnv242 -h -n senvoice-pro-opt -t RUNNING -o %i)
[[ "$MEMREC_ALLOC_JOB" =~ ^[0-9]+$ && "$(squeue -j "$MEMREC_ALLOC_JOB" -h -o '%u %T %j')" == 'hoangnv242 RUNNING senvoice-pro-opt' ]] || exit 2
[[ "$MEMREC_OWNER_CONFIRMED_GPU1_STEP" =~ ^${MEMREC_ALLOC_JOB}\.[1-9][0-9]*$ ]] || exit 2
MEMREC_RUN_ID=cmirank-qwen35-real-nminus1-direct-smoke-v1-20261008-hnv
MEMREC_RUN_DIR="$MEMREC_ROOT/runs/$MEMREC_RUN_ID"
MEMREC_SRUN_PID=''
handback() {
  MEMREC_EXIT_CODE=$?
  trap - EXIT INT TERM
  if [[ -n "$MEMREC_SRUN_PID" ]] && kill -0 "$MEMREC_SRUN_PID" 2>/dev/null; then
    # Only this controller's child srun/owned step, never the reservation/job.
    kill -TERM "$MEMREC_SRUN_PID" 2>/dev/null || true
    wait "$MEMREC_SRUN_PID" 2>/dev/null || true
  fi
  exec 9>"$MEMREC_ROOT/cache/real-memory-gpu-hnv.lock"
  if ! flock -n 9; then
    echo 'Handback blocked: another MemRec GPU controller owns the lock' >&2
    exit 4
  fi
  MEMREC_PROBE=$(srun --input=none --jobid="$MEMREC_ALLOC_JOB" --overlap --ntasks=1 --cpus-per-task=2 --job-name=memrec-keeper-probe-hnv \
    env MEMREC_ROOT="$MEMREC_ROOT" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "$MEMREC_ROOT/envs/llm-hnv/bin/python" scripts/cmirank/14_keeper_handback_cpu.py --run-dir "$MEMREC_RUN_DIR") || exit 4
  printf '%s\n' "$MEMREC_PROBE"
  if [[ "$MEMREC_PROBE" == *'"status": "START_GPU1_KEEPER"'* ]]; then
    [[ ! -e /mnt/data/users/hoangnv242/omni-gen-hnv/STOP ]] || exit 4
    setsid nohup srun --input=none --jobid="$MEMREC_ALLOC_JOB" --overlap --ntasks=1 --cpus-per-task=4 --job-name=omni-gen-1 \
      env GPU=1 D=/mnt/data/users/hoangnv242/omni-gen-hnv bash /mnt/data/users/hoangnv242/omni-gen-hnv/omni_gen.sh \
      > "$MEMREC_ROOT/logs/$MEMREC_RUN_ID-keeper-launch.log" 2>&1 < /dev/null 9>&- &
    printf '%s\n' "$!" > "$MEMREC_ROOT/logs/$MEMREC_RUN_ID-keeper-launcher.pid"
  fi
  srun --input=none --jobid="$MEMREC_ALLOC_JOB" --overlap --ntasks=1 --cpus-per-task=2 --job-name=memrec-keeper-verify-hnv \
    env MEMREC_ROOT="$MEMREC_ROOT" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "$MEMREC_ROOT/envs/llm-hnv/bin/python" scripts/cmirank/14_keeper_handback_cpu.py --run-dir "$MEMREC_RUN_DIR" --verify || exit 4
  exit "$MEMREC_EXIT_CODE"
}
trap handback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
srun --input=none --jobid="$MEMREC_ALLOC_JOB" --overlap --ntasks=1 --cpus-per-task=8 --job-name=memrec-real-policy-smoke-hnv \
  env MEMREC_EXPECTED_COMMIT="$MEMREC_EXPECTED_COMMIT" MEMREC_OWNER_CONFIRMED_GPU1_STEP="$MEMREC_OWNER_CONFIRMED_GPU1_STEP" \
  bash scripts/cmirank/run_real_policy_smoke.sh &
MEMREC_SRUN_PID=$!
wait "$MEMREC_SRUN_PID"
MEMREC_SRUN_PID=''
