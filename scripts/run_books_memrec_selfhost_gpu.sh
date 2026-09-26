#!/usr/bin/env bash
# Real-LLM Books smoke/full-dev/dev700. Invoke ONLY through authorized train_TTS step.
set -euo pipefail

if [[ $# -gt 1 || -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Usage: run_books_memrec_selfhost_gpu.sh [smoke|smoke700|full|resume|dev700|resume700] inside Slurm' >&2
  exit 2
fi
MODE=${1:-smoke}
if [[ "$MODE" != smoke && "$MODE" != smoke700 && "$MODE" != full && "$MODE" != resume && \
      "$MODE" != dev700 && "$MODE" != resume700 ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 2
fi

MEMREC_ROOT=/mnt/data/users/anhnct/memrec-hnv
REPO="$MEMREC_ROOT/repo/MemRec-hnv"
if [[ "$MODE" == smoke ]]; then
  RUN_ID=books-memrec-llm-smoke-v2-hnv
elif [[ "$MODE" == smoke700 ]]; then
  RUN_ID=books-memrec-llm-dev700-smoke-v1-hnv
elif [[ "$MODE" == dev700 || "$MODE" == resume700 ]]; then
  RUN_ID=books-memrec-llm-dev700-v1-hnv
else
  RUN_ID=books-memrec-llm-dev-v1-hnv
fi
CONFIG=configs/memrec_instructrec-books_full_benchmark.yaml
GATE_FILE=full-dev-gate.json
N_EVAL_USERS=2000
WARMUP_SCOPE=all
RUN_TIMEOUT=72h
if [[ "$MODE" == dev700 || "$MODE" == resume700 || "$MODE" == smoke700 ]]; then
  CONFIG=configs/memrec_instructrec-books_dev700.yaml
  GATE_FILE=dev700-gate.json
  N_EVAL_USERS=200
  WARMUP_SCOPE=subset
  RUN_TIMEOUT=570m  # 9.5h inference + bounded startup/cleanup: <=~10 GPU-hours.
fi
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
SMOKE_DIR="$MEMREC_ROOT/runs/books-memrec-llm-smoke-v2-hnv"
SUBSET_SMOKE_DIR="$MEMREC_ROOT/runs/books-memrec-llm-dev700-smoke-v1-hnv"
LOG_DIR="$MEMREC_ROOT/logs"
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
MODEL_REVISION=5a5a776300a41aaa681dd7ff0106608ef2bc90db
MODEL_PATH="$MEMREC_ROOT/models/Qwen3-30B-A3B-Instruct-2507-FP8-hnv"
PYTHON="$MEMREC_ROOT/envs/llm-hnv/bin/python"
VLLM="$MEMREC_ROOT/envs/llm-hnv/bin/vllm"
PORT=18081
CACHE_NAMESPACE="$MODEL_REVISION:vllm0.10.2:tp1:seed42:mem0.60:len16384"

cd "$REPO"
if [[ "$(git status --porcelain --untracked-files=no)" != '' ]]; then
  echo 'Refusing a GPU run with dirty tracked code' >&2
  exit 2
fi
JOB_INFO=$(squeue -j "$SLURM_JOB_ID" -h -o '%u %T %j')
if [[ "$JOB_INFO" != 'anhntc2 RUNNING train_TTS' ]]; then
  echo "Unexpected allocation: $JOB_INFO" >&2
  exit 2
fi
if [[ "$MODE" == resume || "$MODE" == resume700 ]]; then
  if [[ ! -f "$RUN_DIR/manifest.json" || -f "$RUN_DIR/completion.json" ]]; then
    echo 'No incomplete Books dev run to resume' >&2
    exit 2
  fi
  "$PYTHON" - "$RUN_DIR/manifest.json" "$(git rev-parse HEAD)" "$MODEL_REVISION" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest['git_commit'] != sys.argv[2] or manifest['model_revision'] != sys.argv[3]:
    raise SystemExit('Resume code/model differs from the initial full run')
PY
  if [[ -f "$RUN_DIR/server.pid" ]]; then
    old_server_pid=$(< "$RUN_DIR/server.pid")
    if [[ "$old_server_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_server_pid" 2>/dev/null; then
      echo 'Previous server PID is still alive; refusing concurrent resume' >&2
      exit 2
    fi
  fi
elif [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]] || [[ ! -f "$MODEL_PATH/tokenizer_config.json" ]]; then
  echo 'Pinned checkpoint is not downloaded' >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/download-complete-hnv.json" ]] || \
   ! grep -q "$MODEL_REVISION" "$MODEL_PATH/download-complete-hnv.json"; then
  echo 'Pinned checkpoint completion marker is missing or mismatched' >&2
  exit 2
fi
if [[ "$(find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 0 ]]; then
  echo 'Pinned checkpoint has no safetensors weights' >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -x "$VLLM" ]]; then
  echo 'Dedicated vLLM environment is incomplete' >&2
  exit 2
fi
if [[ "$MODE" != smoke ]]; then
  "$PYTHON" scripts/verify_books_smoke_promotion.py \
    --smoke-dir "$SMOKE_DIR" --repo "$REPO" \
    --revision "$MODEL_REVISION" --cache-namespace "$CACHE_NAMESPACE"
fi
if [[ "$MODE" == dev700 || "$MODE" == resume700 ]]; then
  "$PYTHON" scripts/verify_books_smoke_promotion.py \
    --smoke-dir "$SUBSET_SMOKE_DIR" --repo "$REPO" \
    --revision "$MODEL_REVISION" --cache-namespace "$CACHE_NAMESPACE" \
    --allow-cache --reference-predictions "$SMOKE_DIR/test_predictions.jsonl"
fi

mkdir -p "$LOG_DIR"
if [[ "$MODE" == smoke || "$MODE" == smoke700 ]]; then
  ATTEMPT=1
  RUN_LOG="$LOG_DIR/$RUN_ID.log"
  BEFORE="$RUN_DIR/gpu-before.csv"
  APPS_BEFORE="$RUN_DIR/gpu-apps-before.csv"
  AFTER="$RUN_DIR/gpu-after.csv"
  SERVER_LOG="$RUN_DIR/vllm-server.log"
else
  if [[ "$MODE" == resume || "$MODE" == resume700 ]]; then
    ATTEMPT=$(($(find "$RUN_DIR" -maxdepth 1 -name 'gpu-before-attempt-*.csv' | wc -l) + 1))
  else
    ATTEMPT=1
  fi
  RUN_LOG="$LOG_DIR/$RUN_ID-attempt-$ATTEMPT.log"
  BEFORE="$RUN_DIR/gpu-before-attempt-$ATTEMPT.csv"
  APPS_BEFORE="$RUN_DIR/gpu-apps-before-attempt-$ATTEMPT.csv"
  AFTER="$RUN_DIR/gpu-after-attempt-$ATTEMPT.csv"
  SERVER_LOG="$RUN_DIR/vllm-server-attempt-$ATTEMPT.log"
fi
PREFLIGHT_BEFORE="$LOG_DIR/$RUN_ID-preflight-attempt-$ATTEMPT.csv"
PREFLIGHT_APPS="$LOG_DIR/$RUN_ID-gpu-apps-preflight-attempt-$ATTEMPT.csv"
exec > >(tee "$RUN_LOG") 2>&1
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,uuid \
  --format=csv,noheader,nounits > "$PREFLIGHT_BEFORE"
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits \
  > "$PREFLIGHT_APPS"
if [[ "$(wc -l < "$PREFLIGHT_BEFORE")" -ne 4 ]]; then
  echo 'Expected four GPUs in preflight snapshot' >&2
  exit 2
fi
GPU_INDEX=''
for candidate_index in 3 2 1 0; do
  candidate_row=$(grep "^$candidate_index," "$PREFLIGHT_BEFORE" || true)
  if [[ -z "$candidate_row" ]]; then
    echo "Missing GPU $candidate_index in preflight snapshot" >&2
    exit 2
  fi
  candidate_util=$(printf '%s\n' "$candidate_row" | cut -d, -f2 | tr -d ' ')
  candidate_mem=$(printf '%s\n' "$candidate_row" | cut -d, -f3 | tr -d ' ')
  candidate_uuid=$(printf '%s\n' "$candidate_row" | cut -d, -f5 | tr -d ' ')
  if [[ ! "$candidate_util" =~ ^[0-9]+$ || ! "$candidate_mem" =~ ^[0-9]+$ || \
        ! "$candidate_uuid" =~ ^GPU-[0-9a-f-]+$ ]]; then
    echo "Malformed GPU $candidate_index preflight snapshot" >&2
    exit 2
  fi
  # A quiet card with a live/hidden CUDA context is not an empty card.
  if (( candidate_util < 20 && candidate_mem < 512 )) && \
      ! grep -Fq "$candidate_uuid" "$PREFLIGHT_APPS"; then
    GPU_INDEX=$candidate_index
    GPU_UTIL=$candidate_util
    GPU_MEM_BEFORE=$candidate_mem
    break
  fi
done
if [[ -z "$GPU_INDEX" ]]; then
  echo 'No empty H100 (low utilization, <512 MiB and no compute process) for this run' >&2
  exit 2
fi
if [[ "$MODE" == resume || "$MODE" == resume700 ]]; then
  test -d "$RUN_DIR"
else
  mkdir "$RUN_DIR"  # Atomic claim; refuse a concurrent start of this run ID.
fi
cp "$PREFLIGHT_BEFORE" "$BEFORE"
cp "$PREFLIGHT_APPS" "$APPS_BEFORE"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export XDG_CACHE_HOME="$MEMREC_ROOT/cache"
export HF_HOME="$MEMREC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$MEMREC_ROOT/cache/pip"
export TORCH_HOME="$MEMREC_ROOT/cache/torch"
export TRITON_CACHE_DIR="$MEMREC_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$MEMREC_ROOT/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$MEMREC_ROOT/cache/torchinductor"
export CUDA_CACHE_PATH="$MEMREC_ROOT/cache/nv/ComputeCache"
export TMPDIR="$MEMREC_ROOT/cache/tmp"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
export MEMREC_SELFHOST_MODEL="$MODEL_NAME"
export MEMREC_SELFHOST_REVISION="$MODEL_REVISION"
export MEMREC_SELFHOST_BASE_URL="http://127.0.0.1:$PORT/v1"
export MEMREC_SELFHOST_API_KEY=local-placeholder
export MEMREC_LLM_CACHE_DB="$MEMREC_ROOT/cache/books-memrec-primary-v2-hnv.sqlite"
export MEMREC_LLM_CACHE_NAMESPACE="$CACHE_NAMESPACE"
if [[ "$MODE" == smoke ]]; then
  export MEMREC_LLM_CACHE_READ=0
else
  export MEMREC_LLM_CACHE_READ=1
fi
if [[ "$MODE" != smoke && "$MODE" != smoke700 ]]; then
  export MEMREC_BOOKS_RUN_JOURNAL_DB="$RUN_DIR/journal.sqlite"
  export MEMREC_BOOKS_REQUEST_BUDGET_DB="$RUN_DIR/request-budget.sqlite"
  export MEMREC_BOOKS_FULL_RUN_ID="$RUN_ID"
  export MEMREC_BOOKS_FULL_GIT_COMMIT="$(git rev-parse HEAD)"
fi
export PYTHONUNBUFFERED=1

SERVER_PID=''
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    server_command=$(ps -p "$SERVER_PID" -o args= || true)
    if [[ "$server_command" == *"$MODEL_PATH"* ]]; then
      kill -TERM "$SERVER_PID"
      for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$SERVER_PID" 2>/dev/null; then
        server_command=$(ps -p "$SERVER_PID" -o args= || true)
        if [[ "$server_command" == *"$MODEL_PATH"* ]]; then
          kill -KILL "$SERVER_PID"
        fi
      fi
    else
      echo 'Server PID command changed; refusing to kill an unknown process' >&2
      status=1
    fi
  fi
  if [[ -n "$SERVER_PID" ]]; then wait "$SERVER_PID" 2>/dev/null || true; fi
  sleep 5
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$AFTER" || status=1
  if [[ -f "$AFTER" ]]; then
    after_mem=$(grep "^$GPU_INDEX," "$AFTER" | cut -d, -f3 | tr -d ' ')
    if [[ "$after_mem" =~ ^[0-9]+$ ]] && (( after_mem > GPU_MEM_BEFORE + 512 )); then
      echo "VRAM not released: before=$GPU_MEM_BEFORE MiB after=$after_mem MiB" >&2
      status=1
    fi
  fi
  if [[ "$status" -eq 0 && ( "$MODE" == smoke || "$MODE" == smoke700 ) ]]; then
    if ! "$PYTHON" - "$RUN_DIR" "$MEMREC_LLM_CACHE_NAMESPACE" <<'PY'
import hashlib, json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
names = ('manifest.json', 'source-hashes.sha256', 'test_predictions.jsonl',
         'smoke-gate.json', 'gpu-before.csv', 'gpu-apps-before.csv', 'gpu-after.csv')
hashes = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
          for name in names}
manifest = json.loads((run_dir / 'manifest.json').read_text())
promotion = {
    'status': 'passed', 'run_id': manifest['run_id'],
    'git_commit': manifest['git_commit'], 'model_revision': manifest['model_revision'],
    'cache_namespace': sys.argv[2], 'smoke_user_sha256': manifest['smoke_user_sha256'],
    'artifact_sha256': hashes,
}
(run_dir / 'promotion.json').write_text(json.dumps(promotion, indent=2) + '\n')
PY
    then
      status=1
    fi
  fi
  if [[ "$status" -eq 0 && "$MODE" != smoke && "$MODE" != smoke700 ]]; then
    if ! "$PYTHON" - "$RUN_DIR" "$ATTEMPT" "$AFTER" "$GATE_FILE" <<'PY'
import hashlib, json, sys
from pathlib import Path
run_dir, attempt, after, gate_file = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
names = ('manifest.json', 'source-hashes.sha256', 'test_predictions.jsonl',
         gate_file, 'journal.sqlite', 'request-budget.sqlite')
hashes = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
          for name in names}
hashes[after.name] = hashlib.sha256(after.read_bytes()).hexdigest()
apps = run_dir / f'gpu-apps-before-attempt-{attempt}.csv'
hashes[apps.name] = hashlib.sha256(apps.read_bytes()).hexdigest()
manifest = json.loads((run_dir / 'manifest.json').read_text())
completion = {'status': 'passed', 'run_id': manifest['run_id'],
              'git_commit': manifest['git_commit'],
              'model_revision': manifest['model_revision'],
              'attempts': attempt, 'artifact_sha256': hashes}
(run_dir / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
PY
    then
      status=1
    fi
  fi
  printf 'run_id=%s exit_status=%s gpu=%s\n' "$RUN_ID" "$status" "$GPU_INDEX"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "run_id=$RUN_ID job=$SLURM_JOB_ID gpu=$GPU_INDEX commit=$(git rev-parse HEAD)"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("torch", torch.__version__, "cuda", torch.version.cuda)'
"$PYTHON" -c 'import vllm, transformers; print("vllm", vllm.__version__, "transformers", transformers.__version__)'
"$PYTHON" -c 'import socket, sys; s=socket.socket(); e=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); sys.exit(0 if e else 1)' "$PORT"
if [[ "$MODE" != resume && "$MODE" != resume700 ]]; then
  sha256sum configs/memrec_instructrec-books_full_benchmark.yaml "$CONFIG" \
    src/models/llm_client.py src/models/llm_response_cache.py \
    src/models/reranker_llm.py src/models/memrec_agent.py \
    src/train/trainer_memrec.py src/train/books_run_journal.py \
    src/memory/storage.py src/data/books_protocol.py \
    "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json" > "$RUN_DIR/source-hashes.sha256"
  "$PYTHON" - "$RUN_DIR" "$MODEL_REVISION" "$SLURM_JOB_ID" "$GPU_INDEX" "$MODE" <<'PY'
import json, subprocess, sys
from pathlib import Path
from src.data.books_protocol import (
    books_cohorts, books_dev_cost_subset, cohort_digest, BOOKS_CANDIDATE_SHA256
)
run_dir, revision, job_id, gpu_index, mode = sys.argv[1:]
cohorts, manifest = books_cohorts(list(range(7377)))
subset_warmup, subset_eval = books_dev_cost_subset(
    cohorts['all'], cohorts['dev'], 700, 200
)
record = {
    'run_id': Path(run_dir).name,
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
    'model_revision': revision,
    'backend': 'vllm==0.10.2', 'transformers': '4.55.4',
    'tensor_parallel': 1, 'gpu_memory_utilization': 0.60,
    'max_model_len': 16384, 'max_num_seqs': 1, 'seed': 42,
    'response_cache': 'exact-input/write-only-in-smoke' if mode == 'smoke' else 'exact-input/read-write',
    'vllm_cache_root': '/mnt/data/users/anhnct/memrec-hnv/cache/vllm',
    'slurm_job_id': job_id, 'physical_gpu_index': int(gpu_index),
    'candidate_sha256': BOOKS_CANDIDATE_SHA256,
    'dev_cohort_sha256': manifest['cohort_sha256']['dev'],
    'warmup_user_scope': 'subset' if mode == 'dev700' else 'eval' if mode in ('smoke', 'smoke700') else 'all',
    'eval_cohort': 'dev', 'n_eval_users': 30 if mode in ('smoke', 'smoke700') else 200 if mode == 'dev700' else 2000,
    'subset_user_sha256': cohort_digest(subset_warmup) if mode in ('dev700', 'smoke700') else None,
    'subset_eval_sha256': cohort_digest(subset_eval) if mode in ('dev700', 'smoke700') else None,
    'smoke_user_sha256': cohort_digest(cohorts['dev'][:30]) if mode in ('smoke', 'smoke700') else None,
    'smoke_user_ids': cohorts['dev'][:30] if mode in ('smoke', 'smoke700') else None,
}
(Path(run_dir) / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
fi

"$VLLM" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.60 \
  --max-model-len 16384 --max-num-seqs 1 --seed 42 --dtype auto \
  --generation-config vllm --disable-log-requests \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$RUN_DIR/server.pid"
printf '%s\n' "$SERVER_PID" > "$RUN_DIR/server-attempt-$ATTEMPT.pid"
ready=0
for _ in $(seq 1 120); do
  if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -80 "$SERVER_LOG"
    echo 'vLLM server exited before readiness' >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  tail -80 "$SERVER_LOG"
  echo 'vLLM health check timed out' >&2
  exit 1
fi

if [[ "$MODE" == smoke ]]; then
  "$PYTHON" scripts/run_train.py \
    --dataset instructrec-books \
    --config configs/memrec_instructrec-books_full_benchmark.yaml \
    --device cpu --eval-cohort dev --n_eval_users 30 --warmup-user-scope eval \
    --output_dir "$RUN_DIR"
  "$PYTHON" scripts/check_books_memrec_smoke.py --run-dir "$RUN_DIR" \
    > "$RUN_DIR/smoke-gate.json"
  echo '30-user real-LLM smoke passed the output gate; GPU cleanup follows.'
elif [[ "$MODE" == smoke700 ]]; then
  "$PYTHON" scripts/run_train.py \
    --dataset instructrec-books --config "$CONFIG" \
    --device cpu --eval-cohort dev --n_eval_users 30 --warmup-user-scope eval \
    --output_dir "$RUN_DIR"
  "$PYTHON" scripts/check_books_memrec_smoke.py --run-dir "$RUN_DIR" \
    --allow-cache --reference-predictions "$SMOKE_DIR/test_predictions.jsonl" \
    > "$RUN_DIR/smoke-gate.json"
  echo '30-user subset-config real-LLM smoke passed; GPU cleanup follows.'
else
  timeout --signal=TERM --kill-after=120s "$RUN_TIMEOUT" \
    "$PYTHON" scripts/run_train.py \
      --dataset instructrec-books \
      --config "$CONFIG" \
      --device cpu --eval-cohort dev --n_eval_users "$N_EVAL_USERS" \
      --warmup-user-scope "$WARMUP_SCOPE" \
      --output_dir "$RUN_DIR"
  if [[ "$MODE" == dev700 || "$MODE" == resume700 ]]; then
    "$PYTHON" scripts/check_books_memrec_full_dev.py --run-dir "$RUN_DIR" \
      --protocol dev700 > "$RUN_DIR/$GATE_FILE"
  else
    "$PYTHON" scripts/check_books_memrec_full_dev.py --run-dir "$RUN_DIR" \
      --protocol full > "$RUN_DIR/$GATE_FILE"
  fi
  echo 'Books dev output gate passed; GPU cleanup follows.'
fi
