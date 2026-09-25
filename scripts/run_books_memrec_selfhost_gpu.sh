#!/usr/bin/env bash
# Real-LLM Books smoke. Invoke ONLY through the authorized train_TTS Slurm step.
set -euo pipefail

if [[ $# -ne 0 || -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Run without arguments, inside the authorized Slurm allocation only' >&2
  exit 2
fi

MEMREC_ROOT=/mnt/data/users/anhnct/memrec-hnv
REPO="$MEMREC_ROOT/repo/MemRec-hnv"
RUN_ID=books-memrec-llm-smoke-v2-hnv
RUN_DIR="$MEMREC_ROOT/runs/$RUN_ID"
LOG_DIR="$MEMREC_ROOT/logs"
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
MODEL_REVISION=5a5a776300a41aaa681dd7ff0106608ef2bc90db
MODEL_PATH="$MEMREC_ROOT/models/Qwen3-30B-A3B-Instruct-2507-FP8-hnv"
PYTHON="$MEMREC_ROOT/envs/llm-hnv/bin/python"
VLLM="$MEMREC_ROOT/envs/llm-hnv/bin/vllm"
PORT=18081

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
if [[ -e "$RUN_DIR" ]]; then
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

mkdir -p "$LOG_DIR" "$RUN_DIR"
exec > >(tee "$LOG_DIR/$RUN_ID.log") 2>&1
BEFORE="$RUN_DIR/gpu-before.csv"
AFTER="$RUN_DIR/gpu-after.csv"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$BEFORE"
if [[ "$(wc -l < "$BEFORE")" -ne 4 ]]; then
  echo 'Expected four GPUs in preflight snapshot' >&2
  exit 2
fi
GPU_INDEX=$(sort -t, -k2,2n -k3,3n "$BEFORE" | head -1 | cut -d, -f1 | tr -d ' ')
GPU_ROW=$(grep "^$GPU_INDEX," "$BEFORE")
GPU_UTIL=$(printf '%s\n' "$GPU_ROW" | cut -d, -f2 | tr -d ' ')
GPU_MEM_BEFORE=$(printf '%s\n' "$GPU_ROW" | cut -d, -f3 | tr -d ' ')
if [[ ! "$GPU_INDEX" =~ ^[0-3]$ || ! "$GPU_UTIL" =~ ^[0-9]+$ || ! "$GPU_MEM_BEFORE" =~ ^[0-9]+$ ]]; then
  echo 'Malformed selected-GPU snapshot' >&2
  exit 2
fi
if (( GPU_UTIL >= 20 || GPU_MEM_BEFORE >= 2048 )); then
  echo 'No sufficiently idle H100 for this run' >&2
  exit 2
fi
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
export MEMREC_LLM_CACHE_NAMESPACE="$MODEL_REVISION:vllm0.10.2:tp1:seed42:mem0.60:len16384"
export MEMREC_LLM_CACHE_READ=0
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
  if [[ "$status" -eq 0 ]]; then
    if ! "$PYTHON" - "$RUN_DIR" "$MEMREC_LLM_CACHE_NAMESPACE" <<'PY'
import hashlib, json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
names = ('manifest.json', 'source-hashes.sha256', 'test_predictions.jsonl',
         'smoke-gate.json', 'gpu-before.csv', 'gpu-after.csv')
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
sha256sum configs/memrec_instructrec-books_full_benchmark.yaml \
  src/models/llm_client.py src/models/llm_response_cache.py \
  src/models/reranker_llm.py src/models/memrec_agent.py \
  "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json" > "$RUN_DIR/source-hashes.sha256"
"$PYTHON" - "$RUN_DIR" "$MODEL_REVISION" "$SLURM_JOB_ID" "$GPU_INDEX" <<'PY'
import json, subprocess, sys
from pathlib import Path
from src.data.books_protocol import books_cohorts, cohort_digest, BOOKS_CANDIDATE_SHA256
run_dir, revision, job_id, gpu_index = sys.argv[1:]
cohorts, manifest = books_cohorts(list(range(7377)))
record = {
    'run_id': Path(run_dir).name,
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
    'model_revision': revision,
    'backend': 'vllm==0.10.2', 'transformers': '4.55.4',
    'tensor_parallel': 1, 'gpu_memory_utilization': 0.60,
    'max_model_len': 16384, 'max_num_seqs': 1, 'seed': 42,
    'response_cache': 'exact-input/write-only-in-smoke',
    'vllm_cache_root': '/mnt/data/users/anhnct/memrec-hnv/cache/vllm',
    'slurm_job_id': job_id, 'physical_gpu_index': int(gpu_index),
    'candidate_sha256': BOOKS_CANDIDATE_SHA256,
    'dev_cohort_sha256': manifest['cohort_sha256']['dev'],
    'smoke_user_sha256': cohort_digest(cohorts['dev'][:30]),
    'smoke_user_ids': cohorts['dev'][:30],
}
(Path(run_dir) / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY

"$VLLM" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.60 \
  --max-model-len 16384 --max-num-seqs 1 --seed 42 --dtype auto \
  --generation-config vllm --disable-log-requests \
  > "$RUN_DIR/vllm-server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 120); do
  if curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -80 "$RUN_DIR/vllm-server.log"
    echo 'vLLM server exited before readiness' >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  tail -80 "$RUN_DIR/vllm-server.log"
  echo 'vLLM health check timed out' >&2
  exit 1
fi

"$PYTHON" scripts/run_train.py \
  --dataset instructrec-books \
  --config configs/memrec_instructrec-books_full_benchmark.yaml \
  --device cpu --eval-cohort dev --n_eval_users 30 --warmup-user-scope eval \
  --output_dir "$RUN_DIR"
"$PYTHON" scripts/check_books_memrec_smoke.py --run-dir "$RUN_DIR" \
  > "$RUN_DIR/smoke-gate.json"
echo '30-user real-LLM smoke passed the output gate; GPU cleanup follows.'
