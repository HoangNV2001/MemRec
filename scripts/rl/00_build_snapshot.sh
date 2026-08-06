#!/usr/bin/env bash
# M1 step 1 — warm up memory for the RL user set and freeze the graph snapshot.
#
# CPU + gpt-4o-mini only. No GPU (RL_PLAN.md M1, §2.5.1 tier T0/T0-API).
# Cost is ~$3 for 2350 users on instructrec-books; the script refuses to clobber
# an existing snapshot unless FORCE=1, because rebuilding re-spends that budget.
#
#   bash scripts/rl/00_build_snapshot.sh
#   FORCE=1 WORKERS=24 bash scripts/rl/00_build_snapshot.sh
#   LIMIT_USERS=3 bash scripts/rl/00_build_snapshot.sh     # plumbing smoke test
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/rl/m1_env_books.yaml}"
WORKERS="${WORKERS:-24}"
OUTPUT_DIR="${OUTPUT_DIR:-results/m1_warmup_2350}"
# Set PYTHON=... to point at the project env without activating it first, e.g.
#   PYTHON="conda run -n memrec python" bash scripts/rl/00_build_snapshot.sh
PYTHON="${PYTHON:-python}"

ARGS=(--config "$CONFIG" --parallel_workers "$WORKERS" --output_dir "$OUTPUT_DIR")
[[ "${FORCE:-0}" == "1" ]] && ARGS+=(--force)
[[ -n "${LIMIT_USERS:-}" ]] && ARGS+=(--limit_users "$LIMIT_USERS")
# Re-materialise the snapshot offline from an existing warm-up (free, no API):
#   MEMORY_FILE=results/m1_warmup_2350/memory_warmup_only.json FORCE=1 bash scripts/rl/00_build_snapshot.sh
[[ -n "${MEMORY_FILE:-}" ]] && ARGS+=(--memory_file "$MEMORY_FILE")

exec $PYTHON -m src.rl.build_snapshot "${ARGS[@]}"
