#!/usr/bin/env bash
# M2 — reward validation, split by tier so the H100 does the minimum (§2.5.1, §11.6).
#
#   Part A (CPU + API, run FIRST, before renting anything):
#       bash scripts/rl/02_validate_reward.sh reference   # ~$0.5, caches gpt-4o-mini side
#       bash scripts/rl/02_validate_reward.sh stub        # proves the pipeline, $0
#
#   Part B (H100):
#       bash scripts/rl/02_validate_reward.sh hf
#
# DoD: Spearman rho >= 0.6 · r(real) > r(other user) > r(lorem) ~ r(empty) ·
#      throughput >= 20 reward/s at batch 64.
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/rl/m1_env_books.yaml}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-24}"
RANKER_MODEL="${RANKER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

case "${1:-stub}" in
  reference)
    # Caches the gpt-4o-mini half. Refuses to overwrite: rerunning costs money.
    $PYTHON -m src.rl.build_val_reference --config "$CONFIG" --workers "$WORKERS"
    ;;
  stub)
    $PYTHON -m src.rl.validate_reward --config "$CONFIG" --ranker_mode stub \
        --out data/rl/m2_validation_report_stub.json
    ;;
  hf)
    $PYTHON -m src.rl.validate_reward --config "$CONFIG" --ranker_mode hf \
        --ranker_model "$RANKER_MODEL" --device cuda \
        --out data/rl/m2_validation_report.json
    # §5.1 leaves open whether the proxy should see the InstructRec instruction:
    # including it keeps the proxy faithful to LLM_Rec, excluding it stops the
    # instruction from flattening the reward across different M_collab. Measure both.
    $PYTHON -m src.rl.validate_reward --config "$CONFIG" --ranker_mode hf \
        --ranker_model "$RANKER_MODEL" --device cuda --no_instruction \
        --out data/rl/m2_validation_report_no_instruction.json
    ;;
  *)
    echo "usage: $0 {reference|stub|hf}" >&2
    exit 2
    ;;
esac
