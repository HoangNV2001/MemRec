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
# 3B, not the planned 1.5B: M2 Part B measured 1.5B at Spearman 0.307 (DoD 0.6)
# with `lorem` memory outscoring real memory. See docs/RESULTS.md.
RANKER_MODEL="${RANKER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
# The DoD throughput number is defined at batch 64, but 3B in fp32 does not fit
# that on a 24 GB card -- it OOMs. Override on smaller GPUs (batch 32 fits an L4)
# and remember that Validation C is then NOT the DoD measurement.
BATCH_SIZE="${BATCH_SIZE:-64}"

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
        --ranker_model "$RANKER_MODEL" --device cuda --batch_size "$BATCH_SIZE" \
        --out data/rl/m2_validation_report.json \
        --dump_pairs data/rl/m2_pairs.json
    # The instruction control. M2 Part B settled it -- dropping the instruction more
    # than halves the correlation, so include_instruction stays True -- but the run
    # is kept so the claim can be re-checked whenever the ranker changes.
    $PYTHON -m src.rl.validate_reward --config "$CONFIG" --ranker_mode hf \
        --ranker_model "$RANKER_MODEL" --device cuda --batch_size "$BATCH_SIZE" \
        --no_instruction \
        --out data/rl/m2_validation_report_no_instruction.json
    ;;
  *)
    echo "usage: $0 {reference|stub|hf}" >&2
    exit 2
    ;;
esac
