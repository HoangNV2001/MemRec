#!/usr/bin/env bash
# M1 step 2 — render the frozen snapshot into train/val/test jsonl for TRL.
#
# Pure CPU, no API calls: safe and cheap to rerun after any prompt-template change.
#
#   bash scripts/rl/01_build_dataset.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/rl/m1_env_books.yaml}"
# Set PYTHON=... to point at the project env without activating it first, e.g.
#   PYTHON="conda run -n memrec python" bash scripts/rl/01_build_dataset.sh
PYTHON="${PYTHON:-python}"

$PYTHON -m src.rl.build_dataset --config "$CONFIG"

echo
echo "Verifying M1 DoD (splits disjoint + no gold leakage in prompts)..."
$PYTHON -m pytest tests/rl/ -q
