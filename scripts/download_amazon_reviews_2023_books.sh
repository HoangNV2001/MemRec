#!/usr/bin/env bash
# Download the two raw Amazon Reviews'23 Books archives named in
# docs/Amazon_Books_Review_README.md.  This script deliberately does not
# preprocess, decompress, or call an LLM.
#
# Safe default: it only prints the plan.  Download requires --download.
# Interrupted downloads resume from <file>.part when the host supports ranges.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/data/raw/amazon-reviews-2023/books"
DO_DOWNLOAD=0
CHECK_REMOTE=0

REVIEW_URL="https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Books.jsonl.gz"
META_URL="https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz"

usage() {
  cat <<'EOF'
Usage: bash scripts/download_amazon_reviews_2023_books.sh [options]

Download raw Amazon Reviews'23 Books review and metadata archives from the
official URLs linked by docs/Amazon_Books_Review_README.md.

Options:
  --download            Actually download. Without this flag the script is a dry run.
  --output-dir PATH     Destination (default: data/raw/amazon-reviews-2023/books).
  --check-remote        Print HTTP headers before a real download; no file content is fetched.
  -h, --help            Show this help.

Outputs (both are large, compressed JSON Lines):
  Books.jsonl.gz
  meta_Books.jsonl.gz

Examples:
  # Safe: show exactly what would be fetched; no network request and no files written.
  bash scripts/download_amazon_reviews_2023_books.sh

  # Optional: inspect remote Content-Length / range support only.
  bash scripts/download_amazon_reviews_2023_books.sh --check-remote

  # Download; an interrupted transfer resumes from a .part file.
  bash scripts/download_amazon_reviews_2023_books.sh --download
EOF
}

while (($#)); do
  case "$1" in
    --download)
      DO_DOWNLOAD=1
      ;;
    --output-dir)
      shift
      if (($# == 0)); then
        echo "error: --output-dir requires a path" >&2
        exit 2
      fi
      OUTPUT_DIR="$1"
      ;;
    --check-remote)
      CHECK_REMOTE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

print_plan() {
  cat <<EOF
Amazon Reviews'23 Books raw-download plan
  destination: ${OUTPUT_DIR}
  review:      ${REVIEW_URL}
  metadata:    ${META_URL}

No download occurs without --download.  Archives and resumable .part files are
gitignored.  The source README provides no checksum, so a SHA-256 manifest is
written locally after a successful download.
EOF
}

download_one() {
  local url="$1"
  local filename="$2"
  local destination="${OUTPUT_DIR}/${filename}"
  local partial="${destination}.part"
  local -a resume_args=()

  if [[ -f "${destination}" ]]; then
    echo "already present; leaving unchanged: ${destination}"
    return
  fi

  echo "downloading ${filename}"
  if [[ -f "${partial}" ]]; then
    resume_args=(--continue-at -)
  fi
  # --continue-at - resumes an interrupted .part transfer.  A final filename
  # appears only after curl succeeds, so downstream preprocessing never reads a
  # partial archive by mistake.
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    "${resume_args[@]}" \
    --output "${partial}" \
    "${url}"
  mv -- "${partial}" "${destination}"
}

print_plan

if ((DO_DOWNLOAD == 0 && CHECK_REMOTE == 0)); then
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required for download" >&2
  exit 1
fi

if ((CHECK_REMOTE == 1)); then
  echo
  echo "Remote headers (no archive content is fetched):"
  curl --fail --location --head "${REVIEW_URL}"
  curl --fail --location --head "${META_URL}"
  echo
fi

if ((DO_DOWNLOAD == 0)); then
  exit 0
fi

mkdir -p -- "${OUTPUT_DIR}"

download_one "${REVIEW_URL}" "Books.jsonl.gz"
download_one "${META_URL}" "meta_Books.jsonl.gz"

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd -- "${OUTPUT_DIR}"
    sha256sum Books.jsonl.gz meta_Books.jsonl.gz > SHA256SUMS.txt
  )
  echo "wrote ${OUTPUT_DIR}/SHA256SUMS.txt"
else
  echo "warning: sha256sum is unavailable; no local checksum manifest was written" >&2
fi
