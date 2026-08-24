#!/usr/bin/env bash
# Run the offline example end-to-end and list the generated output.
#
#   bash examples/run_example.sh
#
# This needs no GPU and no inference endpoint: it reads examples/scores.json and
# writes report.md + metrics.json + figures/ under examples/output/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

BMD="${BMD:-benchmark-diagnosis}"
if ! command -v "$BMD" >/dev/null 2>&1; then
  if [ -x ".venv/bin/benchmark-diagnosis" ]; then
    BMD=".venv/bin/benchmark-diagnosis"
  else
    echo "[example] 'benchmark-diagnosis' not found on PATH." >&2
    echo "[example] Run 'bash scripts/install.sh' first (or export BMD=path/to/benchmark-diagnosis)." >&2
    exit 1
  fi
fi

"$BMD" --config examples/run-from-scores.yaml run --diagnose

echo
echo "[example] Output written to examples/output/:"
ls -1 examples/output
echo "  figures/:"
ls -1 examples/output/figures
