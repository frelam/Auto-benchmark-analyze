#!/usr/bin/env bash
# Install benchmark-diagnosis with its optional evaluation / chart / serving deps
# and (optionally) prefetch the benchmark datasets.
#
# Usage:
#   bash scripts/install.sh                   # CPU-safe: eval + charts
#   bash scripts/install.sh --with-gpu        # adds vLLM serving + torch (mirt)
#   bash scripts/install.sh --download-data   # also prefetch benchmark datasets
#
# Requires Python 3.10+.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="$HERE/.venv"

log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Install benchmark-diagnosis into $HERE/.venv.

Usage: bash scripts/install.sh [options]

Options:
  --with-gpu        Also install the GPU extras (vLLM serving + torch/mirt).
  --download-data   After installing, best-effort prefetch benchmark datasets.
  -h, --help        Show this help and exit.

The base install always includes evaluation (lm-eval) and chart (matplotlib)
support. Python 3.10+ is required.
EOF
}

WITH_GPU=0
DOWNLOAD_DATA=0
for arg in "$@"; do
  case "$arg" in
    --with-gpu) WITH_GPU=1 ;;
    --download-data) DOWNLOAD_DATA=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

# --- 1. Python version check ------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  die "Python not found: $PYTHON_BIN (install Python 3.10+ first)."
fi
py_major="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')"
py_minor="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')"
if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 10 ]; }; then
  die "Python 3.10+ is required (found $py_major.$py_minor)."
fi
log "Python $py_major.$py_minor found."

# --- 2. Create / reuse the virtualenv ---------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
  log "Creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Reusing existing virtualenv at $VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"
"$VENV_PY" -m pip install --upgrade pip >/dev/null

# --- 3. Install the package with extras -------------------------------------
EXTRAS="eval,plot"
if [ "$WITH_GPU" -eq 1 ]; then
  EXTRAS="$EXTRAS,serve,mirt"
  log "Installing extras [$EXTRAS] (vLLM + torch included)."
else
  log "Installing extras [$EXTRAS] (add --with-gpu for vLLM serving + torch)."
fi
"$VENV_PY" -m pip install -e "$HERE[$EXTRAS]"

# --- 4. Optional dataset prefetch -------------------------------------------
if [ "$DOWNLOAD_DATA" -eq 1 ]; then
  log "Prefetching benchmark datasets (best-effort; never fails the install)."
  "$VENV_PY" "$HERE/scripts/prefetch_tasks.py" || warn "Dataset prefetch skipped; datasets still download lazily at first eval."
fi

# --- 5. Next steps -----------------------------------------------------------
log "Done."
cat <<EOF

Next steps:
  cd "$HERE"
  source .venv/bin/activate
  benchmark-diagnosis --help

  # Run the offline example (no GPU / endpoint needed — reads a JSON scores file):
  benchmark-diagnosis --config examples/run-from-scores.yaml run

  # Full run: auto-deploy weights, evaluate, analyze, and advise:
  benchmark-diagnosis run --model-path meta-llama/Llama-3.1-8B --mode full

  # Reuse an existing inference service instead:
  benchmark-diagnosis run --model my-model --base-url http://10.0.0.5:8000/v1 --mode full
EOF
