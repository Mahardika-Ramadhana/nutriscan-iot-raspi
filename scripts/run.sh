#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
elif [[ -d "venv" ]]; then
    # shellcheck disable=SC1091
    source "venv/bin/activate"
fi

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-ARMV8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m nutriscan.core.app "$@"
