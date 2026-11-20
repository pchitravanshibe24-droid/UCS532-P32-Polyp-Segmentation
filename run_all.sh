#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/Scripts/python.exe ]]; then
        python_bin=.venv/Scripts/python.exe
    elif [[ -x .venv/bin/python ]]; then
        python_bin=.venv/bin/python
    fi
fi
export PYTHONHASHSEED=42
export MPLBACKEND=Agg
export GRADIO_ANALYTICS_ENABLED=False
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$python_bin" -u -m scripts.run_experiments "$@"
