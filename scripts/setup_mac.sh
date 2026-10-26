#!/usr/bin/env bash
set -euo pipefail
python_exe="${1:-python3.11}"
if [ ! -x .venv/bin/python ]; then
    "$python_exe" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required to match Windows"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.11.0'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.check_device
.venv/bin/python -c 'import torch; assert torch.__version__.split("+")[0] == "2.11.0"'
