#!/usr/bin/env bash
set -euo pipefail
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10-3.12 required"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.check_device
.venv/bin/python -c "import torch; assert torch.backends.mps.is_available(), 'MPS unavailable'; print('MPS ready')"
