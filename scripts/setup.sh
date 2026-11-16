#!/usr/bin/env bash
# One-time setup on the GPU machine. Run from the repo root.
set -euo pipefail
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install vllm datasets numpy scipy pandas pyarrow matplotlib pytest
for d in data runner scripts sim analysis gates harness proposer sims; do touch "$d/__init__.py"; done
python - <<'PY'
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import vllm; print("vLLM", vllm.__version__)
PY
