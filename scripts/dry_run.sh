#!/usr/bin/env bash
# Whole pipeline on CPU with a fake LLM, in a throwaway folder. Must finish green before any GPU run.
set -euo pipefail
export HA_FAKE=1 HA_HOME="$(mktemp -d)"
echo "dry run in $HA_HOME"
python -m data.pools
python -m proposer.manual
python -m scripts.run_grid --only base
python -m proposer.propose --n 8
python -m scripts.run_grid
python -m scripts.diagnose
python -m scripts.make_report --runs 50 --ablations
echo "DRY RUN OK -> $HA_HOME/results"
