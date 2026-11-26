#!/usr/bin/env bash
# Full experiment. Every step is resumable: rerun this script after any crash.
# Usage:  bash scripts/run_all.sh            (real GPU run)
#         HA_FAKE=1 HA_HOME=$(mktemp -d) bash scripts/run_all.sh   (CPU dry run)
set -euo pipefail
OUT="${HA_HOME:-.}"
mkdir -p "$OUT/logs" "$OUT/results"
LOG="$OUT/logs/run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "0. tests"
python -m pytest -q tests

step "1. pools (skipped if they exist)"
POOLS="${HA_HOME:-.}/configs/pools_${HA_DATASET:-gsm8k}.json"
[ -f "$POOLS" ] || python -m data.pools

step "2. manual harnesses (base, planted, noops)"
python -m proposer.manual

step "3. base on proposal/dev/audit  (GPU, 3B)"
python -m scripts.run_grid --only base

step "4. planted  (GPU, 3B; skipped in the clean regime)"
[ "${HA_REGIME:-handicap}" = "clean" ] || python -m scripts.run_grid --only planted

step "5. STOP CHECK"
python -m scripts.check_stop || { echo "Fix the problem above before continuing."; exit 1; }

step "6. proposer: 30 LLM edits  (GPU, 7B AWQ, separate process)"
N_LLM=$(python - <<'PY'
from harness.store import list_harnesses, load_harness
print(sum(load_harness(x).tag == "llm" for x in list_harnesses()))
PY
)
if [ "$N_LLM" -lt 30 ]; then python -m proposer.propose --n $((30 - N_LLM)); else echo "already have $N_LLM llm edits"; fi

step "7. full grid  (GPU, 3B; the long step)"
python -m scripts.run_grid

step "8. diagnostics"
python -m scripts.diagnose | tee "$OUT/results/diagnose.txt"

step "9. stage-0 sims + lambda figure (CPU)"
python -m sims.stage0 | tee "$OUT/results/stage0.txt"
python -m sims.plot_stage0

step "10. report + ablations (CPU)"
python -m scripts.make_report --runs 1000 --ablations
python -m scripts.claimed_vs_real
python -m scripts.selection_resolution

echo; echo "ALL DONE. Log: $LOG   Results: ${HA_HOME:-.}/results"
