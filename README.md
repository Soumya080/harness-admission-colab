# harness-admission

When should an automated edit to an LLM harness (system prompt) be accepted?
Reproduces PACE (anytime-valid paired commit gate) against greedy and McNemar acceptance
on GSM8K with Qwen2.5-3B-Instruct as executor and Qwen2.5-7B-Instruct-AWQ as proposer.

## Run
- Colab (T4): open `colab_run.ipynb`, run cells top to bottom. Results persist on Google Drive.
- Linux GPU: `bash scripts/setup.sh && source .venv/bin/activate && bash scripts/run_all.sh`
- CPU dry run (fake model): `HA_FAKE=1 HA_HOME=$(mktemp -d) bash scripts/run_all.sh`

Env vars: `HA_HOME` (where cache/configs/results go), `HA_DTYPE` (`half` on T4),
`HA_GPU_UTIL` (default 0.85), `HA_FAKE=1` (CPU stand-in for vLLM).

## Layout
data/ pools · harness/ harness + store · runner/ executor, cache, extraction · proposer/ edits ·
gates/ greedy, McNemar, PACE · sim/ simulator · analysis/ metrics, report ·
sims/ stage-0 simulations · scripts/ grid, diagnostics, report · tests/
