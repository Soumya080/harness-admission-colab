"""python -m sims.plot_stage0 -> results/fig_lambda.png (log-growth g vs lambda for several p)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os
OUT = Path(os.environ.get("HA_HOME", ".")) / "results"

lams = np.linspace(0.01, 0.95, 200)
plt.figure(figsize=(4.5, 3.2))
for p in (0.6, 0.631, 0.7, 0.8):
    g = p * np.log1p(lams) + (1 - p) * np.log1p(-lams)
    plt.plot(lams, g, label=f"p={p}")
    if p > 0.5: plt.scatter([2*p - 1], [p*np.log(2*p) + (1-p)*np.log(2-2*p)], s=12)  # Kelly optimum
plt.axhline(0, color="k", lw=0.6); plt.axvline(0.5, ls="--", lw=0.6)
plt.xlabel("betting fraction λ"); plt.ylabel("expected log-growth per discordant pair")
plt.legend(); plt.tight_layout()
OUT.mkdir(parents=True, exist_ok=True); plt.savefig(OUT / "fig_lambda.png", dpi=200)
print("saved", OUT / "fig_lambda.png")
