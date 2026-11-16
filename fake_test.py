# synthetic cache: harness 0 = handicapped base (acc .35), harness 1 = planted (acc .75),
# 30 "llm" edits with true acc near base (some slightly better/worse), noisy item-level outcomes
import numpy as np, pandas as pd
from sim.load import load_matrices
from analysis.report import build_table, fig_false_vs_n
rng = np.random.default_rng(1)
H = 32; dev = list(range(100, 300)); aud = list(range(300, 700))
true = np.r_[0.35, 0.75, 0.35 + rng.normal(0, 0.02, 30)]
diff = rng.normal(0, 1.0, 1000)               # item difficulty shared across harnesses
rows = []
for h in range(H):
    for i in dev + aud:
        p = 1 / (1 + np.exp(-(np.log(true[h] / (1 - true[h])) - diff[i])))
        rows.append(("h%d" % h, i, 0.0, 0, int(rng.random() < p)))
        if i in dev:
            for s in range(5):
                rows.append(("h%d" % h, i, 0.7, s, int(rng.random() < p)))
df = pd.DataFrame(rows, columns=["harness", "item", "temp", "sample", "correct"])
ids = ["h%d" % h for h in range(H)]
O, S, AU = load_matrices(df, ids, dev, aud)
tab = build_table(O, S, AU, start=0, edits=list(range(1, H)), runs=200)
print(tab[["regime","n_dev","gate","commits","helpful_pct","null_pct","harmful_pct","final_gain","cost"]].round(1).to_string(index=False))
fig_false_vs_n(tab, "stochastic", "fig.png")
