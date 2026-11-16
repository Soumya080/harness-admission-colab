import itertools, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim.simulate import run_many, MAIN, ABLATIONS
from analysis.metrics import summarize

def build_table(O, S, AU, start, edits, regimes=("controlled", "stochastic"),
                n_devs=(20, 40, 100), gates=MAIN, runs=1000, seed=0):
    rows = []
    for reg, n, g in itertools.product(regimes, n_devs, gates):
        r = summarize(run_many(O, S, start, edits, g, n, reg, runs=runs, seed=seed), AU, start)
        rows.append(dict(regime=reg, n_dev=n, gate=g, **r))
    return pd.DataFrame(rows)

def build_ablation(O, S, AU, start, edits, regime="stochastic", n_dev=40, runs=1000, seed=0):
    rows = []
    for g in MAIN + ABLATIONS:
        r = summarize(run_many(O, S, start, edits, g, n_dev, regime, runs=runs, seed=seed), AU, start)
        rows.append(dict(variant=g, **r))
    # the key control: greedy with a FRESH dev subset per candidate (no reuse)
    r = summarize(run_many(O, S, start, edits, "greedy", n_dev, regime, runs=runs, seed=seed,
                           fresh_dev=True), AU, start)
    rows.append(dict(variant="greedy_fresh_dev", **r))
    return pd.DataFrame(rows)

def fig_false_vs_n(tab, regime, path):
    t = tab[tab.regime == regime]
    plt.figure(figsize=(4.5, 3.2))
    for g, d in t.groupby("gate"):
        plt.plot(d.n_dev, 100 - d.helpful_pct, marker="o", label=g)
    plt.xlabel("dev set size n"); plt.ylabel("commits not verified helpful (%)")
    plt.title(f"{regime} regime"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200)
    plt.close()
