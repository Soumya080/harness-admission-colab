"""Claimed vs real improvement per gate. CPU only.
Claimed = the gain each commit showed on the dev items the gate looked at (what a commit log would report).
Real    = the same commit's gain on the 400-item held-out audit set.
python -m scripts.claimed_vs_real [--runs 1000] [--n-dev 40] [--seed 0]"""
import argparse
import numpy as np
import pandas as pd
from harness.store import list_harnesses, load_harness
from data.pools import load_pools, HOME
from runner import cache
from sim.load import load_matrices
from sim.simulate import GATES

def simulate(O, S, acc, start, edits, gate, n_dev, regime, runs, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(runs):
        dev = rng.choice(O.shape[1], n_dev, replace=False)
        inc, claimed, real = start, [], []
        for cand in rng.permutation(edits):
            order = rng.permutation(dev)
            if regime == "controlled":
                c, i = O[cand, order], O[inc, order]
            else:
                k = S.shape[2]
                c = S[cand, order, rng.integers(k, size=len(order))]
                i = S[inc, order, rng.integers(k, size=len(order))]
            r = gate(zip(c.tolist(), i.tolist()))
            if r.admitted:
                u = r.samples_used
                claimed.append(100 * (c[:u].astype(float) - i[:u]).mean())
                real.append(100 * (acc[cand] - acc[inc]))
                inc = cand
        rows.append(dict(commits=len(claimed), claimed=sum(claimed), real=sum(real),
                         claimed_after_first=sum(claimed[1:]), real_after_first=sum(real[1:])))
    return pd.DataFrame(rows).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    hs = [load_harness(x) for x in sorted(list_harnesses())]
    start = [k for k, h in enumerate(hs) if h.tag == "base"][0]
    edits = np.array([k for k in range(len(hs)) if k != start])
    p = load_pools()
    O, S, AU = load_matrices(cache.load_all(), [h.id for h in hs], p["dev"], p["audit"])
    acc = AU.mean(axis=1)
    out = []
    for regime in ("stochastic", "controlled"):
        for g in ("greedy", "mcnemar", "pace"):
            r = simulate(O, S, acc, start, edits, GATES[g], a.n_dev, regime, a.runs, a.seed)
            out.append(dict(regime=regime, gate=g, **r.round(1).to_dict()))
    df = pd.DataFrame(out)
    (HOME / "results").mkdir(exist_ok=True)
    df.to_csv(HOME / "results" / "claimed_vs_real.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
