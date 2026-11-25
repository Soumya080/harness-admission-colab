"""How well can a small validation set tell good prompts apart? CPU only.
Uses prompts that score >= 80% on the 400-item audit ("good prompts").
Reports their pairwise discordance and true gaps, the paired standard error at several validation sizes,
and how often an n-item comparison (temperature 0.7 samples) picks the worse of two prompts whose audit
accuracies differ by >= 2 points.   python -m scripts.selection_resolution [--n 100] [--reps 200]"""
import argparse, itertools, json
import numpy as np
from harness.store import list_harnesses, load_harness
from data.pools import load_pools, HOME
from runner import cache
from sim.load import load_matrices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    hs = [load_harness(x) for x in sorted(list_harnesses())]
    p = load_pools()
    O, S, AU = load_matrices(cache.load_all(), [h.id for h in hs], p["dev"], p["audit"])
    acc = AU.mean(axis=1)
    top = [k for k in range(len(hs)) if acc[k] >= 0.80]
    pairs = list(itertools.combinations(top, 2))
    q = np.median([np.mean(AU[x] != AU[y]) for x, y in pairs])
    gap = np.median([abs(acc[x] - acc[y]) for x, y in pairs])
    out = {"good_prompts": len(top), "median_discordance": round(float(q), 3),
           "median_true_gap_pts": round(100 * float(gap), 1), "resolution": {}}
    for n in (40, 100, 400):
        se = np.sqrt(q / n)
        out["resolution"][n] = {"paired_se_pts": round(100 * se, 1), "ci95_halfwidth_pts": round(196 * se, 1),
                                "mde_80pct_power_pts": round(280 * se, 1)}
    rng = np.random.default_rng(a.seed); wrong = []
    k = S.shape[2]
    for x, y in pairs:
        if abs(acc[x] - acc[y]) < 0.02:
            continue
        better, worse = (x, y) if acc[x] > acc[y] else (y, x)
        for _ in range(a.reps):
            items = rng.choice(S.shape[1], a.n, replace=False)
            b = S[better, items, rng.integers(k, size=a.n)].mean()
            w = S[worse, items, rng.integers(k, size=a.n)].mean()
            wrong.append(w > b)
    out[f"picks_worse_at_n{a.n}_pct"] = round(100 * float(np.mean(wrong)), 1)
    (HOME / "results").mkdir(exist_ok=True)
    (HOME / "results" / "selection_resolution.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
