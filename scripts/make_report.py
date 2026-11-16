"""CPU only. Cache -> matrices -> simulation -> results/.
python -m scripts.make_report [--runs 1000] [--seed 0] [--ablations]"""
import argparse, json
from pathlib import Path
import numpy as np
from harness.store import list_harnesses, load_harness
from data.pools import load_pools, HOME
from runner import cache
from sim.load import load_matrices
from analysis.report import build_table, build_ablation, fig_false_vs_n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablations", action="store_true")
    a = ap.parse_args()
    out = HOME / "results"; out.mkdir(exist_ok=True)

    hs = [load_harness(x) for x in sorted(list_harnesses())]
    ids = [h.id for h in hs]
    base = [k for k, h in enumerate(hs) if h.tag == "base"]
    assert len(base) == 1, f"need exactly one base harness, found {len(base)}"
    start = base[0]
    edits = np.array([k for k in range(len(hs)) if k != start])
    pools = load_pools()
    O, S, AU = load_matrices(cache.load_all(), ids, pools["dev"], pools["audit"])

    acc = AU.mean(axis=1)
    (out / "audit_acc.json").write_text(json.dumps(
        {f"{h.tag}:{h.id}": round(float(x), 3) for h, x in zip(hs, acc)}, indent=1))
    print(f"base audit acc {acc[start]:.3f} | best edit {acc[edits].max():.3f} | edits {len(edits)}")

    sfx = f"_s{a.seed}" if a.seed else ""
    tab = build_table(O, S, AU, start, edits, runs=a.runs, seed=a.seed)
    tab.round(2).to_csv(out / f"table1{sfx}.csv", index=False)
    print(tab.round(1).to_string(index=False))
    for reg in ("controlled", "stochastic"):
        fig_false_vs_n(tab, reg, out / f"fig_{reg}{sfx}.png")
    if a.ablations:
        ab = build_ablation(O, S, AU, start, edits, runs=a.runs, seed=a.seed)
        ab.round(2).to_csv(out / f"ablations{sfx}.csv", index=False)
        print(ab.round(1).to_string(index=False))

if __name__ == "__main__":
    main()
