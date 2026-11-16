"""Fill the outcome grid for every stored harness. Safe to kill and rerun.
python -m scripts.run_grid [--only base,planted]  (tags)"""
import argparse, time
from harness.store import list_harnesses, load_harness
from data.pools import load_ds, load_pools, gold
from runner import cache
from runner.run import run

JOBS = [("dev", 0.0, 1), ("dev", 0.7, 5), ("audit", 0.0, 1)]   # (split, temp, n_samples)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    ds, pools = load_ds(), load_pools()
    hs = [load_harness(x) for x in list_harnesses()]
    if only: hs = [h for h in hs if h.tag in only]
    print(f"{len(hs)} harnesses")
    for h in hs:
        jobs = JOBS + ([("proposal", 0.0, 1)] if h.tag == "base" else [])
        for split, temp, n in jobs:
            if cache.has(h.id, split, temp):
                continue
            t0 = time.time()
            rows = run(h, pools[split], split, temp, n, ds, gold)
            cache.write(rows, h.id, split, temp)
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"{h.tag:8s} {h.id} {split:5s} t={temp:g} acc={acc:.3f} {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
