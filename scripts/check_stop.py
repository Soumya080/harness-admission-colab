"""Stop check after base + planted. Exit code 1 = do NOT continue.
python -m scripts.check_stop [--min-gain 10] [--max-noans 0.10]"""
import argparse, sys
from harness.store import list_harnesses, load_harness
from runner import cache
from runner.extract import extract_gsm8k_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gain", type=float, default=10.0)
    ap.add_argument("--max-noans", type=float, default=0.10)
    a = ap.parse_args()
    tags = {h.id: h.tag for h in map(load_harness, list_harnesses())}
    df = cache.load_all()
    df = df[df.harness.map(tags).isin(["base", "planted"])]
    df["tag"] = df.harness.map(tags)
    au = df[(df.split == "audit") & (df.temp == 0)]
    acc = au.groupby("tag").correct.mean()
    noans = df.assign(na=df.text.map(lambda t: extract_gsm8k_answer(t) is None)).groupby("tag").na.mean()
    gain = 100 * (acc["planted"] - acc["base"])
    print(f"audit acc  base {acc['base']:.3f}  planted {acc['planted']:.3f}  gain {gain:.1f} pts")
    print(f"no-answer  base {noans['base']:.3f}  planted {noans['planted']:.3f}")
    ok = True
    if gain < a.min_gain:
        print(f"STOP: planted gain {gain:.1f} < {a.min_gain}. The handicap is too weak; fix HANDICAP."); ok = False
    if noans.max() > a.max_noans:
        print("STOP: too many outputs with no extractable number. Look at the `text` column; fix extraction."); ok = False
    print("CHECK PASSED" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
