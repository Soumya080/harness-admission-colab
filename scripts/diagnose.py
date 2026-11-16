"""Data-quality checks on the real grid. python -m scripts.diagnose"""
import numpy as np
from harness.store import list_harnesses, load_harness
from runner import cache
from runner.extract import extract_gsm8k_answer

def main():
    df = cache.load_all()
    tags = {h.id: h.tag for h in map(load_harness, list_harnesses())}
    df["tag"] = df.harness.map(tags)
    df["no_answer"] = df.text.map(lambda t: extract_gsm8k_answer(t) is None)
    g = df.groupby(["tag", "harness", "split", "temp"]).agg(n=("correct", "size"),
                                                           acc=("correct", "mean"),
                                                           no_ans=("no_answer", "mean"))
    print(g.round(3).to_string())

    bad = g[g.no_ans > 0.10]
    print("\n[1] no-answer > 10%:", "NONE" if bad.empty else f"\n{bad}")

    au = df[(df.split == "audit") & (df.temp == 0)].pivot(index="item", columns="harness", values="correct")
    base = [h for h, t in tags.items() if t == "base"][0]
    gains = {h: 100 * (au[h].mean() - au[base].mean()) for h in au.columns if h != base}
    llm = [v for h, v in gains.items() if tags[h] == "llm"]
    if llm:
        print(f"\n[2] LLM-edit audit gain (pts): min {min(llm):.1f}  median {np.median(llm):.1f}  max {max(llm):.1f}")
    print("    planted gain:", [round(v, 1) for h, v in gains.items() if tags[h] == "planted"])
    print("    noop gains:  ", [round(v, 1) for h, v in gains.items() if tags[h] == "noop"])

    d = np.median([float((au[h] != au[base]).mean()) for h in au.columns if h != base])
    print(f"\n[3] median discordance vs base q = {d:.3f} -> audit SE = {100*np.sqrt(d/len(au)):.2f} pts, "
          f"95% CI = +/-{196*np.sqrt(d/len(au)):.1f} pts")

    dev = df[df.split == "dev"].groupby(["harness", "temp"]).correct.mean().unstack()
    if {0.0, 0.7} <= set(dev.columns):
        gap = 100 * (dev[0.0] - dev[0.7]).abs()
        print(f"\n[4] |acc(t=0) - acc(t=0.7)| on dev: median {gap.median():.1f} pts, max {gap.max():.1f} pts")

if __name__ == "__main__":
    main()
