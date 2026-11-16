import numpy as np

def make_labeler(AU, z=1.96):
    """Label a commit (inc -> cand) from PAIRED per-item audit outcomes.
    helpful: whole 95% CI of the gain > 0; harmful: whole CI < 0; null: CI contains 0."""
    cache = {}
    def label(inc, cand):
        key = (inc, cand)
        if key not in cache:
            d = AU[cand].astype(float) - AU[inc].astype(float)
            g, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            cache[key] = ("helpful" if g - z * se > 0 else
                          "harmful" if g + z * se < 0 else "null", g)
        return cache[key]
    return label

def summarize(runs, AU, start):
    label = make_labeler(AU)
    acc = AU.mean(axis=1)
    per_run, counts = [], dict(helpful=0, null=0, harmful=0)
    for commits, final, cost in runs:
        for inc, cand in commits:
            counts[label(inc, cand)[0]] += 1
        per_run.append((len(commits), acc[final] - acc[start], cost))
    tot = max(sum(counts.values()), 1)
    n, fg, c = map(np.array, zip(*per_run))
    return dict(commits=n.mean(), commits_sd=n.std(),
                helpful_pct=100 * counts["helpful"] / tot,
                null_pct=100 * counts["null"] / tot,
                harmful_pct=100 * counts["harmful"] / tot,
                final_gain=100 * fg.mean(), final_gain_sd=100 * fg.std(),
                cost=c.mean())
