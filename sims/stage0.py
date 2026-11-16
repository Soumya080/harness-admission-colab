import numpy as np
from scipy import stats
from gates.pace import pace_gate

def null_stream(rng, n=200, disc=0.2, p=0.5):
    d = rng.random(n) < disc; w = rng.random(n) < p
    return [(1, 0) if (a and b) else (0, 1) if a else (1, 1) for a, b in zip(d, w)]

def validity(R=5000, seed=0):
    """False-commit rate under H0: PACE (peeks every item) vs a t-test peeked every item."""
    rng = np.random.default_rng(seed); pace = tt = 0
    for _ in range(R):
        s = null_stream(rng)
        pace += pace_gate(s, futility_threshold=0.0).admitted
        x = np.array([c - i for c, i in s], float)
        for t in range(10, len(x) + 1):
            if x[:t].std() > 0 and stats.ttest_1samp(x[:t], 0, alternative="greater").pvalue < 0.05:
                tt += 1; break
    return pace / R, tt / R

def lam_sweep(ps=(0.6, 0.7, 0.8), lams=np.linspace(0.05, 0.9, 18)):
    """Expected log-growth per discordant flip: g = p ln(1+l) + (1-p) ln(1-l)."""
    return {p: [(round(l, 2), round(p*np.log1p(l) + (1-p)*np.log1p(-l), 4)) for l in lams] for p in ps}

def winners_curse(n_devs=(20, 40, 100, 200), K=30, R=5000, flip=0.1, seed=0):
    """Best apparent gain (pts) among K useless edits on one dev set of size n."""
    rng = np.random.default_rng(seed); out = {}
    for n in n_devs:
        d = (rng.random((R, K, n)) < flip) * rng.choice([-1, 1], (R, K, n))
        out[n] = 100 * d.mean(axis=2).max(axis=1).mean()
    return out

if __name__ == "__main__":
    print("false-commit rate (PACE, peeking t-test):", validity(R=2000))
    print("best fake gain by dev n:", {k: round(v, 1) for k, v in winners_curse(R=2000).items()})
    print("g at lam=0.5:", {p: round(p*np.log(1.5)+(1-p)*np.log(0.5), 4) for p in (0.6, 0.7, 0.8)})
