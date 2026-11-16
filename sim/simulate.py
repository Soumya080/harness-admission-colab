import numpy as np
from gates.greedy import greedy_gate
from gates.mcnemar import mcnemar_gate
from gates.pace import pace_gate

def _pace(lam=0.5, fut=0.0, alpha=0.05):
    return lambda p: pace_gate(p, alpha=alpha, futility_threshold=fut, betting_fraction=lam)

GATES = {
    "greedy":  lambda p: greedy_gate(p),
    "mcnemar": lambda p: mcnemar_gate(p, alpha=0.05),
    "pace":    _pace(),                                    # PACE as published: no futility
    # --- ablations ---
    "pace_fut":   _pace(fut=0.05),                         # your early-stop variant
    "pace_l25":   _pace(lam=0.25),
    "pace_l75":   _pace(lam=0.75),
    "pace_a10":   _pace(alpha=0.10),
    "ladder_02":  lambda p: greedy_gate(p, min_delta=0.02),   # Blum–Hardt style margin
    "ladder_05":  lambda p: greedy_gate(p, min_delta=0.05),
    "ladder_10":  lambda p: greedy_gate(p, min_delta=0.10),
}
MAIN = ("greedy", "mcnemar", "pace")
ABLATIONS = ("pace_fut", "pace_l25", "pace_l75", "pace_a10", "ladder_02", "ladder_05", "ladder_10")

def one_run(rng, O, S, start, edits, gate, n_dev, regime, fresh_dev=False):
    dev = rng.choice(O.shape[1], n_dev, replace=False)     # fixed for the whole run = reuse
    inc, commits, cost = start, [], 0
    for cand in rng.permutation(edits):
        if fresh_dev:                                       # control: new dev subset per candidate
            dev = rng.choice(O.shape[1], n_dev, replace=False)
        order = rng.permutation(dev)
        if regime == "controlled":
            c, i = O[cand, order], O[inc, order]
        else:                                               # fresh noisy evaluation each time
            k = S.shape[2]
            c = S[cand, order, rng.integers(k, size=len(order))]
            i = S[inc, order, rng.integers(k, size=len(order))]
        res = gate(zip(c.tolist(), i.tolist()))
        cost += res.samples_used
        if res.admitted:
            commits.append((int(inc), int(cand)))
            inc = cand
    return commits, int(inc), cost

def run_many(O, S, start, edits, gate_name, n_dev, regime, runs=1000, seed=0, fresh_dev=False):
    rng = np.random.default_rng(seed)
    gate = GATES[gate_name]
    return [one_run(rng, O, S, start, edits, gate, n_dev, regime, fresh_dev) for _ in range(runs)]
