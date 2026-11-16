import numpy as np
from gates.pace import pace_gate
def stream(rng, p, n=200, disc=0.2):
    d, w = rng.random(n) < disc, rng.random(n) < p
    return [(1, 0) if (a and b) else (0, 1) if a else (1, 1) for a, b in zip(d, w)]
def test_false_commit_rate_below_alpha():
    rng = np.random.default_rng(1)
    rate = np.mean([pace_gate(stream(rng, 0.5), futility_threshold=0.0).admitted for _ in range(2000)])
    assert rate <= 0.05 + 0.01          # alpha plus Monte Carlo slack
def test_has_power_when_real():
    rng = np.random.default_rng(2)
    power = np.mean([pace_gate(stream(rng, 0.85, n=400), futility_threshold=0.0).admitted for _ in range(500)])
    assert power > 0.8
def test_ties_do_not_move_e():
    r = pace_gate([(1, 1), (0, 0)] * 500, futility_threshold=0.0)
    assert not r.admitted
