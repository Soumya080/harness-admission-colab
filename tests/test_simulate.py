import numpy as np
from sim.simulate import run_many
from analysis.metrics import summarize
def test_greedy_commits_more_than_pace_on_noise():
    rng = np.random.default_rng(0)
    H, n, a = 12, 200, 400
    O = (rng.random((H, n)) < 0.5).astype(np.int8)          # all harnesses equal
    S = (rng.random((H, n, 5)) < 0.5).astype(np.int8)
    AU = (rng.random((H, a)) < 0.5).astype(np.int8)
    edits = np.arange(1, H)
    g = summarize(run_many(O, S, 0, edits, "greedy", 40, "stochastic", runs=50), AU, 0)
    p = summarize(run_many(O, S, 0, edits, "pace", 40, "stochastic", runs=50), AU, 0)
    assert g["commits"] > 3 * max(p["commits"], 0.1)
