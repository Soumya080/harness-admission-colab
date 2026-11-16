from sim.simulate import GATES, MAIN, ABLATIONS
def test_all_gates_run():
    pairs = [(1, 0)] * 30 + [(0, 1)] * 5 + [(1, 1)] * 20
    for g in MAIN + ABLATIONS:
        r = GATES[g](iter(pairs))
        assert r.samples_used > 0
def test_ladder_margin_blocks_tiny_gain():
    pairs = [(1, 0)] * 2 + [(1, 1)] * 98          # +2 pts exactly
    assert GATES["greedy"](iter(pairs)).admitted
    assert not GATES["ladder_05"](iter(pairs)).admitted
