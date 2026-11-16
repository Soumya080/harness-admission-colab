from data.pools import make_pools, SIZES
def test_disjoint_and_sized():
    p = make_pools(1319)
    assert {k: len(v) for k, v in p.items()} == SIZES
    sets = [set(v) for v in p.values()]
    assert sum(len(s) for s in sets) == len(set().union(*sets))
def test_deterministic():
    assert make_pools(1319) == make_pools(1319)
