import importlib
from runner.extract import score_numeric
from data.pools import gold

def test_integer_gold_stays_strict():
    assert score_numeric(322886700.0, 322886700.0) == 1
    assert score_numeric(322886701.0, 322886700.0) == 0

def test_non_integer_gold_accepts_rounding():
    assert score_numeric(14053029.67, 14053029.666667) == 1
    assert score_numeric(1.67, 1.6666667) == 1
    assert score_numeric(1.7, 1.6666667) == 0

def test_gsmhard_row_gold_roundtrip():
    for t in (-9867630.0, 14053029.666667, 3244047.1):
        assert gold({"answer": f"#### {t!r}"}) == t

def test_clean_regime_manual(monkeypatch):
    monkeypatch.setenv("HA_REGIME", "clean")
    import proposer.manual as m
    m = importlib.reload(m)
    hs = m.all_manual()
    tags = [h.tag for h in hs]
    assert tags.count("base") == 1 and "planted" not in tags and tags.count("noop") == 5
    assert all(m.HANDICAP not in h.system_prompt for h in hs)
    monkeypatch.setenv("HA_REGIME", "handicap"); importlib.reload(m)
