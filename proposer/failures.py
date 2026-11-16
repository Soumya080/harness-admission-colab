"""Failures the proposer may see: base harness, PROPOSAL pool only, temp 0."""
from data.pools import load_pools
from runner import cache

def load_failures(harness_id, ds, gold_fn):
    proposal = set(load_pools()["proposal"])
    df = cache.load_all()
    df = df[(df.harness == harness_id) & (df.split == "proposal") & (df.temp == 0.0)]
    assert len(df) > 0, "run the base on the proposal pool first: python -m scripts.run_grid --only base"
    assert set(df.item) <= proposal, "non-proposal item shown to proposer"   # leakage firewall
    df = df[df.correct == 0]
    return [dict(item=int(r.item), question=ds[int(r.item)]["question"],
                 text=r.text, gold=gold_fn(ds[int(r.item)])) for r in df.itertuples()]
