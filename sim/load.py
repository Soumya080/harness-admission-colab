import numpy as np, pandas as pd

def load_matrices(df, harness_ids, dev_ids, audit_ids, n_samples=5):
    """df columns: harness, item, temp, sample, correct."""
    hix = {h: k for k, h in enumerate(harness_ids)}
    dix = {i: k for k, i in enumerate(dev_ids)}
    aix = {i: k for k, i in enumerate(audit_ids)}
    H = len(harness_ids)
    O = np.full((H, len(dev_ids)), -1, np.int8)
    S = np.full((H, len(dev_ids), n_samples), -1, np.int8)
    AU = np.full((H, len(audit_ids)), -1, np.int8)
    for r in df.itertuples(index=False):
        h = hix.get(r.harness)
        if h is None: continue
        if r.item in dix:
            if r.temp == 0.0: O[h, dix[r.item]] = r.correct
            elif r.sample < n_samples: S[h, dix[r.item], r.sample] = r.correct
        elif r.item in aix and r.temp == 0.0:
            AU[h, aix[r.item]] = r.correct
    for name, M in (("O", O), ("S", S), ("AU", AU)):
        assert (M >= 0).all(), f"missing cells in {name}: grid incomplete"
    return O, S, AU
