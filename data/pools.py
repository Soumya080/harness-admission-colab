"""Disjoint item pools from a math test set. Run once: python -m data.pools
HA_DATASET selects the dataset: gsm8k (default) or gsmhard (reasoning-machines/gsm-hard, 1,319 items)."""
import json, os, random
from pathlib import Path

HOME = Path(os.environ.get("HA_HOME", Path(__file__).resolve().parent.parent))
DATASET = os.environ.get("HA_DATASET", "gsm8k")
assert DATASET in ("gsm8k", "gsmhard"), f"unknown HA_DATASET={DATASET}"
POOLS_PATH = HOME / "configs" / f"pools_{DATASET}.json"
SIZES = {"proposal": 100, "dev": 200, "audit": 400}

_ds = None
def load_ds():
    """Rows always have 'question' and 'answer' ('... #### <number>'), whatever the dataset."""
    global _ds
    if _ds is None:
        if os.environ.get("HA_FAKE") == "1":      # synthetic rows for dry runs
            _ds = [{"question": f"Q{i}: what is {i}+1?", "answer": f"... #### {i + 1}"} for i in range(1319)]
        elif DATASET == "gsmhard":
            from datasets import load_dataset
            raw = load_dataset("reasoning-machines/gsm-hard", split="train")
            _ds = [{"question": r["input"], "answer": f"#### {r['target']!r}"} for r in raw]
        else:
            from datasets import load_dataset
            _ds = load_dataset("openai/gsm8k", "main", split="test")
    return _ds

def gold(example):
    return float(example["answer"].split("####")[1].strip().replace(",", ""))

def make_pools(n_items, seed=0):
    ids = list(range(n_items))
    random.Random(seed).shuffle(ids)
    out, k = {}, 0
    for name, size in SIZES.items():
        out[name] = ids[k:k + size]; k += size
    P, D, A = (set(out[n]) for n in SIZES)
    assert not (P & D) and not (P & A) and not (D & A), "pools overlap"
    return out

def load_pools():
    return json.loads(POOLS_PATH.read_text())

if __name__ == "__main__":
    pools = make_pools(len(load_ds()))
    POOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOLS_PATH.write_text(json.dumps(pools))
    print(DATASET, {k: len(v) for k, v in pools.items()}, "->", POOLS_PATH)
