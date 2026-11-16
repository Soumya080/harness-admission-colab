"""Disjoint item pools from the GSM8K test split. Run once: python -m data.pools"""
import json, os, random
from pathlib import Path

HOME = Path(os.environ.get("HA_HOME", Path(__file__).resolve().parent.parent))
POOLS_PATH = HOME / "configs" / "pools_gsm8k.json"
SIZES = {"proposal": 100, "dev": 200, "audit": 400}

_ds = None
def load_ds():
    global _ds
    if _ds is None:
        if os.environ.get("HA_FAKE") == "1":      # synthetic GSM8K-shaped rows for dry runs
            _ds = [{"question": f"Q{i}: what is {i}+1?", "answer": f"... #### {i + 1}"} for i in range(1319)]
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
    print({k: len(v) for k, v in pools.items()}, "->", POOLS_PATH)
