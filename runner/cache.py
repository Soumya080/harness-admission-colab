"""Append-only outcome cache: one parquet file per (harness, split, temp).
A file that exists is final and is never rewritten -> reruns skip finished work."""
from pathlib import Path
import os, pandas as pd

HOME = Path(os.environ.get("HA_HOME", Path(__file__).resolve().parent.parent))
ROOT = HOME / "cache" / "outcomes"
COLS = ["harness", "split", "item", "temp", "sample", "text", "correct"]

def path_for(harness_id, split, temp, root=ROOT):
    return Path(root) / f"{harness_id}__{split}__t{temp:g}.parquet"

def has(harness_id, split, temp, root=ROOT):
    return path_for(harness_id, split, temp, root).exists()

def write(rows, harness_id, split, temp, root=ROOT):
    p = path_for(harness_id, split, temp, root)
    if p.exists():
        raise FileExistsError(f"{p} exists; the cache is append-only")
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)              # atomic: a crash never leaves a half file
    return p

def load_all(root=ROOT):
    files = sorted(Path(root).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
