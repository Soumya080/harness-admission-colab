import json
import os
from pathlib import Path
from typing import Any
from harness.core import Harness

HOME = Path(os.environ.get("HA_HOME", Path(__file__).resolve().parent.parent))
DEFAULT_STORE_DIR = HOME / "cache" / "harnesses"


def save_harness(harness: Harness, directory: str | Path = DEFAULT_STORE_DIR) -> str:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{harness.id}.json"
    data = {
        "id": harness.id,
        "system_prompt": harness.system_prompt,
        "parent": harness.parent,
        "tag": harness.tag,
        "note": harness.note,
        "extra": harness.extra,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(file_path)


def load_harness(harness_id: str, directory: str | Path = DEFAULT_STORE_DIR) -> Harness:
    file_path = Path(directory) / f"{harness_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Harness file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Harness(
        system_prompt=data["system_prompt"],
        parent=data.get("parent"),
        tag=data.get("tag", "llm"),
        note=data.get("note", ""),
        extra=data.get("extra", {}),
    )


def list_harnesses(directory: str | Path = DEFAULT_STORE_DIR) -> list[str]:
    path = Path(directory)
    if not path.exists():
        return []
    return [p.stem for p in path.glob("*.json")]
