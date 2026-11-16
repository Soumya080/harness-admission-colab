import json, re

OPS = {"add", "rewrite", "delete", "merge"}

def parse_reply(text, old_prompt):
    """Return dict or None. None means: skip this reply."""
    m = re.search(r"\{.*\}", text, re.DOTALL)          # first { to last }
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    new = (d.get("new_system_prompt") or "").strip()
    if d.get("op") not in OPS:           return None
    if len(new) < 20 or len(new) > 3000: return None   # empty or runaway
    if new == old_prompt.strip():        return None   # not an edit
    return {"op": d["op"], "rationale": str(d.get("rationale", ""))[:300],
            "new_system_prompt": new}
