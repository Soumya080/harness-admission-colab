"""CPU stand-in for vLLM so the whole pipeline can be dry-run without a GPU (HA_FAKE=1).
Accuracy depends on the prompt so gates have something to find:
handicap present -> 0.35, otherwise 0.70; every other wording is a no-op."""
import hashlib, random
HANDICAP_MARK = "Do not show any working"

def _p_correct(system_prompt):
    return 0.35 if HANDICAP_MARK in system_prompt else 0.70

def fake_generate(system_prompt, question, gold, temperature, n, item):
    p = _p_correct(system_prompt)
    # temp 0 must be deterministic per (prompt, item): seed from both
    seed = int(hashlib.md5(f"{system_prompt}|{item}".encode()).hexdigest()[:8], 16)
    outs = []
    for s in range(n):
        r = random.Random(seed if temperature == 0 else seed * 1000 + s + 1)
        ok = r.random() < p
        outs.append(f"Working...\n#### {gold if ok else gold + 1:g}")
    return outs
