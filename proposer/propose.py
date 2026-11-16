import os
"""LLM edits from base failures. GPU: python -m proposer.propose
Dry run (no GPU): HA_FAKE=1 python -m proposer.propose  -> template edits, same code path after parsing."""
import argparse, json, os, random
from data.pools import load_ds, gold
from harness.core import Harness
from harness.store import save_harness, list_harnesses, load_harness
from proposer import prompts, failures, parse

N_EDITS, K_FAIL, BATCH = 30, 10, 8
PROPOSER = "Qwen/Qwen2.5-7B-Instruct-AWQ"

def get_base():
    bs = [h for h in map(load_harness, list_harnesses()) if h.tag == "base"]
    assert len(bs) == 1, f"expected 1 base harness, found {len(bs)}: run python -m proposer.manual"
    return bs[0]

def fake_chat(batch, rng):
    adds = ["Check units.", "Re-read the question.", "Verify arithmetic.", "List the knowns first.",
            "Be concise.", "Double-check the final step.", "Track each quantity.", "Avoid rounding early."]
    out = []
    for _ in batch:
        sp = prompts_base + " " + rng.choice(adds) + f" (v{rng.randint(0, 10**6)})"
        out.append(json.dumps({"op": "add", "rationale": "fake", "new_system_prompt": sp}))
    return out

def main(n_edits=N_EDITS, seed=0):
    global prompts_base
    rng = random.Random(seed)
    ds, b = load_ds(), get_base()
    prompts_base = b.system_prompt
    fails = failures.load_failures(b.id, ds, gold)
    print(f"{len(fails)} proposal-pool failures")
    fake = os.environ.get("HA_FAKE") == "1"
    if not fake:
        from vllm import LLM, SamplingParams
        llm = LLM(PROPOSER, max_model_len=4096, gpu_memory_utilization=float(os.environ.get('HA_GPU_UTIL', 0.85)),
                  seed=seed, dtype=os.environ.get('HA_DTYPE', 'auto'))
        sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)

    seen, made, ok, tried = set(list_harnesses()), 0, 0, 0
    while made < n_edits and tried < 20 * n_edits:
        batch = []
        for _ in range(BATCH):
            sample = rng.sample(fails, min(K_FAIL, len(fails)))
            user = prompts.PROPOSER_USER.format(prompt=b.system_prompt,
                                                failures=prompts.format_failures(sample))
            batch.append([{"role": "system", "content": prompts.PROPOSER_SYSTEM},
                          {"role": "user", "content": user}])
        texts = fake_chat(batch, rng) if fake else [o.outputs[0].text for o in llm.chat(batch, sp)]
        for t in texts:
            tried += 1
            r = parse.parse_reply(t, b.system_prompt)
            if r is None: continue
            ok += 1
            h = Harness(r["new_system_prompt"], parent=b.id, tag="llm", note=f'{r["op"]}: {r["rationale"]}')
            if h.id in seen: continue
            seen.add(h.id); save_harness(h); made += 1
            if made >= n_edits: break
    print(f"made {made} edits; parse success {ok}/{tried}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=N_EDITS)
    main(ap.parse_args().n)
