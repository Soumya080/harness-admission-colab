import os
"""Executor: run one harness on a list of items, score, return rows for the cache.
HA_FAKE=1 swaps vLLM for runner/fake.py (CPU dry run)."""
from runner.extract import extract_and_score

MODEL = "Qwen/Qwen2.5-3B-Instruct"
FAKE = os.environ.get("HA_FAKE") == "1"
_llm = None

def get_llm():
    global _llm                    # load lazily: importing this file must not grab the GPU
    if _llm is None:
        from vllm import LLM
        _llm = LLM(MODEL, max_model_len=4096, gpu_memory_utilization=float(os.environ.get('HA_GPU_UTIL', 0.85)),
                   seed=0, dtype=os.environ.get('HA_DTYPE', 'auto'))
    return _llm

def _generate(h, item_ids, temperature, n_samples, ds, gold):
    if FAKE:
        from runner.fake import fake_generate
        return [fake_generate(h.system_prompt, ds[i]["question"], gold(ds[i]), temperature, n_samples, i)
                for i in item_ids]
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, n=n_samples, max_tokens=512)
    outs = get_llm().chat([h.message(ds[i]["question"]) for i in item_ids], sampling_params=sp, use_tqdm=True)
    return [[c.text for c in o.outputs] for o in outs]

def run(h, item_ids, split, temperature, n_samples, ds, gold):
    texts = _generate(h, item_ids, temperature, n_samples, ds, gold)
    assert len(texts) == len(item_ids)
    rows = []
    for i, outs in zip(item_ids, texts):
        assert len(outs) == n_samples
        g = gold(ds[i])
        for s, t in enumerate(outs):
            _, ok = extract_and_score(t, g)
            rows.append(dict(harness=h.id, split=split, item=int(i), temp=float(temperature),
                             sample=s, text=t, correct=int(ok)))
    return rows
