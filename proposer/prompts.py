PROPOSER_SYSTEM = (
    "You improve the system prompt of a math-solving assistant. "
    "Make ONE focused edit. Reply with a single JSON object and nothing else."
)

PROPOSER_USER = """CURRENT SYSTEM PROMPT:
<<<
{prompt}
>>>

The assistant got these problems wrong:
{failures}

Propose ONE edit. Choose op from: add, rewrite, delete, merge.
Return only this JSON:
{{"op": "...", "rationale": "one sentence", "new_system_prompt": "full new prompt"}}"""

def format_failures(fails):
    out = []
    for k, f in enumerate(fails, 1):
        out.append(f"--- Problem {k} ---\nQuestion: {f['question']}\n"
                   f"Assistant output:\n{f['text'][:800]}\nCorrect answer: {f['gold']}")
    return "\n\n".join(out)
