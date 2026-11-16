import re
from typing import Any


def extract_gsm8k_answer(text: str) -> float | None:
    """Extract numeric answer from generated text for GSM8K tasks.
    Looks for #### pattern, boxed expressions, or the last numeric token.
    """
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        match = re.search(r"[-+]?\d*\.?\d+", after_hash.replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass

    # Look for \boxed{...} pattern
    boxed_match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        last_boxed = boxed_match[-1].strip()
        num_match = re.search(r"[-+]?\d*\.?\d+", last_boxed.replace(",", ""))
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                pass

    # Look for "The answer is ..." pattern
    phrase_match = re.findall(r"(?:the answer is|equals|total is|result is)\s*[:=]?\s*([-$]?\d+[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if phrase_match:
        cand = phrase_match[-1].replace("$", "").replace(",", "").strip()
        try:
            return float(cand)
        except ValueError:
            pass

    # Fallback to the last standalone number
    all_nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if all_nums:
        try:
            return float(all_nums[-1])
        except ValueError:
            pass

    return None


def score_numeric(pred: float | None, gold: float, tol: float = 1e-4) -> int:
    """Returns 1 for correct prediction within tolerance, else 0."""
    if pred is None:
        return 0
    return 1 if abs(pred - gold) <= tol else 0


def extract_and_score(generated_text: str, gold_val: float) -> tuple[float | None, int]:
    pred = extract_gsm8k_answer(generated_text)
    score = score_numeric(pred, gold_val)
    return pred, score
