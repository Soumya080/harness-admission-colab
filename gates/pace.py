from typing import Iterable, Tuple
import numpy as np
from gates.base import GateResult


def pace_gate(
    pairs: Iterable[Tuple[int, int]],
    alpha: float = 0.05,
    max_samples: int | None = None,
    futility_threshold: float = 0.05,
    betting_fraction: float = 0.5,
) -> GateResult:
    """PACE: Paired Anytime-valid Commit Evaluation.
    Maintains an e-process Et over discordant pairs:
    Et = product_{t=1}^T (1 + lambda_t * Z_t)
    where Z_t = +1 if candidate wins discordant pair, -1 if incumbent wins.
    Admitted as soon as Et >= 1 / alpha (Ville's inequality anytime guarantee).
    """
    threshold = 1.0 / alpha
    e_val = 1.0
    samples_used = 0
    discordant_count = 0
    b = 0
    c = 0
    admitted = False

    history_e = [e_val]

    for c_score, i_score in pairs:
        samples_used += 1

        # Check discordant pair
        if c_score != i_score:
            discordant_count += 1
            z_t = 1.0 if c_score > i_score else -1.0
            if z_t > 0:
                b += 1
            else:
                c += 1

            # Predictable betting fraction (Kelly-like or fixed bounded)
            lam = min(max(betting_fraction, 0.01), 0.95)
            e_val *= (1.0 + lam * z_t)
            history_e.append(e_val)

            # Early stopping check for admission
            if e_val >= threshold:
                admitted = True
                break

            # Early futility stopping
            if e_val <= futility_threshold:
                admitted = False
                break

        if max_samples is not None and samples_used >= max_samples:
            break

    return GateResult(
        admitted=admitted,
        samples_used=samples_used,
        metric_value=e_val,
        details={
            "e_value": e_val,
            "threshold": threshold,
            "discordant": discordant_count,
            "b": b,
            "c": c,
            "history_len": len(history_e),
        },
    )
