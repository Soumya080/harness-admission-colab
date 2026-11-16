from typing import Iterable, Tuple
from scipy.stats import binomtest
from gates.base import GateResult


def mcnemar_gate(pairs: Iterable[Tuple[int, int]], alpha: float = 0.05) -> GateResult:
    """Fixed-sample McNemar exact test on discordant pairs.
    Tests H0: P(candidate wins discordant) <= 0.5 vs H1: P > 0.5.
    """
    pair_list = list(pairs)
    n = len(pair_list)
    b = sum(1 for c, i in pair_list if c > i)  # Candidate wins
    c = sum(1 for c, i in pair_list if c < i)  # Incumbent wins
    discordant = b + c

    if discordant == 0:
        return GateResult(
            admitted=False,
            samples_used=n,
            metric_value=1.0,
            details={"b": b, "c": c, "discordant": 0, "p_value": 1.0},
        )

    p_value = float(binomtest(b, discordant, 0.5, alternative="greater").pvalue)
    admitted = p_value < alpha

    return GateResult(
        admitted=admitted,
        samples_used=n,
        metric_value=p_value,
        details={"b": b, "c": c, "discordant": discordant, "p_value": p_value, "alpha": alpha},
    )
