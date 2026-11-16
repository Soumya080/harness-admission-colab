from typing import Iterable, Tuple
from gates.base import GateResult


def greedy_gate(pairs: Iterable[Tuple[int, int]], min_delta: float = 0.0) -> GateResult:
    """Greedy admission: admits candidate if candidate accuracy exceeds
    incumbent accuracy by at least min_delta.
    """
    pair_list = list(pairs)
    n = len(pair_list)
    if n == 0:
        return GateResult(admitted=False, samples_used=0, metric_value=0.0, details={"n": 0})

    cand_wins = sum(c for c, _ in pair_list)
    inc_wins = sum(i for _, i in pair_list)
    delta = (cand_wins - inc_wins) / n
    admitted = delta > min_delta

    return GateResult(
        admitted=admitted,
        samples_used=n,
        metric_value=delta,
        details={
            "candidate_correct": cand_wins,
            "incumbent_correct": inc_wins,
            "delta": delta,
            "n": n,
        },
    )
