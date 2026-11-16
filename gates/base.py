from dataclasses import dataclass
from typing import Protocol, Iterable, Tuple


@dataclass
class GateResult:
    admitted: bool
    samples_used: int
    metric_value: float
    details: dict


class Gate(Protocol):
    def evaluate(self, pairs: Iterable[Tuple[int, int]]) -> GateResult:
        """Evaluates paired binary outcomes (candidate_score, incumbent_score)
        and decides whether to admit the candidate harness.
        """
        ...
