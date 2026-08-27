"""Sample-weighted running means shared by the phase training scripts."""

from __future__ import annotations


class MetricAccumulator:
    """Accumulate per-batch metrics and average them over samples.

    Weighting by batch size matters because the last batch of an epoch is
    usually smaller (``drop_last=False``); averaging batch means would give that
    short batch the same influence as a full one.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.samples = 0

    def update(self, values: dict[str, float], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + value * batch_size
        self.samples += batch_size

    def summary(self) -> dict[str, float]:
        if self.samples == 0:
            raise ValueError("No samples were accumulated.")
        return {key: value / self.samples for key, value in self.totals.items()}
