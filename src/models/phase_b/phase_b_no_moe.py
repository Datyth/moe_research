"""Compatibility import for the A1 no-expert ablation."""

from .ablations.a1_no_expert import PhaseBA1NoExpertStage, PhaseBNoMoEStage

__all__ = ["PhaseBA1NoExpertStage", "PhaseBNoMoEStage"]
