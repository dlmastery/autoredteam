"""Research implementations inspired by 2025-2026 auto red-teaming papers.

Modules
-------
memory
    Lifelong attack memory (AutoRedTeamer-style).
auto_rt
    RL strategy exploration with early termination + progressive rewards (Auto-RT).
autoresearch
    Keep/revert evolutionary loop + AHA-style discovery (Jailbreak-autoresearch / AHA).
vcg
    Vulnerability Concept Graph export (AHA).
strategy_proposer
    Memory-guided strategy proposals (AutoRedTeamer strategy proposer).
production_agent
    AHA-style tool-using victim harness (sandbox FS + canary policy + VCG promote).
aic
    Adaptive Instruction Composition bandit over educational surface tactics (AIC).
cop
    Composition of Principles — sample & compose framing principles (CoP).

These are educational implementations for authorized local testing (canary probes).
They are *inspired by* the papers, not reimplementations of proprietary code.
"""
from __future__ import annotations

from .aic import AdaptiveInstructionComposer, Tactic, educational_tactic_catalog
from .auto_rt import AutoRTExplorer, ProgressiveRewardTracker
from .autoresearch import AutoresearchLoop, KeepRevertLoop, KeepRevertResult
from .cop import (
    PRINCIPLES,
    CompositionOfPrinciples,
    CompositionResult,
    Principle,
    default_principles,
)
from .memory import AttackRecord, LifelongAttackMemory
from .production_agent import (
    AttackEpisode,
    ProductionAgentHarness,
    ProductionAgentVictim,
    SandboxFS,
    run_attack,
)
from .strategy_proposer import StrategyProposer
from .vcg import VulnerabilityConcept, VulnerabilityConceptGraph

# Short aliases matching paper acronyms
CoP = CompositionOfPrinciples
AIC = AdaptiveInstructionComposer

__all__ = [
    "AttackRecord",
    "LifelongAttackMemory",
    "AutoRTExplorer",
    "ProgressiveRewardTracker",
    "AutoresearchLoop",
    "KeepRevertLoop",
    "KeepRevertResult",
    "VulnerabilityConcept",
    "VulnerabilityConceptGraph",
    "StrategyProposer",
    "AttackEpisode",
    "ProductionAgentHarness",
    "ProductionAgentVictim",
    "SandboxFS",
    "run_attack",
    "AdaptiveInstructionComposer",
    "AIC",
    "Tactic",
    "educational_tactic_catalog",
    "CompositionOfPrinciples",
    "CoP",
    "CompositionResult",
    "Principle",
    "PRINCIPLES",
    "default_principles",
]
