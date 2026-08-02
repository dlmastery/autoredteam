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

These are educational implementations for authorized local testing (canary probes).
They are *inspired by* the papers, not reimplementations of proprietary code.
"""
from __future__ import annotations

from .auto_rt import AutoRTExplorer, ProgressiveRewardTracker
from .autoresearch import AutoresearchLoop, KeepRevertResult
from .memory import AttackRecord, LifelongAttackMemory
from .strategy_proposer import StrategyProposer
from .vcg import VulnerabilityConcept, VulnerabilityConceptGraph

__all__ = [
    "AttackRecord",
    "LifelongAttackMemory",
    "AutoRTExplorer",
    "ProgressiveRewardTracker",
    "AutoresearchLoop",
    "KeepRevertResult",
    "VulnerabilityConcept",
    "VulnerabilityConceptGraph",
    "StrategyProposer",
]
