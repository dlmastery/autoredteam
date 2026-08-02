"""Full multi-phase educational red-team pipeline.

Phases
------
  0  setup          - auth gate, ensure Ollama models, smoke tests
  1  compose        - load/compose 100 unique goals (regular + universal)
  2  attack_gen     - ablated attacker authors single-turn probes
  3  defend_single  - official defender answers single-turn probes
  4  multiturn      - crescendo + mutation_loop on non-successes
  5  universal_bon  - Best-of-N surface augmentations (universal layer)
  6  judge          - ensemble scoring (rule + optional LLM judge)
  7  hardneg        - hard-negative mining + re-attack (HASTE-lite)
  8  export         - four-way classifier dataset + policy pairs
  9  dashboard      - interactive HTML + classic reports

Resume any phase via checkpoints under ``runs/<campaign>/pipeline/``.
"""
from __future__ import annotations

from .runner import PHASE_ORDER, run_pipeline

__all__ = ["run_pipeline", "PHASE_ORDER"]
