"""Auto-RT-inspired strategy exploration (Liu, Zhou, et al., arXiv:2501.01830).

Educational subset of Auto-RT ideas:

1. Early-terminated exploration -- stop a strategy branch when intermediate
   reward is hopeless (saves defender calls).
2. Progressive reward tracking -- combine soft judge scores over turns with a
   terminal success bonus; maintain per-strategy posteriors.
3. Strategy-level RL bandit -- select strategies online (Beta-Bernoulli / UCB).

This is NOT a full reimplementation of Auto-RT's multi-model progressive
downgrade stack; it captures the control-loop mechanics for local canary labs.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StrategyStats:
    name: str
    pulls: int = 0
    successes: int = 0
    total_reward: float = 0.0
    early_stops: int = 0
    alpha: float = 1.0  # Beta prior
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0

    @property
    def asr(self) -> float:
        return self.successes / self.pulls if self.pulls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pulls": self.pulls,
            "successes": self.successes,
            "total_reward": round(self.total_reward, 6),
            "mean_reward": round(self.mean, 6),
            "asr": round(self.asr, 6),
            "early_stops": self.early_stops,
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
        }


class ProgressiveRewardTracker:
    """Accumulate intermediate rewards then emit a progressive terminal signal.

    Auto-RT progressive reward tracking: intermediate scores steer search;
    terminal success dominates the final credit assignment.
    """

    def __init__(
        self,
        *,
        intermediate_weight: float = 0.35,
        terminal_bonus: float = 0.65,
        hopeless_threshold: float = 0.08,
        hopeless_patience: int = 2,
    ) -> None:
        self.intermediate_weight = intermediate_weight
        self.terminal_bonus = terminal_bonus
        self.hopeless_threshold = hopeless_threshold
        self.hopeless_patience = hopeless_patience
        self._scores: list[float] = []
        self._low_streak: int = 0

    def reset(self) -> None:
        self._scores.clear()
        self._low_streak = 0

    def observe_intermediate(self, score: float) -> bool:
        """Record a turn score. Returns True if branch should early-terminate."""
        s = min(1.0, max(0.0, float(score)))
        self._scores.append(s)
        if s < self.hopeless_threshold:
            self._low_streak += 1
        else:
            self._low_streak = 0
        return self._low_streak >= self.hopeless_patience

    def finalize(self, success: bool, final_score: float = 0.0) -> float:
        """Progressive reward in [0,1]."""
        mid = sum(self._scores) / len(self._scores) if self._scores else 0.0
        term = 1.0 if success else min(1.0, max(0.0, float(final_score)))
        r = self.intermediate_weight * mid + self.terminal_bonus * term
        return min(1.0, max(0.0, r))


class AutoRTExplorer:
    """Strategy-level explorer with Thompson / UCB and early termination.

    Inspired by Auto-RT's automatic strategy exploration + early-terminated
    exploration for efficiency.
    """

    def __init__(
        self,
        strategies: list[str],
        *,
        mode: str = "thompson",  # thompson | ucb | epsilon_greedy
        seed: int = 0,
        epsilon: float = 0.15,
        ucb_c: float = 1.4,
        reward_tracker: ProgressiveRewardTracker | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must be non-empty")
        self.mode = mode
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.arms: dict[str, StrategyStats] = {s: StrategyStats(name=s) for s in strategies}
        self.tracker = reward_tracker or ProgressiveRewardTracker()
        self.history: list[dict[str, Any]] = []

    @property
    def strategy_names(self) -> list[str]:
        return list(self.arms.keys())

    def select(self) -> str:
        names = self.strategy_names
        if self.mode == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return self.rng.choice(names)
            return max(names, key=lambda n: self.arms[n].mean)
        if self.mode == "ucb":
            total = sum(a.pulls for a in self.arms.values()) + 1
            def ucb(n: str) -> float:
                a = self.arms[n]
                if a.pulls == 0:
                    return float("inf")
                return a.mean + self.ucb_c * math.sqrt(math.log(total) / a.pulls)
            return max(names, key=ucb)
        # thompson (default)
        best_name, best_s = names[0], -1.0
        for n in names:
            a = self.arms[n]
            sample = self.rng.betavariate(max(1e-3, a.alpha), max(1e-3, a.beta))
            if sample > best_s:
                best_s, best_name = sample, n
        return best_name

    def begin_episode(self) -> ProgressiveRewardTracker:
        self.tracker.reset()
        return self.tracker

    def update(
        self,
        strategy: str,
        *,
        success: bool,
        reward: float | None = None,
        early_stopped: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> float:
        arm = self.arms[strategy]
        r = float(reward) if reward is not None else (1.0 if success else 0.0)
        r = min(1.0, max(0.0, r))
        arm.pulls += 1
        arm.total_reward += r
        if success:
            arm.successes += 1
        if early_stopped:
            arm.early_stops += 1
        arm.alpha += r
        arm.beta += 1.0 - r
        self.history.append(
            {
                "strategy": strategy,
                "success": success,
                "reward": r,
                "early_stopped": early_stopped,
                **(meta or {}),
            }
        )
        return r

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "inspired_by": "Auto-RT (arXiv:2501.01830)",
            "arms": {n: a.to_dict() for n, a in self.arms.items()},
            "n_episodes": len(self.history),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.stats(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def load_arms(self, data: dict[str, Any]) -> None:
        arms = data.get("arms") or data
        for name, raw in arms.items():
            if name not in self.arms:
                self.arms[name] = StrategyStats(name=name)
            a = self.arms[name]
            a.pulls = int(raw.get("pulls", 0))
            a.successes = int(raw.get("successes", 0))
            a.total_reward = float(raw.get("total_reward", 0.0))
            a.early_stops = int(raw.get("early_stops", 0))
            a.alpha = float(raw.get("alpha", 1.0))
            a.beta = float(raw.get("beta", 1.0))
