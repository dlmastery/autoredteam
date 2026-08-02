"""Lifelong attack memory -- inspired by AutoRedTeamer (Zhou et al., arXiv:2503.15754).

Stores successful (and hard-failure) attack artifacts across campaigns so later
runs can reuse high-value strategies/prompts instead of rediscovering them.

Educational scope: canary probes only; no real exploit payloads.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class AttackRecord:
    """One remembered attack episode."""

    id: str
    prompt: str
    strategy: str
    category: str = ""
    template: str = ""
    jailbreak_type: str = "regular"
    success: bool = False
    score: float = 0.0
    canary_token: str = ""
    goal_id: str = ""
    defender_response_snippet: str = ""
    source_phase: str = ""
    times_reused: int = 0
    times_succeeded_on_reuse: int = 0
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttackRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def reuse_asr(self) -> float:
        if self.times_reused <= 0:
            return 1.0 if self.success else 0.0
        return self.times_succeeded_on_reuse / self.times_reused


def _hash_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


class LifelongAttackMemory:
    """Persistent library of attacks with memory-guided retrieval.

    AutoRedTeamer insight: continuously discover AND integrate attacks into a
    memory that guides future selection -- not a fixed strategy menu.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: dict[str, AttackRecord] = {}
        if self.path and self.path.is_file():
            self.load(self.path)

    # ---- persistence ---------------------------------------------------- #
    def load(self, path: str | Path | None = None) -> None:
        p = Path(path or self.path or "")
        if not p.is_file():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        for raw in data.get("records") or []:
            rec = AttackRecord.from_dict(raw)
            self.records[rec.id] = rec
        if path:
            self.path = Path(path)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.path or "runs/attack_memory.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "version": 1,
            "inspired_by": "AutoRedTeamer (arXiv:2503.15754) lifelong attack integration",
            "n_records": len(self.records),
            "records": [r.to_dict() for r in self.records.values()],
        }
        p.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        self.path = p
        return p

    # ---- write ---------------------------------------------------------- #
    def remember(
        self,
        *,
        prompt: str,
        strategy: str,
        success: bool,
        score: float = 0.0,
        category: str = "",
        template: str = "",
        jailbreak_type: str = "regular",
        canary_token: str = "",
        goal_id: str = "",
        defender_response: str = "",
        source_phase: str = "",
        meta: dict[str, Any] | None = None,
    ) -> AttackRecord:
        rid = _hash_id(strategy, category, prompt[:200])
        if rid in self.records:
            rec = self.records[rid]
            # refresh score with EMA
            rec.score = 0.7 * rec.score + 0.3 * float(score)
            if success:
                rec.success = True
            if defender_response:
                rec.defender_response_snippet = (defender_response or "")[:240]
            return rec
        rec = AttackRecord(
            id=rid,
            prompt=prompt,
            strategy=strategy,
            category=category,
            template=template,
            jailbreak_type=jailbreak_type,
            success=success,
            score=float(score),
            canary_token=canary_token,
            goal_id=goal_id,
            defender_response_snippet=(defender_response or "")[:240],
            source_phase=source_phase,
            meta=dict(meta or {}),
        )
        self.records[rid] = rec
        return rec

    def record_reuse(self, record_id: str, succeeded: bool) -> None:
        rec = self.records.get(record_id)
        if not rec:
            return
        rec.times_reused += 1
        if succeeded:
            rec.times_succeeded_on_reuse += 1

    # ---- read / select -------------------------------------------------- #
    def successes(self) -> list[AttackRecord]:
        return [r for r in self.records.values() if r.success]

    def hard_failures(self) -> list[AttackRecord]:
        """Failed attempts with non-trivial score -- hard negatives for mining."""
        return [r for r in self.records.values() if (not r.success) and r.score >= 0.2]

    def top_strategies(self, k: int = 5) -> list[tuple[str, float, int]]:
        """Return (strategy, mean_score, n) for successful records."""
        buckets: dict[str, list[float]] = {}
        for r in self.successes():
            buckets.setdefault(r.strategy, []).append(r.score)
        ranked = sorted(
            ((s, sum(v) / len(v), len(v)) for s, v in buckets.items()),
            key=lambda x: (-x[1], -x[2]),
        )
        return ranked[:k]

    def retrieve(
        self,
        *,
        category: str | None = None,
        strategy: str | None = None,
        jailbreak_type: str | None = None,
        only_success: bool = True,
        k: int = 5,
    ) -> list[AttackRecord]:
        """Memory-guided retrieval (AutoRedTeamer attack selection spirit)."""
        pool: Iterable[AttackRecord] = self.records.values()
        out: list[AttackRecord] = []
        for r in pool:
            if only_success and not r.success:
                continue
            if category and r.category and r.category != category:
                continue
            if strategy and r.strategy != strategy:
                continue
            if jailbreak_type and r.jailbreak_type != jailbreak_type:
                continue
            out.append(r)
        out.sort(key=lambda r: (r.reuse_asr * 0.5 + r.score * 0.5, r.times_reused), reverse=True)
        return out[:k]

    def seed_prompts_for_goal(
        self,
        category: str,
        fallback_seeds: list[str],
        *,
        k: int = 2,
    ) -> list[str]:
        """Blend memory hits with operator seeds."""
        hits = self.retrieve(category=category, only_success=True, k=k)
        prompts = [h.prompt for h in hits]
        for s in fallback_seeds:
            if s not in prompts:
                prompts.append(s)
        return prompts[: max(k, len(fallback_seeds))]

    def stats(self) -> dict[str, Any]:
        succ = self.successes()
        return {
            "n_records": len(self.records),
            "n_success": len(succ),
            "n_hard_failures": len(self.hard_failures()),
            "top_strategies": [
                {"strategy": s, "mean_score": round(m, 4), "n": n}
                for s, m, n in self.top_strategies()
            ],
        }

    def ingest_pipeline_items(self, items: list[Any]) -> int:
        """Ingest ItemRecord-like objects from the educational pipeline."""
        n = 0
        for it in items:
            # duck-typed for pipeline.state.ItemRecord or dict
            if isinstance(it, dict):
                get = it.get
            else:
                get = lambda k, default="": getattr(it, k, default)  # noqa: E731
            prompt = get("final_prompt") or get("attack_prompt") or get("seed_prompt") or ""
            if not prompt:
                continue
            self.remember(
                prompt=str(prompt),
                strategy=str(get("final_strategy") or get("strategy") or "unknown"),
                success=bool(get("final_success") or get("success")),
                score=float(get("final_score") or get("score") or 0.0),
                category=str(get("category") or ""),
                template=str(get("template") or ""),
                jailbreak_type=str(get("jailbreak_type") or "regular"),
                canary_token=str(get("canary_token") or ""),
                goal_id=str(get("id") or ""),
                defender_response=str(get("final_response") or get("defender_response") or ""),
                source_phase="pipeline_ingest",
            )
            n += 1
        return n
