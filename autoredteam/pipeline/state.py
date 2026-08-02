"""Checkpointed pipeline state (JSON under runs/<campaign>/pipeline/)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ItemRecord:
    """One unique educational probe tracked across all phases."""

    id: str
    goal: str
    category: str
    severity: str = "medium"
    jailbreak_type: str = "regular"  # regular | universal
    template: str = ""
    canary_token: str = ""
    seed_prompt: str = ""

    # Phase 2/3 single-turn
    attack_prompt: str = ""
    defender_response: str = ""
    strategy: str = "single_turn"
    verdict: str = ""
    score: float = 0.0
    success: bool = False

    # Multi-turn trajectory (phase 4)
    multiturn_turns: list[dict[str, Any]] = field(default_factory=list)
    multiturn_success: bool = False
    multiturn_strategy: str = ""

    # Universal BoN (phase 5)
    bon_variants: list[dict[str, Any]] = field(default_factory=list)
    bon_success: bool = False
    bon_best_prompt: str = ""
    bon_best_response: str = ""

    # Hard-neg re-attack (phase 7)
    hardneg_attempts: list[dict[str, Any]] = field(default_factory=list)
    hardneg_success: bool = False

    # Final aggregated
    final_success: bool = False
    final_prompt: str = ""
    final_response: str = ""
    final_strategy: str = ""
    final_verdict: str = ""
    final_score: float = 0.0
    phases_hit: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ItemRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PipelineState:
    campaign: str
    scope: str
    attacker_model: str = ""
    defender_model: str = ""
    started_at: str = ""
    finished_at: str = ""
    completed_phases: list[str] = field(default_factory=list)
    phase_log: list[dict[str, Any]] = field(default_factory=list)
    items: list[ItemRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def mark_phase(self, name: str, **extra: Any) -> None:
        if name not in self.completed_phases:
            self.completed_phases.append(name)
        self.phase_log.append({"phase": name, "at": _now(), **extra})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "campaign": self.campaign,
            "scope": self.scope,
            "attacker_model": self.attacker_model,
            "defender_model": self.defender_model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "completed_phases": self.completed_phases,
            "phase_log": self.phase_log,
            "items": [it.to_dict() for it in self.items],
            "meta": self.meta,
        }
        path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        # Also write items.jsonl for greppability
        jl = path.with_name("items.jsonl")
        with jl.open("w", encoding="utf-8") as fh:
            for it in self.items:
                fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [ItemRecord.from_dict(x) for x in data.get("items") or []]
        return cls(
            campaign=data.get("campaign", "campaign"),
            scope=data.get("scope", ""),
            attacker_model=data.get("attacker_model", ""),
            defender_model=data.get("defender_model", ""),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            completed_phases=list(data.get("completed_phases") or []),
            phase_log=list(data.get("phase_log") or []),
            items=items,
            meta=dict(data.get("meta") or {}),
        )
