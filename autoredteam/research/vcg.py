"""Vulnerability Concept Graph -- inspired by AHA (Mao et al., arXiv:2607.11698).

AHA's key product is not only attack success rate but a *reusable vulnerability
concept*: claim, enabling condition, falsifier, transfer prediction, evidence.

We implement a lightweight educational VCG suitable for canary-based local labs
and exportable for hardening / classifier feature engineering.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VulnerabilityConcept:
    """One node in the Vulnerability Concept Graph."""

    id: str
    claim: str
    enabling_condition: str
    attacker_surface: str  # e.g. template / strategy / channel
    unsafe_trajectory_summary: str
    falsifier: str  # how to prove the concept wrong (patch test)
    transfer_prediction: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    category: str = ""
    strategy: str = ""
    jailbreak_type: str = "regular"
    confidence: float = 0.5
    confirmed: bool = False
    created_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VulnerabilityConcept":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class VulnerabilityConceptGraph:
    """Store and export VCG nodes (AHA-style research asset)."""

    def __init__(self) -> None:
        self.concepts: dict[str, VulnerabilityConcept] = {}

    def add(self, concept: VulnerabilityConcept) -> VulnerabilityConcept:
        self.concepts[concept.id] = concept
        return concept

    def promote_from_success(
        self,
        *,
        category: str,
        strategy: str,
        template: str,
        jailbreak_type: str,
        prompt: str,
        response: str,
        canary_token: str,
        score: float,
        goal: str = "",
    ) -> VulnerabilityConcept:
        """Build a confirmed concept from a successful educational canary hit."""
        cid = f"vcg-{uuid.uuid4().hex[:12]}"
        claim = (
            f"Defender may emit canary under {jailbreak_type}/{template or strategy} "
            f"framing for category={category}."
        )
        enabling = (
            f"Attack surface: strategy={strategy}, template={template or 'n/a'}, "
            f"jailbreak_type={jailbreak_type}. Success signal: canary token present "
            f"in model output (lab metric)."
        )
        falsifier = (
            "Patch test: re-run the same prompt family after defense update; "
            "concept is falsified if canary hit rate falls below threshold on held-out seeds."
        )
        transfer = (
            f"Predicted to transfer to other goals in category={category} that share "
            f"template={template or strategy}; weaker transfer expected across categories."
        )
        traj = (
            f"goal={goal[:120]} | prompt_len={len(prompt)} | "
            f"response_snip={(response or '')[:160]}"
        )
        concept = VulnerabilityConcept(
            id=cid,
            claim=claim,
            enabling_condition=enabling,
            attacker_surface=f"{jailbreak_type}:{template or strategy}",
            unsafe_trajectory_summary=traj,
            falsifier=falsifier,
            transfer_prediction=transfer,
            evidence=[
                {
                    "prompt": prompt[:2000],
                    "response": (response or "")[:1000],
                    "canary_token": canary_token,
                    "score": score,
                }
            ],
            category=category,
            strategy=strategy,
            jailbreak_type=jailbreak_type,
            confidence=min(0.95, 0.5 + 0.45 * float(score)),
            confirmed=True,
            meta={"educational": True, "inspired_by": "AHA VCG (arXiv:2607.11698)"},
        )
        return self.add(concept)

    def propose_hypothesis(
        self,
        *,
        category: str,
        strategy: str,
        template: str,
        jailbreak_type: str,
    ) -> VulnerabilityConcept:
        """Unconfirmed hypothesis node (AHA: commit to claim before attack)."""
        cid = f"hyp-{uuid.uuid4().hex[:12]}"
        concept = VulnerabilityConcept(
            id=cid,
            claim=(
                f"Hypothesis: {jailbreak_type} attacks using strategy={strategy} "
                f"(template={template}) can elicit policy-bypass canary in {category}."
            ),
            enabling_condition="Unverified; awaiting falsifier experiment.",
            attacker_surface=f"{jailbreak_type}:{template or strategy}",
            unsafe_trajectory_summary="",
            falsifier="If 0/N canary hits on fixed seed set, reject hypothesis.",
            transfer_prediction="Unknown until confirmed.",
            category=category,
            strategy=strategy,
            jailbreak_type=jailbreak_type,
            confidence=0.2,
            confirmed=False,
            meta={"educational": True, "status": "hypothesis"},
        )
        return self.add(concept)

    def confirm(self, concept_id: str, evidence: dict[str, Any], score: float = 0.9) -> None:
        c = self.concepts[concept_id]
        c.confirmed = True
        c.confidence = min(0.95, max(c.confidence, score))
        c.evidence.append(evidence)
        c.enabling_condition = (
            f"Confirmed under educational canary protocol. "
            f"Surface={c.attacker_surface}."
        )

    def reject(self, concept_id: str, reason: str) -> None:
        c = self.concepts[concept_id]
        c.confirmed = False
        c.confidence = min(c.confidence, 0.15)
        c.meta["rejected"] = True
        c.meta["reject_reason"] = reason

    def confirmed(self) -> list[VulnerabilityConcept]:
        return [c for c in self.concepts.values() if c.confirmed]

    def stats(self) -> dict[str, Any]:
        conf = self.confirmed()
        return {
            "n_concepts": len(self.concepts),
            "n_confirmed": len(conf),
            "n_hypotheses": len(self.concepts) - len(conf),
            "by_category": _count(c.category for c in conf),
            "by_surface": _count(c.attacker_surface for c in conf),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "version": 1,
            "inspired_by": "AHA Vulnerability Concept Graph (arXiv:2607.11698)",
            "stats": self.stats(),
            "concepts": [c.to_dict() for c in self.concepts.values()],
        }
        p.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.is_file():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.concepts = {
            c["id"]: VulnerabilityConcept.from_dict(c) for c in data.get("concepts") or []
        }

    def to_classifier_features(self) -> list[dict[str, Any]]:
        """Flatten concepts into rows useful for training / analysis."""
        rows = []
        for c in self.concepts.values():
            rows.append(
                {
                    "concept_id": c.id,
                    "confirmed": c.confirmed,
                    "category": c.category,
                    "strategy": c.strategy,
                    "jailbreak_type": c.jailbreak_type,
                    "attacker_surface": c.attacker_surface,
                    "claim": c.claim,
                    "enabling_condition": c.enabling_condition,
                    "confidence": c.confidence,
                    "n_evidence": len(c.evidence),
                }
            )
        return rows


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        k = x or "(none)"
        out[k] = out.get(k, 0) + 1
    return out
