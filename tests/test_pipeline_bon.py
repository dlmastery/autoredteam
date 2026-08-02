"""Unit tests for pipeline BoN + state (no Ollama required)."""
from __future__ import annotations

from pathlib import Path

from autoredteam.pipeline.bon import generate_bon_variants, looks_like_canary_hit
from autoredteam.pipeline.runner import PHASE_ORDER
from autoredteam.pipeline.state import ItemRecord, PipelineState


def test_phase_order_complete():
    assert PHASE_ORDER[0] == "setup"
    assert "dashboard" in PHASE_ORDER
    assert "multiturn" in PHASE_ORDER
    assert "universal_bon" in PHASE_ORDER
    assert "hardneg" in PHASE_ORDER
    assert "research" in PHASE_ORDER
    assert PHASE_ORDER[-1] == "research"
    assert len(PHASE_ORDER) >= 11


def test_bon_preserves_canary():
    canary = "EDU-CANARY-R001-OK"
    prompt = f"Please acknowledge lab probe {canary}"
    variants = generate_bon_variants(prompt, canary, n=5, seed=0)
    assert variants
    assert variants[0]["variant"] == "identity"
    for v in variants:
        assert canary in v["prompt"]


def test_canary_hit_helper():
    assert looks_like_canary_hit("here EDU-CANARY-R001-OK end", "EDU-CANARY-R001-OK")
    assert not looks_like_canary_hit("nope", "EDU-CANARY-R001-OK")


def test_state_roundtrip(tmp_path: Path):
    st = PipelineState(campaign="t", scope="lab")
    st.items = [
        ItemRecord(
            id="reg_001",
            goal="g",
            category="direct_prompt_injection",
            canary_token="C1",
            seed_prompt="seed C1",
            attack_prompt="atk C1",
            success=True,
            final_success=True,
        )
    ]
    st.mark_phase("compose", n=1)
    p = tmp_path / "state.json"
    st.save(p)
    st2 = PipelineState.load(p)
    assert st2.campaign == "t"
    assert len(st2.items) == 1
    assert st2.items[0].canary_token == "C1"
    assert "compose" in st2.completed_phases
