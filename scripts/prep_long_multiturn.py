#!/usr/bin/env python3
"""Clone full-pipeline state into a long multi-turn campaign checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoredteam.pipeline.state import PipelineState  # noqa: E402


def main() -> None:
    src = ROOT / "runs" / "local-gemma4b-full-pipeline" / "pipeline" / "state.json"
    dst_dir = ROOT / "runs" / "local-gemma4b-long-multiturn" / "pipeline"
    dst_dir.mkdir(parents=True, exist_ok=True)

    state = PipelineState.load(src)
    state.campaign = "local-gemma4b-long-multiturn"
    keep = {"setup", "compose", "attack_gen", "defend_single"}
    state.completed_phases = [p for p in state.completed_phases if p in keep]

    n_st_fail = 0
    for it in state.items:
        st_ok = it.meta.get("single_turn_success")
        if st_ok is None:
            # Infer: prior multiturn phase only ran on single-turn failures.
            if "multiturn" in (it.phases_hit or []) or it.multiturn_turns or it.multiturn_strategy:
                st_ok = False
            elif it.success and not it.multiturn_success and not it.bon_success and not it.hardneg_success:
                st_ok = True
            else:
                st_ok = bool(it.success) and "multiturn" not in (it.phases_hit or [])
            it.meta["single_turn_success"] = bool(st_ok)

        if it.meta["single_turn_success"]:
            continue

        n_st_fail += 1
        it.multiturn_turns = []
        it.multiturn_success = False
        it.multiturn_strategy = ""
        it.meta.pop("multiturn_messages", None)
        it.meta.pop("multiturn_max_turns", None)
        it.meta.pop("multiturn_history", None)
        it.success = False
        it.final_success = False
        it.final_strategy = ""
        it.final_prompt = it.attack_prompt or it.seed_prompt
        it.final_response = ""
        it.final_score = 0.0
        it.final_verdict = ""
        it.phases_hit = [p for p in it.phases_hit if p in ("attack_gen", "defend_single", "compose")]
        it.bon_variants = []
        it.bon_success = False
        it.bon_best_prompt = ""
        it.bon_best_response = ""
        it.hardneg_attempts = []
        it.hardneg_success = False

    sp = dst_dir / "state.json"
    state.save(sp)
    print(f"saved {sp}")
    print(f"items={len(state.items)} single_turn_failures={n_st_fail}")
    print(f"completed_phases={state.completed_phases}")


if __name__ == "__main__":
    main()
