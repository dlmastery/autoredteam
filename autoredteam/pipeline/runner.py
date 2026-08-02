"""Pipeline runner - execute phases 0-9 with checkpoints and resume."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import BASE_DIR
from .phases import PHASE_FUNCS, SCOPE_DEFAULT
from .state import PipelineState

PHASE_ORDER: list[str] = [
    "setup",
    "compose",
    "attack_gen",
    "defend_single",
    "multiturn",
    "universal_bon",
    "judge",
    "hardneg",
    "export",
    "dashboard",
]


def _state_path(run_dir: Path) -> Path:
    return run_dir / "pipeline" / "state.json"


def run_pipeline(
    *,
    campaign: str = "local-gemma4b-full-pipeline",
    from_phase: str | None = None,
    to_phase: str | None = None,
    only_phases: list[str] | None = None,
    resume: bool = True,
    cfg: dict[str, Any] | None = None,
) -> PipelineState:
    """Run the educational multi-phase pipeline.

    Parameters
    ----------
    campaign:
        Run directory name under ``runs/``.
    from_phase / to_phase:
        Inclusive slice of PHASE_ORDER (by name).
    only_phases:
        If set, run exactly these phases (in PHASE_ORDER sequence).
    resume:
        Load checkpoint if present and skip already completed phases
        (unless re-running a phase that is before the checkpoint slice).
    cfg:
        Runtime knobs: limit, skip_pull, skip_author, skip_smoke, bon_n,
        multiturn_max_turns, hardneg_rounds, llm_judge, seed, ...
    """
    cfg = dict(cfg or {})
    run_dir = Path(cfg.get("run_dir") or (BASE_DIR / "runs" / campaign))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("run_dir", str(run_dir))
    cfg.setdefault("dataset_dir", str(BASE_DIR / "datasets"))
    cfg.setdefault("dashboard_path", str(BASE_DIR / "dashboard" / "index.html"))

    sp = _state_path(run_dir)
    if resume and sp.is_file():
        state = PipelineState.load(sp)
        print(f"[resume] loaded {sp} (completed={state.completed_phases})")
    else:
        state = PipelineState(
            campaign=campaign,
            scope=cfg.get("scope") or SCOPE_DEFAULT,
        )

    # Resolve phase list
    if only_phases:
        phases = [p for p in PHASE_ORDER if p in only_phases]
    else:
        phases = list(PHASE_ORDER)
        if from_phase:
            if from_phase not in phases:
                raise ValueError(f"unknown from_phase {from_phase!r}; choose from {PHASE_ORDER}")
            phases = phases[phases.index(from_phase) :]
        if to_phase:
            if to_phase not in PHASE_ORDER:
                raise ValueError(f"unknown to_phase {to_phase!r}; choose from {PHASE_ORDER}")
            # inclusive
            end = PHASE_ORDER.index(to_phase)
            phases = [p for p in phases if PHASE_ORDER.index(p) <= end]

    force = bool(cfg.get("force", False))
    # Explicit from_phase / only_phases implies re-running those phases.
    if from_phase and from_phase in PHASE_ORDER:
        cut = PHASE_ORDER.index(from_phase)
        state.completed_phases = [
            p for p in state.completed_phases if PHASE_ORDER.index(p) < cut
        ]
    if only_phases:
        drop = set(only_phases)
        state.completed_phases = [p for p in state.completed_phases if p not in drop]

    t0 = time.time()
    print(f"[pipeline] campaign={campaign}")
    print(f"[pipeline] phases: {' -> '.join(phases)}")

    for name in phases:
        if name in state.completed_phases and not force:
            print(f"[skip] {name} (already completed; use --force to redo)")
            continue

        fn = PHASE_FUNCS[name]
        print(f"\n======== PHASE: {name} ========")
        state = fn(state, cfg)
        state.save(sp)
        print(f"[checkpoint] {sp}")

    state.finished_at = state.finished_at or __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    state.save(sp)
    elapsed = (time.time() - t0) / 60.0
    n_ok = sum(1 for it in state.items if it.final_success or it.success)
    print(f"\n[pipeline] done in {elapsed:.1f} min | items={len(state.items)} successes~{n_ok}")
    print(f"[pipeline] state: {sp}")
    print(f"[pipeline] dashboard: {cfg.get('dashboard_path')}")
    return state


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Full multi-phase educational red-team pipeline")
    ap.add_argument("--campaign", default="local-gemma4b-full-pipeline")
    ap.add_argument("--from-phase", default=None, choices=PHASE_ORDER)
    ap.add_argument("--to-phase", default=None, choices=PHASE_ORDER)
    ap.add_argument("--only", nargs="+", default=None, help="Run only these phase names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-author", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run phases even if checkpointed")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bon-n", type=int, default=4)
    ap.add_argument("--multiturn-max-turns", type=int, default=3)
    ap.add_argument("--hardneg-rounds", type=int, default=2)
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list-phases", action="store_true")
    args = ap.parse_args(argv)

    if args.list_phases:
        for i, p in enumerate(PHASE_ORDER):
            print(f"  {i}: {p}")
        return

    cfg = {
        "limit": args.limit,
        "skip_pull": args.skip_pull,
        "skip_author": args.skip_author,
        "skip_smoke": args.skip_smoke,
        "force": args.force,
        "bon_n": args.bon_n,
        "multiturn_max_turns": args.multiturn_max_turns,
        "hardneg_rounds": args.hardneg_rounds,
        "llm_judge": args.llm_judge,
        "seed": args.seed,
    }
    run_pipeline(
        campaign=args.campaign,
        from_phase=args.from_phase,
        to_phase=args.to_phase,
        only_phases=args.only,
        resume=not args.no_resume,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
