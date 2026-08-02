#!/usr/bin/env python3
"""CLI entry for the full multi-phase educational pipeline.

  .venv\\Scripts\\python.exe scripts/run_full_pipeline.py --list-phases
  .venv\\Scripts\\python.exe scripts/run_full_pipeline.py --limit 5
  .venv\\Scripts\\python.exe scripts/run_full_pipeline.py
  .venv\\Scripts\\python.exe scripts/run_full_pipeline.py --from-phase export
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoredteam.pipeline.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
