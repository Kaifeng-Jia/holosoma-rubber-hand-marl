#!/usr/bin/env python3
"""Train the isolated mirrored cooperative Kick MAPPO baseline."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ENTRY_POINT = Path(__file__).with_name("train_plan5_push.py")


def main() -> None:
    if "--skill" in sys.argv[1:]:
        raise SystemExit("train_plan5_kick.py fixes --skill=kick; do not pass --skill")
    sys.argv = [str(ENTRY_POINT), "--skill", "kick", *sys.argv[1:]]
    runpy.run_path(str(ENTRY_POINT), run_name="__main__")


if __name__ == "__main__":
    main()
