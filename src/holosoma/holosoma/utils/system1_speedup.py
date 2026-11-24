"""System1 T13 v4 section 4 A: result-preserving speed-ups (no per-step GPU->CPU syncs in a few hot paths).
HOLOSOMA_S1_SPEEDUP=0 restores the previous code paths exactly (used for the same-seed before / after comparison)."""
import os

ENABLED = os.environ.get("HOLOSOMA_S1_SPEEDUP", "1") != "0"
