"""T14 round 6: support (stance) frames of the person's feet, the evaluation's definition (CoCarry): toe speed < 0.01 m/s.
Motion-capture jitter (CORE4D 15 Hz) splits stance into 1-2 frame pieces, so the toes' horizontal positions are first smoothed
with a Gaussian filter (sigma = 8 frames at 30 Hz = 0.27 s; scipy.ndimage.gaussian_filter1d, mode "nearest"), then the speed is
the 30 Hz difference of the smoothed positions. One setting for all tasks."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

SIGMA_FRAMES, THRESHOLD_MPS, FPS = 8.0, 0.01, 30.0


def support_flags(toe_xyz):
    """toe_xyz [T, 2 (left, right), 3] -> flags [T, 2] (frame 0 = False: no speed)."""
    x = gaussian_filter1d(np.asarray(toe_xyz, float)[..., :2], SIGMA_FRAMES, axis=0, mode="nearest")
    v = np.zeros(x.shape[:2])
    v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * FPS
    f = v < THRESHOLD_MPS
    f[0] = False
    return f


def as_solver_sequence(flags):
    return [{"L_Toe": bool(a), "R_Toe": bool(b)} for a, b in flags]
