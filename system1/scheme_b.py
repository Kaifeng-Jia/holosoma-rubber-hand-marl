"""Scheme B (T02 v2): SONIC kinematic planner (lower body / root path) + teleop encoder (3-point targets).

Planner handling follows the official C++ deployment stack (GR00T-WholeBodyControl @1e1ec4f5,
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref):
* planner thread at 10 Hz (planner_dt 0.1 s = every 5 control steps); a replan happens whenever an input
  changed (exact comparison, g1_deploy_onnx_ref.cpp ~L3655-3730) - our waypoints move every tick;
* Initialize(): 4 identical context frames from the robot state, IDLE inference, facing forward
  (localmotion_kplanner.hpp Initialize / InitializeContext);
* context for a replan: sampled from the CURRENT PLANNER MOTION (not the robot) starting at
  gen_frame = current_frame + motion_look_ahead_steps (2), 4 samples at 30 Hz, lerp/slerp
  (UpdateContextFromMotion);
* 30 Hz -> 50 Hz: lerp positions/joints, slerp root quaternion, joint velocity by forward difference x50,
  last frame copies the previous velocity (ResampleGeneratedSequence50Hz);
* new plan blended into the playing one over 8 frames starting at gen_frame - current_frame, then the
  cursor is reset to 0 (CurrentFrameAdvancement); cursor +1 per control step;
* default advanced inputs: random_seed 0, allowed_pred_num_tokens [1,1,1,1,1,1,0,0,0,0,0].
Differences from the deployment (preregistered in results_system1/T02/preregistration.md): the planner is
initialised in the world frame from the robot's actual pose (deployment: canonical origin, zero yaw, default
height, then heading re-alignment), and planning is synchronous with the control step.

Quaternions here are (w, x, y, z); planner qpos joint order is MuJoCo = URDF = holosoma order.
"""
from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLANNER_ONNX = Path("/home/kevin/third_party/GR00T-WholeBodyControl/sonic_hf_extra/planner_sonic.onnx")
PLANNER_SHA256 = "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea"
URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
PLANNER_TICK = 5          # control steps per planner tick (10 Hz)
LOOK_AHEAD = 2            # motion_look_ahead_steps
BLEND = 8                 # blend_num_frames
ALLOWED_TOKENS = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
MAX_FRAMES_30 = 44        # 11 tokens x 4 frames: longest horizon the default mask allows
MODE_IDLE, MODE_WALK = 0, 2
# Time of the 4th (last) waypoint as a fraction of H (preregistration section 8). 1.0 = original rule 2; the value
# that passes the revised gate (tried in the order 0.5, 0.25, 0.125) is written here before any B evaluation.
LAST_WAYPOINT_FRAC = 0.5   # fixed 2026-11-10 21:27 EST: first value passing the revised gate (section 8)


def mujoco_joint_names():
    return [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") in ("revolute", "continuous")]


def qslerp(q0, q1, t):
    """Slerp between unit quaternions (..., 4), shortest path; t broadcastable."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float).copy()
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1); d = np.abs(d)
    t = np.asarray(t, float)[..., None] if np.ndim(t) else t
    lin = d > 0.9995
    th = np.arccos(np.clip(d, -1, 1)); s = np.sin(th)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(lin, 1 - t, np.sin((1 - t) * th) / np.where(s == 0, 1, s))
        b = np.where(lin, t, np.sin(t * th) / np.where(s == 0, 1, s))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Planner:
    """planner_sonic.onnx via onnxruntime on CPU, batch 1 (the model's batch dimension is fixed to 1)."""

    def __init__(self, path=PLANNER_ONNX, threads=1):
        import onnxruntime as ort

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != PLANNER_SHA256:
            raise RuntimeError("planner_sonic.onnx sha256 mismatch")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.calls, self.seconds = 0, 0.0

    def __call__(self, ctx, mode, target_vel, mdir, facing, height, has_target, tpos, thead, seed=0):
        f = dict(context_mujoco_qpos=np.asarray(ctx, np.float32)[None], target_vel=np.array([target_vel], np.float32),
                 mode=np.array([mode], np.int64), movement_direction=np.asarray(mdir, np.float32)[None],
                 facing_direction=np.asarray(facing, np.float32)[None], height=np.array([height], np.float32),
                 random_seed=np.array([seed], np.int64), has_specific_target=np.array([[has_target]], np.int64),
                 specific_target_positions=np.asarray(tpos, np.float32)[None], specific_target_headings=np.asarray(thead, np.float32)[None],
                 allowed_pred_num_tokens=np.array([ALLOWED_TOKENS], np.int64))
        t0 = time.perf_counter()
        q, n = self.sess.run(None, f)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        n = int(np.asarray(n).reshape(-1)[0])
        if n > 64 or not np.isfinite(q[0, :n]).all():
            raise RuntimeError(f"planner output invalid (n={n})")
        return q[0, :n].astype(np.float64)


class PlannerTrack:
    """One robot's planner motion at 50 Hz (root pos, root quat wxyz, joints in SONIC order) and playback cursor."""

    def __init__(self, planner: Planner, sonic_joint_names):
        mj = mujoco_joint_names()
        self.mj_to_sonic = np.array([mj.index(n) for n in sonic_joint_names])   # sonic[j] = mj[mj_to_sonic[j]]
        self.sonic_to_mj = np.array([list(sonic_joint_names).index(n) for n in mj])
        self.planner = planner
        self.motion = None
        self.cur = 0
        self.gen_frame = 0

    # -- 30 Hz planner output -> 50 Hz motion
    def resample(self, q30):
        n30 = len(q30)
        T = int(np.floor(n30 / 30.0 * 50))
        f30 = np.arange(T) / 50.0 * 30
        f0 = np.floor(f30).astype(int); f1 = np.minimum(f0 + 1, n30 - 1)
        w1 = (f30 - f0)[:, None]; w0 = 1 - w1
        pos = w0 * q30[f0, :3] + w1 * q30[f1, :3]
        quat = qslerp(q30[f0, 3:7], q30[f1, 3:7], f30 - f0)
        jmj = w0 * q30[f0, 7:] + w1 * q30[f1, 7:]
        jpos = jmj[:, self.mj_to_sonic]
        jvel = np.zeros_like(jpos)
        jvel[:-1] = (jpos[1:] - jpos[:-1]) * 50.0
        jvel[-1] = jvel[-2]
        return dict(pos=pos, quat=quat, jpos=jpos, jvel=jvel)

    def initialize(self, root_pos, root_quat, jpos_sonic):
        """Deployment Initialize(): IDLE inference from 4 identical frames of the robot state (here in the world frame)."""
        ctx = np.zeros((4, 36))
        ctx[:, :3] = root_pos; ctx[:, 3:7] = root_quat; ctx[:, 7:] = np.asarray(jpos_sonic)[self.sonic_to_mj]
        y = yaw_of(np.asarray(root_quat))
        q30 = self.planner(ctx, MODE_IDLE, -1.0, (0, 0, 0), (np.cos(y), np.sin(y), 0), -1.0, 0, np.zeros((4, 3)), np.zeros(4))
        self.motion = self.resample(q30)   # first time: copied over
        self.cur = 0

    def context(self):
        m, T = self.motion, len(self.motion["pos"])
        self.gen_frame = self.cur + LOOK_AHEAD
        ctx = np.zeros((4, 36))
        for n in range(4):
            f50 = (self.gen_frame / 50.0 + n / 30.0) * 50.0
            f0 = min(int(np.floor(f50)), T - 1); f1 = min(f0 + 1, T - 1)
            w1 = f50 - f0; w0 = 1 - w1
            ctx[n, 3:7] = qslerp(m["quat"][f0], m["quat"][f1], f50 - f0)
            ctx[n, :3] = w0 * m["pos"][f0] + w1 * m["pos"][f1]
            ctx[n, 7:] = (w0 * m["jpos"][f0] + w1 * m["jpos"][f1])[self.sonic_to_mj]
        return ctx

    def replan(self, mode, target_vel, mdir, facing, height, has_target, tpos, thead):
        new = self.resample(self.planner(self.context(), mode, target_vel, mdir, facing, height, has_target, tpos, thead))
        old, Told, Tnew = self.motion, len(self.motion["pos"]), len(new["pos"])
        L = self.gen_frame - self.cur + Tnew
        if L <= 0:
            return
        start = max(0, self.gen_frame - self.cur)
        f = np.arange(L)
        fo = np.clip(f + self.cur, 0, Told - 1)
        fn = np.clip(f + self.cur - self.gen_frame, 0, Tnew - 1)
        wn = np.clip((f - start) / BLEND, 0.0, 1.0)[:, None]
        wo = 1 - wn
        self.motion = dict(pos=wo * old["pos"][fo] + wn * new["pos"][fn],
                           quat=qslerp(old["quat"][fo], new["quat"][fn], wn[:, 0]),
                           jpos=wo * old["jpos"][fo] + wn * new["jpos"][fn],
                           jvel=wo * old["jvel"][fo] + wn * new["jvel"][fn])
        self.cur = 0

    def advance(self):
        self.cur += 1

    def frame(self, k=0):
        return min(self.cur + k, len(self.motion["pos"]) - 1)

    def future_window(self, n_future=10, skip=5):
        idx = [self.frame(i * skip) for i in range(n_future)]
        return self.motion["jpos"][idx], self.motion["jvel"][idx]

    def root(self):
        f = self.frame()
        return self.motion["pos"][f], self.motion["quat"][f]


def waypoints(ref_pos, ref_quat, k, n_ref, horizon_frames_50, shift_xy=np.zeros(2), shift_yaw=0.0, last_frac=None):
    """Preregistered rule 2: 4 waypoints at the reference pelvis after the current time, evenly spaced over the
    horizon (gen frame + i*H/4, i = 1..4, clamped to the last reference frame); optional feedback shift.
    Rule revision (preregistration section 8): the 4th (last) waypoint sits at gen frame + last_frac*H instead of
    gen frame + H; waypoints 1-3 unchanged (the planner output depends on the last waypoint only, section 0).
    last_frac None -> LAST_WAYPOINT_FRAC (1.0 = original rule 2).
    ref_pos [N,3] world, ref_quat [N,4] wxyz; k = current reference frame. Returns (pos [4,3], yaw [4], frames)."""
    lf = LAST_WAYPOINT_FRAC if last_frac is None else float(last_frac)
    g = k + LOOK_AHEAD
    offs = [int(round(i * horizon_frames_50 / 4.0)) for i in range(1, 4)] + [int(round(lf * horizon_frames_50))]
    fr = np.array([min(g + o, n_ref - 1) for o in offs])
    pos = ref_pos[fr].copy()
    pos[:, :2] += shift_xy
    yaw = yaw_of(ref_quat[fr]) + shift_yaw
    return pos, yaw, fr


def ref_planar_speed(ref_pos, k, n_ref):
    a, b = max(k - 1, 0), min(k + 1, n_ref - 1)
    return float(np.linalg.norm(ref_pos[b, :2] - ref_pos[a, :2]) / ((b - a) / 50.0)) if b > a else 0.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def preregistered_constants():
    return dict(planner_onnx=str(PLANNER_ONNX), planner_sha256=PLANNER_SHA256, planner_tick_control_steps=PLANNER_TICK,
                look_ahead=LOOK_AHEAD, blend_frames=BLEND, allowed_tokens=list(ALLOWED_TOKENS),
                waypoint_horizon_frames_50hz=MAX_FRAMES_30 / 30.0 * 50.0, mode_walk=MODE_WALK, random_seed=0,
                last_waypoint_frac=LAST_WAYPOINT_FRAC)


if __name__ == "__main__":
    print(json.dumps(preregistered_constants(), indent=1))
