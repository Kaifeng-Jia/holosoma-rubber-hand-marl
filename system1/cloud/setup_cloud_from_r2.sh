#!/bin/bash
# System1 cloud setup variant for the 5x4090 AutoDL machine (2026-11-10): instead of running
# scripts/setup_isaacsim.sh from scratch, COPY the Isaac Sim env already on this data disk
# (/root/autodl-tmp/r2_deps/hssim: Python 3.11, torch 2.7.0+cu128, isaacsim 5.1.0.0, isaaclab 0.47.2,
# numpy 1.26.0, scipy 1.15.3 - the same versions as the Ubuntu laptop env hssim_s1) and its IsaacLab
# source, then re-point both editable installs (holosoma, isaaclab) to copies under $S1_ROOT.
# The originals (r2_deps, r2_2026*) are only read, never modified. Everything else is as setup_cloud.sh.
#
#   S1_ROOT=/root/autodl-tmp/system1 bash setup_cloud_from_r2.sh
set -euo pipefail
S1_ROOT=${S1_ROOT:-/root/autodl-tmp/system1}
SRC_ENV=/root/autodl-tmp/r2_deps/hssim
SRC_LAB=/root/autodl-tmp/r2_deps/IsaacLab
REPO_URL=${REPO_URL:-https://github.com/Kaifeng-Jia/holosoma-rubber-hand-marl.git}
BRANCH=system1-sonic
SONIC_COMMIT=1e1ec4f5f27e6c3e03048f693bd7c60cce04afda
HF_REV=6733128a3d8a523b1418b06bca3cdf61c8b0987f
SONIC_PT_SHA=e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909
SONIC_CFG_SHA=f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$S1_ROOT"/{deps,third_party,inputs,logs}
LOG="$S1_ROOT/logs/setup_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
echo "start $(date)"

# 1. code
if [[ ! -d $S1_ROOT/duet-system1 ]]; then
  git clone --branch $BRANCH "$REPO_URL" "$S1_ROOT/duet-system1"
fi
cd "$S1_ROOT/duet-system1"
for i in 1 2 3 4 5; do git -c http.version=HTTP/1.1 pull --ff-only && break; echo "git pull attempt $i failed"; sleep 10; done
git rev-parse HEAD

# 2. simulator env: copy, then re-point editable installs to our own copies
ENV=$S1_ROOT/deps/hssim_s1
if [[ ! -d $ENV ]]; then
  echo "copying $SRC_ENV -> $ENV"; cp -a "$SRC_ENV" "$ENV.partial" && mv "$ENV.partial" "$ENV"
fi
if [[ ! -d $S1_ROOT/deps/IsaacLab ]]; then
  cp -a "$SRC_LAB" "$S1_ROOT/deps/IsaacLab.partial" && mv "$S1_ROOT/deps/IsaacLab.partial" "$S1_ROOT/deps/IsaacLab"
fi
PY=$ENV/bin/python
$PY -m pip install --no-deps --no-build-isolation -e "$S1_ROOT/deps/IsaacLab/source/isaaclab"
$PY -m pip install --no-deps --no-build-isolation -e "$S1_ROOT/duet-system1/src/holosoma"
$PY - <<'EOF'
import holosoma, isaaclab, sys
print("holosoma ->", holosoma.__file__); print("isaaclab ->", isaaclab.__file__); print("prefix", sys.prefix)
assert "/system1/" in holosoma.__file__ and "/system1/" in isaaclab.__file__, "editable installs still point outside S1_ROOT"
EOF
if grep -rl "r2_deps\|r2_2026" "$ENV"/lib/python3.11/site-packages/__editable__* 2>/dev/null; then
  echo "editable finder still references the original tree"; exit 3
fi

# 3. SONIC code + small runtime deps (no upgrades) + release checkpoint (sha256 checked)
# GitHub is unreliable from this machine: SONIC code comes as `git archive` of $SONIC_COMMIT (gear_sonic/, LICENSE,
# config.json; LFS pointers not smudged, as a GIT_LFS_SKIP_SMUDGE clone) made on the Ubuntu box, sha256 checked.
SONIC_TAR=$S1_ROOT/inputs/sonic_code_1e1ec4f5.tar
SONIC_TAR_SHA=cd5c39126896bc9cd9edb41718f96b4bc002597fb68a997fc51e30b6b5a80294
if [[ ! -d $S1_ROOT/third_party/GR00T-WholeBodyControl ]]; then
  if [[ -f $SONIC_TAR ]]; then
    echo "$SONIC_TAR_SHA  $SONIC_TAR" | sha256sum -c -
    tar -C "$S1_ROOT/third_party" -xf "$SONIC_TAR"
    printf '{"source": "git archive", "commit": "%s", "tar_sha256": "%s"}\n' $SONIC_COMMIT $SONIC_TAR_SHA > "$S1_ROOT/third_party/GR00T-WholeBodyControl/S1_SOURCE.json"
  else
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$S1_ROOT/third_party/GR00T-WholeBodyControl"
  fi
fi
cd "$S1_ROOT/third_party/GR00T-WholeBodyControl"
if [[ -d .git ]]; then git checkout $SONIC_COMMIT; else grep -q $SONIC_COMMIT S1_SOURCE.json; fi
$PY -m pip install --no-deps einx==0.4.3 frozendict==2.5.1 torch-einops-utils==0.2.6 vector_quantize_pytorch==1.32.2
# hydra-core: imported by gear_sonic (utils/motion_lib/torch_humanoid_batch.py); present in the laptop env, absent in the R2 env
$PY -m pip install --no-deps hydra-core==1.3.3
$PY -m pip install --no-deps --no-build-isolation -e gear_sonic/
for f in sonic_release/config.yaml sonic_release/last.pt config.json LICENSE; do
  mkdir -p "$(dirname "$f")"
  [[ -f $f ]] || curl -sSL --fail -o "$f" "https://huggingface.co/nvidia/GEAR-SONIC/resolve/$HF_REV/$f"
done
echo "$SONIC_PT_SHA  sonic_release/last.pt" | sha256sum -c -
echo "$SONIC_CFG_SHA  sonic_release/config.yaml" | sha256sum -c -
cat > "$S1_ROOT/s1_env.sh" <<EOF2
export S1_ROOT=$S1_ROOT
export S1_SONIC_ROOT=$S1_ROOT/third_party/GR00T-WholeBodyControl
export S1_PY=$ENV/bin/python
export S1_INPUTS=$S1_ROOT/inputs
export OMNI_KIT_ACCEPT_EULA=1
EOF2

# 4. inputs from the Ubuntu box
if [[ -f $S1_ROOT/inputs/cloud_inputs.tar ]]; then
  tar -C "$S1_ROOT/inputs" -xf "$S1_ROOT/inputs/cloud_inputs.tar"
  (cd "$S1_ROOT/inputs" && sha256sum -c SHA256SUMS)
  mkdir -p "$S1_ROOT/duet-system1/logs/WholeBodyTracking/marl_compat_pull_v1"
  cp -p "$S1_ROOT/inputs/priors/marl_compat_pull_v1/model_07999_actor158.pt" "$S1_ROOT/duet-system1/logs/WholeBodyTracking/marl_compat_pull_v1/"
fi

# 5. record versions
$PY - <<'EOF' | tee "$S1_ROOT/logs/versions.json"
import json, platform, sys, torch, importlib.metadata as m
print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
    isaacsim=m.version("isaacsim"), isaaclab=m.version("isaaclab"), numpy=m.version("numpy"), scipy=m.version("scipy"),
    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], host=platform.node(),
    env_source="copy of /root/autodl-tmp/r2_deps/hssim"), indent=1))
EOF
$PY -m pip freeze > "$S1_ROOT/logs/pip_freeze.txt"
echo "setup done $(date); log: $LOG"
