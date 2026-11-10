#!/bin/bash
# System1 cloud setup (AutoDL RTX 4090, Ubuntu 22.04). Run once per machine as the instance user.
#
#   S1_ROOT=/root/autodl-tmp/system1 bash setup_cloud.sh
#
# Layout (everything under $S1_ROOT; nothing outside except conda caches):
#   $S1_ROOT/duet-system1          fresh clone of branch system1-sonic (code)
#   $S1_ROOT/deps/miniconda3       conda; env "hssim_s1" = holosoma's own setup_isaacsim.sh
#                                  (Isaac Sim 5.1.0, Isaac Lab v2.3.0, torch 2.7.0+cu128)
#   $S1_ROOT/third_party/GR00T-WholeBodyControl  SONIC code @1e1ec4f5 + release checkpoint
#   $S1_ROOT/inputs                read-only inputs copied from the Ubuntu box (cloud_inputs.tar)
# Credentials are NOT stored here: the git clone needs a token configured by Kevin.
set -euo pipefail
S1_ROOT=${S1_ROOT:-/root/autodl-tmp/system1}
REPO_URL=${REPO_URL:-https://github.com/Kaifeng-Jia/holosoma-rubber-hand-marl.git}
BRANCH=system1-sonic
SONIC_COMMIT=1e1ec4f5f27e6c3e03048f693bd7c60cce04afda
HF_REV=6733128a3d8a523b1418b06bca3cdf61c8b0987f
SONIC_PT_SHA=e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909
SONIC_CFG_SHA=f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7
mkdir -p "$S1_ROOT"/{deps,third_party,inputs,logs}
LOG="$S1_ROOT/logs/setup_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# 1. code
if [[ ! -d $S1_ROOT/duet-system1 ]]; then
  git clone --branch $BRANCH "$REPO_URL" "$S1_ROOT/duet-system1"
fi
cd "$S1_ROOT/duet-system1" && git pull --ff-only && git rev-parse HEAD

# 2. simulator env with holosoma's own installer (same versions as the paper backend)
export WORKSPACE_DIR=$S1_ROOT/deps CONDA_ENV_NAME=hssim_s1 OMNI_KIT_ACCEPT_EULA=1
bash "$S1_ROOT/duet-system1/scripts/setup_isaacsim.sh"
PY=$S1_ROOT/deps/miniconda3/envs/hssim_s1/bin/python
# holosoma editable from this clone (the installer may already do this; --no-deps keeps pins)
$PY -m pip install --no-deps -e "$S1_ROOT/duet-system1/src/holosoma"

# 3. SONIC code + the four small runtime deps used by its network modules (no dependency upgrades)
if [[ ! -d $S1_ROOT/third_party/GR00T-WholeBodyControl ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$S1_ROOT/third_party/GR00T-WholeBodyControl"
fi
cd "$S1_ROOT/third_party/GR00T-WholeBodyControl" && git checkout $SONIC_COMMIT
$PY -m pip install --no-deps einx==0.4.3 frozendict==2.5.1 torch-einops-utils==0.2.6 vector_quantize_pytorch==1.32.2
$PY -m pip install --no-deps --no-build-isolation -e gear_sonic/
for f in sonic_release/config.yaml sonic_release/last.pt config.json LICENSE; do
  mkdir -p "$(dirname "$f")"
  [[ -f $f ]] || curl -sSL --fail -o "$f" "https://huggingface.co/nvidia/GEAR-SONIC/resolve/$HF_REV/$f"
done
echo "$SONIC_PT_SHA  sonic_release/last.pt" | sha256sum -c -
echo "$SONIC_CFG_SHA  sonic_release/config.yaml" | sha256sum -c -
# environment file sourced by every launch script (sonic_controller.py reads S1_SONIC_ROOT)
cat > "$S1_ROOT/s1_env.sh" <<EOF2
export S1_ROOT=$S1_ROOT
export S1_SONIC_ROOT=$S1_ROOT/third_party/GR00T-WholeBodyControl
export S1_PY=$S1_ROOT/deps/miniconda3/envs/hssim_s1/bin/python
export S1_INPUTS=$S1_ROOT/inputs
EOF2

# 4. inputs from the Ubuntu box (priors, D checkpoints, frozen states, V3 tables) — see make_cloud_inputs.py
if [[ -f $S1_ROOT/inputs/cloud_inputs.tar ]]; then
  tar -C "$S1_ROOT/inputs" -xf "$S1_ROOT/inputs/cloud_inputs.tar"
  (cd "$S1_ROOT/inputs" && sha256sum -c SHA256SUMS)
  mkdir -p "$S1_ROOT/duet-system1/logs/WholeBodyTracking/marl_compat_pull_v1"
  cp -p "$S1_ROOT/inputs/priors/marl_compat_pull_v1/model_07999_actor158.pt" "$S1_ROOT/duet-system1/logs/WholeBodyTracking/marl_compat_pull_v1/"
fi

# 5. record versions
$PY - <<'EOF' | tee "$S1_ROOT/logs/versions.json"
import json, platform, subprocess, sys, torch, importlib.metadata as m
print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
    isaacsim=m.version("isaacsim"), isaaclab=m.version("isaaclab"), numpy=m.version("numpy"),
    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], host=platform.node()), indent=1))
EOF
$PY -m pip freeze > "$S1_ROOT/logs/pip_freeze.txt"
echo "setup done; log: $LOG"
