#!/bin/bash
# System1 cloud setup for a FRESH AutoDL machine (empty data disk, 2026-11-20, T10b pilot): setup_cloud.sh with four fixes,
# the SONIC code from the sha256-checked git archive (as setup_cloud_from_r2.sh) and the same s1_env.sh as setup_cloud_from_r2.sh.
#   - scripts/source_common.sh hard-sets WORKSPACE_DIR=$HOME/.holosoma_deps, so setup_cloud.sh's WORKSPACE_DIR export is ignored
#     and the env would land on the 30 GB system disk: $HOME/.holosoma_deps is made a symlink to $S1_ROOT/deps instead.
#   - pip cache and TMPDIR on the data disk (isaacsim wheels are several GB).
#   - /etc/network_turbo (AutoDL proxy for GitHub / HuggingFace / pytorch / NVIDIA indexes; the aliyun pip mirror is no_proxy).
#   - hydra-core==1.3.3 --no-deps (imported by gear_sonic; as setup_cloud_from_r2.sh).
# Versions are those of holosoma's own scripts/setup_isaacsim.sh (Isaac Sim 5.1.0, Isaac Lab v2.3.0, torch 2.7.0+cu128), the same
# installer as the laptop env. Code: public GitHub clone of branch system1-sonic (no credentials stored).
#
#   S1_ROOT=/root/autodl-tmp/system1 bash setup_cloud_fresh.sh      (inputs/sonic_code_1e1ec4f5.tar and inputs/cloud_inputs.tar copied first)
set -euo pipefail
S1_ROOT=${S1_ROOT:-/root/autodl-tmp/system1}
REPO_URL=${REPO_URL:-https://github.com/Kaifeng-Jia/holosoma-rubber-hand-marl.git}
BRANCH=system1-sonic
SONIC_COMMIT=1e1ec4f5f27e6c3e03048f693bd7c60cce04afda
HF_REV=6733128a3d8a523b1418b06bca3cdf61c8b0987f
SONIC_PT_SHA=e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909
SONIC_CFG_SHA=f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7
SONIC_TAR=$S1_ROOT/inputs/sonic_code_1e1ec4f5.tar
SONIC_TAR_SHA=cd5c39126896bc9cd9edb41718f96b4bc002597fb68a997fc51e30b6b5a80294
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$S1_ROOT"/{deps,third_party,inputs,logs,cache/pip,tmp}
export PIP_CACHE_DIR=$S1_ROOT/cache/pip TMPDIR=$S1_ROOT/tmp
LOG="$S1_ROOT/logs/setup_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true
# The turbo proxy re-signs TLS with its own CA (issuer "autodl"); curl / git trust it via the system bundle but libmamba does not
# (first run 2026-11-20 15:0x: "self-signed certificate in certificate chain" on conda.anaconda.org). These hosts are fast
# without the proxy (measured 4-8 MB/s), so the proxy is used only for GitHub and HuggingFace.
export no_proxy="${no_proxy:-},anaconda.org,anaconda.com,pytorch.org,nvidia.com"
export NO_PROXY="$no_proxy"
echo "start $(date)"

# 0. installer workspace -> data disk
if [[ -e $HOME/.holosoma_deps && ! -L $HOME/.holosoma_deps ]]; then echo "$HOME/.holosoma_deps exists and is not a symlink"; exit 2; fi
ln -sfn "$S1_ROOT/deps" "$HOME/.holosoma_deps"

# 1. code
if [[ ! -d $S1_ROOT/duet-system1 ]]; then
  for i in 1 2 3 4 5; do git -c http.version=HTTP/1.1 clone --branch $BRANCH "$REPO_URL" "$S1_ROOT/duet-system1" && break
    echo "git clone attempt $i failed"; rm -rf "$S1_ROOT/duet-system1"; sleep 10; done
fi
cd "$S1_ROOT/duet-system1"
for i in 1 2 3 4 5; do git -c http.version=HTTP/1.1 pull --ff-only && break; echo "git pull attempt $i failed"; sleep 10; done
git rev-parse HEAD

# 2. simulator env with holosoma's own installer
export CONDA_ENV_NAME=hssim_s1 OMNI_KIT_ACCEPT_EULA=1
bash "$S1_ROOT/duet-system1/scripts/setup_isaacsim.sh"
ENV=$S1_ROOT/deps/miniconda3/envs/hssim_s1
PY=$ENV/bin/python
$PY -m pip install --no-deps -e "$S1_ROOT/duet-system1/src/holosoma"
$PY - <<'EOF'
import holosoma, sys
print("holosoma ->", holosoma.__file__); print("prefix", sys.prefix)
assert "/system1/duet-system1/" in holosoma.__file__, "holosoma editable install does not point to the clone"
EOF

# 3. SONIC code (git archive of $SONIC_COMMIT, sha256 checked) + small runtime deps (no upgrades) + release checkpoint
if [[ ! -d $S1_ROOT/third_party/GR00T-WholeBodyControl ]]; then
  echo "$SONIC_TAR_SHA  $SONIC_TAR" | sha256sum -c -
  tar -C "$S1_ROOT/third_party" -xf "$SONIC_TAR"
  printf '{"source": "git archive", "commit": "%s", "tar_sha256": "%s"}\n' $SONIC_COMMIT $SONIC_TAR_SHA > "$S1_ROOT/third_party/GR00T-WholeBodyControl/S1_SOURCE.json"
fi
cd "$S1_ROOT/third_party/GR00T-WholeBodyControl"
grep -q $SONIC_COMMIT S1_SOURCE.json
$PY -m pip install --no-deps einx==0.4.3 frozendict==2.5.1 torch-einops-utils==0.2.6 vector_quantize_pytorch==1.32.2 hydra-core==1.3.3
$PY -m pip install --no-deps --no-build-isolation -e gear_sonic/
for f in sonic_release/config.yaml sonic_release/last.pt config.json LICENSE; do
  mkdir -p "$(dirname "$f")"
  [[ -f $f ]] || curl -sSL --fail --retry 5 -o "$f" "https://huggingface.co/nvidia/GEAR-SONIC/resolve/$HF_REV/$f"
done
echo "$SONIC_PT_SHA  sonic_release/last.pt" | sha256sum -c -
echo "$SONIC_CFG_SHA  sonic_release/config.yaml" | sha256sum -c -
cat > "$S1_ROOT/s1_env.sh" <<EOF2
export S1_ROOT=$S1_ROOT
export S1_SONIC_ROOT=$S1_ROOT/third_party/GR00T-WholeBodyControl
export S1_PY=$PY
export S1_INPUTS=$S1_ROOT/inputs
export OMNI_KIT_ACCEPT_EULA=1
EOF2

# 4. inputs from the Ubuntu box
if [[ -f $S1_ROOT/inputs/cloud_inputs.tar ]]; then
  tar -C "$S1_ROOT/inputs" -xf "$S1_ROOT/inputs/cloud_inputs.tar"
  (cd "$S1_ROOT/inputs" && sha256sum -c SHA256SUMS)
fi

# 5. record versions
$PY - <<'EOF' | tee "$S1_ROOT/logs/versions.json"
import json, platform, sys, torch, importlib.metadata as m
print(json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
    isaacsim=m.version("isaacsim"), isaaclab=m.version("isaaclab"), numpy=m.version("numpy"), scipy=m.version("scipy"),
    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], host=platform.node(),
    env_source="fresh scripts/setup_isaacsim.sh (system1/cloud/setup_cloud_fresh.sh)"), indent=1))
EOF
$PY -m pip freeze > "$S1_ROOT/logs/pip_freeze.txt"
echo "setup done $(date); log: $LOG"
