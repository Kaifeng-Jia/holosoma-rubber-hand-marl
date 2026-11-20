# T10b 试点用的云端环境（2026-11-20，两张 RTX 4090）

本文件只写事实。

- **机器：** AutoDL `autodl-container-cgna4vuzjk-b9c168f4`，2 × RTX 4090 24 GB，驱动 595.71.05，Ubuntu 22.04，128 核，1 TB 内存。
- **第一次登录时的状态（14:57 New York 时间）：**
  - 数据盘 `/root/autodl-tmp` 只用了 20 KB，没有 `system1/`、`s1_env.sh`、`r2_deps/` 或旧的 `hssim_s1`；
  - 没有 `/root/autodl-fs`；
  - 15:47 在 `/` 下 5 层以内搜索，除本次安装生成的以外，没有别的 `s1_env.sh`、`r2_deps` 或 `hssim_s1`。
  - 所以没能沿用 T03/T04 那样直接复制旧环境的做法，而是从头安装。
- **安装脚本：** `system1/cloud/setup_cloud_fresh.sh`（本分支）。
  - 代码从公开的 GitHub 仓库克隆，分支 `system1-sonic`。
  - Isaac 环境用 holosoma 自带的 `scripts/setup_isaacsim.sh`，和本机 `hssim_s1` 是同一个安装脚本。
  - SONIC 代码用 T03 用过的那份 `git archive`，sha256 `cd5c3912…0294`。
  - SONIC 权重和配置从 HuggingFace 固定的 revision `6733128a` 下载，sha256 和脚本里固定的值一致：
    - `last.pt`：`e6bdab3f…8909`
    - `config.yaml`：`f0818779…29c7`
  - 权重只放在云端，没有提交，也没有转发。

## 安装经过（America/New_York）

| 次数 | 时间 | 结果 | 日志 |
|---|---|---|---|
| 1 | 15:02–15:05 | 失败。mamba 经 AutoDL 学术加速代理访问 conda-forge 时 TLS 校验失败：代理用自己的 CA（issuer `autodl`）重新签发证书，libmamba 不认 | `setup_20261121T040247.log.gz` |
| 2 | 15:07–15:31 | 跑完了，但 torch 版本不对。holosoma 自己的 `pip install -e .[unitree,booster]` 把 torch 从 2.7.0+cu128 升到了 2.15.0，triton 从 3.3.0 升到了 3.9.0；和本机 `hssim_s1` 比，共 149 个包版本不同 | `setup_20261121T040729.log.gz`，`pip_freeze_attempt2_before_pins.txt`，`versions_attempt2_before_pins.json` |
| 3 | 15:33–15:40 | 通过。第 4b 步按本机 `hssim_s1` 的版本 `--no-deps` 重装了这 149 个包（`system1/cloud/hssim_s1_laptop_pins.txt`），并断言 torch / triton / numpy 的版本 | `setup_20261121T043356.log.gz`，`versions.json`，`pip_freeze.txt` |

- 第 1 次失败后的修改：conda、pytorch、NVIDIA 这几个源改成不走代理直连，实测 4–8 MB/s；只有 GitHub 和 HuggingFace 走代理。
- 安装工作区 `$HOME/.holosoma_deps` 是指向数据盘 `/root/autodl-tmp/system1/deps` 的符号链接。原因是 `scripts/source_common.sh` 把 `WORKSPACE_DIR` 写死成了 `$HOME/.holosoma_deps`，否则环境会装到 30 GB 的系统盘上。
- **移开的文件：** 从本机上传 `cloud_inputs.tar` 太慢，传到 16 KB 就停了，而试点用不到它。这个半截文件移到了云端的 `/root/autodl-tmp/system1/_to_delete/cloud_inputs.tar.partial_upload_20261120`。

## 版本对比（`pip_versions_compare.csv`）

| 包 | 新云端（本次） | 本机 `hssim_s1` | 旧云端（T03/T04，2026-11-10） |
|---|---|---|---|
| Python | 3.11.18 | 3.11.15 | 3.11.17 |
| torch / torchvision / triton | 2.7.0+cu128 / 0.22.0+cu128 / 3.3.0 | 相同 | 相同 |
| CUDA（torch） | 12.8 | 12.8 | 12.8 |
| numpy / scipy | 1.26.0 / 1.15.3 | 相同 | 相同 |
| isaacsim / isaaclab | 5.1.0.0 / 0.47.2 | 相同 | 相同 |
| warp-lang | 1.10.0 | 相同 | 相同 |
| nvidia-cudnn-cu12 / nvidia-cublas-cu12 | 9.7.1.26 / 12.8.3.14 | 相同 | 相同 |
| einx / vector_quantize_pytorch / hydra-core / frozendict / torch-einops-utils | 0.4.3 / 1.32.2 / 1.3.3 / 2.5.1 / 0.2.6 | 相同 | 相同 |

- **新云端和本机：** 除可编辑安装的路径以外，pip 包版本逐个相同。新云端多出 10 个包：httpcore2、httpx2、lazy-importer、nvidia-cuda-cccl、5 个 opentelemetry 包、truststore。
- **新云端和旧云端：** 上表的数值相关包相同。另有 127 个包至少在一个环境里版本不同或缺失，主要是 jupyter、wandb、类型标注、cu13 的 nvidia 包等工具类包。旧云端是 `r2_deps/hssim` 的复制品，它和本机之间本来就有这些差别。逐包列在 `pip_versions_compare.csv`。

## 文件

- `versions.json`、`pip_freeze.txt`：最终环境。
- `*_attempt2_before_pins.*`：第 2 次安装后、固定版本前的环境。
- `pip_freeze_laptop_hssim_s1_20261120.txt`：用来对比的本机环境。
- `pip_versions_compare.csv`：三个环境之间所有不同的包。
- `setup_*.log.gz`：三次安装的完整日志。
