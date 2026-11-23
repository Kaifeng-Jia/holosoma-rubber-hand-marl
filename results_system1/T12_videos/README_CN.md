# T12-v：D Table 回放视频

本文件只写事实。

## 任务

- **来源：** Kevin 2026-11-23 直接贴在对话里的任务书，`docs/system1/` 下没有对应文件。
- **要求：** 只渲染，不训练，不改评测协议。
- **对象：** D，Table，种子 721。
  - 原始策略：论文 D 的 `model_12000.pt`，sha256 `f163d11a…`，即任务书写的"d_paper_final it12000"。
  - 随机化 pilot：`model_14000.pt`，sha256 `2f343369…`。
- **条件：** Isaac（`d_final_5070` 协议）和 MuJoCo（T10 设置），用 ±2 冻结测试集，状态文件 sha256 `389e8da5…dac1`。
- **选例：** 任务书规定为"±2 测试集第 0、1、2 号状态"。
  - 我取的是 `perturbed_000`、`perturbed_001`、`perturbed_002`，也就是 50 个 ±2 扰动状态里的前三个。
  - 状态文件里排第 0 的是 nominal，它不在 ±2 扰动集里，所以没算进去。
  - 这个读法我没有向 Kevin 确认过。

## 来源：全部用最终评测存下的轨迹，没有重跑

渲染需要两台机器人的根位姿和关节角，以及物体位姿。12 段的已存轨迹里这些都有，所以没有任何一段重新推理或重跑。

| 策略 | 仿真器 | 评测目录 |
|---|---|---|
| 原始 | Isaac | `logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2` |
| 原始 | MuJoCo | `logs_system1/t10/mj_eval/d_table_seed721_pm2` |
| pilot | Isaac | `logs_system1/t10b/finals/d_table_seed721_rand_pm2` |
| pilot | MuJoCo | `logs_system1/t10b/mj_finals/d_table_seed721_rand_pm2` |

- **原始策略的 ±2 评测为什么不在 `d_paper_final` 下：**
  - `d_paper_final/` 只有 ±8 那组。原始策略的 ±2 评测在 `d_curve/.../it12000_pm2`。
  - 两者 checkpoint 相同（`f163d11a…`），也就是 T03 `c2_main` 和 T10b 用的那份。
- **已核对：**
  - 4 个来源的 checkpoint 和状态文件 sha256 都和上表一致；
  - 每段的成功与否、终止步，和 T10b finals 的 `episodes.csv` 一致；
  - 同一状态的 4 段，第 0 行的位姿相同（MuJoCo 和 Isaac 只差 2e-8）；
  - 4 段存下的参考物体轨迹逐帧相同；
  - MuJoCo 轨迹的关节顺序和 `robot.py` 里 G1 的 `dof_names` 相同。
- 每段的轨迹路径和 sha256 记在 `manifest.json` 里。

## 每段的结果

| 状态 | 原始 · Isaac | 原始 · MuJoCo | pilot · Isaac | pilot · MuJoCo |
|---|---|---|---|---|
| perturbed_000 | 成功，687 步 | 失败，第 347 步，物体位置 | 失败，第 363 步，物体位置 | 失败，第 429 步，物体位置 |
| perturbed_001 | 成功，687 步 | 失败，第 380 步，物体位置 | 失败，第 490 步，物体位置 | 成功，687 步 |
| perturbed_002 | 失败，第 335 步，物体位置 | 失败，第 378 步，物体位置 | 成功，687 步 | 失败，第 467 步，物体位置 |

- "物体位置"指物体位置偏差 > 0.25 m，判定规则同 T10b。
- 这三个状态是事先定的，不代表 50 个状态的整体情况。整体完成数见 T10b `retrain/finals/`：
  - 原始：Isaac 35、MuJoCo 0；
  - pilot：Isaac 31、MuJoCo 34。

## 视频

只放在本机 `exports_system1/T12_videos/`，不进 git：

| 状态 | 全景 | 近景 |
|---|---|---|
| perturbed_000 | `d_table_seed721_pm2_perturbed_000_full.mp4` | `d_table_seed721_pm2_perturbed_000_close.mp4` |
| perturbed_001 | `d_table_seed721_pm2_perturbed_001_full.mp4` | `d_table_seed721_pm2_perturbed_001_close.mp4` |
| perturbed_002 | `d_table_seed721_pm2_perturbed_002_full.mp4` | `d_table_seed721_pm2_perturbed_002_close.mp4` |

- **布局：** 2×2，行是原始 / pilot，列是 Isaac / MuJoCo。
- **每格左上角标注：**
  - 策略和仿真器；
  - 成功或失败，失败时写终止步和原因；
  - 当前时间和步数。
  - 回合结束后，这一格停在最后一个执行步，并注明"回合已结束，停在最后一帧"。
- **时间：** 50 Hz 存下的状态按 25 fps 播放，正常速度：视频第 j 帧显示第 2j 行，不插值、不平滑、不改姿态。
  - 4 格共用一个时钟，视频长度按 4 段中最长的一段，即 687 步。每个视频 344 帧，13.76 s。
  - 成功回合最后那一行多出来的终止状态不画，和 U5 的做法相同。
- **分辨率：** 每格 1280×720，拼起来是 2560×1440，H.264，crf 20。
- **画面风格：** 沿用分支里 U5 / 论文 Fig. 4 的渲染管线 `system1/render/u5/`：Unitree 原色机器人、浅蓝桌子、棋盘地面。两台机器人颜色相同。
- **相机：** 同一状态的 4 格用同一组相机参数。
  - **全景：** 用 U5 的 `camera_for` 规则，把 4 段每隔 10 行和最后一行的全部网格包围盒都框进去，留 7% 边，方位角 135°，俯角 −22°，固定不动。

    | 状态 | lookat | 距离 |
    |---|---|---|
    | perturbed_000 | (0.046, −0.370, 0.661) | 2.759 m |
    | perturbed_001 | (0.002, −0.387, 0.661) | 2.934 m |
    | perturbed_002 | (0.027, −0.352, 0.661) | 3.050 m |

    论文 F3 的桌子相机是 lookat (0.079, −0.371, 0.662)、距离 2.713 m。

  - **近景：** 对准参考物体，在当前参考帧的 x、y 处，高度 0.55 m；距离 2.0 m，方位角 180°，俯角 −30°。
    - 这组参数是在 perturbed_000 第 150、300、320 行的几张静图上比较取景后定的，要求同时看到两台机器人、两侧的手和桌面。
    - 近景跟着参考走，所以某一格回合结束后，画面停住了，镜头还在移动，那一格的场景会慢慢偏向一边。
- **CPU 渲染：** 用 Mesa EGL 的 llvmpipe 软件渲染，`CUDA_VISIBLE_DEVICES` 为空，GL vendor 记录为 `Mesa / llvmpipe`。
  - 6 个视频并行，00:26 开始，00:38 结束，每个 11–12 分钟。
  - 没有占用 GPU。
- **代码：**
  - 渲染：`system1/t12/render_t12.py`；
  - 并行启动：`system1/t12/run_t12.sh`；
  - 冒烟测试：`--max-frames 3`，已在本机跑过。

## 原始策略在 MuJoCo 里失败时，桌子从什么时候开始滑

**方法：** `system1/t12/slip_onset.py`，只读存下的状态。拿同一策略、同一状态的 Isaac 回合做对照。规则写在脚本开头，下面是摘要：

1. **手是否接触桌子：** 由存下的根位姿和关节角做正向运动学，算橡胶手碰撞网格到桌子碰撞网格的有符号距离。距离 ≥ −1.5 cm，即在 1.5 cm 以内或已穿入，算接触。
2. **手在桌面上滑：** 在桌子坐标系里，看手连杆原点的速度。接触着并且速度 > 0.1 m/s，算这只手在滑。从第 190 行（参考开始抬起）起，第一次连续 5 行有手在滑，记为"手开始滑"。
3. **桌子在地面上跑到参考前面：** 物体水平速度比参考水平速度快 0.2 m/s 以上，从第 190 行起第一次连续 5 行如此，记为"桌子开始跑到参考前面"。
4. **其他：** 物体偏离参考第一次超过 0.1 m 的行；MuJoCo 和 Isaac 物体位置第一次相差 > 5 cm、> 10 cm 的行；桌子的最大升高。

阶段名取参考在那一行所处的阶段，判定规则同 T10b。时间 = 行号 / 50。

| 状态 | 仿真器 | 结果 | 手开始滑 | 桌子开始跑到参考前面 | 物体偏离参考 > 0.1 m | MuJoCo 与 Isaac 相差 > 5 cm / > 10 cm | 桌子最大升高（参考） | 桌子最大水平速度（参考） |
|---|---|---|---|---|---|---|---|---|
| 000 | MuJoCo | 第 347 步失败 | 4.86 s，举起中 | 4.74 s，举起中 | 5.72 s，举起中 | 4.90 s / 5.44 s，都在举起中 | 0.0 cm（6.2） | 0.95 m/s（0.76） |
| 000 | Isaac | 成功 | 3.90 s，抬起 | 4.78 s，举起中 | 5.32 s，举起中 | 同上 | 0.5 cm（6.4） | 0.72 m/s（0.76） |
| 001 | MuJoCo | 第 380 步失败 | 4.28 s，贴地推行 | 4.78 s，举起中 | 6.00 s，举起中 | 5.90 s / 6.18 s，都在举起中 | 0.4 cm（6.4） | 0.80 m/s（0.76） |
| 001 | Isaac | 成功 | 4.10 s，抬起 | 4.78 s，举起中 | 6.40 s，举起中 | 同上 | 0.7 cm（6.4） | 0.77 m/s（0.76） |
| 002 | MuJoCo | 第 378 步失败 | 3.94 s，抬起 | 4.80 s，举起中 | 5.90 s，举起中 | 5.82 s / 5.94 s，都在举起中 | 0.5 cm（6.2） | 0.85 m/s（0.76） |
| 002 | Isaac | 第 335 步失败 | 4.14 s，抬起 | 4.76 s，举起中 | 6.10 s，举起中 | 同上 | 2.8 cm（6.2） | 0.65 m/s（0.76） |

- **MuJoCo 的终止时刻：** 6.94 s（第 347 步）、7.60 s（第 380 步）、7.56 s（第 378 步），都在参考的举起中段（第 231–415 帧）。
- **接触比例（第 190 行起）：**
  - 机器人 0 的两只手接触比例在 0–8% 之间。
  - 机器人 1 的两只手：MuJoCo 86–100%，Isaac 66–100%。
- **机器人 1 的手在桌子坐标系里的位移（第 190 行到最后一行）：**
  - MuJoCo：11.5 / 5.3、13.6 / 11.4、16.8 / 13.8 cm；
  - Isaac：8.1 / 12.6、6.9 / 11.7、17.8 / 13.5 cm。
- 逐行数据在 `slip_onset_per_step.csv.gz`：四只手到桌子的距离和滑动速度、物体和参考的水平速度、物体误差、物体高度。

## 和任务书的偏离

- **选例：** 任务书写的是"±2 测试集第 0、1、2 号状态"，我取的是 perturbed_000–002，见"任务"一节。
- **原始策略 ±2 的来源：** 来自 `d_curve/.../it12000_pm2`，checkpoint 和 `d_paper_final` 相同，见"来源"一节。
- **视频尺寸：** 2×2 视频是 2560×1440，每格 1280×720。
- **两个相机各出一个视频：** 每个状态有全景、近景两个视频，共 6 个。
- **滑动的判定规则是我定的：** 任务书没有给，规则见上一节。

## Google Drive 副本（任务 D0，2026-11-23）

- **任务书：** Kevin 直接贴在对话里，`docs/system1/` 下没有对应文件。Kevin 已批准。
- **rclone：** v1.76.0，`~/.local/bin/rclone`，未用 sudo。
  - 来源是 downloads.rclone.org 的官方发布包 `rclone-v1.76.0-linux-amd64.zip`，sha256 `b00f00b3266c3764413afe599d3ec316f6f43b3e0ea8a0bc1d186d2133e46df7`。
  - 这个值和发布目录里的 `SHA256SUMS` 一致；`SHA256SUMS` 的 GPG 签名有效，签名密钥指纹 `FBF737ECE9F8AB18604BD2AC93935E02FF3B54FA`。
  - 装好的二进制 sha256 `62e4d3e5432f2c685f1834b8bb342058ff0611c99d048aeb44bb8d8fd611527e`。
- **远端 `gdrive`：** 类型 Google Drive，`client_id` 留空，即 rclone 共用的 client_id；scope `drive.file`；没有配 shared drive。
  - 授权走浏览器自动流程，Kevin 本人登录，02:05:56 完成。
  - rclone 提示它的共用 client_id 正在退役，"2026 年内会停用"。
- **Drive 文件夹：** `DuetMimic_transfer/T12_videos/`，在"我的云端硬盘"下。
  - 文件夹、子文件夹和 7 个文件都只有所有者一个权限，没有共享，也没有链接分享。
- **上传：** 6 个 mp4 来自 `exports_system1/T12_videos/`，`manifest.json` 来自本目录。
  - D0 任务书写的是 `exports_system1/T12_videos/` 下的 `manifest.json`，但那里没有这个文件。
  - 用 `rclone copy --checksum --files-from`，02:06:46 开始，02:07:31 结束，共 51.9 MiB 加 39 KB。
- **核对：**
  - `rclone check --checksum`（md5）：两组都是 0 处不同，分别 6 个和 1 个文件匹配。
  - Drive 记录的 sha256 和字节数，和本地 7 个文件逐个相同。
  - 6 个 mp4 的 sha256 和 `manifest.json` 里记的视频 sha256 相同。
- **重跑核对：** 之后用 `system1/transfer/drive_upload.py` 又跑了一遍。
  - copy 没有传任何东西（"There was nothing to transfer"）。
  - check 和 sha256 比对都通过，`drive_manifest.csv` 由这一次生成。
- **日志：** 只在本机，不进 git。
  - 第一次上传：`logs_system1/d0_drive/T12_videos/`。
  - 脚本那次：`logs_system1/drive/T12_videos/20261123_020930/record.json`，里面有每条命令和返回码。

## 文件

| 文件 | 内容 |
|---|---|
| `drive_manifest.csv` | 传到 Google Drive 的 7 个文件：文件名、字节数、sha256、Drive 路径 |
| `manifest.json` | 6 个视频各一项：视频路径和 sha256、每格的来源轨迹路径和 sha256、评测 summary 和 sha256、checkpoint、状态号、成功与否、终止步和原因、相机参数、帧号规则、GL vendor、代码 sha256 |
| `slip_onset.csv` | 上一节表格的数，每个状态 × 仿真器一行 |
| `slip_onset_per_step.csv.gz` | 逐行数据 |
| `slip_onset_provenance.json` | 脚本、URDF、`references.json` 的 sha256 和规则参数 |
| `SHA256SUMS` | 本目录文件的校验和 |
