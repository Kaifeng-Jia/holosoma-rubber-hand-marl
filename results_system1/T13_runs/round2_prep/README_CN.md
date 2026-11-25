# T13 第二轮准备：C-out-Table、C-out-Push 的代码和本机冒烟

- **依据：**
  - 任务书 `docs/system1/T13_parallel_runs_zh.md` 第 1、8 节，sha256 `468a04ec…`；
  - Kevin 2026-11-25 消息"T13 第二轮准备"。
- **时间（纽约，2026-11-25）：**代码 15:20–15:37；冒烟 15:37:50–15:47:27；D 对照测速 15:48–15:50。
- **机器：**kevin-Legion-R9000P-ADR10，RTX 5070 Laptop 8 GB，环境 `hssim_s1`。同时有 T14 第 9e 步在 CPU 上跑（6 个进程，负载约 8 / 32 核）。
- **代码：**commit `102b406`。冒烟是在 `2d29c4e` 加未提交改动上跑的；这些改动和 `102b406` 的提交逐字相同（`code_round2.diff`）。

## 结论（只列事实）

- 两个 run 的本机冒烟都通过（`smoke_c_out_table.json`、`smoke_c_out_push.json`），检查项见第 3 节。
- 第 0 轮（输出层为零）的 C-out 和方案 A 在同样 3 个验证状态上逐数组相同，最大差 0.0（`iter0_*_vs_A.json`）。
- 第一轮三个 run 的计划不受影响：`cloud_runs.py --dry-run` 生成的命令，和第一轮实际启动时的命令逐项相同。

## 1. 配置（`cloud_plan_c_out_*.json`，4096 个环境）

| run | 训练命令 | 验证 |
|---|---|---|
| c_out_table | d_table 的命令 + `--sonic-residual --sonic-tf32 --sonic-residual-kind out --c-train-diag --t13v4-cout` | `--controller sonic_residual --t13v4-cout`，其余同 d_table |
| c_out_push | d_push 的命令（同样 25 轮 critic 预热，再 `--resume`）+ 同样 5 项 | 同上，其余同 d_push |

- **和第一轮相同的部分：**观测 v1 加噪声 v1、T10b 随机化、critic 修正、接触三项、4096 个环境、停止规则（轮数减半）、验证状态、种子 721。
- **C-out 的部分：**`--sonic-residual … out`、`--sonic-tf32`、`--c-train-diag` 照 T03 C-out 的命令。
  - SONIC 收到不改动的参考；PD 目标 = SONIC 输出 + D 的动作缩放 × a。
  - 输出层初始化为零，第 0 轮等于方案 A。
  - hip_pitch 的 PD 用 SONIC 的值（`with_sonic_pd`，同 T03）。
- **共享奖励，不用 PA。**
- **新开关 `--t13v4-cout`，默认关：**两个训练脚本和评测脚本里原来写着"C-out path: separate switch"，就是这个开关。
  - 关时行为不变：原来的保护照旧，T13 v4 的开关不能和 `--sonic-residual` 一起用。
  - 开时只允许 C-out（`out`）、橡胶手、不带 PA。checkpoint 里会记上 `system1_t13v4_cout`。

## 2. 和任务书的差别、我做的选择（请 Kevin 看）

- **C-out-Push 的初始化：**T03 没有 C-out Push，只有 C-out Pull。Pull 和 Push 用同一个训练脚本，所以照 T03 C-out Pull 的做法：
  - actor 不加载单机器人先验，随机初始化（T03 Pull 用的是 `--from-scratch`）；
  - actor 的观测归一化用先验的（冻结，按观测 v1 换算：同伴位置沿用先验，物体 3 维均值 0、方差 1）；
  - 输出层为零；
  - 先做 critic 预热，再联合训练，学习率 actor 1e-5、critic 1e-3。
  - 第一轮 d_push 是加载先验（4 层都等于先验）。C-out-Push 的冒烟 checkpoint 里，隐层都和先验不同，归一化器和换算后的先验逐项相同。
  - 没有直接用 `--from-scratch`：它会在 checkpoint 里写 V3 Pull 消融的合同，评测脚本对 Push 会拒收。所以随机初始化放在 `--t13v4-cout` 下。
  - **如果 Kevin 要 C-out-Push 也从先验加载，**改成加载只是一处开关，开卡前告诉我。
- **4096 个环境的显存和速度本机测不了**（8 GB）。开卡后先在每台云机上跑一次 `cloud_smoke.py`（4096 个环境，第一轮也是这样做的），通过后再起训练。
- **SONIC 权重要在云机上：**C-out 训练要用 SONIC 的 checkpoint。第一轮的云机只跑了 D，开机后我先核对权重文件和 sha256；如果没有，先报 Kevin，不自己拷。

## 3. 本机冒烟（512 个环境；轮数缩短，其余是云上的原命令）

| 项 | C-out-Table | C-out-Push |
|---|---|---|
| critic 预热 | — | 3 轮，rc 0，36 s，GPU 峰值 4612 MiB；预热后输出层仍为零 |
| 训练 | 20 轮，rc 0，98 s，3.63 s/轮，GPU 峰值 4758 MiB | 20 轮（`--resume`），rc 0，97 s，3.55 s/轮，GPU 峰值 4693 MiB |
| 指标 | 全是有限值；接触三项的 14 个指标都在 | 同左 |
| PhysX 溢出类警告 | 0 | 0 |
| checkpoint | C-out 标记在；actor 输入 159、critic 585；T14 参考 sha256 和 d_table 相同 | C-out 标记在；actor 输入 159、critic 585 |
| 第 0 轮 vs 方案 A（3 个验证状态） | 逐数组最大差 0.0，回合字段无差别 | 用预热后的 checkpoint：同左 |
| 云上验证命令（3 个验证状态） | rc 0，76 s，summary 完整 | rc 0，37 s，summary 完整 |

- **训练配置读回（`run_config.json`）：**
  - T10b 随机化在（配方和第一轮相同）；
  - 观测 v1、训练噪声开、critic 585；
  - 接触标签和第一轮相同；
  - `system1_scheme_c` 记了 C-out 的 PD、动作缩放、残差定义。
- **验证的 run_manifest：**`controller=sonic_residual`，`system1_scheme_c_kind=out`，观测 v1 噪声关，`cout=true`。

**本机测速（`timing_512.csv`，512 个环境，同一台机器、同样的负载）：**

| 配置 | s/轮 |
|---|---|
| D-Table（d_table 的命令） | 3.75 |
| C-out-Table | 3.63 |
| C-out-Push | 3.55 |

- 512 个环境时两者几乎一样。4096 个环境时 SONIC 推理的批量是本机的 8 倍，这个比例不能外推，以云上冒烟为准。
- **参考：**第一轮在云上（4096 个环境）d_table 中位数 7.03 s/轮，d_push 5.89 s/轮，训练用了 11–14 小时。

## 4. 开卡后的步骤

1. 每台云机：`git pull` 到 `system1-sonic` 最新；核对 SONIC 权重；跑 `cloud_smoke.py --run c_out_table|c_out_push --num-envs 4096`。
2. 冒烟通过：用 `cloud_runs.py --run … --num-envs 4096` 起训练（nohup，记 PID），和第一轮相同。
3. 收尾照第一轮：定时任务看 `decision.json`，结束后自动拷回、核对 sha256、在本机评测。
   - Isaac 评测命令取自 run 自己最后一次验证的命令，已带上 `sonic_residual`。
   - MuJoCo 端的 C-out 路径（T10c）还要接上观测 v1，开训后用冒烟 checkpoint 做好并测试，不挡开卡。

## 5. 文件

- `smoke_c_out_table.json`、`smoke_c_out_push.json`：冒烟记录（命令、返回码、用时、显存、checkpoint 检查、验证）。
- `iter0_c_out_table_vs_A.json`、`iter0_c_out_push_vs_A.json`：第 0 轮等价检查。
- `cloud_plan_c_out_table.json`、`cloud_plan_c_out_push.json`：云上命令（dry-run；解释器路径是本机的，云上换成 `S1_PY`）。
- `code_round2.diff`、`timing_512.csv`、`timing_d_table_512.cmd`、`SHA256SUMS`。
