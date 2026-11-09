# T00 返包：代码地图（只读）

## receipt

| 项 | 值 |
|---|---|
| 任务书 | `docs/system1/T00_codebase_map_zh.md` |
| 任务书 sha256 | `5f235f10183c02b3296c7174b0f450aaeeaf77397bd57252e67ca151f674b444` |
| 开始 | 2026-11-09 15:32:14 EST |
| 结束 | 2026-11-09 15:52 EST，约 20 分钟 |
| hostname | `kevin-Legion-R9000P-ADR10` |
| GPU | RTX 5070 Laptop 8151 MiB；开始时空闲 6698 MiB，只有桌面进程占用 |
| 开始时 git HEAD | `c7e763f9c72a64fc1fd398e638d08404d311c7e6`，`system1-sonic`，工作区干净 |
| Python 环境 | 未运行训练或仿真。读取 npz/URDF 时用 `~/.holosoma_deps/miniconda3/envs/hssim/bin/python -I`，只读；其余用系统 `python3` 3.10.12 |

## 做了什么

- 交付 `docs/system1/codebase_map_zh.md`，按任务书 12 项写成。
- 18 个 Full checkpoint、6 个运行时参考、6 个物体 URDF、机器人 URDF、12 个冻结状态文件（±2/±8）、评测脚本、3 个单人先验，都重新计算了 sha256，并与 manifest 或 run_config 核对，全部一致。

## 与任务书的偏离

- **无。** 没有修改代码，没有运行训练或仿真。
- **只读操作的副作用：** 第一次在三个旧目录运行 `git status` 时没有设 `GIT_OPTIONAL_LOCKS=0`，可能刷新了 `/home/kevin/holosoma/.git/worktrees/*/index` 的 stat 缓存。git 对象和工作区文件内容没有变化。见地图第 12 节第 13 条。
- **网络访问：** 对两个 remote 各运行了一次 `git ls-remote`（只读）；在本目录运行了一次 `git fetch origin`。
