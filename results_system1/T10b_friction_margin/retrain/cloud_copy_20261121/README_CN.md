# 云端试点数据拷回本机（2026-11-21）

本文件只写事实。

- **依据：** Kevin 2026-11-21："D 停了以后：先把 checkpoint、训练日志和验证记录全部拷回本机，核对 sha256，然后告诉我，我来关机。"
- **云端：**
  - 实例 `autodl-container-cgna4vuzjk-b9c168f4`，`ssh -p 19419 root@connect.nmb2.seetacloud.com`；
  - 代码目录在 `dc3ceae`，没有未提交改动。
- **试点结束：**
  - D Table 721：18:51 第 14000 轮验证是 31，停止规则触发，调度器按记下的 PID 停了训练。训练日志记到第 14699 轮；最后一个 checkpoint 是 `model_14000.pt`。
  - C-in Bin 721：训满 12000 轮，14:49 第 12000 轮验证后结束。
  - 18:53 调度器写完两个 `decision.json` 后退出。之后云端没有我们的进程在跑。
- **拷了什么：** 云端整个 `/root/autodl-tmp/system1/logs/`（716 MB，1777 个文件），用 rsync 拷到本机 `exports_system1/t10b_cloud_logs/logs/`，不进 git。内容包括：
  - 两个 run 的全部 checkpoint（D 15 个，C-in 13 个，每 1000 轮一个，含第 0 轮）；
  - `metrics.jsonl`、`run_config.json`、`randomization_samples.json`、`status.json`；
  - 训练 stdout，全部验证输出和记录（D 7 次，C-in 6 次）；
  - 调度器日志，`decision.json`；
  - 云端环境安装日志和版本记录，以及冒烟测试 `t10b_smoke_v2_1/`。
  - 第一遍 15:40 左右开始，C-in 部分 18:22 拷完并核对；D 停下后 18:53–18:54 补拷。
- **核对：**
  - 在云端对 `logs/` 下每个文件算 sha256，清单是本目录的 `cloud_logs_sha256.txt`；在本机用 `sha256sum -c` 逐个核对，1777 个全部一致。
  - 两边的文件清单也相同。
- **最终 checkpoint：**

| run | 文件 | sha256 |
|---|---|---|
| D Table 721 | `t10b_pilot/d_table_seed721_rand/train/model_14000.pt` | `2f343369af2865c448dcc0406a9e6c788ea67675a93312dca76f58bf8ded1806` |
| C-in Bin 721 | `t10b_pilot/cin_bucket_seed721_rand/train/model_12000.pt` | `be438b4a06356520d5b16646a6cda6ac22c299bf242cf1ec938b339cae735a9d` |

- **没拷的：** 云端 `system1/` 下的其他目录都不是试点产生的数据：
  - `deps`（25 GB，环境）；
  - `cache`（17 GB）；
  - `third_party`（456 MB）；
  - `inputs`（5.4 MB 的 SONIC 代码包，从本机传上去的）；
  - `_to_delete`（352 KB，一个没传完的上传包）；
  - `tmp`（空）。

| 文件 | 内容 |
|---|---|
| `cloud_logs_sha256.txt` | 云端 `logs/` 下每个文件的 sha256 |
| `*__decision.json` | 调度器的最终记录：停止规则、最终 checkpoint、验证曲线 |
| `t10b_pilot_runner.log` | 调度器日志 |
