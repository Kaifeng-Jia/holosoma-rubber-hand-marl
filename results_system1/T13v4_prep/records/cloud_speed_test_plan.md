# T13 第四版第 4 节 B：云上测速的线程数补充（Kevin 2026-11-24）

**本机不再做提速。**云上第一张卡的测速，在"2048 / 4096 个环境各跑 100 轮"之外，线程数再各跑一次（同一配置，同一种子）：

| run | 线程设置 |
|---|---|
| threads-default | 不加参数：PyTorch 默认线程数，不设 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` |
| threads-vcpu | `OMP_NUM_THREADS=<N> MKL_NUM_THREADS=<N>`，训练脚本加 `--torch-threads <N>`；N 为实例分配的 vCPU 数，用 `nproc` 读，记进报告 |

- **环境数：**两次都用第 4 节 B 选定的环境数；如果还没选定，用 2048。
- **轮数：**各 100 轮，s/轮取第 3 轮以后的中位数。
- **要记录的：**
  - 每个 run 的 `run_config.json` 里的 `torch_threads`（requested / actual / 三个环境变量）。本机的 `hssim_s1` 里本来就有 `OPENBLAS_NUM_THREADS=32`，不加参数时 PyTorch 实际用 16 个线程（本机 2026-11-24 开关测试）；云上照实记录，不另改；
  - GPU 和 CPU 型号、`nproc`；
  - GPU 利用率和进程 CPU 占用。
- **这一项不改变训练结果的定义：**线程数只影响 CPU 端的计算速度。
- 两个设置哪个快就用哪个，并写进 T13 的状态文件。
