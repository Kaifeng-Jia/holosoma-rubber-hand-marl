# 验证 a 基准的预登记（2026-11-09 16:12:24 EST，写于任何 SONIC 运行之前）

- 官方 Mode 2 命令原样运行一次：checkpoint sonic_release/last.pt，sample_data/robot_filtered + smpl_filtered，num_envs=128。记录官方 metrics_eval.json。
- 官方 release config 的 encoder_sample_probs 为 g1/teleop/smpl 各 1.0，每个 env 随机选编码器，所以官方 MPJPE 混合了三种编码器。
- 接入用 G1 关节参考接口，因此另跑一次只改 ++manager_env.commands.motion.encoder_sample_probs={g1:1.0,teleop:0.0,smpl:0.0}，其余与官方命令相同。
- **验证 a 的基准值 = G1-only 原生运行的 mpjpe_g（官方 im_eval 指标）。** 判为通过：本仓库接入的 MPJPE（同一段示例动作，同一 14 个 body、同一定义）≤ 1.2 × 基准值，并且不摔倒。
- 官方混合编码器的数值一并报告，不用于判定。
