# 历史记录原文摘录

来源：`/home/kevin/holosoma-rubber-hand-marl/MULTI_AGENT_EMERGENCE_ROADMAP.md`
原文件 SHA256：`3b7c6be286c4c7bf8cdeac8a8d771464035d00d711f9beddb105195c5f69b773`

以下保留原行号；这是当时记录，不是本次重新评测。

## 原文件第 1448–1502 行

```text
1448:   为 `0.43130`。正式 full-actor 必须从该 checkpoint resume，不得从容量测试 resume。
1449: 
1450: #### 2026-08-22：Pull 2,048-env 完整训练与首轮物理验收
1451: 
1452: - 正式目录：`logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048`；训练从
1453:   iteration 50 连续运行至 8050，共 8,000 行有限 metrics，`status.passed=true`，耗时约
1454:   `10.33 h`。每千轮保存一次，`model_01050.pt` 至 `model_08050.pt` 共 8 个 checkpoint；
1455: - 最终 `model_08050.pt` SHA256 为
1456:   `727630e9cec654d88bfb454d5eaabd70d7db629478598a2039140653a57cbf43`。最后 100 轮相较
1457:   最初 100 轮，平均 reward 从 `0.07109` 升至 `0.15086`，tracking failure 从
1458:   `419.28/iteration` 降至 `0.56/iteration`，value loss 从 `0.10996` 降至 `0.00423`；
1459: - seed 721、frame-0、actor mean-action、单环境连续 PhysX 回放中，`model_08050.pt` 完成
1460:   316/316 帧，无 reset 或 tracking failure。桌子实际/参考净位移为
1461:   `0.58082/0.58428 m`，沿轨进度比 `99.40%`，最终平面误差 `0.00799 m`，最终 yaw
1462:   误差 `0.262°`，平面 RMSE `0.01467 m`。录制文件
1463:   `logs/Plan5Pull/eval_full8000_seed721/model_08050_object_centric.npz` SHA256 为
1464:   `b6c6c056a609849b3c19cb80516bd687a7d5ddacd5f1d5ebe5d06ed82e6d3e0f`；
1465: - `model_07050.pt` 在相同协议下也完成 316/316 帧，但沿轨进度比 `89.88%`、最终平面
1466:   误差 `0.06377 m`、最终 yaw 误差 `4.05°`，因此数值主候选为 `08050`。保留 `07050`
1467:   仅用于比较动作自然度、抖动与角色分工；
1468: - 本结果只覆盖一个 seed 的确定性固定初始化回放，不能解释为统计稳健性证明；用户随后已在
1469:   ViSER 中确认 Pull 动作质量良好。Pull 固定多 seed 统计仍待完成，但它不再阻塞独立 Push
1470:   的干净训练。
1471: 
1472: #### 2026-08-27：Push live-state 完整训练、重复评测与暂停决定
1473: 
1474: - 实时状态修正：双机器人执行时，每台 actor 在每个控制步读取对应实体机器人的实时关节位置
1475:   和速度；这些量属于本机 proprioception，不是 centralized critic 的特权信息。修正后用户确认
1476:   `model_08050.pt` 的手臂高频振荡明显消失；网络仍为 shared actor
1477:   `158→512→256→128→29`、team critic `527→512→256→128→1`，没有把桌子状态加入 actor；
1478: - 干净训练链：critic-only 目录
1479:   `logs/Plan5Push/a1_livepd_20kg_critic50_seed721_env2048/`，完整 8,000 轮目录
1480:   `logs/Plan5Push/a1_livepd_20kg_full8000_from_critic50_seed721_env2048/`，续训目录
1481:   `logs/Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/`。续训从 iteration
1482:   `8050` 连续到 `15050`，共 7,000 条有限 metrics，`status.passed=true`；环境数 `2,048`、
1483:   每环境每轮 `24` steps、seed `721`、20 kg 宽桌、材料 `[0.5, 0.5, 0.0]`、原 11 项
1484:   WBT/object reward、adaptive actor LR 和 critic LR `1e-3` 均保持不变；
1485: - 保留的当前关键 checkpoint SHA256：`08050` 为
1486:   `58265a7d200695793b85b0821aea927fa90129f916eb30b438b1f621354e8266`；`13050` 为
1487:   `0ba8ae21239c7b6c0198f07086abbb0cfc3875dc908626df9e9fc0891692aede`；`14050` 为
1488:   `8fb2e29cd2af7b9cd05655b13ddfdd0b9d00de9d93028139890f11e8d99f3333`；`15050` 为
1489:   `b6be5f9b385122e8398ecc0494bc8fad9fdd13e72cd8766130f501f84f4b75e9`；
1490: - 固定评测协议：`13050/14050/15050` 分别使用 seeds `721/722/723`，每个 seed 独立启动
1491:   3 次，共 27 次；均从 frame 0、deterministic actor mean、相同 reference、初态和 PhysX
1492:   常量开始，最多 309 帧。这里的 seed 不随机化 reference、初态、摩擦或质量，因此该实验
1493:   衡量的是相同固定场景下跨 Isaac/PhysX 启动的重复稳定性，不是环境泛化；
1494: 
1495: | checkpoint | 完整成功 | 平均帧数 | 中位帧数 | 平均沿轨进度 | 首个终止原因 |
1496: |---|---:|---:|---:|---:|---|
1497: | `13050` | **1/9** | **182.00** | **172** | **76.31%** | 8 次 object-position，1 次完成 |
1498: | `14050` | 0/9 | 83.78 | 90 | 2.86% | 3 次 robot、6 次 object-position |
1499: | `15050` | 0/9 | 87.78 | 91 | 2.94% | 3 次 robot、6 次 object-position |
1500: 
1501: - 解释：PPO 优化的是 2,048 个随机 phase 环境中的期望回报，不保证固定 frame-0 物理回放随
1502:   iteration 单调改善。`14050/15050` 的训练平均 reward 与 object reward 没有显示实现错误，
```

## 原文件第 1650–1672 行

```text
1650:   不依赖本地 `logs/` 才能加载；
1651: - 训练保持 Plan 5 契约：共享 `158→512→256→128→29` Actor、`527→512→256→128→1`
1652:   centralized critic、原 Push/Pull reference-guided reward 与 joint tracking termination。两台实体
1653:   robot 使用 `main_mesh_collision_rubberhand.urdf`；桌子质量 `20 kg`，静/动摩擦
1654:   `0.5/0.5`，restitution `0`；物理 `200 Hz`、控制 `50 Hz`；
1655: - 正式 run 为 `logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/`：seed `721`、
1656:   `2,048 env × 24 steps × 8,000 iterations`，每 `2,000` iterations 保存一次，完整保存
1657:   `model_00000/02000/04000/06000/08000.pt`，训练账本 `status.passed=true`；
1658: - 最终 `model_08000.pt` SHA256 为
1659:   `9c6a31809ba69eaf3fd510391ce5b5f56069d4378c514ce09c3b4f28e407b3a3`。其单环境
1660:   deterministic actor-only rollout 完整执行 motion step `0..297`，无提前 done、timeout 或
1661:   joint-bad-tracking；rollout SHA256 为
1662:   `0531239d1cafe0490dae00c56d83bbc8899b71be59be03cb5739d8dfa7c4cb46`，并已通过用户视觉验收；
1663: - Git 复现边界：checkpoint、训练 metrics 和评估 rollout 仍位于被忽略的 `logs/`，不上传
1664:   GitHub；训练入口要求用户另行恢复 Kick `158-D` source checkpoint，固定 SHA256 为
1665:   `1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491`。本节的“成功”表示本轮
1666:   完整训练与单条确定性物理回放已通过，不替代后续多 seed 统计评估。
```
