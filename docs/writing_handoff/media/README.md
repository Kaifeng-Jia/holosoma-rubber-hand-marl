# 展示素材清单（2026-09-25）

打开[离线视频相册](index.html)。不需要 Viser 服务、网络或 GPU。
每组为完整既有物理评测回放重新渲染，不是参考动画。原50Hz状态抽取为25fps视频，正常速度，无姿态插值或平滑；相机逐组记录。
每组6张1280×720 PNG、1920×720六宫格、1280×720 MP4。显示可视网格，不额外显示训练碰撞代理。

| 素材 | 分类 | 视频 | 六宫格 | 来源 |
|---|---|---|---|---|
| 桶交接：当前首选 | 主展示 / B-RZ0 | [MP4](bucket_best/replay.mp4) | [JPG](bucket_best/contact_sheet.jpg) | [manifest](bucket_best/manifest.json) |
| 双人扶椅子 | 主展示 / C-01 | [MP4](chair/replay.mp4) | [JPG](chair/contact_sheet.jpg) | [manifest](chair/manifest.json) |
| OMOMO 双人推动 | 精选回放 / Push-livePD | [MP4](omomo_push/replay.mp4) | [JPG](omomo_push/contact_sheet.jpg) | [manifest](omomo_push/manifest.json) |
| OMOMO 双人拉动 | 主展示 / Pull | [MP4](omomo_pull/replay.mp4) | [JPG](omomo_pull/contact_sheet.jpg) | [manifest](omomo_pull/manifest.json) |
| OMOMO 双人腿部推动 | 主展示 / Kick | [MP4](omomo_kick/replay.mp4) | [JPG](omomo_kick/contact_sheet.jpg) | [manifest](omomo_kick/manifest.json) |
| 桶交接 A：关系与高度 | 奖励/行为补充 / B-A | [MP4](bucket_A/replay.mp4) | [JPG](bucket_A/contact_sheet.jpg) | [manifest](bucket_A/manifest.json) |
| 桶交接 B：软接触调制 | 奖励/行为补充 / B-B | [MP4](bucket_B/replay.mp4) | [JPG](bucket_B/contact_sheet.jpg) | [manifest](bucket_B/manifest.json) |
| 5kg小桌：跟踪与操作方式 | 分析示例 / T-06 | [MP4](smalltable_5kg/replay.mp4) | [JPG](smalltable_5kg/contact_sheet.jpg) | [manifest](smalltable_5kg/manifest.json) |

## 选例说明

- 桶交接：当前首选：9.92s。双删除奖励，仍保留物体位置项竖直权重2；episode 000。完整评测5/5。
- 双人扶椅子：7.80s。基础11项；固定条件完整回放。两人参与程度可作行为分析。
- OMOMO 双人推动：6.16s。live-PD 13050 唯一完整复测选例，seed722 repeat3；全组1/9，不是旧平滑13050。
- OMOMO 双人拉动：6.30s。8050完整回放；基础11项；单人拉动先验派生成双人参考。
- OMOMO 双人腿部推动：5.94s。8000完整回放；源参考为踢动，实际接触方式保持原样展示。
- 桶交接 A：关系与高度：9.92s。完整A配方，选择episode 004；全组2/5完整。
- 桶交接 B：软接触调制：9.92s。A上增加软接触门控，选择episode 002；全组1/5完整。
- 5kg小桌：跟踪与操作方式：13.72s。完整桶A配方用于小桌，episode 001；全组5/5完整，但未持续整桌抬起。

时长为首末记录状态间隔；25fps编码尾部可能多显示不超过一帧。原回放未移动，manifest保留源NPZ/SHA、元数据、训练URDF SHA、帧号、fps和镜头参数。
导出脚本精确版本见[index.json](index.json)的renderer_script_snapshot，路径相对仓库根。
Demo3/4诊断数据在[证据包](../evidence/README.md)，不混作优质动作展示。腕部前后机器人同相机图尚未包含；现有依据见[方法报告](../../experiments/retarget_method_comparison_20260925.md)。
