#!/usr/bin/env python3
"""Generate an offline gallery and code/text/media writing handoff, not a training package."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs/writing_handoff/media"
DEMOS = [
    ("bucket_best", "桶交接：当前首选", "主展示 / B-RZ0", "双删除奖励，仍保留物体位置项竖直权重2；episode 000。完整评测5/5。"),
    ("chair", "双人扶椅子", "主展示 / C-01", "基础11项；固定条件完整回放。两人参与程度可作行为分析。"),
    ("omomo_push", "OMOMO 双人推动", "精选回放 / Push-livePD", "live-PD 13050 唯一完整复测选例，seed722 repeat3；全组1/9，不是旧平滑13050。"),
    ("omomo_pull", "OMOMO 双人拉动", "主展示 / Pull", "8050完整回放；基础11项；单人拉动先验派生成双人参考。"),
    ("omomo_kick", "OMOMO 双人腿部推动", "主展示 / Kick", "8000完整回放；源参考为踢动，实际接触方式保持原样展示。"),
    ("bucket_A", "桶交接 A：关系与高度", "奖励/行为补充 / B-A", "完整A配方，选择episode 004；全组2/5完整。"),
    ("bucket_B", "桶交接 B：软接触调制", "奖励/行为补充 / B-B", "A上增加软接触门控，选择episode 002；全组1/5完整。"),
    ("smalltable_5kg", "5kg小桌：跟踪与操作方式", "分析示例 / T-06", "完整桶A配方用于小桌，episode 001；全组5/5完整，但未持续整桌抬起。"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gallery():
    scripts = [ROOT / "scripts/export_recorded_demo_media.py", ROOT / "docs/writing_handoff/provenance/export_recorded_demo_media_v1.py"]
    versions = {sha(p.read_bytes()): str(p.relative_to(ROOT)) for p in scripts}
    records, articles = [], []
    md = ["# 展示素材清单（2026-09-25）", "",
          "打开[离线视频相册](index.html)。不需要 Viser 服务、网络或 GPU。",
          "每组为完整既有物理评测回放重新渲染，不是参考动画。原50Hz状态抽取为25fps视频，正常速度，无姿态插值或平滑；相机逐组记录。",
          "每组6张1280×720 PNG、1920×720六宫格、1280×720 MP4。显示可视网格，不额外显示训练碰撞代理。", "",
          "| 素材 | 分类 | 视频 | 六宫格 | 来源 |", "|---|---|---|---|---|"]
    for name, title, category, note in DEMOS:
        folder = MEDIA / name
        m = json.loads((folder / "manifest.json").read_text())
        if m["script_sha256"] not in versions:
            raise ValueError(f"Renderer snapshot missing: {name}")
        for rel, digest in m["outputs_sha256"].items():
            if sha((folder / rel).read_bytes()) != digest:
                raise ValueError(f"SHA mismatch: {name}/{rel}")
        frames = m["still_frame_indices"]
        if len(frames) != 6:
            raise ValueError("Six stills required")
        duration = (m["recording_frames"] - 1) / m["recording_fps"]
        records.append({"id": name, "title_cn": title, "category_cn": category, "note_cn": note,
                        "replay": f"{name}/replay.mp4", "contact_sheet": f"{name}/contact_sheet.jpg",
                        "manifest": f"{name}/manifest.json", "recorded_duration_s": duration,
                        "renderer_script_snapshot": versions[m["script_sha256"]]})
        md.append(f"| {title} | {category} | [MP4]({name}/replay.mp4) | [JPG]({name}/contact_sheet.jpg) | [manifest]({name}/manifest.json) |")
        stills = "".join(f'<a href="{name}/frame_{i:04d}.png"><img loading="lazy" src="{name}/frame_{i:04d}.png" alt="frame {i}"></a>' for i in frames)
        articles.append(f'<article id="{name}"><h2>{html.escape(title)}</h2><p class="tag">{html.escape(category)} · 原状态首末间隔 {duration:.2f}s</p><p>{html.escape(note)}</p><video controls preload="none" playsinline poster="{name}/frame_{frames[2]:04d}.png"><source src="{name}/replay.mp4" type="video/mp4"></video><p><a href="{name}/replay.mp4">打开 MP4</a> · <a href="{name}/contact_sheet.jpg">六宫格</a> · <a href="{name}/manifest.json">来源与哈希</a></p><div class="stills">{stills}</div></article>')
    md += ["", "## 选例说明", ""]
    md += [f"- {r['title_cn']}：{r['recorded_duration_s']:.2f}s。{r['note_cn']}" for r in records]
    md += ["", "时长为首末记录状态间隔；25fps编码尾部可能多显示不超过一帧。原回放未移动，manifest保留源NPZ/SHA、元数据、训练URDF SHA、帧号、fps和镜头参数。",
           "导出脚本精确版本见[index.json](index.json)的renderer_script_snapshot，路径相对仓库根。",
           "Demo3/4诊断数据在[证据包](../evidence/README.md)，不混作优质动作展示。腕部前后机器人同相机图尚未包含；现有依据见[方法报告](../../experiments/retarget_method_comparison_20260925.md)。", ""]
    (MEDIA / "README.md").write_text("\n".join(md))
    (MEDIA / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    head = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>协作机器人 · 实验展示素材</title>
<style>body{margin:0;background:#eef2f6;color:#182b40;font:16px/1.65 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:28px}article{background:white;padding:24px;margin:28px 0;border-radius:14px}h2{margin:0}video{width:100%;max-height:720px;background:#e8edf2;border-radius:8px}a{color:#195b9a}nav a{display:inline-block;margin-right:18px}.tag{color:#607388}.stills{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stills img{width:100%}aside{padding:18px;background:#e1eaf3;border-radius:8px}footer{padding:20px;color:#627287}@media(max-width:600px){main{padding:12px}.stills{grid-template-columns:repeat(2,1fr)}}</style>
<main><h1>从人类示范到双机器人协作</h1><p>2026-09-25 · 已训练策略的物理评测回放 · Mac 离线展示</p><aside>保存状态重新渲染，不是新物理评测或参考动画。无姿态修补或动作平滑；镜头可跟随机器人中心。选例与整组结果分开记录。点击切片看原分辨率PNG。</aside><nav>'''
    nav = "".join(f'<a href="#{r[0]}">{html.escape(r[1])}</a>' for r in DEMOS)
    tail = '<footer>方法、奖励与完整评测见上级README和evidence。所有素材从本地加载，不请求在线服务。</footer></main></html>'
    (MEDIA / "index.html").write_text(head + nav + "</nav>" + "".join(articles) + tail)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "paper_handoff_exports")
    p.add_argument("--name", default="CORE4D_MAC_WRITING_HANDOFF_20260925")
    p.add_argument("--gallery-only", action="store_true")
    args = p.parse_args()
    gallery()
    if args.gallery_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / (args.name + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    suffixes = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".sh", ".cfg", ".txt", ".xml", ".urdf", ".ini", ".rst"}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    selected = {x for x in tracked if x and (ROOT / x).is_file() and not (ROOT / x).is_symlink()
                and (Path(x).suffix in suffixes or Path(x).name in {"LICENSE", "NOTICE", ".gitignore"})}
    selected.update(str(f.relative_to(ROOT)) for f in (ROOT / "docs/writing_handoff").rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    selected.update({"scripts/export_recorded_demo_media.py", "scripts/package_writing_handoff.py",
                     "docs/experiments/retarget_method_comparison_20260925.md", "docs/experiments/retarget_method_comparison_20260925.json",
                     "docs/experiments/assets/chair_wrist_orientation_20260925.svg"})
    contents = {}
    for rel in sorted(selected):
        if any(x in Path(rel).parts for x in ("logs", "logs_eval", ".git", "__pycache__")) or Path(rel).suffix in {".pt", ".npz", ".npy", ".onnx", ".STL", ".obj", ".dae"}:
            raise ValueError(f"Excluded material reached package: {rel}")
        contents[rel] = (ROOT / rel).read_bytes()
    contents["WORKTREE_DIFF.patch"] = subprocess.check_output(["git", "diff", "--", "story.md", "DEMO_AND_EXPERIMENT_INVENTORY.md", "MULTI_AGENT_EMERGENCE_ROADMAP.md"], cwd=ROOT)
    manifest = {"purpose": "source-reading and evidence/media; not a runnable simulator", "snapshot_date": "2026-09-25",
                "repository": "https://github.com/Kaifeng-Jia/holosoma-rubber-hand-marl",
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip(),
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "source_worktree_clean": False, "new_handoff_files_are_uncommitted": True,
                "entry_document": "docs/writing_handoff/README.md", "offline_gallery": "docs/writing_handoff/media/index.html",
                "omitted": ["checkpoints", "datasets", "NPZ", "meshes", "full logs", "environments", "git history", "private meeting originals"],
                "files": [{"path": k, "bytes": len(v), "sha256": sha(v)} for k, v in sorted(contents.items())]}
    contents["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    contents["SHA256SUMS"] = "".join(f"{sha(v)}  {k}\n" for k, v in sorted(contents.items())).encode()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel, data in sorted(contents.items()):
            z.writestr(args.name + "/" + rel, data)
    digest = sha(archive.read_bytes())
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "sha256": digest, "bytes": archive.stat().st_size, "entries": len(contents), "media_groups": len(DEMOS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
