"""Finalize local V3 delivery only after the complete backup is hash verified."""
import datetime,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
R=Path('/home/kevin/holosoma-core4d-base');D=R/'logs/V3_20261102/cloud_backup_20261104';B=D/'raw';O=R/'logs/V3_20261102/closeout_20261104';O.mkdir(exist_ok=True)
PY='/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def patchtext(path,text):
 path=Path(path)
 if path.exists():
  old=path.read_text();patch='*** Begin Patch\n*** Update File: '+str(path)+'\n@@\n'+''.join('-'+l+'\n' for l in old.splitlines())+''.join('+'+l+'\n' for l in text.splitlines())+'*** End Patch\n'
 else:patch='*** Begin Patch\n*** Add File: '+str(path)+'\n'+''.join('+'+l+'\n' for l in text.splitlines())+'*** End Patch\n'
 subprocess.run(['apply_patch'],input=patch,text=True,check=True)
def main():
 deadline=time.time()+3*3600
 while time.time()<deadline:
  s=read(D/'status.json')
  if s.get('final_complete') and s.get('state')=='complete_verified' and (B/'v3_closeout_receipt.json').exists():break
  write(O/'finalization_status.json',dict(state='waiting_for_final_verified_backup',backup=s.get('state'),cycle=s.get('cycle'),timestamp=time.time()))
  time.sleep(30)
 else:raise RuntimeError('Bounded3h wait expired; no completion claimed')
 write(O/'finalization_status.json',dict(state='verifying_final_models',timestamp=time.time()))
 import torch
 torch.set_num_threads(2);registry=[]
 for p in sorted((B/'v3_continuous_final_eval_20261103_ops').glob('*.exit.json')):
  e=read(p)
  if not e.get('validated'):continue
  assert e['returncode']==0;cp=B/e['checkpoint'].removeprefix('/root/autodl-tmp/');assert sha(cp)==e['checkpoint_sha256']
  state=torch.load(cp,map_location='cpu',weights_only=False);target=e['job']['target'];assert state['iter']==target
  for group in ['actor_model_state_dict','critic_model_state_dict','actor_obs_normalizer_state_dict','critic_obs_normalizer_state_dict']:
   for value in state[group].values():
    if torch.is_tensor(value) and value.is_floating_point():assert torch.isfinite(value).all(),(cp,group)
  params=state['actor_model_state_dict'];assert params['actor_module.module.0.weight'].shape[1]==158 and params['std'].shape[0]==29
  config=cp.parent/'run_config.json';assert config.exists()
  registry.append(dict(run_id=e['job']['id'],task=e['job']['case'],checkpoint=str(cp),checkpoint_sha256=sha(cp),final_iteration=target,loaded_on_cpu=True,finite_parameters=True,actor_input=158,action_output=29,run_config=str(config),run_config_sha256=sha(config),training_exit=str(B/e['training_exit'].removeprefix('/root/autodl-tmp/')),evaluation_exit=str(p)))
 assert len(registry)==21;write(O/'final_checkpoint_registry.json',registry)
 with (O/'final_aggregation.stdout.log').open('w') as log:
  subprocess.run([PY,'-B',str(R/'scripts/finalize_v3_results_20261104.py'),'--hand-metrics'],stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['python3','-B',str(R/'scripts/export_v3_training_curves_20261104.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 agg=read(O/'aggregation_status.json');assert agg['evaluation_units']==120 and agg['episodes']==6120 and not agg['missing_input'],agg
 assert agg['hand_status_counts'].get('missing',0)==0
 media=R/'paper_handoff_exports/V3_20261102/U5_F3_EXEC_20261104';assert read(media/'validation.json')['images']==30 and read(media/'video_validation.json')['video_count']==6
 for task in ['push','pull','kick','chair','bucket','table']:
  vm=read(media/'videos'/(task+'.json'));assert sha(media/'videos'/(task+'.mp4'))==vm['video_sha256']
 completion=read(B/'v3_closeout_receipt.json');assert completion['gpu_work_complete']
 package=R/'paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104';package.mkdir(exist_ok=False)
 for name in ['results','curves','checks','sources','configs','U5_media']:(package/name).mkdir()
 for p in O.iterdir():
  if p.suffix in ['.csv','.json'] and p.name!='finalization_status.json':
   dest='curves' if p.name.startswith('training_') else 'results';shutil.copy2(p,package/dest/p.name)
 for name in ['status.json','latest_inventory.json',f'verification_{s["cycle"]:03d}.json']:
  shutil.copy2(D/name,package/'checks'/('backup_'+name))
 shutil.copy2(B/'v3_closeout_receipt.json',package/'checks/cloud_completion.json')
 for row in registry:shutil.copy2(row['run_config'],package/'configs'/(row['run_id']+'.json'))
 for p in ['finalize_v3_results_20261104.py','export_v3_training_curves_20261104.py','backup_v3_cloud_20261104.py','finish_v3_local_20261104.py','render_v3_f3_execution.py','render_v3_f3_video.py','run_v3_f3_batch.py']:shutil.copy2(R/'scripts'/p,package/'sources'/p)
 for name in ['outputs','videos','video_ops','ops','sources']:shutil.copytree(media/name,package/'U5_media'/name)
 for name in ['validation.json','video_validation.json','jobs.json','input_hashes.json']:shutil.copy2(media/name,package/'U5_media'/name)
 # Per-frame Palm/FK inputs remain in the hash-verified local backup; this is the small Mac package.
 when=datetime.datetime.now(datetime.timezone.utc).isoformat()
 notes=f'''# V3 GPU实验收尾与Mac交接

更新时间：{when}。

## 本轮已完成

- 21个正式训练最终checkpoint全部在本地，SHA256匹配，CPU加载成功，参数有限，Actor158/动作29核验通过。
- 标准评测21项、扩展评测63项、hard评测18项，真实进程退出0；加18项已核验历史Full标准档，共120个评测单元、6120回合（6000扰动+120 nominal）。
- hard按Full-only规则冻结为±8cm/±8°，选择依据见results/hard_level_selection.json。
- 每回合/每种子/三种子汇总与精确双侧McNemar配对统计已生成。Nominal单列，不混进50回合分母。
- 手部位置误差从实际保存姿态离线FK复算；使用rubber-hand link origin，不用腕点冒充手掌。
- 30张1920×1080图、6份轨迹投影JSON、6个原生50Hz视频已校验。seed721首个完成±2扰动回合，五帧固定规则，没有挑最优回合或修改姿态。
- 云端空闲队列已在确认全部GPU任务完成后停止；未关闭云实例，未删除远端数据，未推送GitHub。

## Mac从哪里读取

- `results/group_summary.csv`：先每seed汇总，再三seed平均与样本标准差ddof=1。
- `results/per_seed_summary.csv`、`results/episodes.csv`：逐项追溯；失败前缀误差保留在逐回合表，但不混入完成回合误差汇总。
- `results/paired_completion.csv`：同任务/档位/seed/state_id配对150个样本，精确双侧McNemar。`paired_episode_details.csv`保留两侧完成标记和双方均完成时的误差。
- `results/final_checkpoint_registry.json`：21个最终模型的本地路径、hash、迭代和加载检查；包内不重复放模型，完整模型在Ubuntu备份目录。
- `curves/training_scalars.csv`：原metrics字段等价导出，保留warmup和恢复段；`accepted_before_resume_boundary=false`是未进入最终模型谱系的旧段尾部。没有凭完成率猜回合长度。
- `U5_media/outputs/`：六任务各5图、相机、投影、来源和退出回执；`U5_media/videos/`为对应视频。
- `checks/`：本地备份验证和云端完成回执。

## 仍不能声明全部论文材料完成的项目

- Push/Pull的固定人体Palm目标/角色映射尚未核验，Palm字段保留missing；Kick是not_applicable。Chair/Bucket/SmallTable的固定人体Palm指标已汇总，包含历史Full标准档补算。
- F2a/F2b、F2d复用核验与F3_human素材未在此收尾包完成核验/交齐；已有F2c在旁边的U5_F2c_C01目录。不能把本包说成全部U5完成。
- 历史主结果若没有原始真实episode-length字段，曲线中为空并注明；不将step reward伪装成episode return。历史早期训练段仍按原证据追溯。
- U1历史理由未记录仍为reason_not_documented；不编造历史决策。
- copied-single-prior每任务只有一个独立先验，三seed键是授权重复评测键，不是三个独立训练先验。结果不能解释成三个独立模型的统计证据。
- 本包整理实验事实和材料，不修改论文结论、不新增P2实验。

## 数据与实例

完整本地备份：`{D}`。大文件和逐帧数组保留在该目录；本包偏重Mac写作所需CSV/配置/素材。
SHA256SUMS覆盖本包所有文件；压缩包另附SHA256。确认收到包后，云端不再有本轮必须继续运行的GPU任务，可由Kevin在平台停止计费实例；不要把停止实例误当作已删除备份。
'''
 patchtext(package/'README_CN.md',notes)
 dictionary='''# 数据字典

完成率分母：每seed固定50个扰动状态；nominal不计。标准差：三个seed的sample SD（ddof=1），不是150回合SE。

Object：逐回合三维位置RMSE，厘米；逐回合表保留失败前缀，但汇总误差只计完成回合。
Hand：两机器人×两固定手link origin×全部保存帧的欧氏距离均值，厘米；world坐标，无平面重对齐。参考帧由reference_frame对齐，含初始和真实pre-reset终止状态。
Palm：机器人手link旋转乘固定掌面基，相对于同一人体示范掌面目标的SO(3)测地角；每回合所有四手有效帧平均，度。Kick无目标为not_applicable；Push/Pull目标尚缺为missing；失败为not_completed，不能写0。
Paired errors：仅双方均完成的同state_id样本；差值=variant−Full；没有共同完成时为空。
训练length/return：分别使用Episode/mean_length_last100_steps与Episode/mean_return_raw_last100；mean_step_reward为reward_mean原值。源字段缺失则留空，不能互相替代。
配置/参考等hash列在旧summary未直接记录时可为空；具体来源保留在source_index、原run_manifest、checkpoint registry和备份，不能把空值当作已核验。
绝对路径表示Ubuntu原始证据位置；Mac包内相对路径从README进入。压缩包不含checkpoint和全部原始轨迹，它们保存在Ubuntu的hash验证备份中。
'''
 patchtext(package/'DATA_DICTIONARY.md',dictionary)
 manifest={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*')) if p.is_file()};(package/'SHA256SUMS').write_text(''.join(v+'  '+k+'\n' for k,v in manifest.items()))
 archive=package.with_suffix('.tar.gz')
 with tarfile.open(archive,'w:gz',compresslevel=6) as t:t.add(package,arcname=package.name)
 archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
 result=dict(state='gpu_experiments_backed_up_and_packaged',all_paper_requirements_complete=False,archive=str(archive),archive_sha256=sha(archive),package=str(package),checkpoint_count=21,evaluation_units=120,episodes=6120,backup_verified=True,remote_instance_untouched=True,github_pushed=False,pending=['Push/Pull Palm human target mapping','Remaining F2/F3 human material audit','Historical missing episode-length source fields'],finished_utc=when)
 write(O/'finalization_status.json',result)
 main=R/'logs/V3_20261102/README_zh.md';old=main.read_text();header='# V3 2026-11-02：执行进度\n';assert old.startswith(header)
 update='\n## 最新：GPU实验与备份收尾（2026-11-04）\n\n21个正式训练、21项标准+63项扩展+18项hard评测已完成，最终模型在本地逐个hash和CPU加载验证。加历史Full标准档共120单元/6120回合；30图、6投影和6视频已交本地。三种子、配对、曲线与Mac包见 `closeout_20261104/finalization_status.json`。本次不是全部论文材料完成声明：Push/Pull Palm目标、剩余F2/F3人体素材核验仍待补；历史缺失字段如实保留。P2取消，不启动新实验；远端未关机/未删除，未GitHub推送。下方均为历史进度。\n'
 patchtext(main,header+update+old[len(header):]);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(O/'finalization_status.json',dict(state='error',error=repr(e),timestamp=time.time()));raise
