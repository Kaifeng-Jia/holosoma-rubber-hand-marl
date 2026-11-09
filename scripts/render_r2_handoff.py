#!/usr/bin/env python3
"""R2 native stills/provenance from saved recordings; no policy or physics stepping."""
import argparse,json,subprocess,sys,shutil,hashlib
from pathlib import Path
import numpy as np
from export_recorded_demo_media import sha,ROOT

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    old=ROOT/'logs/PaperSupplement20260929/unified_evaluation_20261003';new=ROOT/'logs/ExperimentsR2_evaluation_20261026_retry01'
    u=out/'U7_media';records=[];oldchecks=[]
    renderer=ROOT/'scripts/export_recorded_demo_media.py'
    archived=ROOT/'paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005/render_tools/export_recorded_demo_media.py'
    def render(src,obj,dest,frames,title,label,info):
        cmd=[sys.executable,'-B',str(renderer),'--rollout',str(src),'--object-urdf',str(obj),'--output',str(dest),
             '--title',title,'--checkpoint-label',label,'--width','1920','--height','1080','--frames',','.join(map(str,frames)),
             '--stills-only','--no-overlay']
        if not (dest/'manifest.json').exists():subprocess.run(cmd,check=True)
        m=read(dest/'manifest.json');assert m['rollout_sha256']==sha(src) and m['still_frame_indices']==frames
        record=dict(info,render_manifest=str((dest/'manifest.json').relative_to(out)),
                    source=str(src),source_sha256=sha(src),camera=m['camera'],frame_cameras=m['camera_frames'],
                    fovy=m['fovy_degrees'],lighting=m['lighting'],labels_axes_ui=False,command=cmd)
        records.append(record)
    selections={
      'Push-livePD_seed721':([0,185,309],['俯身准备','俯身随桌推进','末端起身'],['双臂在桌面上方，尚未形成可由单帧确认的负载接触','身体前倾、桌子发生平移；不由图像推断手部受力','身体较中段直立，参考片段到达末端']),
      'Pull-baseline_seed721':([0,190,316],['拉动准备','俯身随桌后移','结束起身'],['手臂垂向桌边，初始状态','双机器人弯腰靠近桌腿/桌边，桌子位置已变化','两机器人起身，手臂离开低处操作位']),
      'C-01_seed721':([0,313,391],['倒椅初始状态','扶正椅子末段','椅子立稳与动作结束'],['椅背/椅腿方向显示椅子侧倒','椅子接近直立，一台机器人仍俯身','椅子已直立，机器人上身恢复，抬手动作存在']),
      'T-07_seed721':([0,412,687],['桌旁准备','随桌移动','片段末端'],['两机器人站在桌两侧','两机器人随物体平移，单帧不足以证明双手共同承重','物体和机器人到片段终点，不把姿态图称为成功放下'])}
    for mid,(frames,labels,basis) in selections.items():
        path=old/'media'/mid/'manifest.json';m=read(path)
        assert sha(Path(m['rollout_source']))==m['rollout_sha256']
        checks={k:sha(path.parent/k)==v for k,v in m['outputs_sha256'].items()}
        assert all(checks.values())
        oldchecks.append(dict(model_id=mid,manifest=str(path),manifest_sha256=sha(path),outputs_verified=checks,
                       archived_script=str(archived),archived_script_sha256=sha(archived),
                       manifest_script_sha256=m['script_sha256'],exact_script_match=sha(archived)==m['script_sha256']))
        dest=u/'old_selected_frames'/mid
        render(Path(m['rollout_source']),Path(m['object_urdf_source']),dest,frames,mid,'old nominal',
             dict(model_id=mid,seed=721,rollout_id='nominal',checkpoint_sha256=m['recorded_metadata']['checkpoint_sha256'],
                  stages=[dict(frame=f,label=l,visual_basis=b) for f,l,b in zip(frames,labels,basis)],
                  original_selection_preserved=True))
        shutil.copy2(path,dest/'old_render_manifest.json')
        shutil.copy2(m['rollout_source'],dest/'source_recording.npz')
    # Fixed selection rule: first completed perturbed rollout of the lowest registered seed.
    for group,mid in [('U5_push_registered','Push-registered_seed721'),('U6_kick','Kick-baseline_seed721')]:
        folder=new/group/'results'/mid;eps=read(folder/'episodes.json')
        ep=next(e for e in eps if e['kind']=='perturbed' and e['completed_reference'])
        src=folder/Path(ep['trajectory']).name;m=read(new/'media'/mid/'manifest.json')
        with np.load(src) as z:count=len(z['dof_pos'])
        frames=np.linspace(0,count-1,4).round().astype(int).tolist()
        render(src,Path(m['object_urdf_source']),u/'rollouts'/mid,frames,mid,ep['state_id'],
             dict(model_id=mid,seed=721,rollout_id=ep['state_id'],checkpoint_sha256=m['recorded_metadata']['checkpoint_sha256'],
                  selection='First completed perturbed episode in frozen state order for seed721; illustrative, not aggregate result',
                  outcome=ep,stages_pending_visual_review=True))
    # Failed nominal is retained, not replaced by a successful perturbation.
    mid='Push-registered_seed721';folder=new/'U5_push_registered/results'/mid
    ep=next(e for e in read(folder/'episodes.json') if e['kind']=='nominal');assert not ep['completed_reference']
    src=folder/Path(ep['trajectory']).name;m=read(new/'media'/mid/'manifest.json')
    with np.load(src) as z:arr={k:z[k].copy() for k in z.files}
    frames=[max(1,ep['steps']-20),ep['steps']]
    render(src,Path(m['object_urdf_source']),u/'failure_case/actual',frames,mid,'failed nominal',
           dict(model_id=mid,seed=721,rollout_id='nominal',checkpoint_sha256=m['recorded_metadata']['checkpoint_sha256'],
                outcome=ep,selection='Last frame and 0.4s before termination; no assumption of force or collision causality'))
    with np.load(out/'checks/push_reference_for_fk.npz') as z:ref={k:z[k].copy() for k in z.files}
    indices=arr['reference_frame'];aligned={k:v.copy() for k,v in arr.items()}
    aligned['root_pos']=ref['agent_body_pos_w'][indices,:,0];aligned['root_quat_xyzw']=ref['agent_body_quat_w'][indices,:,0]
    aligned['dof_pos']=ref['agent_joint_pos'][indices];aligned['object_pos_w']=ref['object_pos_w'][indices]
    aligned['object_quat_xyzw']=ref['object_quat_w'][indices]
    aligned['_metadata_json']=json.dumps(dict(artifact_kind='aligned_reference_not_execution',fps=50))
    aligned_path=u/'failure_case/aligned_reference.npz';np.savez_compressed(aligned_path,**aligned)
    render(aligned_path,Path(m['object_urdf_source']),u/'failure_case/reference',[frames[-1]],'Aligned reference','NOT execution',
           dict(model_id=mid,seed=721,rollout_id='nominal',reference_only=True,alignment='recorded reference_frame; pelvis is URDF root',
                matched_execution_frame=frames[-1],reference_frame=int(indices[frames[-1]])))
    write(u/'old_media_provenance_check.json',oldchecks)
    write(u/'render_manifest.json',dict(script=str(Path(__file__).resolve()),script_sha256=sha(Path(__file__)),command=[sys.executable,*sys.argv],records=records))
    write(u/'camera.json',[{k:r[k] for k in ['render_manifest','camera','frame_cameras','fovy','lighting']} for r in records])
    print('R2 stills finished',flush=True)
if __name__=='__main__':main()
