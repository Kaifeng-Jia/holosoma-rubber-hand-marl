"""Authorized F2c re-render: C-01 wrists, fixed historical camera, CPU only."""
import argparse,copy,csv,hashlib,json,os,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'paper_handoff_exports/FIGURES_R2_20261027_approved'
sys.path.insert(0,str(OLD/'sources'))
import figround_common as frozen
from scipy.spatial.transform import Rotation
from OpenGL import GL

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):Path(p).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def arrays(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def source(p):return dict(path=str(p),sha256=sha(p))

def make_visual(path,physical_urdf,solid_pose,ghost_pose,ghost_scale,mesh):
    root=ET.Element('robot',name=path.stem);ET.SubElement(root,'link',name='root')
    physical=ET.parse(physical_urdf)
    for name,q in [('solid',solid_pose)]+([] if ghost_pose is None else [('ghost',ghost_pose)]):
        link=ET.SubElement(root,'link',name=name)
        if name=='solid':
            for original in physical.findall('.//visual'):
                v=copy.deepcopy(original)
                for m in v.findall('.//mesh'):m.set('filename',mesh.name)
                link.append(v)
        else:
            v=ET.SubElement(link,'visual');g=ET.SubElement(v,'geometry')
            ET.SubElement(g,'mesh',filename=mesh.name,scale=f'{ghost_scale:.17g} {ghost_scale:.17g} {ghost_scale:.17g}')
        joint=ET.SubElement(root,'joint',name=name+'_fixed',type='fixed')
        ET.SubElement(joint,'parent',link='root');ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz=frozen.numbers(q[:3]),rpy=frozen.numbers(Rotation.from_quat(q[3:][[1,2,3,0]]).as_euler('xyz')))
    ET.indent(root,space='  ');ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    assert not root.findall('.//collision') and not root.findall('.//inertial')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ['raw','assets','sources','diagnostics']:(out/name).mkdir()
    pkg=ROOT/'src/holosoma';base=pkg/'holosoma/data/motions/g1_29dof/whole_body_tracking'
    full=base/'core4d_chair021_20231020_074_a1';variant=base/'v3_20261102/chair_one_sided'
    desc=read(variant/'variant.json')
    for prefix in ['source_pair','runtime_reference','object_urdf','training_promotion','diff']:
        assert sha(pkg/desc[prefix+'_file'])==desc[prefix+'_sha256']
    fullpair=full/'core4d_pair_compact_fps30.npz';smallpair=pkg/desc['source_pair_file']
    assert sha(fullpair)==desc['parent_contract']['source_pair_sha256']
    f=arrays(fullpair);v=arrays(smallpair)
    assert np.array_equal(f['robot_qpos'],v['robot_qpos'])
    assert np.array_equal(f['object_qpos'][:,3:],v['object_qpos'][:,3:])
    frame=117;fps=int(f['fps']);assert fps==int(v['fps'])==30
    time_s=frame/fps;runtime_frame=round(time_s*50);assert runtime_frame==195
    nom=ROOT/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/two_stage/person2/person2_nominal_scaled.npz'
    q2=arrays(nom)['qpos'][frame,36:43]
    scales=f['human_to_robot_scales'];mean=float(f['shared_object_scale'])
    robot=f['robot_qpos'][frame];mesh_source=full/'chair021_m.obj'
    mesh=out/'assets'/mesh_source.name;shutil.copyfile(mesh_source,mesh)
    robotpath=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
    historical=read(OLD/'raw/R6_chair_shared_mean.json');camera=historical['camera']['settings']
    frozen.OUT=out
    rows=[];sources=[fullpair,smallpair,nom,variant/'variant.json',OLD/'raw/R6_chair_shared_mean.json',robotpath,mesh_source]
    for name,physical,q,ghost in [
        ('F2c_chair_one_sided_ghost',pkg/desc['object_urdf_file'],v['object_qpos'][frame],q2),
        ('F2c_chair_shared',full/'chair021_training.urdf',f['object_qpos'][frame],None),
    ]:
        visual=out/'assets'/(name+'_visual_only.urdf')
        make_visual(visual,physical,q,ghost,float(scales[1]),mesh)
        scene=frozen.newscene(robotpath,visual,2,'cplus_sharedscene',1920,1080)
        try:
            frozen.styling(scene,camera['lookat'])
            ghost_count=0
            for item in scene.mapping_records:
                if item['role']=='object':
                    isghost=item['link']=='ghost';assert item['link'] in ['solid','ghost'],item
                    rgba=frozen.colors('#B07D3E' if isghost else frozen.cfg['object_hex'])
                    rgba[3]=.42 if isghost else 1.
                    scene.model.geom_rgba[item['geom_id']]=rgba
                    item.update(rgba=rgba,object_role='nominal_participant2_ghost' if isghost else 'physical_reference_solid')
                    ghost_count+=int(isghost)
            assert (ghost_count>0)==(ghost is not None)
            assert np.all(scene.model.geom_contype==0) and np.all(scene.model.geom_conaffinity==0)
            scene.set_pose(robot,np.asarray([0,0,0,1,0,0,0]))
            image=out/'raw'/(name+'.png');cam=frozen.render(scene,camera,image)
            assert np.allclose(cam['world_to_camera'],historical['camera']['world_to_camera'],rtol=0,atol=1e-6)
            gl={k:GL.glGetString(c).decode() for k,c in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]}
            assert 'llvmpipe' in gl['renderer'].lower(),gl
            hands=[];E=np.asarray(cam['world_to_camera']);K=np.asarray(cam['K'])
            for role in [0,1]:
                for side in ['left','right']:
                    xyz=scene.hand_center(role,side);c=E[:3,:3]@xyz+E[:3,3];pixel=K@c;uv=pixel[:2]/pixel[2]
                    hands.append(dict(role=role,side=side,link=side+'_rubber_hand_link',world_xyz_m=xyz.tolist(),pixel_xy=uv.tolist(),in_front=bool(c[2]>0),inside_raster=bool(c[2]>0 and 0<=uv[0]<1920 and 0<=uv[1]<1080)))
            record=dict(asset_id=name,kind='reference_geometry',task='chair',configuration='C-01 with-wrist fixed; '+('U3a participant1 physical object plus nominal participant2 ghost' if ghost is not None else 'Full shared mean object'),training_seed=None,state_id=None,selection_rule='author-fixed source frame117 at30Hz; no policy or outcome selection',state_index=None,reference_frame=runtime_frame,source_frame=frame,fps=fps,runtime_fps=50,time_s=time_s,input_paths=[str(x) for x in sources]+[str(physical)],input_sha256=[sha(x) for x in sources]+[sha(physical)],checkpoint_sha256=None,robot_urdf_sha256=sha(robotpath),object_urdf_sha256=sha(physical),visual_urdf=source(visual),camera_id='R6_chair_camera_C01_1920x1080_v3',camera_intrinsics=cam['K'],camera_extrinsics=cam['world_to_camera'],projection=cam['projection'],image_size=[1920,1080],crop_transform=np.eye(3).tolist(),camera=cam,light_material_spec=frozen.cfg,render_command=[sys.executable,*sys.argv],code_sha256=sha(__file__),image_sha256=sha(image),image=str(image.relative_to(out)),physical_object_scale=float(scales[0]) if ghost is not None else mean,physical_object_qpos_wxyz=q.tolist(),physical_mass_kg=5,physical_object_asset=source(physical),ghost=None if ghost is None else dict(scale=float(scales[1]),qpos_wxyz=ghost.tolist(),alpha=.42,color_hex='#B07D3E',collision_enabled=False,dynamics_enabled=False,training_asset_member=False,source=source(nom)),role_geom_mapping=scene.mapping_records,mesh_sources=scene.mesh_records,robot_qpos_sha256=hashlib.sha256(robot.tobytes()).hexdigest(),robot_qpos_identical_between_panels=True,labels_ui=False,no_physics_steps=True,no_policy=True,software_backend=gl,approval='user approved rerender both panels using C-01 and 1920x1080 on 2026-11-02',historical_image_not_reused=True)
            write(image.with_suffix('.json'),record)
            write(out/'raw'/(name+'_overlay.json'),dict(image_sha256=sha(image),camera_id=record['camera_id'],note='Optional display overlays only; hand link origins are not contact evidence; not baked into image',hand_link_origins=hands))
            rows.append(record)
        finally:scene.renderer.close()
    assert rows[0]['camera']==rows[1]['camera'] and rows[0]['robot_qpos_sha256']==rows[1]['robot_qpos_sha256']
    values={'one_sided':100*max(abs(float(scales[0])/s-1) for s in scales),'shared_mean':100*max(abs(mean/s-1) for s in scales)}
    with (out/'size_mismatch.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['configuration','s1','s2','physical_scale','max_size_mismatch_percent'])
        for key,s in [('one_sided',float(scales[0])),('shared_mean',mean)]:w.writerow([key,*map(float,scales),s,values[key]])
    for file in [Path(__file__),OLD/'sources/figround_common.py',ROOT/'scripts/render_wrist_reference_pairs.py',ROOT/'scripts/export_recorded_demo_media.py',frozen.R0/'sources/previous_r0_helpers.py']:
        shutil.copyfile(file,out/'sources'/file.name)
    shutil.copyfile(frozen.R0/'R0/material_lighting.json',out/'sources/material_lighting.json')
    np.savez_compressed(out/'sources/selected_frame117.npz',robot_qpos=robot,physical_one_sided=v['object_qpos'][frame],physical_shared=f['object_qpos'][frame],ghost_participant2=q2,scales=scales,mean_scale=mean,source_frame=frame,source_fps=fps,runtime_frame=runtime_frame,runtime_fps=50)
    manifest=dict(status='rendered_pending_visual_review',created_at=datetime.now(timezone.utc).isoformat(),instruction_commit='6d3836a8328b32c901b8140d3e0f41aff4ef01d6',scope='F2c only; no training code/assets changed',records=rows,size_mismatch_percent=values,projection_change='same camera settings/extrinsics and45deg vertical FOV; native 1920x1080, recomputed K; no resize/crop',source_hashes=[source(x) for x in sources])
    write(out/'manifest.json',manifest)
    files=[p for p in out.rglob('*') if p.is_file()];(out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(files)))
    print(json.dumps({'output':str(out),'images':[x['image'] for x in rows],'same_robot_and_camera':True,'size_mismatch_percent':values}))
if __name__=='__main__':main()
