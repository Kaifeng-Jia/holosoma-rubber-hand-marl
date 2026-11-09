"""Optional MP4 from the exact same F3 episode/camera; native 50 Hz states."""
import argparse,hashlib,json,os,subprocess,sys,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
 m=read(ROOT/'outputs'/a.task/'manifest.json');j=m['job'];n=j['horizon_frames']
 helper=ROOT/'paper_handoff_exports/FIGURES_R2_20261027_approved/sources/figround_common.py'
 code=helper.read_text().replace("ROOT=Path('/home/kevin/holosoma-core4d-base')",'ROOT=Path('+repr(str(ROOT))+')').replace("assert os.environ.get('MUJOCO_GL')=='osmesa' and os.environ.get('CUDA_VISIBLE_DEVICES')==''","assert os.environ.get('MUJOCO_GL')=='egl'")
 assert hashlib.sha256(code.encode()).hexdigest()==m['executed_helper_sha256']
 f=types.ModuleType('v3_video_helper');exec(compile(code,str(helper),'exec'),f.__dict__)
 import mujoco,imageio_ffmpeg
 from OpenGL import GL
 assert sha(ROOT/j['trajectory'])==j['trajectory_sha256']
 with np.load(ROOT/j['trajectory'],allow_pickle=False) as z:arr={k:z[k] for k in z.files}
 q=np.concatenate([arr['root_pos'],arr['root_quat_xyzw'][...,[3,0,1,2]],arr['dof_pos']],axis=-1)
 o=np.concatenate([arr['object_pos_w'],arr['object_quat_xyzw'][...,[3,0,1,2]]],axis=-1)
 out=ROOT/'videos';out.mkdir(exist_ok=True);dest=out/(a.task+'.mp4')
 settings=m['records'][0]['camera']['settings']
 scene=f.newscene(ROOT/j['robot_urdf'],ROOT/j['object_urdf'],2,'g1_native_execution',1920,1080)
 f.styling(scene,settings['lookat']);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 cam.lookat[:]=settings['lookat'];cam.distance=settings['distance'];cam.azimuth=settings['azimuth'];cam.elevation=settings['elevation']
 cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','50','-i','-','-an','-c:v','libx264','-threads','2','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for i in range(n):
   scene.set_pose(q[i],o[i]);scene.renderer.update_scene(scene.data,camera=cam)
   if i==0:
    vendor=GL.glGetString(GL.GL_VENDOR).decode();assert 'NVIDIA' in vendor
   p.stdin.write(scene.renderer.render().tobytes())
   if i%100==0:print(a.task,i,n,flush=True)
 finally:p.stdin.close();rc=p.wait();scene.renderer.close()
 assert rc==0,rc
 record=dict(task=a.task,state_id=j['state_id'],training_seed=721,checkpoint_sha256=j['checkpoint_sha256'],trajectory_sha256=j['trajectory_sha256'],fps=50,frames=n,frame_indices=list(range(n)),time_s=(np.arange(n)/50).tolist(),terminal_extra_state_excluded=True,reference_horizon_frames=n,camera_id=m['records'][0]['camera_id'],camera=m['records'][0]['camera'],no_interpolation=True,no_smoothing=True,no_pose_modification=True,no_policy=True,no_physics_steps=True,video_sha256=sha(dest),code_sha256=sha(__file__),encoding_command=cmd,width=1920,height=1080,labels_ui=False)
 dest.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
