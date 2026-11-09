"""Detached five-GPU U5 renderer; never launches or changes a training job."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OPS=ROOT/'ops'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def run(task,gpu):
 env=os.environ.copy();env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 env.pop('LIBGL_ALWAYS_SOFTWARE',None)
 cmd=[sys.executable,'-B',str(ROOT/'scripts/render_v3_f3_execution.py'),'--task',task,'--output',str(OUT/task)]
 with (OPS/(task+'.stdout.log')).open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  record=dict(task=task,gpu=gpu,pid=p.pid,command=cmd,start_epoch=time.time());write(OPS/(task+'.launch.json'),record)
  rc=p.wait();record.update(returncode=rc,end_epoch=time.time(),manifest_exists=(OUT/task/'manifest.json').is_file());write(OPS/(task+'.exit.json'),record)
 return record
def main():
 OUT.mkdir(exist_ok=True);OPS.mkdir(exist_ok=True)
 assignments=[('push',0),('pull',1),('kick',2),('chair',3),('bucket',5)]
 # Explicitly exclude GPU4, which is running the final training job.
 result=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  futures={pool.submit(run,t,g):(t,g) for t,g in assignments};queued=False
  while futures:
   done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    t,g=futures.pop(f);result.append(f.result());write(OPS/'status.json',dict(completed=result,active=[list(x) for x in futures.values()]))
    if not queued:futures[pool.submit(run,'table',g)]=('table',g);queued=True
 write(OPS/'final.json',dict(results=result,all_passed=all(r['returncode']==0 and r['manifest_exists'] for r in result)))
if __name__=='__main__':main()
