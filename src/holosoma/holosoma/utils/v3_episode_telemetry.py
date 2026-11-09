"""Opt-in raw episode telemetry, never modifying rewards, actions or reset rules."""
import gzip,json
from collections import deque
import torch

class EpisodeTelemetry:
    def __init__(self,env,path):
        self.env=env;self.original_step=env.step;self.iteration=0
        self.length=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        self.returns=torch.zeros(env.num_envs,device=env.device)
        self.window=deque(maxlen=100);self.return_window=deque(maxlen=100)
        self.stream=gzip.open(path,'xt',encoding='utf-8')
        self.phase=self._phase()
        env.step=self.step
    def _phase(self):
        return self.env.command_manager.get_state('paired_motion_command').time_steps.detach().clone().reshape(-1)
    def step(self,actions):
        result=self.original_step(actions)
        # Snapshot raw values before learner adds timeout bootstrap in place.
        rewards=result[1].detach().reshape(-1);done=result[2].reshape(-1).bool()
        self.length+=1;self.returns+=rewards
        ids=done.nonzero(as_tuple=False).reshape(-1)
        if ids.numel():
            lengths=self.length[ids].cpu().tolist();returns=self.returns[ids].cpu().tolist()
            timeouts=result[3].get('time_outs',torch.zeros_like(done)).reshape(-1)
            row={'iteration':self.iteration,'env_ids':ids.cpu().tolist(),'length_steps':lengths,'return_raw':returns,'timeout':timeouts[ids].cpu().tolist(),'initial_reference_phase':self.phase[ids].cpu().tolist()}
            self.stream.write(json.dumps(row,separators=(',',':'))+'\n')
            self.window.extend(lengths);self.return_window.extend(returns)
            self.length[ids]=0;self.returns[ids]=0
            self.phase[ids]=self._phase()[ids]
        return result
    def summary(self):
        self.stream.flush()
        return {'Episode/mean_length_last100_steps':sum(self.window)/len(self.window) if self.window else None,'Episode/mean_return_raw_last100':sum(self.return_window)/len(self.return_window) if self.return_window else None,'Episode/completed_window_count':len(self.window)}
    def close(self):
        self.env.step=self.original_step;self.stream.close()

CONTRACT={'version':'v3_episode_telemetry_v1','length':'actual control steps since reset, not reference phase or episode_length_buf offset','scope':'all completed episodes including failures and timeouts; unfinished prefixes excluded','window':'last100 finished episodes in time then env-index order across physical environments','units':'control steps; divide by control_hz for seconds','returns':'sum of raw environment rewards before PPO timeout bootstrap','reset_sampling':'unchanged; initial reference phase recorded per episode','resume':'new segment resets environment; rolling window restarts; all segments retained'}
