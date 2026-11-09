from pathlib import Path
import copy,importlib.util,json,sys,time
from dataclasses import replace
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src/holosoma'))
import torch
torch.set_num_threads(1)
from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle
from holosoma.agents.mappo.v3_plan5_ablation import V3Plan5PPO,ZeroTeammateRunner
from holosoma.agents.mappo.ppo import Plan5PPO
from holosoma.agents.mappo.runner import Plan5PolicyRunner
from holosoma.agents.mappo.core4d_smalltable_evaluation import deterministic_core4d_smalltable_actions
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object
P=R/'logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt'
SHA='f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364'
cfg=replace(g1_29dof_wbt_w_object.algo.config,num_learning_epochs=1,num_mini_batches=1)
def bundle(scratch=False):
 torch.manual_seed(721);return initialize_plan5_model_bundle(P,cfg,expected_sha256=SHA,from_scratch=scratch)
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
warm=bundle();fresh=bundle(True);again=bundle(True)
assert not same(warm.actor.state_dict(),fresh.actor.state_dict())
assert same(fresh.actor.state_dict(),again.actor.state_dict())
assert same(warm.critic.state_dict(),fresh.critic.state_dict())
assert same(warm.actor_obs_normalizer.state_dict(),fresh.actor_obs_normalizer.state_dict())
s=importlib.util.spec_from_file_location('oldtest',R/'src/holosoma/tests/agents/mappo/test_ppo.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
obs=t._observations(2);changed={**obs,'teammate_obs':obs['teammate_obs']+9}
# Deliberately nonzero normalization statistics expose a raw-only mask bug.
warm.actor_obs_normalizer._mean[...,154:]=2
runner=ZeroTeammateRunner(warm)
a=runner.decide(obs,update_critic_normalizer=False);b=runner.decide(changed,update_critic_normalizer=False)
assert torch.equal(a.actions,b.actions)
assert not a.actor_observations[...,154:].any() and not a.normalized_actor_observations[...,154:].any()
normal=Plan5PolicyRunner(warm).decide(obs,update_critic_normalizer=False)
assert torch.equal(a.normalized_actor_observations[...,:154],normal.normalized_actor_observations[...,:154])
assert torch.equal(a.critic_observations,normal.critic_observations)
assert torch.equal(a.normalized_critic_observations,normal.normalized_critic_observations)
def forbidden(*args,**kwargs):raise AssertionError('critic during evaluation')
warm.critic.evaluate=forbidden
ev=deterministic_core4d_smalltable_actions(warm,changed,zero_teammate_obs=True)
assert torch.equal(ev,a.actions)
default=bundle();old=Plan5PPO(default,cfg,num_envs=2,num_steps_per_env=2);new=V3Plan5PPO(default,cfg,num_envs=2,num_steps_per_env=2)
assert same(old.training_state_dict(iteration=0),new.training_state_dict(iteration=0))
assert torch.equal(old.runner.decide(obs,update_critic_normalizer=False).actions,new.runner.decide(obs,update_critic_normalizer=False).actions)
for scratch,zero in [(True,False),(False,True)]:
 m=bundle(scratch);l=V3Plan5PPO(m,cfg,num_envs=2,num_steps_per_env=2,from_scratch=scratch,zero_teammate_obs=zero)
 initial=copy.deepcopy(m.actor.state_dict());crit=copy.deepcopy(m.critic.state_dict());norm=copy.deepcopy(m.actor_obs_normalizer.state_dict())
 env=t.FakeTeamEnvironment(2);obs=t._observations(2)
 for iteration in range(1,52):
  obs=l.collect_rollout(env,obs);l.update(update_actor=iteration>50)
  if iteration==50:
   assert same(initial,m.actor.state_dict());assert not same(crit,m.critic.state_dict())
 assert not same(initial,m.actor.state_dict());assert same(norm,m.actor_obs_normalizer.state_dict())
 state=copy.deepcopy(l.training_state_dict(iteration=51))
 restored=V3Plan5PPO(bundle(scratch),cfg,num_envs=2,num_steps_per_env=2,from_scratch=scratch,zero_teammate_obs=zero)
 assert restored.load_training_state_dict(state)==51
 assert same(state,restored.training_state_dict(iteration=51))
 try:new.load_training_state_dict(state)
 except ValueError:pass
 else:raise AssertionError('default accepted ablation state')
 try:restored.load_training_state_dict(old.training_state_dict(iteration=0))
 except ValueError:pass
 else:raise AssertionError('ablation accepted default state')
print(json.dumps({'passed':True,'time':time.time(),'tests':['scratch deterministic differs from prior','critic initialization identical','frozen prior normalizer retained','zero teammate before and after normalization','other154 and critic unchanged','actor-only evaluation masked','default checkpoint/actions unchanged','50 critic-only then actor update51 both variants','exact checkpoint restore and wrong-contract rejection'],'prior_sha256':SHA,'warm_std_mean':float(bundle().actor.std.mean()),'fresh_std_mean':float(bundle(True).actor.std.mean())}))
