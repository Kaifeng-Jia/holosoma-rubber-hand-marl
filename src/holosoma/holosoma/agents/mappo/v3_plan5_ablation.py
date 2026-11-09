"""Opt-in Pull ablations, preserving the original PPO and centralized critic."""
import torch
from holosoma.agents.mappo.ppo import Plan5PPO
from holosoma.agents.mappo.runner import Plan5PolicyRunner


def ablation_contract(*, from_scratch=False, zero_teammate_obs=False):
    if from_scratch and zero_teammate_obs:
        raise ValueError('Combined ablation not authorized')
    return {'version':'v3_pull_single_factor_v1','from_scratch':bool(from_scratch),
            'zero_teammate_obs':bool(zero_teammate_obs),
            'actor_normalizer':'original prior, frozen, unchanged',
            'critic':'original centralized527',
            'mask':'last4 before and after normalization' if zero_teammate_obs else 'none'}


class ZeroTeammateRunner(Plan5PolicyRunner):
    def _prepare_observations(self, observations, *, update_critic_normalizer=True):
        masked={**observations,'teammate_obs':torch.zeros_like(observations['teammate_obs'])}
        raw,normalized,critic,ncritic,n=super()._prepare_observations(
            masked,update_critic_normalizer=update_critic_normalizer)
        normalized=normalized.clone();normalized[...,154:]=0
        return raw,normalized,critic,ncritic,n


class V3Plan5PPO(Plan5PPO):
    def __init__(self,*args,from_scratch=False,zero_teammate_obs=False,**kwargs):
        self.v3_contract=ablation_contract(from_scratch=from_scratch,zero_teammate_obs=zero_teammate_obs)
        self.v3_enabled=bool(from_scratch or zero_teammate_obs)
        super().__init__(*args,**kwargs)
        if zero_teammate_obs:self.runner=ZeroTeammateRunner(self.models,layout=self.layout)

    def training_state_dict(self,*,iteration):
        state=super().training_state_dict(iteration=iteration)
        if self.v3_enabled:state['v3_ablation']=self.v3_contract.copy()
        return state

    def _validate_training_state(self,state):
        super()._validate_training_state(state)
        expected=self.v3_contract if self.v3_enabled else None
        if state.get('v3_ablation')!=expected:
            raise ValueError('V3 ablation checkpoint contract mismatch')
