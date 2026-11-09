"""Pre-update config comparison against frozen seed721; no training math changes."""
import hashlib,json,os
from pathlib import Path
def flatten(x,p=''):
 if isinstance(x,dict):
  out={}
  for k,v in x.items():out.update(flatten(v,p+'.'+k if p else k))
  return out
 return {p:x}
def validate(actual):
 base=Path(os.environ['PUSH721_CONFIG']);old=json.loads(base.read_text());out=Path(actual['output_dir']);a=flatten(old);b=flatten(actual);diff=[];bad=[]
 for k in sorted(set(a)|set(b)):
  if k in a and k in b and a[k]==b[k]:continue
  reason=None
  if k in ['seed','output_dir','started_unix_time']:reason='allowed_seed_directory_timestamp'
  elif k=='git_commit' or k.startswith('runtime_process.'):reason='disclosed_source_or_device_provenance'
  elif k in ['placement_provenance','runtime_reference_check'] and k not in a and b.get(k) is None:reason='R2_audit_field_added_null_no_behavior_change'
  elif os.environ.get('PUSH_IS_SMOKE')=='1' and k=='iterations_requested':reason='preflight_only_not_formal_training'
  item=dict(field=k,before=a.get(k),after=b.get(k),reason=reason);diff.append(item)
  if reason is None:bad.append(k)
 result=dict(passed=not bad,baseline=str(base),baseline_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),differences=diff,unapproved_fields=bad,scope='Compared before first PPO update; no physics, reward, model, reference or optimizer configuration changed.')
 (out/'config_diff.json').write_text(json.dumps(result,indent=2)+'\n')
 if bad:raise RuntimeError('Config mismatch before learning: '+','.join(bad))
