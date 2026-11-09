"""Run byte-frozen R2 trainer with one pre-update audit callback."""
from pathlib import Path
import sys
root=Path('/root/autodl-tmp/push_seeds_20261107')
trainer=root/'scripts/train_plan5_push.py'
code=trainer.read_text()
needle='        _write_json(output_dir / "run_config.json", run_config)'
assert code.count(needle)==1
code=code.replace(needle,needle+'\n        from push_20261107_guard import validate\n        validate(run_config)')
sys.path.insert(0,str(root/'scripts'));sys.argv[0]=str(trainer)
exec(compile(code,str(trainer),'exec'),dict(__name__='__main__',__file__=str(trainer)))
