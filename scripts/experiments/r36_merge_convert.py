"""Round 36: a ``wide`` lab student (``triforce.nets.wide.WidePolicy``) as today's ``ZeldaPolicy`` of the same shape, so
the Rust actor plays it (the bench's and the chains' fast path).

    python scripts/experiments/r36_merge_convert.py models/r36merge/w768 [--cache target/merge/cache] [--rows 20000]

``WidePolicy`` is ``ZeldaPolicy``'s encoders under towers of ``depth`` layers ``hidden_dim`` wide and the same heads:
the weights map one to one (``encoders.X`` -> ``X``, ``policy_tower`` / ``value_tower`` -> ``mlp_extractor.policy_net``
/ ``value_net``).  Writes ``<run>-z`` beside the run (its ``distill.json`` copied) after checking the two networks
give the same joint log-probabilities and values on held-out cache rows.  The check runs both networks: a job of the
machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 20m -- .venv/bin/python
scripts/experiments/r36_merge_convert.py ...``)."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r36_merge_distill import HOLDOUT_ENV, Data  # pylint: disable=wrong-import-position

from triforce import registry  # pylint: disable=wrong-import-position
from triforce.jobs import require_job  # pylint: disable=wrong-import-position
from triforce.model import ModelConfig, ZeldaPolicy  # pylint: disable=wrong-import-position


def convert(wide) -> ZeldaPolicy:
    c = wide.config
    settings = {**c['encoders'], 'hidden_dim': c['hidden_dim'], 'obs_spec': c.obs_spec}
    if c['depth'] != 3:
        settings['tower_depth'] = c['depth']
    zelda = ZeldaPolicy(config=ModelConfig.from_dict(settings))
    renamed = {}
    for name, value in wide.state_dict().items():
        if name.startswith('encoders.'):
            name = name[len('encoders.'):]
        elif name.startswith('policy_tower.'):
            name = 'mlp_extractor.policy_net.' + name[len('policy_tower.'):]
        elif name.startswith('value_tower.'):
            name = 'mlp_extractor.value_net.' + name[len('value_tower.'):]
        renamed[name] = value
    zelda.load_state_dict(renamed, strict=True)
    return zelda


@torch.no_grad()
def check(wide, zelda, cache: Path, count: int) -> dict:
    data = Data(cache, mmap=True)
    rows = np.flatnonzero(data.env == HOLDOUT_ENV)[:count]
    worst_logp = worst_value = 0.0
    for begin in range(0, len(rows), 4096):
        obs, mask, _, _ = data.batch(rows[begin:begin + 4096], 'cpu')
        a, b = wide.joint_log_probs(obs, mask), zelda.joint_log_probs(obs, mask)
        legal = torch.isfinite(a)
        if not torch.equal(legal, torch.isfinite(b)):
            raise SystemExit('the two networks disagree on which actions are legal')
        worst_logp = max(worst_logp, float((a[legal] - b[legal]).abs().max()))
        worst_value = max(worst_value, float((wide.get_value(obs) - zelda.get_value(obs)).abs().max()))
    return {'rows': int(len(rows)), 'max_abs_logp': worst_logp, 'max_abs_value': worst_value}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('run', help='the wide student run dir (models/r36merge/ID)')
    parser.add_argument('--cache', default='target/merge/cache')
    parser.add_argument('--rows', type=int, default=20000)
    args = parser.parse_args(argv)
    require_job()
    run = Path(args.run)
    wide, checkpoint = registry.load_model(run / 'ckpt.pt')
    wide.eval()
    zelda = convert(wide).eval()
    result = check(wide, zelda, Path(args.cache), args.rows)
    print(json.dumps(result), flush=True)
    if result['max_abs_logp'] > 1e-3 or result['max_abs_value'] > 1e-3:
        raise SystemExit('converted network differs')
    root = run.parent.parent
    extra = {'distill': checkpoint.get('distill'), 'converted_from': str(run), 'conversion_check': result}
    path = registry.save_checkpoint(run.parent.name, run.name + '-z', zelda, iterations=checkpoint.get('iterations'),
                                    steps=checkpoint.get('steps'), root=root, extra=extra)
    if (run / 'distill.json').exists():
        shutil.copy(run / 'distill.json', path.parent / 'distill.json')
    print(f'{path}: {zelda.parameter_count() if hasattr(zelda, "parameter_count") else ""}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
