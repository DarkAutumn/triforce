"""Teacher-labelled datasets for offline work (docs/lab.md "Datasets").

    python -m triforce.lab.record --families zol stalfos-keese-gel --actor teacher --rollouts 20 --out DIR

Plays rollouts like lab training (``collect.MultiCollector``: each family's pool, envs split by ``--parallel``) with
the teachers acting (``--actor teacher``) or a given student (``--actor student --student CKPT``), and writes one
``DIR/shard-<n>.npz`` per rollout: every observation key, ``action_mask``, ``action`` (the one taken),
``teacher_joint`` (the family teacher's joint log-probs, ``[n, 352]``, ``-inf`` illegal), ``teacher_value``,
``family`` (index into ``families`` in ``DIR/dataset.json``), ``health_lost`` and ``hits`` (the action taken's).
With ``--try-others K:P`` (as lab training's: the look-ahead tries K other candidates on a share P of the decisions,
the play unchanged) also ``tried_actions`` ``[n, K, 3]`` and ``tried_lost`` ``[n, K]`` (1 lost health, 0 not, -1
untried): hit labels on the actor's own states, recorded with no update."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .. import registry
from ..events import progress_job
from ..jobs import add_job_arguments, begin_job, work
from .collect import MultiCollector, Teachers, parse_families


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m triforce.lab.record', description=__doc__.split('\n\n')[0])
    parser.add_argument('--families', nargs='+', required=True, metavar='KEY[=YAML]')
    parser.add_argument('--actor', default='teacher', choices=['teacher', 'student'])
    parser.add_argument('--student', default=None, help="checkpoint acting with --actor student")
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--steps-per-env', type=int, default=512)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--states-dir', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--logs-dir', default=None, help="the log folder's root (default logs/, $TRIFORCE_LOGS_DIR)")
    parser.add_argument('--try-others', default=None, metavar='K:P',
                        help="the look-ahead's verdict on K other candidates on a share P of the decisions")
    add_job_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    begin_job(args)
    if args.actor == 'student' and not args.student:
        raise SystemExit("--actor student needs --student CKPT")
    out = Path(args.out)
    families = parse_families(args.families, args.parallel)
    keys = [f.key for f in families]
    root = registry.models_root(args.models_dir)
    total = args.rollouts * args.steps_per_env * args.parallel
    with progress_job('lab-record', 'eval', total, unit='decisions', root=args.logs_dir,
                      work=work(train_steps=total)) as progress:
        out.mkdir(parents=True, exist_ok=True)
        teachers = Teachers(keys, args.device, root)
        student = registry.load_model(args.student)[0] if args.student else teachers.networks[keys[0]]
        from .train import parse_try_others  # pylint: disable=import-outside-toplevel
        collector = MultiCollector(families, out / 'run', args.steps_per_env, 0.99, 0.95, seed=args.seed,
                                   states_dir=args.states_dir, teachers=teachers, actor=args.actor,
                                   tried=parse_try_others(args.try_others))
        (out / 'dataset.json').write_text(json.dumps({
            'families': keys, 'actor': args.actor, 'student': args.student, 'teachers': teachers.paths,
            'try_others': args.try_others,
            'envs': collector.shares, 'steps_per_env': args.steps_per_env, 'seed': args.seed}, indent=2),
            encoding='utf-8')
        try:
            for shard in range(args.rollouts):
                collector.start(student)
                collector.take()
                collector.ready(student, args.device)
                batch = collector.buffer.flatten('cpu')
                arrays = {f'obs/{k}': v.numpy() for k, v in batch['obs'].items()}
                arrays.update(action_mask=batch['masks'].numpy(), action=batch['actions'].numpy(),
                              teacher_joint=batch['teacher_joint'].numpy(),
                              teacher_value=batch['teacher_value'].numpy(), family=batch['family'].numpy(),
                              health_lost=batch['health_lost'].numpy(), hits=batch['hits'].numpy())
                if 'tried_lost' in batch:
                    arrays.update(tried_actions=batch['tried_actions'].numpy(), tried_lost=batch['tried_lost'].numpy())
                np.savez_compressed(out / f'shard-{shard:04d}.npz', **arrays)
                print(f"[record] shard {shard}: {batch['masks'].shape[0]} rows", flush=True)
                progress.update(progress.done + batch['masks'].shape[0])
        finally:
            collector.close()
    return out


def load(directory) -> tuple[dict, dict]:
    """(dataset.json, every shard's arrays concatenated) of a recorded dataset."""
    directory = Path(directory)
    meta = json.loads((directory / 'dataset.json').read_text(encoding='utf-8'))
    shards = [np.load(path) for path in sorted(directory.glob('shard-*.npz'))]
    return meta, {key: np.concatenate([s[key] for s in shards]) for key in shards[0].files}


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(0 if main() else 1)
