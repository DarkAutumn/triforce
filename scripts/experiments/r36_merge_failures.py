"""Round 36, "One network by imitation": how a student's failed bench episodes end, beside its teacher's.

    python scripts/experiments/r36_merge_failures.py --network CKPT --slots zol@d3first ow-lynel ... --out F.json

Plays the bench's (or ``--manifest``'s) entry of each slot's family on the bench's seeds, with the student and with
the slot's teacher (``registry.policy_checkpoint``), and counts the outcomes of the episodes that fail (the
evaluation records' ``outcome``: a death, the step budget, a wrong exit, ...)."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from triforce import registry
from triforce.evaluate import evaluate_specialist
from triforce.events import logged_job
from triforce.jobs import add_job_arguments, begin_job, work
from triforce.lab.bench import entry_task, load_manifest


def outcomes(config, checkpoint, manifest, directory, parallel) -> dict:
    result = evaluate_specialist(config, checkpoint, episodes_per_state=manifest['episodes'], parallel=parallel,
                                 seed=manifest['seed'], act=manifest['act'], states_dir=directory,
                                 clips=manifest['clips'])
    records = result['records']
    return {'episodes': len(records), 'success': sum(bool(r['success']) for r in records),
            'failures': dict(Counter(str(r['outcome']) for r in records if not r['success']))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--network', required=True)
    parser.add_argument('--slots', nargs='+', required=True)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--late-manifest', default='target/merge/late17.json',
                        help="the rooms of a paired family's default run")
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--out', required=True)
    add_job_arguments(parser)
    args = parser.parse_args(argv)
    begin_job(args)
    early, late = load_manifest(args.manifest), load_manifest(args.late_manifest)
    plan = []
    for slot in args.slots:
        family, part = registry.split_policy(slot)
        paired = any(e['key'] == f'{family}@d3first' for e in early['entries'])
        manifest = late if paired and part is None else early
        entry = next(e for e in manifest['entries'] if e['key'] == family)
        plan.append((slot, manifest, entry))
    episodes = sum(len(e['states']) * m['episodes'] for _, m, e in plan) * 2
    result = {}
    with logged_job('r36-merge-failures', 'eval', [], log=False, work=work(eval_episodes=episodes)):
        for slot, manifest, entry in plan:
            config, directory = entry_task(entry)
            result[slot] = {
                'student': outcomes(config, args.network, manifest, directory, args.parallel),
                'teacher': outcomes(config, str(registry.policy_checkpoint(slot)), manifest, directory,
                                    args.parallel)}
            print(f"[failures] {slot}: {json.dumps(result[slot])}", flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    sys.exit(main())
