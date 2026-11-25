"""Round 40, Carrier43 (docs/experiments/round40.md "Why the Darknut teacher walks past 3x5b's bomb carriers"): the
owner's Darknut fighter (r40_darknut_fighter.py, room '5b': kill one, take the drop, out N) played in darknut@d3first's
training env on 3:0x5b's first pass with the drop cycle fixed (specialists/experiments/r43-carrier/5bs-c<CYCLE>.yaml),
every reward term the env pays summed a visit.  A job of the queue:

    triforce job run ... --express --timeout 10m -- .venv/bin/python scripts/experiments/r43_fighter_rewards.py \
        CYCLE PER_STATE SEED0 OUT.jsonl [WORKERS]
"""
import collections
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r40_darknut_fighter as F  # noqa: E402

TASK = Path(__file__).resolve().parents[2] / 'specialists' / 'experiments' / 'r43-carrier'


class Summing:
    """The env, every step's reward terms summed (info['rewards']), the start's RAM kept."""

    def __init__(self, env):
        self._env = env
        self.terms = collections.Counter()
        self.total = 0.0
        self.start = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *a, **k):
        out = self._env.reset(*a, **k)
        self.terms.clear()
        self.total = 0.0
        self.start = bytes(self._env.state.ram)
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.total += float(reward)
        for name, value in (info.get('rewards') or {}).items():
            self.terms[name] += float(value)
        return obs, reward, terminated, truncated, info


def _task(cycle: int):
    F.task_file = lambda room, bombs: TASK / f'5bs-c{cycle}.yaml'


def _visits(task):
    cycle, visits = task
    _task(cycle)
    env = Summing(F.make_env('5b', 0))
    out = []
    for seed, start in visits:
        rec = F.play(env, '5b', seed, start)
        ram, s = bytes(env._env.state.ram), env.start        # pylint: disable=protected-access
        rec.update(cycle=cycle, start_cycle=s[0x52A], start_help=s[0x50], start_bombs=s[0x658], end_cycle=ram[0x52A],
                   total=round(env.total, 4), terms={k: round(v, 4) for k, v in env.terms.items()})
        rec.pop('where', None)
        out.append(rec)
    return out


def main() -> int:
    cycle, per_state, seed0, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    from triforce import jobs
    jobs.require_job()
    _task(cycle)
    visits = list(F.starts('5b', 0, seed0, per_state))
    tasks = [(cycle, visits[i:i + 3]) for i in range(0, len(visits), 3)]
    done = 0
    with multiprocessing.get_context('spawn').Pool(workers) as pool, open(out, 'a', encoding='utf-8') as f:
        for part in pool.imap_unordered(_visits, tasks):
            f.write(''.join(json.dumps(r) + '\n' for r in part))
            f.flush()
            done += len(part)
            jobs.progress(done, len(visits), 'visits')
    return 0


if __name__ == '__main__':
    sys.exit(main())
