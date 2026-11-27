"""Round 40 ("Every shooter"; docs/lab.md "Continuing with a new input"): the overworld teacher widened with the
`shooter` token group acts as its source.

    triforce job run --agent A --task T --subtask widen-proof --express --timeout 15m -- \\
        .venv/bin/python scripts/experiments/r49_widen_proof.py OUT [--source RUN_DIR]

1. Loads the source (default the store's `gen1-overworld/s1r2-c1`), widens it as `python -m triforce.lab.train --group
   overworld --load ...` does (`lab.recipe.widen_loaded` with the group file's tokens) and saves the widened checkpoint
   in OUT (`ckpt.pt`, its `act` file in the play rule, `widened.json`), for chains (`eval-chain --checkpoint
   ow-zora@d3first=OUT ...` for each of the group's families).
2. Records observations: the group's ten d3first tasks (`obs.shooter` on), up to eight seed states each, the source
   acting (sampled from its policy) up to 300 decisions an episode.
3. Compares every head, the value and the hit logits of source and widened on them, on the CPU and the GPU, in whole
   batches and in batches of 20 and 1; exit 0 when the CPU's are equal bit for bit in every batch and every argmax
   action agrees (the GPU's differences are printed beside the source's own between batch sizes).
"""

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from triforce import _native, jobs, registry
from triforce.lab import groups
from triforce.lab.recipe import widen_loaded
from triforce.specialist import load_specialist

ROOT = Path(__file__).resolve().parents[2]
STATES = ROOT / 'triforce' / 'states'


def widen(source_dir: Path, out: Path):
    """(source, widened, the group): the widened checkpoint saved in ``out``."""
    source, checkpoint = registry.load_model(source_dir / 'ckpt.pt')
    group = groups.get('overworld')
    wide, added = widen_loaded(source, 'hitnet', group.network['arch_config'])
    if added != ['shooter']:
        raise SystemExit(f"{source_dir}: expected to add the shooter group, added {added}")
    out.mkdir(parents=True, exist_ok=True)
    saved = {**checkpoint, 'model_config': wide.config.to_dict(), 'state_dict': wide.state_dict(),
             'obs_schema_version': _native.OBS_SCHEMA_VERSION}
    saved.pop('optimizer', None)
    torch.save(saved, out / 'ckpt.pt')
    sha = hashlib.sha1((out / 'ckpt.pt').read_bytes()).hexdigest()
    (out / 'act').write_text(f'{registry.PLAY_ACT} {sha}\n')     # a store run acts by the play rule
    if (source_dir / 'version').exists():
        (out / 'version').write_text((source_dir / 'version').read_text())
    (out / 'widened.json').write_text(json.dumps({
        'source': str(source_dir / 'ckpt.pt'), 'added': added, 'sha1': sha,
        'source_sha1': hashlib.sha1((source_dir / 'ckpt.pt').read_bytes()).hexdigest()}, indent=1))
    print(f"{source_dir}: tokens {source.config['tokens']} -> {wide.config['tokens']}; {out / 'ckpt.pt'} {sha}")
    return source.eval(), wide.eval(), group


def record(source, group) -> list:
    """The observations of the group's tasks, the source acting (sampled, seeded)."""
    torch.manual_seed(0)
    observations, filled = [], 0
    for member in group.members:
        family = member.split('@')[0]
        config = load_specialist(str(ROOT / 'specialists' / 'd3first' / f'{family}.yaml'))
        if not config.to_dict()['obs'].get('shooter'):
            raise SystemExit(f"{family}@d3first: obs.shooter is off")
        states = sorted({Path(p).stem for pattern in config.start_states
                         for p in glob.glob(str(STATES / f'{pattern}.state'))})
        for seed, state in enumerate(states[::max(1, len(states) // 8)][:8]):
            env = _native.Env(config, start_state=state, seed=seed, strict_actions=False, states_dir=STATES)
            obs, _ = env.reset(seed=seed)
            for _ in range(300):
                observations.append(obs)
                filled += int(np.any(obs['entity_shooter']))
                with torch.no_grad():
                    out = source.evaluate(obs, deterministic=False)
                obs, _, terminated, truncated, _ = env.step([int(a) for a in out.actions[0].tolist()])
                if terminated or truncated:
                    break
    shooter = np.stack([o['entity_shooter'] for o in observations])
    print(f"{len(observations)} observations, {filled} with a shooter row filled; rows loaded "
          f"{int((shooter[..., 0] > 0).sum())}, will_throw 1 {int((shooter[..., 2] == 1).sum())}")
    return observations


def run(network, batch: dict, device: str, size: int):
    """Every head of ``network`` in batches of ``size`` on ``device``, and its argmax actions."""
    network.to(device)
    out, actions = {}, []
    rows = len(next(iter(batch.values())))
    with torch.no_grad():
        for lo in range(0, rows, size):
            part = {k: v[lo:lo + size] for k, v in batch.items()}
            h = network._heads(network._batch(part))  # pylint: disable=protected-access
            for key, value in (('kind', h.kind), ('direction', h.direction), ('duration', h.duration),
                               ('value', h.value), ('hit', h.aux['hit'])):
                out.setdefault(key, []).append(value.cpu())
            actions.append(network.evaluate(part, deterministic=True).actions.cpu())
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(actions)


def largest(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--source', type=Path,
                        default=Path.home() / 'triforce-shared' / 'runs' / 'gen1-overworld' / 's1r2-c1')
    args = parser.parse_args(argv)
    jobs.require_job()
    source, wide, group = widen(args.source, args.out)
    observations = record(source, group)
    batch = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
    ok = True
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        whole, _ = run(source, batch, device, len(observations))
        for size in (len(observations), 20, 1):
            (a, pa), (b, pb) = run(source, batch, device, size), run(wide, batch, device, size)
            worst, same_actions = largest(a, b), bool((pa == pb).all())
            print(f"{device}, batches of {size}: largest head difference {worst:.3g}, argmax actions differing "
                  f"{int((pa != pb).any(dim=-1).sum())} of {len(observations)}; the source against itself in whole "
                  f"batches {largest(a, whole):.3g}" + (' (bit for bit)' if worst == 0 else ''))
            ok = ok and same_actions and (worst == 0 or device == 'cuda')
    print('the CPU bit for bit, every argmax action the same' if ok else 'DIFFERENT')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
