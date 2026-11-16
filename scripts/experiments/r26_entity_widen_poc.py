"""Proof of concept for docs/observation-next.md §5: a checkpoint whose entity features are widened by zero columns
acts bit for bit as the original in the Rust policy, whatever the new columns hold.

    TRIFORCE_MODELS_DIR=models .venv/bin/python scripts/experiments/r26_entity_widen_poc.py FAMILY [--extra N]
        [--episodes E] [--steps S] [--seed SEED]

The family's promoted checkpoint (A) and a copy (B) whose `entity_encoder.input_proj.weight` gets N zero input
columns after the 33 entity features (before the type embedding, where `model.EntityEncoder` concatenates new
features) and whose `obs_spec` says `entity_features: [12, 33 + N]`. The env plays E episodes from the family's
start states; every decision both policies see the same observation, B's with N extra columns of random values in
every slot (present or not). Compared bit for bit: the joint log-probs over all 352 actions, the value, and the
action each picks from the same Gumbel noise. B's action is played. Exit 0 if every decision matched.

Without the entity zero-column drop in `crates/policy` (the policy's `live` drop extended to `input_proj`) the
values differ in the last bits (the dot product's lanes shift); with it they are identical.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                           cwd=Path(__file__).parent, check=True).stdout.strip())
sys.path.insert(0, str(ROOT))
from triforce import _native, registry      # noqa: E402  pylint: disable=wrong-import-position

PROJ = 'entity_encoder.input_proj.weight'


def policies(run_dir: Path, extra: int):
    checkpoint = registry._torch_load(run_dir / registry.CHECKPOINT_NAME)  # pylint: disable=protected-access
    config = checkpoint['model_config']
    weights = {k: v.detach().to('cpu', torch.float32).contiguous().numpy() for k, v in checkpoint['state_dict'].items()}
    original = _native.Policy(json.dumps(config), weights)
    slots, features = config['obs_spec']['entity_features']['shape']
    wide_config = json.loads(json.dumps(config))
    wide_config['obs_spec']['entity_features']['shape'] = [slots, features + extra]
    proj = weights[PROJ]
    wide = dict(weights)
    wide[PROJ] = np.ascontiguousarray(np.concatenate(
        [proj[:, :features], np.zeros((proj.shape[0], extra), np.float32), proj[:, features:]], axis=1))
    return original, _native.Policy(json.dumps(wide_config), wide), features


def batch(obs):
    return {k: np.asarray(v)[None] for k, v in obs.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('family')
    parser.add_argument('--extra', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--steps', type=int, default=400, help='decisions per episode at most')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    models = registry.models_root()
    run_dir = models / args.family / (models / args.family / 'promoted').read_text().strip()
    a, b, features = policies(run_dir, args.extra)
    rng = np.random.default_rng(args.seed)
    env = _native.Env(args.family, seed=args.seed)
    decisions = mismatches = 0
    outcomes = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        for _ in range(args.steps):
            wide = dict(obs)
            ef = np.asarray(obs['entity_features'], np.float32)
            noise_cols = rng.uniform(-1.0, 1.0, (ef.shape[0], args.extra)).astype(np.float32)
            wide['entity_features'] = np.ascontiguousarray(np.concatenate([ef, noise_cols], axis=1))
            ja, va = a.joint_log_probs(batch(obs))
            jb, vb = b.joint_log_probs(batch(wide))
            noise = rng.gumbel(size=(1, a.noise_size)).astype(np.float32)
            pa = a.evaluate(batch(obs), noise=noise)['actions'][0]
            pb = b.evaluate(batch(wide), noise=noise)['actions'][0]
            same = (ja.tobytes() == jb.tobytes() and va.tobytes() == vb.tobytes() and list(pa) == list(pb))
            decisions += 1
            if not same:
                mismatches += 1
                if mismatches <= 3:
                    finite = np.isfinite(ja) & np.isfinite(jb)
                    print(f'episode {episode} decision {decisions}: max |dlogp| '
                          f'{np.abs(ja[finite] - jb[finite]).max():.3g}, value {va[0]!r} vs {vb[0]!r}, '
                          f'actions {list(pa)} vs {list(pb)}')
            obs, _, terminated, truncated, info = env.step(tuple(int(x) for x in pb))
            if terminated or truncated:
                break
        outcomes.append(info.get('outcome') if isinstance(info, dict) else None)
    print(f'{args.family} ({run_dir.name}): {decisions} decisions over {args.episodes} episodes, '
          f'{features} -> {features + args.extra} entity features, {mismatches} differed')
    return 0 if mismatches == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
