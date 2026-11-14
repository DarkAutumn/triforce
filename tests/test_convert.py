"""``triforce convert-model`` (``triforce/convert.py``) on a real promoted 1.4.1 run: the widened model acts exactly as
the source on recorded observations (the Rust policy, bit for bit, the new kinds masked), its new rows start as
documented, the Adam state is padded, the source is untouched, and anything but the source layout is refused.

The source is the first promoted 1.4.1 run under ``$TRIFORCE_CONVERT_SOURCE`` (a models root), else the models
root (``registry.models_root``); without one the tests skip (a fresh clone has no models)."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from triforce import _native, convert, registry
from triforce.actions import NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from triforce.observation import HISTORY_FEATURES, LINK_FEATURES

from .conftest import REPO_STATES_DIR

OLD_KINDS = len(convert.SOURCE_KINDS)


def _source_root() -> Path:
    return Path(os.environ.get('TRIFORCE_CONVERT_SOURCE') or registry.models_root())


def _source_run() -> Path:
    """A promoted run stamped 1.4.1.x (the smallest family that has one), else skip."""
    root = _source_root()
    for name in ('boss-aquamentus', 'wallmaster', 'zol', 'goriya'):
        pointer = root / name / registry.PROMOTED_NAME
        if pointer.is_file():
            run = root / name / pointer.read_text(encoding='utf-8').strip()
            version = registry.model_version(run)
            if version and version.startswith(convert.SOURCE_VERSION + '.'):
                return run
    pytest.skip(f"no promoted {convert.SOURCE_VERSION} run under {root}")


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def converted(tmp_path_factory):
    run = _source_run()
    before = _sha1(run / registry.CHECKPOINT_NAME)
    target = convert.convert_run(run, tmp_path_factory.mktemp('converted'))
    assert _sha1(run / registry.CHECKPOINT_NAME) == before, "the source is untouched"
    return run, target


def _native_policy(checkpoint: dict):
    weights = {name: value.float().numpy() for name, value in checkpoint['state_dict'].items()}
    return _native.Policy(json.dumps(checkpoint['model_config']), weights)


def _old_layout(obs: dict) -> dict:
    """Current observations in the source layout: the added history and link columns and the new kinds' mask
    entries removed."""
    keep = {key: [names.index(n) for n in convert.old_names(key)]
            for key, names in (('history', HISTORY_FEATURES), ('link', LINK_FEATURES))}
    old = {key: (value[:, keep[key]] if key in keep else value) for key, value in obs.items()}
    mask = obs['action_mask']
    k, d, u = NUM_KINDS, NUM_DIRECTIONS, NUM_DURATIONS
    dirs = mask[:, k:k + k * d].reshape(-1, k, d)[:, :OLD_KINDS]
    durs = mask[:, k + k * d:].reshape(-1, k, u)[:, :OLD_KINDS]
    old['action_mask'] = np.concatenate([mask[:, :OLD_KINDS], dirs.reshape(len(mask), -1), durs.reshape(len(mask), -1)],
                                        axis=1)
    return old


def test_the_widened_model_acts_as_the_source(converted):
    run, target = converted
    source = registry._torch_load(run / registry.CHECKPOINT_NAME)    # pylint: disable=protected-access
    old = _native_policy(source)
    new = registry.load_model(target)[0].to_native()
    rows = []
    env = _native.Env(source['specialist'], states_dir=str(REPO_STATES_DIR), seed=5)
    obs, _ = env.reset(seed=5)
    for step in range(120):
        rows.append(obs)
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        noise = _native.keyed_noise([3], [step], new.noise_size)
        action = new.evaluate(batch, noise=noise)['actions'][0]
        obs, _, terminated, truncated, _ = env.step([int(a) for a in action])
        if terminated or truncated:
            obs, _ = env.reset(seed=6 + step)
    observations = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}
    mask = observations['action_mask']
    assert not mask[:, OLD_KINDS:NUM_KINDS].any(), "the family masks the new kinds"
    n = len(mask)
    for noise_of in (None, lambda p: _native.keyed_noise(list(range(n)), [2] * n, p.noise_size)):
        pick = (lambda p: {'deterministic': True}) if noise_of is None else (lambda p: {'noise': noise_of(p)})
        a = old.evaluate(_old_layout(observations), **pick(old))
        b = new.evaluate(observations, **pick(new))
        for key in ('actions', 'log_prob', 'entropy', 'value'):
            assert np.asarray(a[key]).tobytes() == np.asarray(b[key]).tobytes(), key
        assert a['kind_log_probs'][:, :OLD_KINDS].tobytes() == b['kind_log_probs'][:, :OLD_KINDS].tobytes()
        assert (b['kind_log_probs'][:, OLD_KINDS:] < -1e8).all(), "the new kinds are masked"


def test_new_rows_optimizer_and_files(converted):
    run, target = converted
    source = registry._torch_load(run / registry.CHECKPOINT_NAME)    # pylint: disable=protected-access
    widened = registry.load_checkpoint(target)
    state, old = widened['state_dict'], source['state_dict']
    assert (widened['action_schema_version'], widened['obs_schema_version']) == (_native.ACTION_SCHEMA_VERSION,
                                                                               _native.OBS_SCHEMA_VERSION)
    assert state['kind_head.weight'].shape[0] == NUM_KINDS
    wait, bomb = 1, 4
    for k in range(OLD_KINDS, NUM_KINDS):
        assert torch.equal(state['kind_head.weight'][k], old['kind_head.weight'][wait])
        assert state['kind_head.bias'][k] == old['kind_head.bias'][wait] - convert.NEW_KIND_BIAS_BELOW_WAIT
        embedding_from = wait if _native.ACTION_KINDS[k] == 'RECORDER' else bomb
        assert torch.equal(state['kind_embedding.weight'][k], old['kind_embedding.weight'][embedding_from])
    for name in convert.KIND_ROWS:
        assert torch.equal(state[name][:OLD_KINDS], old[name])
    for name in convert.FIRST_LAYERS:
        assert state[name].shape[1] == old[name].shape[1] + 6 and torch.count_nonzero(state[name]) == \
            torch.count_nonzero(old[name])
    names = [n for n, _ in registry.load_model(target)[0].named_parameters()]
    for index, entry in widened['optimizer']['state'].items():
        name = names[int(index)]
        assert entry['exp_avg'].shape == state[name].shape and entry['exp_avg_sq'].shape == state[name].shape
        if name in convert.KIND_ROWS:
            assert not entry['exp_avg_sq'][OLD_KINDS:].any() and torch.equal(
                entry['exp_avg_sq'][:OLD_KINDS], source['optimizer']['state'][index]['exp_avg_sq'])
    assert registry.model_version(target) == _native.VERSION
    assert registry.act_mode(target) == registry.act_mode(run)
    assert (target.parent / registry.PROMOTED_NAME).read_text(encoding='utf-8').strip() == target.name
    info = json.loads((target / 'convert.json').read_text(encoding='utf-8'))
    assert info['source_sha1'] == _sha1(run / registry.CHECKPOINT_NAME)
    assert info['sha1'] == _sha1(target / registry.CHECKPOINT_NAME)


def test_anything_but_the_source_layout_is_refused(converted, tmp_path):
    run, target = converted
    with pytest.raises(convert.ConvertError, match='version stamp 1.4.2'):
        convert.convert_run(target, tmp_path / 'again')
    with pytest.raises(convert.ConvertError, match='exists'):
        convert.convert_run(run, target.parent.parent)
    source = registry._torch_load(run / registry.CHECKPOINT_NAME)    # pylint: disable=protected-access
    for edit, problem in ((lambda c: c['model_config']['obs_spec']['history'].update(shape=[36]), "'history'"),
                          (lambda c: c.update(action_schema_version=3), 'action_schema_version'),
                          (lambda c: c['state_dict'].update({'kind_head.bias': torch.zeros(9)}), 'kind_head.bias')):
        bad = json.loads(json.dumps({k: v for k, v in source.items() if k not in ('state_dict', 'optimizer')}))
        bad['state_dict'] = dict(source['state_dict'])
        edit(bad)
        with pytest.raises(convert.ConvertError, match=problem):
            convert.convert_checkpoint(bad, '1.4.1.0')
    with pytest.raises(convert.ConvertError, match='version stamp missing'):
        convert.convert_checkpoint(source, None)


def test_loading_an_unconverted_model_says_how_to_convert(converted):
    run, _ = converted
    with pytest.raises(registry.IncompatibleModelError, match='convert-model'):
        registry.require_compatible(run)
    with pytest.raises(registry.SchemaMismatchError, match='convert-model'):
        registry.load_checkpoint(run)
