"""``triforce convert-model`` (``triforce/convert.py``): a checkpoint of the previous observation layout widened by
feature name acts exactly as its source (the Rust policy, bit for bit, whatever the added columns hold), its Adam
state is padded with zeros, the source is untouched, and anything but the source layout is refused.

Two layouts: ``synthetic`` (the current layout as the source, with columns added in the middle and at the end of
``entity_features``, ``link`` and ``room_state``: the mechanism, whatever the release adds) and ``current`` (the
release's own ``convert.ADDED``). The real-run tests convert the first promoted run of the source version under
``$TRIFORCE_CONVERT_SOURCE`` (a models root), else the models root (``registry.models_root``); without one they skip
(a fresh clone has no models)."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from triforce import _native, convert, registry
from triforce.model import ZeldaPolicy

from .conftest import REPO_STATES_DIR

WIDENED = (convert.ENTITY_KEY, 'link', 'room_state')
# Synthetic additions: (key, index in the widened names, name).
SYNTHETIC = ((convert.ENTITY_KEY, 5, 'test_entity_mid'), (convert.ENTITY_KEY, None, 'test_entity_a'),
             (convert.ENTITY_KEY, None, 'test_entity_b'), ('link', 3, 'test_link_mid'), ('link', None, 'test_link_end'),
             ('room_state', 40, 'test_room_mid'))


def _current_added() -> bool:
    """Whether this code's layout holds every name of ``convert.ADDED`` (the release that adds them)."""
    return all(name in convert.FEATURE_NAMES[key] for key, names in convert.ADDED.items() for name in names)


@pytest.fixture(params=['synthetic', 'current'])
def layout(request, monkeypatch):
    """The converter's target layout: ``synthetic`` patches ``convert`` to the current layout plus ``SYNTHETIC``."""
    if request.param == 'current':
        if not _current_added():
            pytest.skip("this code's layout is still the source layout")
        return request.param
    names = {key: list(value) for key, value in convert.FEATURE_NAMES.items()}
    for key, index, name in SYNTHETIC:
        names[key].insert(len(names[key]) if index is None else index, name)
    spec = {key: dict(entry) for key, entry in convert.OBS_SPEC.items()}
    for key in WIDENED:
        shape = list(spec[key]['shape'])
        shape[-1] = len(names[key])
        spec[key]['shape'] = shape
    added = {key: tuple(name for k, _, name in SYNTHETIC if k == key) for key in WIDENED}
    monkeypatch.setattr(convert, 'FEATURE_NAMES', names)
    monkeypatch.setattr(convert, 'OBS_SPEC', spec)
    monkeypatch.setattr(convert, 'ADDED', added)
    monkeypatch.setattr(convert, 'SOURCE_OBS_SCHEMA', _native.OBS_SCHEMA_VERSION)
    return request.param


def _source_checkpoint(seed: int) -> dict:
    """A random network of the source layout with a random Adam state."""
    spec = {key: {'shape': convert.source_shape(key), 'vocab': entry['vocab']}
            for key, entry in convert.OBS_SPEC.items()}
    torch.manual_seed(seed)
    network = ZeldaPolicy(config={'obs_spec': spec})
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    params = list(network.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4).state_dict()
    optimizer['state'] = {i: {'step': torch.tensor(7.0), 'exp_avg': torch.randn_like(p), 'exp_avg_sq': torch.rand_like(p)}
                          for i, p in enumerate(params)}
    return {'specialist': 'goriya', 'obs_schema_version': convert.SOURCE_OBS_SCHEMA,
            'action_schema_version': convert.SOURCE_ACTION_SCHEMA, 'model_config': network.config.to_dict(),
            'state_dict': network.state_dict(), 'optimizer': optimizer}


def _native_policy(checkpoint: dict):
    weights = {name: value.float().numpy() for name, value in checkpoint['state_dict'].items()}
    return _native.Policy(json.dumps(checkpoint['model_config']), weights)


def _random_source_obs(rows: int, rng) -> dict:
    """Observations of the source layout: random values in every float column, some empty slots."""
    obs = {}
    for key, entry in convert.OBS_SPEC.items():
        shape = [rows, *convert.source_shape(key)]
        if key == 'action_mask':
            obs[key] = np.ones(shape, dtype=bool)
        elif entry['vocab'] is not None:
            obs[key] = rng.integers(0, entry['vocab'], size=shape, dtype=np.int64)
        else:
            obs[key] = rng.uniform(-1, 1, size=shape).astype(np.float32)
    presence = obs[convert.ENTITY_KEY][..., 0]
    presence[:] = rng.integers(0, 2, size=presence.shape)
    return obs


def _widen_obs(old: dict, rng) -> dict:
    """``old`` in the current layout: each old column at its name's place, random values in the added ones."""
    new = dict(old)
    for key in WIDENED:
        names = convert.FEATURE_NAMES[key]
        value = rng.uniform(-1, 1, size=(*old[key].shape[:-1], len(names))).astype(np.float32)
        value[..., [names.index(n) for n in convert.old_names(key)]] = old[key]
        new[key] = value
    return new


def _same_bits(a: dict, b: dict):
    for key in ('actions', 'log_prob', 'entropy', 'value', 'kind_log_probs', 'direction_log_probs',
                'duration_log_probs', 'head_log_prob', 'head_entropy'):
        assert np.asarray(a[key]).tobytes() == np.asarray(b[key]).tobytes(), key


def test_the_converted_model_acts_as_the_source_bit_for_bit(layout):
    source = _source_checkpoint(1)
    converted = convert.convert_checkpoint(source, f'{convert.SOURCE_VERSION}.0')
    old, new = _native_policy(source), _native_policy(converted)
    added = sum(len(names) for names in convert.ADDED.values())
    assert added > 0
    rng = np.random.default_rng(2)
    rows = 64
    obs_old = _random_source_obs(rows, rng)
    obs_new = _widen_obs(obs_old, rng)
    assert all(obs_new[key].shape[-1] > obs_old[key].shape[-1] for key in WIDENED), layout
    _same_bits(old.evaluate(obs_old, deterministic=True), new.evaluate(obs_new, deterministic=True))
    noise = _native.keyed_noise(list(range(rows)), [3] * rows, old.noise_size)
    _same_bits(old.evaluate(obs_old, noise=noise), new.evaluate(obs_new, noise=noise))
    for a, b in zip(old.joint_log_probs(obs_old), new.joint_log_probs(obs_new)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_weights_and_adam_moments_are_placed_by_name(layout):
    source = _source_checkpoint(3)
    converted = convert.convert_checkpoint(source, f'{convert.SOURCE_VERSION}.0')
    old, new = source['state_dict'], converted['state_dict']
    assert converted['obs_schema_version'] == _native.OBS_SCHEMA_VERSION
    assert {k: v['shape'] for k, v in converted['model_config']['obs_spec'].items()} == \
        {k: list(v['shape']) for k, v in convert.OBS_SPEC.items()}
    embed = converted['model_config']['entity_embed_dim']
    entity = convert.FEATURE_NAMES[convert.ENTITY_KEY]
    columns = {convert.ENTITY_LAYER: [entity.index(n) for n in convert.old_names(convert.ENTITY_KEY)]
               + [len(entity) + j for j in range(embed)]}
    prefix = old[convert.FIRST_LAYERS[0]].shape[1] - sum(len(convert.old_names(k)) for k in
                                                         sorted(convert.FEATURE_NAMES) if k != convert.ENTITY_KEY)
    for name in convert.FIRST_LAYERS:
        columns[name] = convert.feature_columns(prefix, convert.OBS_SPEC)[1]
    names = [n for n, _ in ZeldaPolicy(config=converted['model_config']).named_parameters()]
    for name, cols in columns.items():
        added = [c for c in range(new[name].shape[1]) if c not in cols]
        assert len(added) == new[name].shape[1] - old[name].shape[1] > 0
        assert torch.equal(new[name][:, cols], old[name]) and not new[name][:, added].any(), name
        index = names.index(name)
        for moment in ('exp_avg', 'exp_avg_sq'):
            wide = converted['optimizer']['state'][index][moment]
            assert wide.shape == new[name].shape
            assert torch.equal(wide[:, cols], source['optimizer']['state'][index][moment])
            assert not wide[:, added].any(), (name, moment)
    for name, value in old.items():
        if name not in columns:
            assert torch.equal(new[name], value), name


def test_anything_but_the_source_layout_is_refused(layout):
    source = _source_checkpoint(4)
    version = f'{convert.SOURCE_VERSION}.0'
    converted = convert.convert_checkpoint(source, version)
    with pytest.raises(convert.ConvertError, match='the source layout has shape'):
        convert.convert_checkpoint(converted, version)
    for edit, problem in (
            (lambda c: c['model_config']['obs_spec']['link'].update(shape=[7]), "'link'"),
            (lambda c: c['model_config']['obs_spec'][convert.ENTITY_KEY].update(shape=[12, 30]), "'entity_features'"),
            (lambda c: c.update(action_schema_version=2), 'action_schema_version'),
            (lambda c: c.update(obs_schema_version=4), 'obs_schema_version'),
            (lambda c: c['state_dict'].update({'kind_head.bias': torch.zeros(8)}), 'kind_head.bias')):
        bad = json.loads(json.dumps({k: v for k, v in source.items() if k not in ('state_dict', 'optimizer')}))
        bad['state_dict'] = dict(source['state_dict'])
        edit(bad)
        with pytest.raises(convert.ConvertError, match=problem):
            convert.convert_checkpoint(bad, version)
    with pytest.raises(convert.ConvertError, match='version stamp missing'):
        convert.convert_checkpoint(source, None)
    with pytest.raises(convert.ConvertError, match='version stamp 1.4.2.0'):
        convert.convert_checkpoint(source, '1.4.2.0')


# --------------------------------------------------------------------------------------- a real promoted run


def _source_root() -> Path:
    return Path(os.environ.get('TRIFORCE_CONVERT_SOURCE') or registry.models_root())


def _source_run() -> Path:
    """A promoted run stamped with the source version (the smallest family that has one), else skip."""
    if not _current_added():
        pytest.skip("this code's layout is still the source layout")
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
def converted_run(tmp_path_factory):
    run = _source_run()
    before = _sha1(run / registry.CHECKPOINT_NAME)
    target = convert.convert_run(run, tmp_path_factory.mktemp('converted'))
    assert _sha1(run / registry.CHECKPOINT_NAME) == before, "the source is untouched"
    return run, target


def _old_layout(obs: dict) -> dict:
    """Current observations in the source layout: the added columns of every widened key removed."""
    old = dict(obs)
    for key in WIDENED:
        names = convert.FEATURE_NAMES[key]
        old[key] = obs[key][..., [names.index(n) for n in convert.old_names(key)]]
    return old


def test_a_converted_run_acts_as_its_source_on_real_observations(converted_run):
    run, target = converted_run
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
    entity = convert.FEATURE_NAMES[convert.ENTITY_KEY]
    added = [entity.index(n) for n in convert.ADDED[convert.ENTITY_KEY]]
    assert observations[convert.ENTITY_KEY][..., added].any(), "the new entity columns are computed"
    n = len(rows)
    for pick in ({'deterministic': True}, {'noise': _native.keyed_noise(list(range(n)), [2] * n, old.noise_size)}):
        _same_bits(old.evaluate(_old_layout(observations), **pick), new.evaluate(observations, **pick))


def test_a_converted_run_is_stamped_and_keeps_its_files(converted_run):
    run, target = converted_run
    assert registry.model_version(target) == _native.VERSION
    assert registry.act_mode(target) == registry.act_mode(run)
    assert (target.parent / registry.PROMOTED_NAME).read_text(encoding='utf-8').strip() == target.name
    info = json.loads((target / 'convert.json').read_text(encoding='utf-8'))
    assert info['source_sha1'] == _sha1(run / registry.CHECKPOINT_NAME)
    assert info['sha1'] == _sha1(target / registry.CHECKPOINT_NAME)
    assert info['added_features'] == {key: list(names) for key, names in convert.ADDED.items()}
    with pytest.raises(convert.ConvertError, match=f'version stamp {_native.VERSION}'):
        convert.convert_run(target, target.parent.parent.parent / 'again')
    with pytest.raises(convert.ConvertError, match='exists'):
        convert.convert_run(run, target.parent.parent)


def test_loading_an_unconverted_model_says_how_to_convert(converted_run):
    run, _ = converted_run
    with pytest.raises(registry.IncompatibleModelError, match='convert-model'):
        registry.require_compatible(run)
    with pytest.raises(registry.SchemaMismatchError, match='convert-model'):
        registry.load_checkpoint(run)
    network, _ = registry.load_model(run, check_schema=False)
    current = {key: {'shape': list(entry['shape']), 'vocab': entry['vocab']} for key, entry in _native.OBS_SPEC.items()}
    with pytest.raises(ValueError, match='convert-model'):
        from triforce.model import widen_policy  # pylint: disable=import-outside-toplevel
        widen_policy(network, current)
