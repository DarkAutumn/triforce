"""Widens checkpoints of the previous action and observation layout to the current one: ``triforce convert-model``.

    triforce [--models-dir DIR] convert-model (RUN_DIR... | --promoted) [--out DIR]

Round 24 (1.4.2.0) added the action kinds RECORDER, FOOD and CANDLE (kind-head indices 8-10, after CLIP; every
existing index kept) and their observation: the ``history`` kind one-hot grew from 8 to 11 (``kind_RECORDER``,
``kind_FOOD``, ``kind_CANDLE`` after ``kind_CLIP``) and ``link`` ends in ``recorder_usable``, ``food_usable``,
``candle_usable``. A 1.4.1 checkpoint (action schema 2, observation schema 4) is widened, not retrained:

- ``mlp_extractor.policy_net.0.weight`` and ``mlp_extractor.value_net.0.weight``: every old feature column moves to
  its feature name's new column; the added features' columns are zero (the network ignores them; the Rust policy
  drops all-zero columns, so it computes bit for bit what the source did).
- ``kind_head.weight`` / ``kind_head.bias``: the old rows kept; each new kind's row is WAIT's with the bias
  ``NEW_KIND_BIAS_BELOW_WAIT`` lower, so its logit is WAIT's minus 5 in every state. WAIT is legal in every state,
  so a new kind is never the argmax and is drawn at most ``e^-5`` (0.7%) as often as WAIT when a family first
  unmasks it, little enough not to disturb the model, enough for PPO to try it and learn from there.
- ``kind_embedding.weight`` (the kind's conditioning of the direction and duration heads): the old rows kept; FOOD
  and CANDLE start as BOMB's (placed ahead of Link in a cardinal direction, as a bomb is), RECORDER as WAIT's (it has
  neither head).
- ``model_config.obs_spec`` takes the current shapes (history 38, link 87, action_mask 143); the schema versions
  become the current ones; every other weight is copied unchanged.
- The Adam state (``optimizer``) is padded, not dropped: the changed tensors' ``exp_avg`` and ``exp_avg_sq`` get
  zeros where the weights got new rows or columns. ``train --load`` of a converted run so continues exactly as from
  the source; the new entries get no gradient while their kinds are masked (masked logits are filled, the new
  inputs are zero), and once a family unmasks them their first Adam steps are up to ~3x the learning rate (no second
  moment yet, the step count already high), settling within a few dozen updates.

The source run dir is left untouched: each converted run is a full copy, ``<out>/<specialist>/<run>/``, with the new
``ckpt.pt``, its ``version`` stamped with this code's, its ``act`` file re-bound to the new sha1 (same mode: the
converted model acts as the source did), the specialist's ``promoted`` pointer when it names the run, and
``convert.json`` (source sha1 and version). ``eval.json`` stays the source's (bound to the source sha1: measure the
converted run again for a canonical evaluation of its own). A checkpoint that is not exactly the old layout (schemas,
shapes, the feature names it implies, its version stamp) is refused. A widened layout added elsewhere in the same
release joins ``ADDED`` (names are matched; a name the current observation doesn't have is ignored).
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from . import __version__, registry
from ._native import ACTION_KINDS, ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION, OBS_SPEC
from .observation import HISTORY_FEATURES, LINK_FEATURES, OBJECTIVE_FEATURES, ROOM_STATE_FEATURES, TARGET_FEATURES

# What the convertible checkpoints were saved with.
SOURCE_VERSION = registry.CONVERTIBLE_VERSION
SOURCE_ACTION_SCHEMA = 2
SOURCE_OBS_SCHEMA = 4
SOURCE_KINDS = ('MOVE', 'WAIT', 'SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'CLIP')
# The observation features added since (per vector key, by name).
ADDED = {
    'history': ('kind_RECORDER', 'kind_FOOD', 'kind_CANDLE'),
    'link': ('recorder_usable', 'food_usable', 'candle_usable'),
    'room_state': ('maze_step',),
}
# A new kind's logit starts at WAIT's minus this.
NEW_KIND_BIAS_BELOW_WAIT = 5.0
# The kind whose embedding row each new kind starts from.
NEW_KIND_EMBEDDING_FROM = {'RECORDER': 'WAIT', 'FOOD': 'BOMB', 'CANDLE': 'BOMB'}
FIRST_LAYERS = ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight')
KIND_ROWS = ('kind_head.weight', 'kind_head.bias', 'kind_embedding.weight')
FEATURE_NAMES = {'history': HISTORY_FEATURES, 'link': LINK_FEATURES, 'objective': OBJECTIVE_FEATURES,
                 'room_state': ROOM_STATE_FEATURES, 'target': TARGET_FEATURES}
SPECIAL_KEYS = ('room_id', 'entity_types', 'entity_features', 'grid', 'action_mask')
SKIP_NAMES = (registry.CHECKPOINT_NAME, registry.ACT_NAME, registry.VERSION_NAME)


class ConvertError(ValueError):
    """The checkpoint is not one this converter widens."""


def old_names(key: str) -> list[str]:
    """The source layout's feature names of vector key ``key``: the current ones without the added ones."""
    added = set(ADDED.get(key, ()))
    return [name for name in FEATURE_NAMES[key] if name not in added]


def _spec(shape) -> list:
    return [int(n) for n in shape]


def check_source(checkpoint: dict, version) -> None:
    """Raises ConvertError unless ``checkpoint`` (and its run's ``version`` stamp) is exactly the source layout."""
    problems = []
    if version is None or version.split('.')[:3] != SOURCE_VERSION.split('.'):
        problems.append(f"version stamp {version or 'missing'} is not {SOURCE_VERSION}.x")
    if checkpoint.get('action_schema_version') != SOURCE_ACTION_SCHEMA:
        problems.append(f"action_schema_version {checkpoint.get('action_schema_version')} != {SOURCE_ACTION_SCHEMA}")
    if checkpoint.get('obs_schema_version') != SOURCE_OBS_SCHEMA:
        problems.append(f"obs_schema_version {checkpoint.get('obs_schema_version')} != {SOURCE_OBS_SCHEMA}")
    spec = (checkpoint.get('model_config') or {}).get('obs_spec') or {}
    if set(spec) != set(OBS_SPEC):
        problems.append(f"observation keys {sorted(spec)} != {sorted(OBS_SPEC)}")
    for key, entry in spec.items():
        current = OBS_SPEC.get(key)
        if current is None:
            continue
        if key in FEATURE_NAMES:
            want = [len(old_names(key))]
        elif key == 'action_mask':
            want = [len(SOURCE_KINDS) * (1 + 8 + 4)]
        else:
            want = _spec(current['shape'])
        if _spec(entry.get('shape', ())) != want or entry.get('vocab') != current['vocab']:
            problems.append(f"'{key}' is {entry}, the source layout has shape {want} vocab {current['vocab']}")
    state = checkpoint.get('state_dict') or {}
    kinds = len(SOURCE_KINDS)
    for name in KIND_ROWS:
        if name not in state or state[name].shape[0] != kinds:
            problems.append(f"{name} has {None if name not in state else tuple(state[name].shape)}, not {kinds} rows")
    if problems:
        raise ConvertError('; '.join(problems))


def feature_columns(prefix: int, spec: dict) -> tuple[int, list[int]]:
    """(the current feature count, where each source feature column lands among the current ones): the ``prefix``
    columns (room, entities, grid) stay first; the vector keys (sorted, as ``model.py`` concatenates them) map by
    feature name."""
    keys = sorted(k for k in spec if k not in SPECIAL_KEYS)
    new_at, at = {}, prefix
    for key in keys:
        names = FEATURE_NAMES[key]
        new_at[key] = {name: at + i for i, name in enumerate(names)}
        at += len(names)
    columns = list(range(prefix))
    for key in keys:
        columns += [new_at[key][name] for name in old_names(key)]
    return at, columns


def widen_state(state: dict, prefix: int, spec: dict) -> tuple[dict, dict]:
    """(the widened ``state_dict``, name -> (dim, positions of the source entries)) for every changed tensor."""
    width, columns = feature_columns(prefix, spec)
    kinds = list(ACTION_KINDS)
    old_rows = [kinds.index(k) for k in SOURCE_KINDS]
    if old_rows != list(range(len(SOURCE_KINDS))):
        raise ConvertError(f"the current kinds {kinds} don't keep the source's indices {SOURCE_KINDS}")
    new = dict(state)
    placed = {}
    for name in FIRST_LAYERS:
        old = state[name]
        if old.shape[1] != len(columns):
            raise ConvertError(f"{name} has {old.shape[1]} input columns, the source layout {len(columns)}")
        wide = torch.zeros(old.shape[0], width, dtype=old.dtype)
        wide[:, columns] = old
        new[name], placed[name] = wide, (1, columns)
    wait, rows = kinds.index('WAIT'), list(range(len(SOURCE_KINDS)))
    for name in KIND_ROWS:
        old = state[name]
        wide = torch.empty((len(kinds), *old.shape[1:]), dtype=old.dtype)
        wide[:len(SOURCE_KINDS)] = old
        for k in range(len(SOURCE_KINDS), len(kinds)):
            source = wait if name.startswith('kind_head') else kinds.index(NEW_KIND_EMBEDDING_FROM[kinds[k]])
            wide[k] = old[source]
            if name == 'kind_head.bias':
                wide[k] -= NEW_KIND_BIAS_BELOW_WAIT
        new[name], placed[name] = wide, (0, rows)
    return new, placed


def widen_optimizer(optimizer, state: dict, widened: dict, placed: dict, names: list[str]):
    """The Adam state dict with zero moments for the widened tensors' new entries (None stays None). ``names`` are
    the network's parameter names in ``parameters()`` order, which the state's indices follow."""
    if optimizer is None:
        return None
    entries = optimizer.get('state', {})
    if len(names) != len(optimizer['param_groups'][0]['params']) or len(optimizer['param_groups']) != 1:
        raise ConvertError("the optimizer state doesn't follow the network's parameters")
    out = {'state': {}, 'param_groups': [dict(group) for group in optimizer['param_groups']]}
    for index, entry in entries.items():
        name = names[int(index)]
        entry = dict(entry)
        for moment in ('exp_avg', 'exp_avg_sq'):
            value = entry[moment]
            if tuple(value.shape) != tuple(state[name].shape):
                raise ConvertError(f"optimizer {moment} of {name} is {tuple(value.shape)}, the weight "
                                   f"{tuple(state[name].shape)}")
            if name in placed:
                dim, positions = placed[name]
                wide = torch.zeros(widened[name].shape, dtype=value.dtype)
                index_ = [slice(None)] * wide.dim()
                index_[dim] = positions
                wide[tuple(index_)] = value
                entry[moment] = wide
        out['state'][index] = entry
    return out


def convert_checkpoint(checkpoint: dict, version) -> dict:
    """The widened checkpoint of a source ``checkpoint`` (dict) whose run is stamped ``version``."""
    from .model import ZeldaPolicy  # pylint: disable=import-outside-toplevel
    check_source(checkpoint, version)
    spec = {key: {'shape': _spec(entry['shape']), 'vocab': entry['vocab']} for key, entry in OBS_SPEC.items()}
    network = ZeldaPolicy(config={**checkpoint['model_config'], 'obs_spec': spec})
    prefix = network.feature_dim - sum(spec[key]['shape'][0] for key in network.vector_keys)
    state, placed = widen_state(checkpoint['state_dict'], prefix, spec)
    network.load_state_dict(state)
    names = [name for name, _ in network.named_parameters()]
    return {**checkpoint,
            'obs_schema_version': OBS_SCHEMA_VERSION,
            'action_schema_version': ACTION_SCHEMA_VERSION,
            'model_config': network.config.to_dict(),
            'state_dict': {name: value.contiguous() for name, value in state.items()},
            'optimizer': widen_optimizer(checkpoint.get('optimizer'), checkpoint['state_dict'], state, placed, names)}


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def convert_run(run_dir, out_root) -> Path:
    """Converts the run at ``run_dir`` (``<models>/<specialist>/<run>``) into ``<out_root>/<specialist>/<run>``:
    a full copy with the widened ``ckpt.pt``; returns the new run dir. The source is only read."""
    run_dir = Path(run_dir)
    source = run_dir / registry.CHECKPOINT_NAME
    if not source.is_file():
        raise ConvertError(f"{run_dir}: no {registry.CHECKPOINT_NAME}")
    checkpoint = registry._torch_load(source)      # pylint: disable=protected-access
    found = registry.model_version(source)
    try:
        converted = convert_checkpoint(checkpoint, found)
    except ConvertError as error:
        raise ConvertError(f"{run_dir}: {error}") from None
    target = Path(out_root) / run_dir.parent.name / run_dir.name
    if target.resolve() == run_dir.resolve():
        raise ConvertError(f"{run_dir}: --out must not be the models dir the run is in")
    if target.exists():
        raise ConvertError(f"{target} exists")
    target.mkdir(parents=True)
    for item in run_dir.iterdir():
        if item.name in SKIP_NAMES:
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, target / item.name)
    checkpoint_out = target / registry.CHECKPOINT_NAME
    torch.save(converted, checkpoint_out)
    registry.write_model_version(checkpoint_out)
    act = run_dir / registry.ACT_NAME
    if act.is_file():
        fields = act.read_text(encoding='utf-8').split()
        if len(fields) == 2 and fields[1] == _sha1(source):
            registry.set_act_mode(checkpoint_out, fields[0])
        else:
            shutil.copy2(act, target / registry.ACT_NAME)
    pointer = run_dir.parent / registry.PROMOTED_NAME
    if pointer.is_file() and pointer.read_text(encoding='utf-8').strip() == run_dir.name:
        shutil.copy2(pointer, target.parent / registry.PROMOTED_NAME)
    (target / 'convert.json').write_text(json.dumps({
        'source_sha1': _sha1(source), 'source_version': found, 'version': __version__,
        'sha1': _sha1(checkpoint_out), 'action_schema_version': ACTION_SCHEMA_VERSION,
        'obs_schema_version': OBS_SCHEMA_VERSION, 'new_kind_bias_below_wait': NEW_KIND_BIAS_BELOW_WAIT,
        'added_features': {key: [n for n in names if n in FEATURE_NAMES.get(key, ())] for key, names in ADDED.items()},
    }, indent=2) + '\n', encoding='utf-8')
    return target


def main(argv=None) -> int:
    """``triforce convert-model``: exit 0 when every run converted, 1 when any was refused, 2 on bad arguments."""
    parser = argparse.ArgumentParser(prog='triforce convert-model', description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('runs', nargs='*', help="run dirs (<models>/<specialist>/<run>)")
    parser.add_argument('--promoted', action='store_true', help="every promoted run of the models dir")
    parser.add_argument('--models-dir', default=None, help="for --promoted: default $TRIFORCE_MODELS_DIR or "
                                                            "<repo>/models")
    parser.add_argument('--out', required=True, help="models root the converted runs are written to "
                                                      "(<out>/<specialist>/<run>)")
    args = parser.parse_args(argv)
    runs = [Path(r) for r in args.runs]
    if args.promoted:
        from .specialist import list_specialists  # pylint: disable=import-outside-toplevel
        for name in list_specialists():
            try:
                run = registry.promoted_run(name, args.models_dir)
            except FileNotFoundError:
                continue
            runs.append(registry.models_root(args.models_dir) / name / run)
    if not runs:
        parser.error("give run dirs or --promoted")
    failed = False
    for run in runs:
        try:
            print(f"{run} -> {convert_run(run, args.out)}")
        except ConvertError as error:
            print(f"triforce convert-model: error: {error}", file=sys.stderr)
            failed = True
    return 1 if failed else 0
