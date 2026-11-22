"""Widens checkpoints of the previous observation layout to the current one: ``triforce convert-model``.

    triforce [--models-dir DIR] convert-model (RUN_DIR... | --promoted) [--out DIR]

Round 26 (1.5.1.0, observation schema 6; decisions 2026-11-16 "Schema 6: the one observation change for the rest of
the game") added inputs to three blocks and changed no existing column: 8 entity features after the 33 of schema 5
(``touch_harms``, ``weapon_target``, ``touch_damage``, ``visible``, ``phase``, ``phase_time``, ``in_line``,
``ringleader``), 6 at the end of ``link`` (``sword_blocked``, ``sword_blocked_time``, ``arrow_usable``,
``rupee_reserve``, ``rupees_spare``, ``rupees_few``) and 4 at the end of ``room_state`` (``frame_phase_32``,
``frame_phase_64``, ``wallmaster_cooldown``, ``wallmaster_lure``). A 1.5.0 checkpoint (observation schema 5, action
schema 3) is widened, not retrained:

- ``entity_encoder.input_proj.weight`` (the per-slot token: the entity features, then the type embedding): every old
  feature column moves to its feature name's new column, the type embedding's columns follow the new features; the
  added features' columns are zero.
- ``mlp_extractor.policy_net.0.weight`` and ``mlp_extractor.value_net.0.weight``: every old vector feature column
  moves to its feature name's new column; the added ones are zero.
- The Rust policy drops all-zero input columns of all three layers (``Policy::live``, ``Policy::entity_live``), so a
  converted model computes bit for bit what the source did whatever the new columns hold (in torch the zero columns
  add exact zeros too).
- ``model_config.obs_spec`` takes the current shapes; the observation schema becomes the current one; every other
  weight is copied unchanged.
- The Adam state (``optimizer``) is padded, not dropped: the widened tensors' ``exp_avg`` and ``exp_avg_sq`` get zeros
  in the new columns. ``train --load`` of a converted run so continues exactly as from the source; a new column gets
  gradient as soon as its input is non-zero, and its first Adam steps are up to ~3x the learning rate (no second
  moment yet, the step count already high), settling within a few dozen updates (as round 24's).

The source run dir is left untouched: each converted run is a full copy, ``<out>/<specialist>/<run>/``, with the new
``ckpt.pt``, its ``version`` stamped with this code's, its ``act`` file re-bound to the new sha1 (same mode: the
converted model acts as the source did), the specialist's ``promoted`` pointer when it names the run, and
``convert.json`` (source sha1 and version, the added features). ``eval.json`` stays the source's (bound to the source
sha1: measure the converted run again for a canonical evaluation of its own). A checkpoint that is not exactly the old
layout (version stamp, schemas, shapes, the feature names it implies, the action heads) is refused. Inputs added
elsewhere in the same release join ``ADDED`` (names are matched; a name the current observation doesn't have is
ignored).
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from . import __version__, registry
from ._native import ACTION_KINDS, ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION
from ._native import OBS_SPEC as _OBS_SPEC
from .model import policy_spec
from .observation import (ENTITY_FEATURES, HISTORY_FEATURES, LINK_FEATURES, OBJECTIVE_FEATURES, ROOM_STATE_FEATURES,
                          TARGET_FEATURES)

# The layout a ``ZeldaPolicy`` reads (schema 7's token keys are not its).
OBS_SPEC = policy_spec(_OBS_SPEC)

# What the convertible checkpoints were saved with.
SOURCE_VERSION = registry.CONVERTIBLE_VERSION
SOURCE_ACTION_SCHEMA = 3
SOURCE_OBS_SCHEMA = registry.CONVERTIBLE_OBS_SCHEMA
# The observation features added since (per key, by name).
ADDED = {
    'entity_features': ('touch_harms', 'weapon_target', 'touch_damage', 'visible', 'phase', 'phase_time', 'in_line',
                        'ringleader'),
    'link': ('sword_blocked', 'sword_blocked_time', 'arrow_usable', 'rupee_reserve', 'rupees_spare', 'rupees_few'),
    'room_state': ('frame_phase_32', 'frame_phase_64', 'wallmaster_cooldown', 'wallmaster_lure'),
}
FIRST_LAYERS = ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight')
ENTITY_LAYER = 'entity_encoder.input_proj.weight'
ENTITY_KEY = 'entity_features'
FEATURE_NAMES = {ENTITY_KEY: ENTITY_FEATURES, 'history': HISTORY_FEATURES, 'link': LINK_FEATURES,
                 'objective': OBJECTIVE_FEATURES, 'room_state': ROOM_STATE_FEATURES, 'target': TARGET_FEATURES}
SPECIAL_KEYS = ('room_id', 'entity_types', 'entity_features', 'grid', 'action_mask')
SKIP_NAMES = (registry.CHECKPOINT_NAME, registry.ACT_NAME, registry.VERSION_NAME)


class ConvertError(ValueError):
    """The checkpoint is not one this converter widens."""


def old_names(key: str) -> list[str]:
    """The source layout's feature names of key ``key``: the current ones without the added ones."""
    added = set(ADDED.get(key, ()))
    return [name for name in FEATURE_NAMES[key] if name not in added]


def _spec(shape) -> list:
    return [int(n) for n in shape]


def source_shape(key: str) -> list:
    """The shape of observation key ``key`` in the source layout."""
    current = _spec(OBS_SPEC[key]['shape'])
    if key == ENTITY_KEY:
        return [current[0], len(old_names(key))]
    if key in FEATURE_NAMES:
        return [len(old_names(key))]
    return current


def check_source(checkpoint: dict, version) -> None:
    """Raises ConvertError unless ``checkpoint`` (and its run's ``version`` stamp) is exactly the source layout."""
    problems = []
    if version is None or version.split('.')[:3] != SOURCE_VERSION.split('.'):
        problems.append(f"version stamp {version or 'missing'} is not {SOURCE_VERSION}.x")
    if ACTION_SCHEMA_VERSION != SOURCE_ACTION_SCHEMA:
        problems.append(f"this code's action schema {ACTION_SCHEMA_VERSION} is not the source's {SOURCE_ACTION_SCHEMA}"
                        " (this converter widens observations only)")
    if checkpoint.get('action_schema_version') != SOURCE_ACTION_SCHEMA:
        problems.append(f"action_schema_version {checkpoint.get('action_schema_version')} != {SOURCE_ACTION_SCHEMA}")
    if checkpoint.get('obs_schema_version') != SOURCE_OBS_SCHEMA:
        problems.append(f"obs_schema_version {checkpoint.get('obs_schema_version')} != {SOURCE_OBS_SCHEMA}")
    spec = (checkpoint.get('model_config') or {}).get('obs_spec') or {}
    if set(spec) != set(OBS_SPEC):
        problems.append(f"observation keys {sorted(spec)} != {sorted(OBS_SPEC)}")
    for key, entry in spec.items():
        if key not in OBS_SPEC:
            continue
        want, vocab = source_shape(key), OBS_SPEC[key]['vocab']
        if _spec(entry.get('shape', ())) != want or entry.get('vocab') != vocab:
            problems.append(f"'{key}' is {entry}, the source layout has shape {want} vocab {vocab}")
    state = checkpoint.get('state_dict') or {}
    kinds, bias = len(ACTION_KINDS), state.get('kind_head.bias')
    if bias is None or bias.shape[0] != kinds:
        problems.append(f"kind_head.bias has {None if bias is None else tuple(bias.shape)}, not {kinds} rows")
    if problems:
        raise ConvertError('; '.join(problems))


def feature_columns(prefix: int, spec: dict) -> tuple[int, list[int]]:
    """(the current tower feature count, where each source feature column lands among the current ones): the
    ``prefix`` columns (room, entities, grid) stay first; the vector keys (sorted, as ``model.py`` concatenates them)
    map by feature name."""
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


def entity_columns(embed: int) -> tuple[int, list[int]]:
    """(the current entity token width, where each source token column lands): the features by name, then the
    ``embed`` type embedding columns after the current features (``model.EntityEncoder`` concatenates them so)."""
    names = FEATURE_NAMES[ENTITY_KEY]
    at = {name: i for i, name in enumerate(names)}
    return len(names) + embed, [at[name] for name in old_names(ENTITY_KEY)] + [len(names) + j for j in range(embed)]


def _widen_columns(old: torch.Tensor, width: int, columns: list[int], name: str) -> torch.Tensor:
    if old.shape[1] != len(columns):
        raise ConvertError(f"{name} has {old.shape[1]} input columns, the source layout {len(columns)}")
    wide = torch.zeros(old.shape[0], width, dtype=old.dtype)
    wide[:, columns] = old
    return wide


def widen_state(state: dict, prefix: int, spec: dict, embed: int) -> tuple[dict, dict]:
    """(the widened ``state_dict``, name -> (dim, positions of the source entries)) for every changed tensor."""
    new = dict(state)
    placed = {}
    width, columns = feature_columns(prefix, spec)
    for name in FIRST_LAYERS:
        new[name], placed[name] = _widen_columns(state[name], width, columns, name), (1, columns)
    width, columns = entity_columns(embed)
    new[ENTITY_LAYER], placed[ENTITY_LAYER] = _widen_columns(state[ENTITY_LAYER], width, columns, ENTITY_LAYER), \
        (1, columns)
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
    state, placed = widen_state(checkpoint['state_dict'], prefix, spec, network.config.entity_embed_dim)
    network.load_state_dict(state)
    names = [name for name, _ in network.named_parameters()]
    return {**checkpoint,
            'obs_schema_version': OBS_SCHEMA_VERSION,
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
        'source_obs_schema_version': SOURCE_OBS_SCHEMA, 'obs_schema_version': OBS_SCHEMA_VERSION,
        'added_features': {key: [n for n in names if n in FEATURE_NAMES[key]] for key, names in ADDED.items()},
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
