"""Model registry (design §9): ``<models>/<specialist>/<run-id>/ckpt.pt`` plus ``<models>/<specialist>/promoted``.

A checkpoint is a dict with ``obs_schema_version``, ``action_schema_version``, ``specialist``, ``model_config``
(including the observation layout, so the network can be rebuilt from it alone), ``state_dict``,
``optimizer``, ``iterations``, ``steps``, ``metrics``.  Loading a checkpoint raises ``SchemaMismatchError`` unless
its action schema is the current one and its observation schema is the current one or an older one whose layout
(``model_config['obs_spec']``) is a subset of the current ``OBS_SPEC``: every key it reads still exists with the
same shape and vocabulary (schema 1 is schema 2 without ``target``).  The network reads only its own keys.

The models root is, in order: the ``root`` argument, ``$TRIFORCE_MODELS_DIR``, ``<repo>/models``.

How a run's model acts (better-play §6b) is ``<run dir>/act``: ``sample`` (keyed sampling, the default when the
file is absent) or ``argmax``, chosen by evaluating both and set with ``promote --act``.  Chains, the watch player
and evaluation follow it unless told otherwise.

CLI: ``python -m triforce.registry list <specialist>`` / ``promote <specialist> <run-id> [--act MODE]``.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch

from ._native import ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION, OBS_SPEC

MODELS_DIR_ENV = 'TRIFORCE_MODELS_DIR'
CHECKPOINT_NAME = 'ckpt.pt'
PROMOTED_NAME = 'promoted'
ACT_NAME = 'act'
ACT_MODES = ('sample', 'argmax')
ACT_CHOICES = ('model',) + ACT_MODES


def act_override(choice: str) -> Optional[bool]:
    """A command line ``--act`` choice as a ``deterministic`` setting: None for ``model`` (each model's act mode),
    else whether every model acts by argmax."""
    if choice not in ACT_CHOICES:
        raise ValueError(f"--act {choice!r} is not one of {ACT_CHOICES}")
    return None if choice == 'model' else choice == 'argmax'


class SchemaMismatchError(ValueError):
    """A checkpoint was saved with a different observation or action schema."""


def obs_schema_version() -> int:
    """The current observation schema version (the native observation builder's)."""
    return OBS_SCHEMA_VERSION


def models_root(root: Optional[str | os.PathLike] = None) -> Path:
    """Resolves the models directory."""
    if root is not None:
        return Path(root)
    if os.environ.get(MODELS_DIR_ENV):
        return Path(os.environ[MODELS_DIR_ENV])
    return Path(__file__).resolve().parent.parent / 'models'


def new_run_id() -> str:
    """A sortable, unique-enough run id."""
    return time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid() % 10000:04d}'


def run_dir(specialist: str, run_id: str, root=None) -> Path:
    """``<models>/<specialist>/<run-id>``."""
    return models_root(root) / specialist / run_id


def checkpoint_path(specialist: str, run_id: str, root=None) -> Path:
    """``<models>/<specialist>/<run-id>/ckpt.pt``."""
    return run_dir(specialist, run_id, root) / CHECKPOINT_NAME


def save_checkpoint(specialist: str, run_id: str, network, optimizer=None, *, iterations: int = 0, steps: int = 0,
                    metrics: Optional[dict] = None, root=None, extra: Optional[dict] = None) -> Path:
    """Writes ``ckpt.pt`` atomically and returns its path.  ``network`` is a ``model.ZeldaPolicy``."""
    # pylint: disable=too-many-arguments
    path = checkpoint_path(specialist, run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'obs_schema_version': obs_schema_version(),
        'action_schema_version': ACTION_SCHEMA_VERSION,
        'specialist': specialist,
        'run_id': run_id,
        'model_config': network.config.to_dict(),
        'state_dict': {k: v.detach().cpu() for k, v in network.state_dict().items()},
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'iterations': int(iterations),
        'steps': int(steps),
        'metrics': dict(metrics or {}),
        **(extra or {}),
    }
    tmp = path.with_suffix('.tmp')
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)
    return path


def _resolve(path) -> Path:
    path = Path(path)
    return path / CHECKPOINT_NAME if path.is_dir() else path


def _obs_layout_problem(checkpoint: dict) -> Optional[str]:
    """Why an older checkpoint's observation layout can't read current observations (None if it can)."""
    spec = (checkpoint.get('model_config') or {}).get('obs_spec')
    if not spec:
        return "no obs_spec in model_config"
    for key, entry in spec.items():
        current = OBS_SPEC.get(key)
        if current is None:
            return f"key '{key}' is not in the current observation"
        if list(entry.get('shape', ())) != list(current['shape']) or entry.get('vocab') != current['vocab']:
            return (f"key '{key}' is shape {list(entry.get('shape', ()))} vocab {entry.get('vocab')}, "
                    f"current is shape {list(current['shape'])} vocab {current['vocab']}")
    return None


def load_checkpoint(path, check_schema: bool = True) -> dict:
    """Loads a checkpoint (file or run directory).  Raises SchemaMismatchError when the action schema differs or
    the observation schema is newer, or older with a layout that isn't a subset of the current one."""
    path = _resolve(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if check_schema:
        action = checkpoint.get('action_schema_version')
        if action != ACTION_SCHEMA_VERSION:
            raise SchemaMismatchError(f"{path}: action_schema_version is {action}, "
                                      f"code expects {ACTION_SCHEMA_VERSION}")
        obs = checkpoint.get('obs_schema_version')
        if obs != obs_schema_version():
            problem = (_obs_layout_problem(checkpoint) if isinstance(obs, int) and obs < obs_schema_version()
                       else "not an older schema")
            if problem is not None:
                raise SchemaMismatchError(f"{path}: obs_schema_version is {obs}, code expects "
                                          f"{obs_schema_version()} ({problem})")
    return checkpoint


def load_model(path, check_schema: bool = True, obs_spec: Optional[dict] = None):
    """Rebuilds the ``ZeldaPolicy`` stored in a checkpoint.  Returns (network, checkpoint).  With ``obs_spec`` (the
    layout the caller wants) a checkpoint built for a smaller layout comes back widened to it
    (``model.widen_policy``: new inputs start at zero); compare ``network.config.obs_spec`` with the checkpoint's
    to tell."""
    from .model import ZeldaPolicy, widen_policy  # pylint: disable=import-outside-toplevel
    checkpoint = load_checkpoint(path, check_schema)
    network = ZeldaPolicy(config=checkpoint['model_config'])
    network.load_state_dict(checkpoint['state_dict'])
    if obs_spec is not None and obs_spec != network.config.obs_spec:
        network = widen_policy(network, obs_spec)
    return network, checkpoint


def list_runs(specialist: str, root=None) -> list[str]:
    """Run ids that have a checkpoint, oldest first."""
    base = models_root(root) / specialist
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / CHECKPOINT_NAME).is_file())


def promote(specialist: str, run_id: str, root=None, act: Optional[str] = None) -> Path:
    """Marks run_id as the specialist's promoted model (``<models>/<specialist>/promoted`` holds the id); ``act``
    also sets how the run acts (``set_act_mode``)."""
    ckpt = checkpoint_path(specialist, run_id, root)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint for {specialist}/{run_id}: {ckpt}")
    load_checkpoint(ckpt)
    if act is not None:
        set_act_mode(ckpt, act)
    pointer = models_root(root) / specialist / PROMOTED_NAME
    tmp = pointer.with_suffix('.tmp')
    tmp.write_text(run_id + '\n', encoding='utf-8')
    os.replace(tmp, pointer)
    return pointer


def act_mode(checkpoint) -> str:
    """How the model in ``checkpoint`` (file or run dir) acts: its run dir's ``act``, ``sample`` without one."""
    path = _resolve(checkpoint).parent / ACT_NAME
    if not path.is_file():
        return 'sample'
    mode = path.read_text(encoding='utf-8').strip()
    if mode not in ACT_MODES:
        raise ValueError(f"{path}: act mode {mode!r} is not one of {ACT_MODES}")
    return mode


def set_act_mode(checkpoint, mode: str) -> Path:
    """Sets how the model in ``checkpoint`` (file or run dir) acts; returns the ``act`` file."""
    if mode not in ACT_MODES:
        raise ValueError(f"act mode {mode!r} is not one of {ACT_MODES}")
    path = _resolve(checkpoint).parent / ACT_NAME
    path.write_text(mode + '\n', encoding='utf-8')
    return path


def promoted_run(specialist: str, root=None) -> str:
    """The promoted run id.  Raises FileNotFoundError if none is promoted."""
    pointer = models_root(root) / specialist / PROMOTED_NAME
    if not pointer.is_file():
        raise FileNotFoundError(f"No promoted model for specialist '{specialist}' ({pointer})")
    return pointer.read_text(encoding='utf-8').strip()


def promoted_checkpoint(specialist: str, root=None) -> Path:
    """Path of the promoted run's ``ckpt.pt``."""
    path = checkpoint_path(specialist, promoted_run(specialist, root), root)
    if not path.is_file():
        raise FileNotFoundError(f"Promoted run for '{specialist}' has no checkpoint: {path}")
    return path


def main(argv=None):
    """``list`` / ``promote`` subcommands."""
    parser = argparse.ArgumentParser(prog='python -m triforce.registry')
    parser.add_argument('--models-dir', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list').add_argument('specialist')
    promote_parser = sub.add_parser('promote')
    promote_parser.add_argument('specialist')
    promote_parser.add_argument('run_id')
    promote_parser.add_argument('--act', choices=ACT_MODES, default=None, help="also set how the run acts")
    args = parser.parse_args(argv)

    if args.command == 'list':
        try:
            current = promoted_run(args.specialist, args.models_dir)
        except FileNotFoundError:
            current = None
        for run_id in list_runs(args.specialist, args.models_dir):
            path = checkpoint_path(args.specialist, run_id, args.models_dir)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            mark = '*' if run_id == current else ' '
            success = ckpt.get('metrics', {}).get('success_rate')
            print(f"{mark} {run_id}  iterations={ckpt.get('iterations')} steps={ckpt.get('steps')} "
                  f"success_rate={success} act={act_mode(path)}")
    else:
        print(promote(args.specialist, args.run_id, args.models_dir, args.act))


if __name__ == '__main__':
    main()
