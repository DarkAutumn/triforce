"""Model registry (design §9): ``<models>/<specialist>/<run-id>/ckpt.pt`` plus ``<models>/<specialist>/promoted``.

A checkpoint is a dict with ``obs_schema_version``, ``action_schema_version``, ``specialist``, ``model_config``
(including the observation layout, so the network can be rebuilt from it alone), ``state_dict``,
``optimizer``, ``iterations``, ``steps``, ``metrics``.  Loading a checkpoint raises ``SchemaMismatchError`` unless
its action schema is the current one and its observation schema is the current one or an older one whose layout
(``model_config['obs_spec']``) is a subset of the current ``OBS_SPEC``: every key it reads still exists with the
same shape and vocabulary (schema 1 is schema 2 without ``target``).  The network reads only its own keys.

The models root is, in order: the ``root`` argument, ``$TRIFORCE_MODELS_DIR``, ``<repo>/models``.

How a run's model acts (better-play §6b) is ``<run dir>/act``: ``<mode> <sha1 of ckpt.pt>``, ``mode`` being
``sample`` (keyed sampling, the default when the file is absent) or ``argmax``, chosen by evaluating both and set
with ``promote --act``.  The mode belongs to the weights it was measured for: when ``ckpt.pt`` no longer has that
hash (retrained or replaced in place) or the file is the old one-word format, ``act_mode`` warns and returns
``sample``.  Chains, the watch player and evaluation follow it unless told otherwise.

Checkpoints load with ``torch.load(weights_only=True)`` (no pickled code runs); ``save_checkpoint`` keeps
``metrics`` plain floats/ints so that stays true.

Every run dir also carries ``version``: the triforce version (``triforce.__version__``) that saved the checkpoint,
written by ``save_checkpoint``.  A model whose first three numbers differ from the code's is refused
(``require_compatible``, called wherever a model is loaded to act or to continue training) unless
``$TRIFORCE_ALLOW_INCOMPATIBLE_MODELS`` is ``1`` (``--allow-incompatible-models``); ``triforce stamp`` stamps runs
known to be compatible.  The stamp is a file of its own because ``act`` and ``eval.json`` bind to ``ckpt.pt``'s
sha1: stamping must not rewrite the checkpoint.

CLI: ``python -m triforce.registry list <specialist>`` / ``promote <specialist> <run-id> [--act MODE]``.
"""

import argparse
import hashlib
import json
import numbers
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import torch

from . import __version__
from ._native import ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION, OBS_SPEC

MODELS_DIR_ENV = 'TRIFORCE_MODELS_DIR'
ALLOW_INCOMPATIBLE_ENV = 'TRIFORCE_ALLOW_INCOMPATIBLE_MODELS'
CHECKPOINT_NAME = 'ckpt.pt'
PROMOTED_NAME = 'promoted'
ACT_NAME = 'act'
VERSION_NAME = 'version'
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


class IncompatibleModelError(ValueError):
    """A model's version is not compatible with this code's (the first three numbers differ, or it has none)."""


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
        'metrics': _plain_metrics(metrics or {}),
        **(extra or {}),
    }
    tmp = path.with_suffix('.tmp')
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)
    write_model_version(path)
    return path


def _plain_metrics(metrics: dict) -> dict:
    """``metrics`` with numpy/torch scalars as Python floats/ints, so the checkpoint loads with
    ``weights_only=True``."""
    def plain(value):
        if isinstance(value, (bool, str)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, 'item'):      # 0-d tensors / arrays
            return plain(value.item())
        raise TypeError(f"checkpoint metric value {value!r} ({type(value).__name__}) is not a plain number")
    return {str(key): plain(value) for key, value in metrics.items()}


def model_version(checkpoint) -> Optional[str]:
    """The triforce version the model in ``checkpoint`` (file or run dir) was saved or stamped with: its run dir's
    ``version`` file, None when there is none."""
    try:
        return (_resolve(checkpoint).parent / VERSION_NAME).read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        return None


def write_model_version(checkpoint, version: str = __version__) -> Path:
    """Stamps the run dir of ``checkpoint`` (file or run dir) with ``version`` (written atomically); returns the
    ``version`` file."""
    path = _resolve(checkpoint).parent / VERSION_NAME
    tmp = path.with_name(f'.{VERSION_NAME}.{os.getpid()}.tmp')
    tmp.write_text(version + '\n', encoding='utf-8')
    os.replace(tmp, path)
    return path


def compatible(a: str, b: str) -> bool:
    """Whether two versions can share models: their first three numbers (MAJOR.MINOR.COMPAT) are equal."""
    return a.split('.')[:3] == b.split('.')[:3]


def set_allow_incompatible(on: bool) -> None:
    """``--allow-incompatible-models``: when ``on``, incompatible models load with a warning, here and in every
    child process (the environment variable is inherited).  Never switches it back off."""
    if on:
        os.environ[ALLOW_INCOMPATIBLE_ENV] = '1'


def require_compatible(checkpoint) -> None:
    """Raises ``IncompatibleModelError`` unless the model in ``checkpoint`` (file or run dir) carries a version
    compatible with this code's; with ``$TRIFORCE_ALLOW_INCOMPATIBLE_MODELS=1`` it warns instead."""
    found = model_version(checkpoint)
    if found is not None and compatible(found, __version__):
        return
    message = (f"{_resolve(checkpoint)}: model version {found or 'unversioned'} is not compatible with triforce "
               f"{__version__} (the first three numbers must match)")
    if os.environ.get(ALLOW_INCOMPATIBLE_ENV) == '1':
        warnings.warn(message, stacklevel=2)
        return
    raise IncompatibleModelError(f"{message}; stamp a model known to be compatible with `triforce stamp`, or pass "
                                 "--allow-incompatible-models")


def _resolve(path) -> Path:
    path = Path(path)
    return path / CHECKPOINT_NAME if path.is_dir() else path


_SHA1_CACHE: dict = {}


def checkpoint_sha1(checkpoint) -> str:
    """Hex sha1 of a checkpoint's ``ckpt.pt`` bytes (file or run dir); cached per (path, mtime, size)."""
    path = _resolve(checkpoint).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    digest = _SHA1_CACHE.get(key)
    if digest is None:
        sha = hashlib.sha1()
        with open(path, 'rb') as file:
            for chunk in iter(lambda: file.read(1 << 20), b''):
                sha.update(chunk)
        digest = _SHA1_CACHE[key] = sha.hexdigest()
    return digest


def _torch_load(path: Path) -> dict:
    """``torch.load`` with the safe loader; a file that needs pickled code raises ``pickle.UnpicklingError``
    naming the path."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise pickle.UnpicklingError(f"{path}: not a plain checkpoint (refusing to unpickle code): {error}") from error


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
    checkpoint = _torch_load(path)
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


def eval_mode(result: dict) -> Optional[str]:
    """The act mode an evaluation result was measured in (``act``, else its ``deterministic`` flag)."""
    if result.get('act') in ACT_MODES:
        return result['act']
    deterministic = result.get('deterministic')
    return None if deterministic is None else ('argmax' if deterministic else 'sample')


def act_mode(checkpoint) -> str:
    """How the model in ``checkpoint`` (file or run dir) acts: its run dir's ``act`` when that was set for these
    weights (its sha1 matches ``ckpt.pt``), else ``sample`` (a warning says why when an ``act`` file exists)."""
    ckpt = _resolve(checkpoint)
    path = ckpt.parent / ACT_NAME
    if not path.is_file():
        return 'sample'
    fields = path.read_text(encoding='utf-8').split()
    mode = fields[0] if fields else ''
    if mode not in ACT_MODES:
        raise ValueError(f"{path}: act mode {mode!r} is not one of {ACT_MODES}")
    if len(fields) != 2:
        warnings.warn(f"{path}: act file without the checkpoint hash (old format); the model samples", stacklevel=2)
        return 'sample'
    if fields[1] != checkpoint_sha1(ckpt):
        warnings.warn(f"{path}: act mode {mode!r} was set for other weights (sha1 {fields[1]}, {ckpt.name} is "
                      f"{checkpoint_sha1(ckpt)}); the model samples", stacklevel=2)
        return 'sample'
    return mode


def set_act_mode(checkpoint, mode: str) -> Path:
    """Sets how the model in ``checkpoint`` (file or run dir) acts, bound to the current ``ckpt.pt``'s sha1;
    returns the ``act`` file (written atomically).  An ``eval.json`` measured in the other mode is renamed to
    ``eval-<its mode>.json`` (an existing file of that name is kept as ``eval-<its mode>.replaced.json``), so
    ``eval.json`` stays the evaluation in the model's act mode."""
    if mode not in ACT_MODES:
        raise ValueError(f"act mode {mode!r} is not one of {ACT_MODES}")
    ckpt = _resolve(checkpoint)
    path = ckpt.parent / ACT_NAME
    tmp = path.with_name(f'.{ACT_NAME}.{os.getpid()}.tmp')
    tmp.write_text(f'{mode} {checkpoint_sha1(ckpt)}\n', encoding='utf-8')
    os.replace(tmp, path)
    evaluation = ckpt.parent / 'eval.json'
    if evaluation.is_file():
        try:
            old = eval_mode(json.loads(evaluation.read_text(encoding='utf-8')))
        except ValueError:
            old = None
        if old is not None and old != mode:
            target = evaluation.with_name(f'eval-{old}.json')
            if target.is_file():
                os.replace(target, target.with_name(f'eval-{old}.replaced.json'))
            os.replace(evaluation, target)
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
            ckpt = _torch_load(path)
            mark = '*' if run_id == current else ' '
            success = ckpt.get('metrics', {}).get('success_rate')
            print(f"{mark} {run_id}  iterations={ckpt.get('iterations')} steps={ckpt.get('steps')} "
                  f"success_rate={success} act={act_mode(path)}")
    else:
        print(promote(args.specialist, args.run_id, args.models_dir, args.act))


if __name__ == '__main__':
    main()
