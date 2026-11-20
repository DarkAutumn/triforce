"""Model registry (design §9): ``<models>/<specialist>/<run-id>/ckpt.pt`` plus ``<models>/<specialist>/promoted``.

A checkpoint is a dict with ``obs_schema_version``, ``action_schema_version``, ``specialist``, ``model_config``
(including the observation layout, so the network can be rebuilt from it alone), ``state_dict``,
``optimizer``, ``iterations``, ``steps``, ``metrics``.  Loading a checkpoint raises ``SchemaMismatchError`` unless
its action schema is the current one and its observation schema is the current one or an older one whose layout
(``model_config['obs_spec']``) is a subset of the current ``OBS_SPEC``: every key it reads still exists with the
same shape and vocabulary (schema 1 is schema 2 without ``target``).  The network reads only its own keys.

The models root is, in order: the ``root`` argument, ``$TRIFORCE_MODELS_DIR``, ``<repo>/models``.

How a run's model acts (better-play §6b) is ``<run dir>/act``: ``<mode> <sha1 of ckpt.pt>``, ``mode`` being an act
mode spelling (docs/rust-actor.md "Act modes": ``sample``, the default when the file is absent, ``argmax``,
``temperature:T``, ``floor:P``, ``minp:R``, ``topp:P``, ``escape:N``, ``relax:N`` and ``+`` combinations, optionally
with a foresight part ``+guard:K`` or ``+plan:K:S[:R]`` and followed by ``+low:H:MODE`` for low health;
``parse_act`` gives the canonical spelling), chosen by evaluating and set with ``promote --act``.  The mode belongs to
the weights it was measured for: when ``ckpt.pt`` no longer has that
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

Numbered checkpoints (``train --checkpoint-every N``, round 25): ``<run dir>/checkpoints/it<iteration:06d>/`` each
hold a copy of ``ckpt.pt`` as saved at that iteration with its own ``version`` and ``act`` (bound to that copy's
sha1), so a numbered checkpoint dir is used wherever a run dir is (``eval-chain --checkpoint NAME=DIR``, ``evaluate
--checkpoint DIR``, ``train --load DIR``); ``snapshot_checkpoint`` writes one, ``list_snapshots`` lists them.

CLI: ``python -m triforce.registry list <specialist>`` / ``promote <specialist> <run-id> [--act MODE]``.
"""

import argparse
import hashlib
import json
import numbers
import os
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Optional

import torch

from . import __version__
from ._native import ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION, OBS_SPEC
from ._native import act_mode as act_mode_spelling
from ._native import routes as native_routes

MODELS_DIR_ENV = 'TRIFORCE_MODELS_DIR'
ALLOW_INCOMPATIBLE_ENV = 'TRIFORCE_ALLOW_INCOMPATIBLE_MODELS'
CHECKPOINT_NAME = 'ckpt.pt'
PROMOTED_NAME = 'promoted'
ACT_NAME = 'act'
VERSION_NAME = 'version'
SNAPSHOTS_NAME = 'checkpoints'
# The act mode of the whole project (round 23, decided Nov 14; docs/decisions.md "One act mode"): evaluation,
# chains, the early-stop confirmation, a run without an ``act`` file, and play (``PLAY_ACT``).
ACT_RULE = 'temperature:0.5+relax:16'
DEFAULT_ACT = ACT_RULE
# The honest planner (round 25, the owner's decision of Nov 16; docs/decisions.md "The honest planner beside the
# models"): the project's rule with one-step foresight against one future whose random numbers are redrawn
# (docs/rust-actor.md "Foresight"). A second, labelled act mode beside the models alone (chain reports, the golden
# path, play, a teacher to try); never a default.
PLAN_ACT = f'{ACT_RULE}+plan:16:1'
# ``--act model``: each model acts by its own act mode.
MODEL_ACT = 'model'
# The main route (phase X, the owner, Nov 19): the route every tool plays and measures when none is named
# (`play-zelda.py`, `zelda-headless.py` and the golden path, `zelda-map.py`, `triforce eval-chain`, `eval-batch
# --chains` and the chains script `scripts/experiments/r56_chains.sh`). `d2first` until a golden path on `d3first`
# reaches the ending ("d3 will be the main run when it works ... d2 will be the secondary"); then 'd3first' here.
# The engine's `DEFAULT_ROUTE` (`zelda::game_map`) is another thing: the route of bare start-state names, `d2first`
# for good.
MAIN_ROUTE = 'd2first'


def parse_act(text: str, allow_model: bool = False) -> str:
    """The canonical spelling of act mode ``text`` (``_native.act_mode``; ``model`` too with ``allow_model``);
    raises ValueError for anything else."""
    if allow_model and text == MODEL_ACT:
        return MODEL_ACT
    return act_mode_spelling(text)


def act_argument(allow_model: bool = True):
    """An argparse ``type`` for ``--act``: the canonical spelling, or an argparse error naming the problem."""
    def parse(text: str) -> str:
        try:
            return parse_act(text, allow_model)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from None
    parse.__name__ = 'act mode'
    return parse


# ``%%``: argparse formats help strings.
ACT_HELP = ("act mode: sample, argmax, temperature:T, floor:P, minp:R, topp:P, escape:N, relax:N or a '+' "
            "combination, optionally with foresight (+guard:K: try the action against the game's own future and take "
            "the next of the K best that loses no health; +plan:K:S[:R]: the same against S futures with the game's "
            "random numbers redrawn, refused when it loses health in more than the share R of them) and followed by "
            "+low:H:MODE (MODE while Link holds H hearts or fewer, or H%% of his maximum) (docs/rust-actor.md \"Act "
            "modes\")")


# How the play tools (play-zelda.py with its live cell, zelda-headless.py, the golden path) act by default: never
# argmax, so every run differs (round 23, docs/experiments/round23.md §10). Every play tool reads this one constant.
PLAY_ACT = ACT_RULE


def draws(mode: str) -> bool:
    """Whether act mode ``mode`` draws every decision at random (``sample``, the rules, and ``relax``: a rule's draw,
    a plain one when stalled), unlike ``argmax`` and ``escape`` (argmax until stalled); with ``+low:H:MODE`` both
    parts must draw."""
    return all(style != 'argmax' and 'escape:' not in style for style in _styles(parse_act(mode)))


def _styles(mode: str) -> list:
    """The styles of canonical act mode ``mode``: the normal one, and the low-health one when it has ``+low:H:``."""
    normal, low, rest = mode.partition('+low:')
    return [normal, rest.split(':', 1)[1]] if low else [normal]


def act_override(choice: str) -> Optional[str]:
    """A command line ``--act`` choice as an override: None for ``model`` (each model's act mode), else the
    canonical spelling every model acts by."""
    choice = parse_act(choice, allow_model=True)
    return None if choice == MODEL_ACT else choice


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


def snapshot_dir(run: Path, iteration: int) -> Path:
    """``<run dir>/checkpoints/it<iteration:06d>``: the numbered checkpoint of ``iteration``."""
    return Path(run) / SNAPSHOTS_NAME / f'it{int(iteration):06d}'


def snapshot_checkpoint(checkpoint, iteration: int, keep: int = 0) -> Path:
    """Copies the run's ``ckpt.pt`` (``checkpoint``: the file or its run dir, as just saved at ``iteration``) into
    its numbered checkpoint dir (``snapshot_dir``) with the run's version stamp and act mode, the act bound to the
    copy's sha1; ``keep`` > 0 then removes all but the newest ``keep`` numbered checkpoints.  Returns the dir."""
    ckpt = _resolve(checkpoint)
    target = snapshot_dir(ckpt.parent, iteration)
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / f'.{CHECKPOINT_NAME}.{os.getpid()}.tmp'
    shutil.copyfile(ckpt, tmp)
    os.replace(tmp, target / CHECKPOINT_NAME)
    write_model_version(target, model_version(ckpt) or __version__)
    set_act_mode(target, act_mode(ckpt))
    if keep > 0:
        for _, old in list_snapshots(ckpt.parent)[:-keep]:
            shutil.rmtree(old)
    return target


def list_snapshots(run) -> list[tuple[int, Path]]:
    """(iteration, dir) of a run dir's numbered checkpoints that hold a ``ckpt.pt``, oldest first."""
    base = Path(run) / SNAPSHOTS_NAME
    if not base.is_dir():
        return []
    found = []
    for path in base.iterdir():
        if path.name.startswith('it') and path.name[2:].isdigit() and (path / CHECKPOINT_NAME).is_file():
            found.append((int(path.name[2:]), path))
    return sorted(found)


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


# Models of this version prefix and observation schema `triforce convert-model` widens to this code's layout
# (triforce/convert.py).
CONVERTIBLE_VERSION = '1.5.0'
CONVERTIBLE_OBS_SCHEMA = 5
CONVERT_HINT = ("a 1.5.0 model is widened to this code's observation by `triforce convert-model RUN_DIR... --out DIR` "
                "(or `triforce --models-dir MODELS convert-model --promoted --out DIR`)")


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
    if found is not None and compatible(found, CONVERTIBLE_VERSION + '.0'):
        raise IncompatibleModelError(f"{message}; {CONVERT_HINT}")
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
                hint = f"; {CONVERT_HINT}" if obs == CONVERTIBLE_OBS_SCHEMA else ""
                raise SchemaMismatchError(f"{path}: obs_schema_version is {obs}, code expects "
                                          f"{obs_schema_version()} ({problem}){hint}")
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


def promote(specialist: str, run_id: str, root=None, act: Optional[str] = None, route: Optional[str] = None) -> Path:
    """Marks run_id as the specialist's promoted model (``<models>/<specialist>/promoted`` holds the id; with ``route``,
    phase X, ``promoted.<route>``: that route's run, over ``promoted`` there); ``act`` also sets how the run acts
    (``set_act_mode``)."""
    if route is not None and route not in native_routes():
        raise ValueError(f"unknown route {route!r} (routes.yaml has {native_routes()})")
    ckpt = checkpoint_path(specialist, run_id, root)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint for {specialist}/{run_id}: {ckpt}")
    load_checkpoint(ckpt)
    if act is not None:
        set_act_mode(ckpt, act)
    pointer = promoted_pointer(specialist, root, route)
    tmp = pointer.with_name(f'.{pointer.name}.{os.getpid()}.tmp')
    tmp.write_text(run_id + '\n', encoding='utf-8')
    os.replace(tmp, pointer)
    return pointer


def eval_mode(result: dict) -> Optional[str]:
    """The act mode an evaluation result was measured in (``act``, else the ``deterministic`` flag of results
    written before act modes had rules)."""
    try:
        return parse_act(result['act'])
    except (KeyError, TypeError, ValueError):
        pass
    deterministic = result.get('deterministic')
    return None if deterministic is None else ('argmax' if deterministic else 'sample')


def act_mode(checkpoint) -> str:
    """How the model in ``checkpoint`` (file or run dir) acts: its run dir's ``act`` when that was set for these
    weights (its sha1 matches ``ckpt.pt``), else ``DEFAULT_ACT`` (a warning says why when an ``act`` file exists)."""
    ckpt = _resolve(checkpoint)
    path = ckpt.parent / ACT_NAME
    if not path.is_file():
        return DEFAULT_ACT
    fields = path.read_text(encoding='utf-8').split()
    try:
        mode = parse_act(fields[0] if fields else '')
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    if len(fields) != 2:
        warnings.warn(f"{path}: act file without the checkpoint hash (old format); the model acts by {DEFAULT_ACT}",
                      stacklevel=2)
        return DEFAULT_ACT
    if fields[1] != checkpoint_sha1(ckpt):
        warnings.warn(f"{path}: act mode {mode!r} was set for other weights (sha1 {fields[1]}, {ckpt.name} is "
                      f"{checkpoint_sha1(ckpt)}); the model acts by {DEFAULT_ACT}", stacklevel=2)
        return DEFAULT_ACT
    return mode


def set_act_mode(checkpoint, mode: str) -> Path:
    """Sets how the model in ``checkpoint`` (file or run dir) acts, bound to the current ``ckpt.pt``'s sha1;
    returns the ``act`` file (written atomically).  An ``eval.json`` measured in the other mode is renamed to
    ``eval-<its mode>.json`` (an existing file of that name is kept as ``eval-<its mode>.replaced.json``), so
    ``eval.json`` stays the evaluation in the model's act mode."""
    mode = parse_act(mode)
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


def promoted_pointer(specialist: str, root=None, route: Optional[str] = None) -> Path:
    """The file naming the specialist's promoted run: ``promoted``, or (phase X) for ``route`` its own
    ``promoted.<route>``, which overrides ``promoted`` on that route where it exists."""
    return models_root(root) / specialist / (PROMOTED_NAME if route is None else f'{PROMOTED_NAME}.{route}')


def promoted_run(specialist: str, root=None, route: Optional[str] = None) -> str:
    """The promoted run id: on ``route``, its ``promoted.<route>`` when there is one (phase X: a run promoted for one
    route only), else ``promoted`` (every route's).  Raises FileNotFoundError if none is promoted."""
    if route is not None and promoted_pointer(specialist, root, route).is_file():
        return promoted_pointer(specialist, root, route).read_text(encoding='utf-8').strip()
    pointer = promoted_pointer(specialist, root)
    if not pointer.is_file():
        raise FileNotFoundError(f"No promoted model for specialist '{specialist}' ({pointer})")
    return pointer.read_text(encoding='utf-8').strip()


def route_promotions(specialist: str, root=None) -> dict[str, str]:
    """Route -> run of every ``promoted.<route>`` of the specialist (phase X)."""
    base = models_root(root) / specialist
    prefix = f'{PROMOTED_NAME}.'
    if not base.is_dir():
        return {}
    return {p.name[len(prefix):]: p.read_text(encoding='utf-8').strip()
            for p in sorted(base.iterdir()) if p.name.startswith(prefix) and p.suffix != '.tmp' and p.is_file()}


def promoted_checkpoint(specialist: str, root=None, route: Optional[str] = None) -> Path:
    """Path of the promoted run's ``ckpt.pt`` (on ``route``: ``promoted_run``)."""
    path = checkpoint_path(specialist, promoted_run(specialist, root, route), root)
    if not path.is_file():
        raise FileNotFoundError(f"Promoted run for '{specialist}' has no checkpoint: {path}")
    return path


# Round 32: which run plays a route step.  The step's policy key is the engine's (``zelda::GameMap::step_policy``,
# ``_native.step_policy``): ``family@route`` before the route's ``own_models_until`` step (routes.yaml), else
# ``family``.  ``policy_run`` turns a key into the run that plays it; every tool that plays goes through the two.
POLICY_ROUTE_SEP = '@'      # zelda::game_map::POLICY_ROUTE_SEP


def route_policy(family: str, route: str) -> str:
    """The policy key of ``family``'s run for ``route``'s own part (``darknut@d3first``)."""
    return f'{family}{POLICY_ROUTE_SEP}{route}'


def split_policy(key: str) -> tuple[str, Optional[str]]:
    """A policy key's family and, for a route part's key, its route."""
    family, sep, route = key.partition(POLICY_ROUTE_SEP)
    return family, (route if sep else None)


def own_models_until(route: Optional[str]) -> Optional[int]:
    """The index of the step up to which ``route``'s own runs play (routes.yaml ``own_models_until``); None: a route
    whose own runs (``promoted.<route>``) play every step, no route, or a name routes.yaml lacks (a chain file's route
    read back by a tool that plays nothing)."""
    from ._native import own_models_until as native_until  # pylint: disable=import-outside-toplevel
    return None if route is None or route not in native_routes() else native_until(route)


def policy_run(key: str, root=None, route: Optional[str] = None) -> str:
    """The run that plays policy ``key`` on ``route``: for ``family@R`` the family's ``promoted.<R>``; for a family
    its ``promoted`` on a route with ``own_models_until`` (its own runs play only before that step), else
    ``promoted_run(family, root, route)`` (``promoted.<route>`` over ``promoted``).  FileNotFoundError without one."""
    family, part = split_policy(key)
    if part is not None:
        pointer = promoted_pointer(family, root, part)
        if not pointer.is_file():
            raise FileNotFoundError(f"No run of '{family}' promoted for route {part} ({pointer})")
        return pointer.read_text(encoding='utf-8').strip()
    return promoted_run(family, root, None if own_models_until(route) is not None else route)


def policy_checkpoint(key: str, root=None, route: Optional[str] = None) -> Path:
    """Path of the ``ckpt.pt`` of the run that plays policy ``key`` on ``route`` (``policy_run``)."""
    family, _ = split_policy(key)
    path = checkpoint_path(family, policy_run(key, root, route), root)
    if not path.is_file():
        raise FileNotFoundError(f"The run of '{key}' has no checkpoint: {path}")
    return path


def route_part_key(family: str, root=None, route: Optional[str] = None) -> Optional[str]:
    """``family@route`` when ``route`` has ``own_models_until`` and the family a run promoted for it (a second policy
    plays the family's rooms before that step), else None."""
    if own_models_until(route) is None or not promoted_pointer(family, root, route).is_file():
        return None
    return route_policy(family, route)


def main(argv=None):
    """``list`` / ``promote`` subcommands."""
    parser = argparse.ArgumentParser(prog='python -m triforce.registry')
    parser.add_argument('--models-dir', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list').add_argument('specialist')
    promote_parser = sub.add_parser('promote')
    promote_parser.add_argument('specialist')
    promote_parser.add_argument('run_id')
    promote_parser.add_argument('--act', type=act_argument(allow_model=False), default=None,
                                help=f"also set how the run acts ({ACT_HELP})")
    promote_parser.add_argument('--route', default=None,
                                help="promote the run for this route only (promoted.<route>, over promoted there)")
    args = parser.parse_args(argv)

    if args.command == 'list':
        try:
            current = promoted_run(args.specialist, args.models_dir)
        except FileNotFoundError:
            current = None
        routes = route_promotions(args.specialist, args.models_dir)
        for run_id in list_runs(args.specialist, args.models_dir):
            path = checkpoint_path(args.specialist, run_id, args.models_dir)
            ckpt = _torch_load(path)
            mark = '*' if run_id == current else ' '
            success = ckpt.get('metrics', {}).get('success_rate')
            only = [route for route, run in routes.items() if run == run_id]
            print(f"{mark} {run_id}  iterations={ckpt.get('iterations')} steps={ckpt.get('steps')} "
                  f"success_rate={success} act={act_mode(path)}"
                  + (f" promoted on {','.join(only)}" if only else ''))
    else:
        print(promote(args.specialist, args.run_id, args.models_dir, args.act, args.route))


if __name__ == '__main__':
    main()
