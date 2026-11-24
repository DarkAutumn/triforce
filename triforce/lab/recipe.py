"""A lab run's recipe: the refusals that stop a run whose tasks, network or demonstrations disagree, and its manifest
(docs/lab.md "The recipe on record", "Refusals").

The refusals (``Refused``, each naming what to change):

- ``check_routes``: a task without ``state_routes`` and no ``--state-route`` draws every route's states, ``d2first``'s
  included (the generation plays ``d3first`` alone); ``--all-routes`` lets it.
- ``check_warm_start``: a task's ``warm_start`` is never loaded by the lab (one network for every task; ``--load`` names
  its start), and the generation trains from scratch; ``--allow-warm-start`` lets the run start without it.
- ``check_obs_agree``: the tasks of one run set an ``obs.`` switch differently (one network reads one observation).
- ``check_network_reads``: the network reads an opt-in block (``SWITCH_KEYS``: ``obs.threat``'s ``entity_threat``,
  ``obs.chase``'s ``chase`` and ``entity_chase``) whose switch is off in the tasks (it would read zeros), or the tasks
  turn on a switch whose blocks the network does not read.
- ``check_demos``: a demonstration file lacks a key the network reads (``PPO``'s cloning term runs the network on the
  file's rows: ``ppo.demo_log_prob``), or holds a switch's block all zeros where the tasks turn the switch on (recorded
  with it off).

``build`` writes the run's manifest in ``triforce.manifest``'s shape: ``code``, ``rom``, ``task`` (the group, the
route, every task by ``manifest.task`` with its hashes), ``data`` (every task's seed, harvested and curriculum data,
the demonstration files with their groups), ``networks`` (the network trained: its architecture and settings, the
checkpoint it starts from, the teachers), ``training`` (the PPO settings, auxiliary targets, look-ahead, teacher
term, sampling), ``arguments``, ``seeds``, ``job``."""

import hashlib
import json
from typing import Optional

import numpy as np

from .. import manifest, nets, paths
from ..model import CHASE_KEY, ENTITY_CHASE_KEY, ENTITY_THREAT_KEY, MASK_KEY, TOKEN_KEYS, obs_spec_from_space
from ..observation import observation_space

# The switches whose blocks only a network that asks for them reads (``model.TOKEN_KEYS``): switch -> its keys.
# ``obs.target`` and ``obs.objective`` fill vector blocks every network reads; ``check_obs_agree`` covers them.
SWITCH_KEYS = {'threat': (ENTITY_THREAT_KEY,), 'chase': (CHASE_KEY, ENTITY_CHASE_KEY)}


class Refused(Exception):
    """A lab run that would train something other than it claims (the module docstring); the message says what to
    change."""


def _obs(family) -> dict:
    return family.config.to_dict()['obs']


# ------------------------------------------------------------------------------------------------- refusals

def check_routes(families, state_route: Optional[str], all_routes: bool):
    """A task with no ``state_routes`` and no ``--state-route`` draws every route's states (``d2first``'s too)."""
    if state_route or all_routes:
        return
    bare = [f.key for f in families if not f.config.state_routes]
    if bare:
        raise Refused(f"{', '.join(bare)}: the task names no state_routes and the run no --state-route, so it would "
                      f"draw every route's states, d2first's included; pass --state-route d3first (--group sets it), "
                      f"give the task state_routes, or pass --all-routes to train on every route's states")


def check_warm_start(families, allow: bool):
    """A task's ``warm_start`` (the lab starts from ``--load`` or from scratch, never from a task's)."""
    if allow:
        return
    warm = [f'{f.key} ({f.config.warm_start})' for f in families if f.config.warm_start]
    if warm:
        raise Refused(f"{', '.join(warm)}: the task has a warm_start, which the lab never loads (one network for "
                      f"every task: --load names its start) and the generation trains from scratch; set the task's "
                      f"warm_start to null, or pass --allow-warm-start to train without it")


def check_obs_agree(families):
    """Every task of the run sets every ``obs.`` switch alike."""
    settings = {f.key: _obs(f) for f in families}
    lines = []
    for name in sorted({k for obs in settings.values() for k in obs}):
        values: dict = {}
        for key, obs in settings.items():
            values.setdefault(json.dumps(obs.get(name)), []).append(key)
        if len(values) > 1:
            lines.append(f"obs.{name}: " + '; '.join(f"{json.loads(v)} in {', '.join(keys)}"
                                                      for v, keys in values.items()))
    if lines:
        raise Refused("the tasks disagree on the observation, and one network reads one observation: set each "
                      "switch alike in every task (the generation: obs.threat and obs.chase true)\n  "
                      + '\n  '.join(lines))


def network_reads(network) -> set[str]:
    """The observation keys ``network`` reads: its layout's keys but the opt-in blocks (``model.TOKEN_KEYS``) and the
    mask, plus the opt-in blocks a token network asks for (any submodule's ``token_keys`` and ``link_keys``:
    ``hitnet``, also inside a ``hitshift``)."""
    spec = network.config.to_dict()['obs_spec']
    reads = {k for k in spec if k not in TOKEN_KEYS and k != MASK_KEY}
    for module in network.modules():
        reads |= set(getattr(module, 'token_keys', ())) | set(getattr(module, 'link_keys', ()))
    return reads


def check_network_reads(network, families):
    """The network reads an opt-in block exactly when the tasks' switch fills it (``check_obs_agree`` first)."""
    reads = network_reads(network)
    obs = _obs(families[0])
    arch = network.config.to_dict().get('arch') or 'zelda'
    for switch, keys in SWITCH_KEYS.items():
        read = sorted(k for k in keys if k in reads)
        if read and not obs.get(switch):
            raise Refused(f"the network ({arch}) reads {', '.join(read)}, which obs.{switch} fills, and obs.{switch} "
                          f"is off in the tasks ({', '.join(f.key for f in families)}): it would read zeros; turn "
                          f"obs.{switch} on in every task, or leave the '{switch}' token group out of --arch-config")
        if obs.get(switch) and not read:
            raise Refused(f"obs.{switch} is on in the tasks and the network ({arch}) reads none of "
                          f"{', '.join(keys)}: train a network that reads them (--arch hitnet with the '{switch}' "
                          f"token group in --arch-config tokens), or turn obs.{switch} off in every task")


def check_demos(network, families, files):
    """Each demonstration file holds every key the network reads, and a switch's blocks recorded on where the tasks
    turn the switch on (a block all zeros over the file's rows was recorded with the switch off)."""
    if not files:
        return
    reads = network_reads(network)
    obs = _obs(families[0])
    problems = []
    for path in files:
        with np.load(path) as data:
            held = {k[len('obs/'):] for k in data.files if k.startswith('obs/')}
            missing = sorted(reads - held)
            if missing:
                problems.append(f"{path}: no {', '.join(missing)}")
                continue
            for switch, keys in SWITCH_KEYS.items():
                off = [k for k in keys if k in reads and obs.get(switch) and not np.any(data[f'obs/{k}'])]
                if off:
                    problems.append(f"{path}: {', '.join(off)} all zeros (recorded with obs.{switch} off)")
    if problems:
        raise Refused("demonstrations the network cannot learn from as recorded (the cloning term runs it on their "
                      "rows): record them again with the tasks' observation switches, or leave the tasks' files out "
                      "with --no-task-demos\n  " + '\n  '.join(problems))


def create_network(arch: str, arch_config: Optional[dict] = None, families=None):
    """A fresh network of ``arch`` (``nets``) with ``arch_config`` on today's observation layout; with ``families``
    (``lab.collect.parse_families``) the tasks' switches are checked against it first (``check_obs_agree``,
    ``check_network_reads``: ``Refused``)."""
    network = nets.create(arch, obs_spec_from_space(observation_space()), **(arch_config or {}))
    if families:
        check_obs_agree(families)
        check_network_reads(network, families)
    return network


# --------------------------------------------------------------------------------------------------- manifest

def task_entry(family) -> dict:
    """``manifest.task`` of a family's task, its curriculum folder (``lab.collect.route_only``'s folder of links in
    this checkout) written as ``manifest.label`` does, so two checkouts' runs of one recipe compare."""
    entry = manifest.task(family.config, family.yaml)
    curriculum = entry['effective'].get('curriculum') or {}
    if curriculum.get('trajectories'):
        curriculum['trajectories'] = manifest.label(curriculum['trajectories'])
        text = json.dumps(entry['effective'], sort_keys=True)
        entry['effective_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return entry


def curriculum_folder(config) -> Optional[str]:
    """The tracks a task's curriculum reads (the task's ``curriculum.trajectories``, else the family's folder of the
    trajectories data, as ``lab.collect.parse_families`` checks them), None with the curriculum off."""
    if not config.curriculum.enabled:
        return None
    return str(config.curriculum.trajectories or paths.data_dir('trajectories') / config.name)


def student(network) -> dict:
    """The network trained: its architecture, settings (the observation layout by its hash), size and the keys it
    reads (``network_reads``)."""
    config = dict(network.config.to_dict())
    spec = config.pop('obs_spec', None)
    return {'arch': config.get('arch') or 'zelda', 'config': config,
            'obs_spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            'parameters': network.parameter_count(), 'reads': sorted(network_reads(network))}


def build(args, families, network, *, group=None, envs: dict, source=None, teachers: Optional[dict] = None,
          demos=(), demo_groups=None, training: dict, seeds: dict, sources: Optional[dict] = None) -> dict:
    """The run's manifest (the module docstring).  ``envs``: envs per key; ``source``: the checkpoint ``--load``
    names; ``teachers``: key -> checkpoint; ``training``: the resolved training settings; ``sources``: where a group
    run's ``arch``, ``arch_config``, ``aux`` and ``try_others`` came from (``lab.train.apply_network``), recorded as
    ``networks.student.sources``."""
    # pylint: disable=too-many-arguments
    data = {'tasks': {f.key: manifest.data(f.config, args.states_dir, curriculum_folder(f.config)) for f in families}}
    if demos:
        data['demos'] = manifest.demo_entries(demos, demo_groups)
    networks = {'student': student(network)}
    if sources is not None:
        networks['student']['sources'] = dict(sources)
    if source is not None:
        networks['warm_start'] = manifest.network(source)
    if teachers:
        networks['teachers'] = {key: manifest.network(path) for key, path in teachers.items()}
    task = {'group': None if group is None else group.record(), 'route': args.state_route,
            'tasks': {f.key: task_entry(f) for f in families}, 'envs': dict(envs)}
    return manifest.build('lab-train', arguments_=vars(args), seeds=seeds, task=task, data=data, networks=networks,
                          training=training, models=manifest.models(args.models_dir))


def read_reference(reference: str) -> dict:
    """The manifest of ``--same-recipe-as RUN``: a run dir or ``run.json`` (``manifest.read``), or a store run named
    ``NAME/RUN``."""
    try:
        return manifest.read(reference)
    except ValueError as error:
        raise Refused(f"--same-recipe-as {reference}: {error}") from None


def check_same(recipe: dict, reference: dict, excepted, what: str):
    """Refuses a run whose manifest differs from ``reference`` in more than its own identity and seeds
    (``manifest.same_recipe``) and the ``excepted`` keys; ``what`` names the reference in the message."""
    differences, unknown = manifest.same_recipe(recipe, reference, excepted)
    for entry in unknown:
        print(f"[lab] {what}: unknown there: {entry['path']} (this run: {manifest.short(entry['a'])})", flush=True)
    if differences:
        lines = '\n'.join(f"  {d['path']}: {manifest.short(d['a'])} (this run) vs {manifest.short(d['b'])}"
                          for d in differences)
        raise Refused(f"{what}: this run differs in {len(differences)} more than --except "
                      f"{','.join(excepted or ()) or '(nothing)'} allows (name the changed keys with --except, or "
                      f"run the recipe as recorded):\n{lines}")
