"""Rollouts over several families at once, and their teachers (docs/lab.md "Several families", "Teachers").

``Family``: one member of a training set: its policy key (``zol``, or ``zol@d3first`` for a route's own run), its
task (the family's yaml: rooms, start states, hp and item randomisation, masks, rewards, max_steps) and its share of
the envs.  ``MultiCollector`` gives every family its own ``EnvPool`` (``TrainTask``) and ``ppo.Collector`` and rolls
them out together; the PPO learner sees one ``BufferSet``, whose rows carry their family and, with ``Teachers``,
the teacher's joint log-probs and value.  ``shares``: envs per family (fixed for a run); ``NeedSampler`` turns the
rows each family's rollout takes (decisions an env) by its need, from the run's own episodes, between iterations.
"""

import fnmatch
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .. import _native, lineage, paths, registry
from ..inputs import (MissingInput, check_curriculum, check_harvested, check_start_states, declaration,
                      small_harvested_entries, untracked)
from ..paths import SPECIALISTS_DIR
from ..metrics import RolloutMetrics
from ..ppo import Collector
from ..specialist import SpecialistConfig, load_specialist

CHUNK = 4096


@dataclass
class Family:
    """A member of a training set (the module docstring)."""
    key: str
    config: SpecialistConfig
    envs: int = 1
    yaml: str = ''          # the task: a yaml path, or a family name (its own yaml)

    @property
    def family(self) -> str:
        return registry.split_policy(self.key)[0]


def task_yaml(key: str) -> str:
    """The task of policy ``key``: a route part's own yaml (``specialists/<route>/<family>.yaml``, the wood-sword
    runs' recipe) when there is one, else the family's."""
    family, route = registry.split_policy(key)
    if route is not None and (SPECIALISTS_DIR / route / f'{family}.yaml').is_file():
        return str(SPECIALISTS_DIR / route / f'{family}.yaml')
    return family


def route_only(config: SpecialistConfig, route: str, states_dir=None, tracks_dir=None) -> SpecialistConfig:
    """``config`` on ``route``'s states alone (``state_routes: [route]``: its start-state globs and its harvested
    starts in the route's namespace, ``<route>_<name>``): the globs no state of the route matches (rooms the route
    never plays) are dropped, and the curriculum keeps the tracks explored from the route's states (a folder of links,
    ``<tracks_dir>/<route>/<family>``, default ``<checkout>/target/route-tracks``; none: no curriculum).  The tracks
    come from the yaml's ``curriculum.trajectories``, else the family's folder of ``paths.data_dir('trajectories')``
    (this checkout's ``trajectories/`` while it has one, else the store's current snapshot)."""
    folder = Path(states_dir or _native.paths()['states_dir'])
    prefix = f'{route}_'
    names = [p.stem[len(prefix):] for p in folder.glob(f'{prefix}*.state')]
    kept = [g for g in config.start_states if fnmatch.filter(names, g)]
    if not kept:
        raise SystemExit(f"{config.name}: no start state of route {route} matches {config.start_states}")
    left_out = plain_way_only(config, route, names)
    if left_out:
        # A glob that matched a plain-way-only state becomes the names it keeps.
        kept = [n for g in kept for n in ([g] if not set(fnmatch.filter(names, g)) & set(left_out)
                                          else sorted(set(fnmatch.filter(names, g)) - set(left_out)))]
        print(f"[lab] {config.name}: clips on in every episode (clips.off_share 0): {len(left_out)} seed states the "
              f"route enters only on its plain way are left out: {', '.join(left_out)}", flush=True)
        if not kept:
            raise SystemExit(f"{config.name}: every start state of route {route} is on a plain way: {left_out}")
    config = config.replace(state_routes=[route], start_states=kept)
    if not config.curriculum.enabled:
        return config
    source = Path(config.curriculum.trajectories or paths.data_dir('trajectories') / config.name)
    tracks = sorted(p.parent for p in source.glob('*/trajectory.json')
                    if (start := Path(json.loads(p.read_text(encoding='utf-8')).get('start_state', '')).stem)
                    .startswith(prefix) and start[len(prefix):] not in left_out)
    if not tracks:
        # No track of the route: a curriculum of none only when the task declares every one of its states untracked.
        seeds = [Path(p).stem for p in config.resolve_start_states(states_dir)]
        undeclared = [s for s in seeds if not any(fnmatch.fnmatchcase(s, g) for g in untracked(config))]
        if undeclared:
            raise SystemExit(f"{config.name}: the curriculum is on and {source} holds no track of route {route}'s "
                             f"states; explore them, or add to the task's curriculum.untracked: "
                             f"{declaration(undeclared, seeds)}")
        return config.replace(curriculum={'enabled': False})
    out = Path(tracks_dir or paths.scratch('route-tracks')).resolve() / route / config.name
    out.mkdir(parents=True, exist_ok=True)
    # The folder is exactly the route's tracks of today's source: a link of a track gone from it is removed, one that
    # points elsewhere (an earlier snapshot) is pointed again.
    wanted = {track.name: track.resolve() for track in tracks}
    for link in out.iterdir():
        if link.is_symlink() and (link.name not in wanted or link.resolve() != wanted[link.name]):
            link.unlink(missing_ok=True)
    for name, target in wanted.items():
        if not (out / name).is_symlink():
            try:
                (out / name).symlink_to(target)
            except FileExistsError:     # another run of this checkout made it meanwhile
                pass
    # The task's other curriculum settings (fraction, threshold, the untracked declaration) stay.
    return config.replace(curriculum={**config.to_dict()['curriculum'], 'enabled': True, 'trajectories': str(out)})


def plain_way_only(config: SpecialistConfig, route: str, names: list[str]) -> list[str]:
    """The seed states of ``names`` (the route's, without its namespace) a task with clips on in every episode
    (``clips.off_share`` 0: the owner, Nov 24, 2.0.1.0 trains with clips on) never trains: those whose route steps
    (by room, side and visit) are all on a stretch's plain way, so the start fixes the clips switch off (4:0x20 from S,
    4:0x30 from E, 4:0x21 from W). Only of the task's own globs; none for a task that trains clips off."""
    if config.clips.off_share > 0:
        return []
    ways = {}
    for step in _native.route_steps(route):
        key = (step['level'], step['room'], step['in_cave'], step['entry'], step['visit'])
        ways.setdefault(key, set()).add(step['way'])
    matched = sorted({n for g in config.start_states for n in fnmatch.filter(names, g)})
    return [n for n in matched if ways.get(_native.state_name(n)) == {'plain'}]


def parse_families(items, parallel: int, shares: Optional[dict] = None, state_route: Optional[str] = None,
                   curriculum: bool = True, states_dir=None, tracks_dir=None) -> list[Family]:
    """``key[=yaml]`` items (a key's task: the given yaml, else ``task_yaml``) with ``parallel`` envs split by
    ``shares`` (key -> weight; default equal), every family at least one; with ``state_route``, each task on that
    route's states alone (``route_only``, its links under ``tracks_dir``).  Every input a task declares is checked
    (``triforce.inputs``: its start-state globs and harvested states under ``states_dir``, with ``curriculum`` its
    tracks); a missing one stops the run."""
    families = []
    for item in items:
        key, _, path = item.partition('=')
        config = load_specialist(path or task_yaml(key))
        try:
            check_start_states(config, states_dir)
            if state_route:
                config = route_only(config, state_route, states_dir, tracks_dir)
            small = small_harvested_entries(check_harvested(config, states_dir))
            if small:
                print(f"[lab] {key}: harvested room entries under {_native.MIN_HARVESTED_ENTRY} states are not drawn: "
                      + ', '.join(f'{stem} ({n})' for stem, n in small.items()), flush=True)
            if curriculum and config.curriculum.enabled:
                check_curriculum(config, config.curriculum.trajectories or paths.data_dir('trajectories') / config.name,
                                 states_dir)
        except MissingInput as error:
            raise SystemExit(f"{key}: {error}" if key.split('@')[0] != config.name else str(error)) from None
        families.append(Family(key, config, yaml=path or task_yaml(key)))
    keys = [f.key for f in families]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"a family is named twice: {keys}")
    if parallel < len(families):
        raise SystemExit(f"--parallel {parallel} is less than one env a family ({len(families)})")
    weights = [float((shares or {}).get(f.key, 1.0)) for f in families]
    total = sum(weights)
    envs = [max(1, int(parallel * w / total)) for w in weights]
    while sum(envs) < parallel:
        envs[max(range(len(envs)), key=lambda i: weights[i] * parallel / total - envs[i])] += 1
    while sum(envs) > parallel:
        envs[max(range(len(envs)), key=lambda i: envs[i])] -= 1
    for family, count in zip(families, envs):
        family.envs = count
    return families


def teacher_paths(keys, root=None, checkpoints: Optional[dict] = None, route: Optional[str] = None) -> dict:
    """Each key's teacher checkpoint (``Teachers``): the given one, else ``registry.policy_checkpoint(key, root,
    route)``."""
    return {key: (checkpoints or {}).get(key) or registry.policy_checkpoint(key, root, route) for key in keys}


class Teachers:
    """Each family's teacher (the network that plays its rooms today: ``registry.policy_checkpoint(key, root,
    route)``, on ``route`` a family's ``late.<route>`` over its ``promoted``, or the given checkpoints), on ``device``,
    labelling rows in batches: joint log-probs ``[n, 352]`` and values.  ``weights``: key -> the weight of its rows'
    teacher term (default 1; 0: no teacher term for that key)."""

    def __init__(self, keys, device, root=None, checkpoints: Optional[dict] = None,
                 weights: Optional[dict] = None, route: Optional[str] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.device = torch.device(device)
        self.weights = {key: float((weights or {}).get(key, 1.0)) for key in keys}
        self.networks: dict[str, torch.nn.Module] = {}
        self.paths: dict[str, str] = {}
        # Each teacher as run.json names it (round 40, `lineage.identity`: family, run, iterations, SHA-1).
        self.identities: dict[str, dict] = {}
        loaded: dict[str, tuple] = {}
        chosen = teacher_paths(keys, root, checkpoints, route)
        for key in keys:
            path = chosen[key]
            real = str(Path(path).resolve())
            if real not in loaded:
                network, checkpoint = registry.load_model(path)
                loaded[real] = network.to(self.device).eval(), lineage.identity(checkpoint, path)
            self.networks[key], self.identities[key] = loaded[real]
            self.paths[key] = str(path)
        self._native: dict[str, object] = {}

    def native(self, key: str):
        """The teacher of ``key`` as a pool policy (teacher-acting rollouts)."""
        if key not in self._native:
            self._native[key] = self.networks[key].to_native()
        return self._native[key]

    @torch.no_grad()
    def label(self, key: str, obs: dict, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(joint log-probs ``[n, J]``, values ``[n]``) of ``key``'s teacher on the rows, on the CPU."""
        network = self.networks[key]
        joints, values = [], []
        for begin in range(0, masks.shape[0], CHUNK):
            rows = slice(begin, begin + CHUNK)
            part = {k: v[rows].to(self.device) for k, v in obs.items()}
            joints.append(network.joint_log_probs(part, masks[rows].to(self.device)).cpu())
            values.append(network.get_value(part).cpu())
        return torch.cat(joints), torch.cat(values)


class BufferSet:
    """The families' readied rollout buffers as one batch (``PPO.optimize`` reads ``flatten``)."""

    def __init__(self, keys: list[str], buffers: list, labels: Optional[list] = None,
                 weights: Optional[list] = None):
        self.keys = keys
        self.buffers = buffers
        self.labels = labels
        self.rows = sum(b.rows for b in buffers)
        self.family = torch.cat([torch.full((b.rows,), i, dtype=torch.int64) for i, b in enumerate(buffers)])
        self.weight = None      # teacher term weight per row; None: 1 everywhere
        if weights is not None and any(w != 1.0 for w in weights):
            self.weight = torch.tensor(weights, dtype=torch.float32)[self.family]

    @staticmethod
    def rows(buffer) -> tuple[dict, torch.Tensor]:
        """(observations, masks) of one buffer's stored steps, flat."""
        length = buffer.memory_length
        obs = {key: value[:, :length].reshape(-1, *value.shape[2:]) for key, value in buffer.observation.items()}
        return obs, buffer.masks[:, :length].reshape(-1, buffer.masks.shape[-1])

    def flatten(self, device) -> dict:
        parts = [b.flatten(device) for b in self.buffers]
        batch = {'obs': {key: torch.cat([p['obs'][key] for p in parts]) for key in parts[0]['obs']}}
        for key in parts[0]:
            if key != 'obs':
                batch[key] = torch.cat([p[key] for p in parts])
        batch['family'] = self.family.to(device)
        if self.labels is not None:
            batch['teacher_joint'] = torch.cat([joint for joint, _ in self.labels]).to(device)
            batch['teacher_value'] = torch.cat([value for _, value in self.labels]).to(device)
            if self.weight is not None:
                batch['teacher_weight'] = self.weight.to(device)
        return batch


class MultiCollector:
    """``ppo.Collector``'s interface over the families' pools (the module docstring).  ``actor``: 'student' (the
    network trained acts) or 'teacher' (each family's teacher acts; ``teachers`` needed)."""

    def __init__(self, families: list[Family], run_dir: Path, steps_per_env: int, gamma: float, lam: float, *,
                 seed: int, states_dir=None, curricula: Optional[dict] = None,
                 teachers: Optional[Teachers] = None, actor: str = 'student', tried: Optional[dict] = None,
                 chunk_length: int = 16, sampler: Optional['NeedSampler'] = None):
        # pylint: disable=too-many-arguments
        if actor not in ('student', 'teacher') or (actor == 'teacher' and teachers is None):
            raise ValueError(f"actor {actor!r}: 'student', or 'teacher' with teachers")
        self.families = families
        self.keys = [f.key for f in families]
        self.teachers = teachers
        self.actor = actor
        self.pools = []
        self.collectors = []
        for i, family in enumerate(families):
            task_dir = Path(run_dir) / 'tasks' / family.key
            task_dir.mkdir(parents=True, exist_ok=True)
            task = _native.TrainTask(family.config, str(task_dir), curriculum=(curricula or {}).get(family.key),
                                     states_dir=None if states_dir is None else str(states_dir))
            pool = _native.EnvPool(task, num_envs=family.envs, threads=family.envs, seed=seed + 7919 * i)
            self.pools.append(pool)
            self.collectors.append(Collector(pool, steps_per_env, gamma, lam, budget=family.config.max_steps,
                                             tried=tried, chunk_length=chunk_length))
        self.steps_per_env = steps_per_env
        self.tried = tried
        self.chunk_length = chunk_length
        self.sampler = sampler
        self.num_envs = sum(f.envs for f in families)
        self.buffer: Optional[BufferSet] = None
        self.episodes: list = []
        self.by_family: dict[str, RolloutMetrics] = {}
        self.rollout_seconds = self.recompute_seconds = 0.0
        self._threads: list[threading.Thread] = []
        self._errors: list[BaseException] = []
        self._began = 0.0
        self._served: Optional[tuple] = None      # the network server's counters when the rollout started
        self.serve_stats: dict = {}

    @property
    def shares(self) -> dict[str, int]:
        """Envs per family (fixed for a run; ``NeedSampler`` changes the decisions an env instead)."""
        return {f.key: f.envs for f in self.families}

    def steps_for(self, network) -> dict[str, int]:
        """Decisions an env of each family in the next rollout: ``steps_per_env`` each, or by ``sampler``'s shares
        of the same total rows (a recurrent network's in whole sequences)."""
        if self.sampler is None:
            return {f.key: self.steps_per_env for f in self.families}
        unit = self.chunk_length if getattr(network, 'recurrent', False) else 1
        return self.sampler.steps({f.key: f.envs for f in self.families}, self.steps_per_env * self.num_envs, unit)

    def start(self, network):
        """Starts every family's rollout (each pool on its own thread, the GIL released while it steps)."""
        student = network.to_native() if self.actor == 'student' else None
        for collector, steps in zip(self.collectors, self.steps_for(network).values()):
            collector.resize(steps)
        self._errors = []
        self._began = time.time()
        self._served = self._server_counters()
        extra = {} if self.tried is None else {'tried': self.tried}

        def run(collector, index, policy):
            try:
                collector.pool.rollout([policy], collector._buffers[index].memory_length,  # pylint: disable=protected-access
                                       collector._views[index], **extra)  # pylint: disable=protected-access
            except BaseException as err:  # pylint: disable=broad-exception-caught
                self._errors.append(err)

        self._threads = []
        for family, collector in zip(self.families, self.collectors):
            policy = student if student is not None else self.teachers.native(family.key)
            thread = threading.Thread(target=run, args=(collector, collector._next, policy),  # pylint: disable=protected-access
                                      name=f'rollout-{family.key}', daemon=True)
            thread.start()
            self._threads.append(thread)

    def wait(self):
        for thread in self._threads:
            thread.join()
        self._threads = []
        if self._errors:
            raise self._errors[0]

    @staticmethod
    def _server_counters() -> Optional[tuple]:
        """(batches, rows, busy seconds, graphs replayed) of the process's network server, None when none serves
        (``nets.serve``)."""
        from ..nets import serve  # pylint: disable=import-outside-toplevel
        server = serve._DEFAULT  # pylint: disable=protected-access
        return None if server is None else (server.batches, server.rows, server.busy,
                                            float(server.graphs and server.replaying))

    def take(self):
        self.wait()
        self.rollout_seconds = time.time() - self._began
        now, began = self._server_counters(), self._served
        self.serve_stats = {}
        if now is not None and began is not None and self.rollout_seconds > 0:
            batches, rows, busy = (a - b for a, b in zip(now[:3], began[:3]))
            # The served network's side of the rollout: rows a batch, batches a second, the share of the rollout's
            # wall time the server thread spent answering (the rest it gathered rows or waited), and whether it
            # replayed CUDA graphs (1) or ran the eager forward (0).
            self.serve_stats = {'charts/serve_rows_per_batch': rows / batches if batches else 0.0,
                                'charts/serve_batches_per_second': batches / self.rollout_seconds,
                                'charts/serve_busy': busy / self.rollout_seconds, 'charts/serve_graphs': began[3]}
            from ..nets import serve  # pylint: disable=import-outside-toplevel
            audit = serve._DEFAULT.audit  # pylint: disable=protected-access
            if audit is not None:       # ``$TRIFORCE_SERVE_AUDIT``: the run's graphed answers so far against eager
                self.serve_stats.update({f'charts/serve_audit_{key}': value for key, value in audit.items()})
        self.episodes = []
        for i, collector in enumerate(self.collectors):
            collector._result = {'seconds': self.rollout_seconds}  # pylint: disable=protected-access
            collector.take()
            self.episodes += [{**e, 'family': self.keys[i]} for e in collector.episodes]
            if self.sampler is not None:
                self.sampler.observe(self.keys[i], collector.episodes)

    def ready(self, network, device) -> RolloutMetrics:
        began = time.time()
        self.by_family = {key: collector.ready(network, device) for key, collector in zip(self.keys, self.collectors)}
        labels = None
        if self.teachers is not None:
            labels = []
            for key, collector in zip(self.keys, self.collectors):
                obs, masks = BufferSet.rows(collector.buffer)
                labels.append(self.teachers.label(key, obs, masks))
        self.buffer = BufferSet(self.keys, [c.buffer for c in self.collectors], labels,
                                None if self.teachers is None else [self.teachers.weights[k] for k in self.keys])
        self.recompute_seconds = time.time() - began
        merged = RolloutMetrics()
        for collector in self.collectors:
            buffer = collector.buffer
            merged.add_steps(buffer.actions.reshape(-1, 3).numpy(), buffer.frames.reshape(-1).numpy(),
                             buffer.interrupts.reshape(-1).numpy())
            for event in collector.episodes:
                merged.add_episode(event['info'])
        merged.cohorts = [c for m in self.by_family.values() for c in m.cohorts]
        merged.extra = {}
        return merged

    def family_metrics(self) -> dict:
        """The last rollout's summary metrics per family, as ``family/<key>/<metric>``, with each family's share of
        the rows (``rows_share``), the hits Link took an episode (``hits_taken_per_episode``: the decisions that lost
        health over the episodes the rollout ended, the bench's hits a visit) and the Bubble touches
        (``bubble_touches_per_episode``: the decisions a Bubble's touch landed in, round 38), the aimless score's
        (round 38: ``waste_frames_per_episode``, the frames calm decisions added to their stretches' waste, and
        ``calm_share``, the share of decisions that were calm), the frames its decisions
        played (``frames_per_decision``) and, with ``tried``, the frames the look-ahead's trials emulated
        (``tried_frames_per_decision``); with a sampler, its ``need`` and ``recent_success``."""
        result = {f'family/{key}/{name}': value for key, metrics in self.by_family.items()
                  for name, value in metrics.summary().items() if isinstance(value, (int, float))}
        if self.buffer is not None:
            for key, buffer in zip(self.keys, self.buffer.buffers):
                result[f'family/{key}/rows_share'] = buffer.rows / self.buffer.rows
                ended = float(buffer.dones[:, 1:].sum())
                if ended:
                    result[f'family/{key}/hits_taken_per_episode'] = float((buffer.health_lost > 0).sum()) / ended
                    result[f'family/{key}/bubble_touches_per_episode'] = float(buffer.blocked.sum()) / ended
                    result[f'family/{key}/waste_frames_per_episode'] = float(buffer.waste.sum()) / ended
                result[f'family/{key}/calm_share'] = float(buffer.calm.float().mean())
                result[f'family/{key}/frames_per_decision'] = float(buffer.frames.float().mean())
                if buffer.tried_frames is not None:
                    result[f'family/{key}/tried_frames_per_decision'] = float(buffer.tried_frames.float().mean())
        if self.sampler is not None:
            for key in self.keys:
                result[f'family/{key}/need'] = self.sampler.need(key)
                recent = self.sampler.success(key)
                if recent is not None:
                    result[f'family/{key}/recent_success'] = recent
        return result

    def close(self):
        for pool in self.pools:
            pool.close()


class NeedSampler:
    """Rows per family by need (docs/lab.md "Sampling across families"): a family's need is how far its success over
    its last ``window`` episodes of the run is below its target (``targets``; a family with no episode yet needs
    its whole target); its share of the rows is ``(floor + need) / Σ (floor + need)``.  With every family at its
    target the shares are equal; ``floor`` keeps every family in the rollout."""

    def __init__(self, targets: dict[str, float], floor: float = 0.05, window: int = 100):
        if floor <= 0:
            raise ValueError("the need floor must be above 0 (every family keeps some rows)")
        self.targets = dict(targets)
        self.floor = floor
        self.recent = {key: deque(maxlen=window) for key in targets}

    @staticmethod
    def teacher_targets(keys, reference: Path) -> dict[str, float]:
        """Each key's success on the yardstick in ``reference`` (``lab/reference.json``: today's networks), the
        student's target: "below its teacher"."""
        data = json.loads(Path(reference).read_text(encoding='utf-8'))['keys']
        missing = [k for k in keys if k not in data]
        if missing:
            raise SystemExit(f"{reference} has no {missing}")
        return {k: float(data[k]['success']) for k in keys}

    def observe(self, key: str, episodes: list):
        """Adds a rollout's finished episodes of ``key`` (``episode`` events)."""
        for event in episodes:
            self.recent[key].append(event['info'].get('outcome') == 'success')

    def state(self) -> dict:
        """What a resumed run restores (``restore``): each family's recent outcomes, oldest first."""
        return {key: [bool(x) for x in recent] for key, recent in self.recent.items()}

    def restore(self, state: dict):
        """The recent outcomes ``state`` saved; a family it does not name starts empty."""
        for key, recent in self.recent.items():
            recent.clear()
            recent.extend(bool(x) for x in state.get(key, ()))

    def success(self, key: str) -> Optional[float]:
        recent = self.recent[key]
        return sum(recent) / len(recent) if recent else None

    def need(self, key: str) -> float:
        success = self.success(key)
        return self.targets[key] if success is None else max(0.0, self.targets[key] - success)

    def shares(self) -> dict[str, float]:
        weights = {key: self.floor + self.need(key) for key in self.targets}
        total = sum(weights.values())
        return {key: w / total for key, w in weights.items()}

    def steps(self, envs: dict[str, int], rows: int, unit: int = 1) -> dict[str, int]:
        """Decisions an env per family (``envs``: envs per family) so each takes about its share of ``rows``, in
        multiples of ``unit`` (at least one)."""
        shares = self.shares()
        return {key: max(unit, round(shares[key] * rows / envs[key] / unit) * unit) for key in envs}
