"""Rollouts over several families at once, and their teachers (docs/lab.md "Several families", "Teachers").

``Family``: one member of a training set: its policy key (``zol``, or ``zol@d3first`` for a route's own run), its
task (the family's yaml: rooms, start states, hp and item randomisation, masks, rewards, max_steps) and its share of
the envs.  ``MultiCollector`` gives every family its own ``EnvPool`` (``TrainTask``) and ``ppo.Collector`` and rolls
them out together; the PPO learner sees one ``BufferSet``, whose rows carry their family and, with ``Teachers``,
the teacher's joint log-probs and value.  ``shares``: envs per family (fixed for a run); ``NeedSampler`` turns the
rows each family's rollout takes (decisions an env) by its need, from the run's own episodes, between iterations.
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .. import _native, registry
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


def parse_families(items, parallel: int, shares: Optional[dict] = None) -> list[Family]:
    """``key[=yaml]`` items (a key's task: the given yaml, else ``task_yaml``) with ``parallel`` envs split by
    ``shares`` (key -> weight; default equal), every family at least one."""
    families = []
    for item in items:
        key, _, path = item.partition('=')
        config = load_specialist(path or task_yaml(key))
        families.append(Family(key, config))
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


class Teachers:
    """Each family's teacher (the network that plays its rooms today: ``registry.policy_checkpoint(key)``, or the
    given checkpoints), on ``device``, labelling rows in batches: joint log-probs ``[n, 352]`` and values."""

    def __init__(self, keys, device, root=None, checkpoints: Optional[dict] = None):
        self.device = torch.device(device)
        self.networks: dict[str, torch.nn.Module] = {}
        self.paths: dict[str, str] = {}
        loaded: dict[str, torch.nn.Module] = {}
        for key in keys:
            path = (checkpoints or {}).get(key) or registry.policy_checkpoint(key, root)
            real = str(Path(path).resolve())
            if real not in loaded:
                loaded[real] = registry.load_model(path)[0].to(self.device).eval()
            self.networks[key] = loaded[real]
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

    def __init__(self, keys: list[str], buffers: list, labels: Optional[list] = None):
        self.keys = keys
        self.buffers = buffers
        self.labels = labels
        self.rows = sum(b.rows for b in buffers)
        self.family = torch.cat([torch.full((b.rows,), i, dtype=torch.int64) for i, b in enumerate(buffers)])

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

    def take(self):
        self.wait()
        self.rollout_seconds = time.time() - self._began
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
        self.buffer = BufferSet(self.keys, [c.buffer for c in self.collectors], labels)
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
        the rows (``rows_share``) and, with a sampler, its ``need`` and ``recent_success``."""
        result = {f'family/{key}/{name}': value for key, metrics in self.by_family.items()
                  for name, value in metrics.summary().items() if isinstance(value, (int, float))}
        if self.buffer is not None:
            for key, buffer in zip(self.keys, self.buffer.buffers):
                result[f'family/{key}/rows_share'] = buffer.rows / self.buffer.rows
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
