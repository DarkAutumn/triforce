"""Lab training: one network over several families at once, with today's networks as teachers (docs/lab.md).

    python -m triforce.lab.train --name zol+skg --families zol stalfos-keese-gel --parallel 8 --iterations 80 \\
        --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 60

Each ``--families`` item is a policy key, optionally with its task yaml (``zol@d3first=path/zol.yaml``); a key's
own yaml stays its task definition.  The run lives in ``<models>/<name>/<run-id>`` and every member family gets a
link ``<models>/<family>/<run-id>`` to it, so ``promote``, ``evaluate``, ``eval-chain``, ``select-checkpoint`` and
``triforce list`` treat it as each family's run (``runner.Policies`` loads it once).

Modes: ``ppo`` (default; the teacher term KL(teacher ‖ student) beside PPO with ``--teacher-coeff``, held
``--teacher-hold`` iterations, then fading linearly to 0 over ``--teacher-fade``) and ``imitate`` (no PPO: the
student minimises the KL alone; ``--actor student`` collects on the student's own states, DAgger-style, the
distribution it will play; ``--actor teacher`` on the teachers', a fixed distribution for capacity tests).
Metrics are logged per family (``family/<key>/...``) beside the merged ones.  The job's log folder (``--logs-dir``)
holds a plan of kind ``lab-train`` and a stream a family (``RunLog``), so the training GUI draws the run: one group,
a row a family.

Also (docs/lab.md): auxiliary targets (``--aux OUTPUT=TARGET:COEFF``) and look-ahead labels for candidates not taken
(``--try-others K:P``, training only); a recurrent network (``LabNet.initial_state``) trains on sequences of
``--bptt`` decisions from stored states; ``--sampling need`` shares the rows by each family's need (``--need-target-for``
a family's own target); ``--teacher-weight`` scales or removes a family's teacher term; ``--rehearse`` keeps tasks
outside the rollout in the loss by their stored teacher rows; demonstrations (``--demos``), the early stop
(``--stop-success``) and numbered checkpoints (``--checkpoint-every``) as in ``triforce train``.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from .. import lineage, nets, registry
from ..best_of_n import success_by_state
from ..demos import load_demos
from ..events import EventLog, logged_job
from ..jobs import announce, work
from ..jobs import progress as job_progress
from ..metrics import MetricsLogger, format_summary
from ..model import obs_spec_from_space
from ..observation import observation_space
from ..ppo import PPO, IterationResult, PPOConfig, joint_kl, minibatch_forward
from ..train import CONFIRM_EPISODES, EarlyStop, curriculum_spec
from .auxiliary import AuxLoss, aux_loss, held_out
from .collect import MultiCollector, NeedSampler, Teachers, parse_families
from .rehearse import Rehearsal

REFERENCE = Path(__file__).with_name('reference.json')
KIND = 'lab-train'          # the plan kind of a lab run's log folder
# The run's numbers (besides the family's own) every family's progress event carries: the teacher term and the
# auxiliary losses are the one network's.
SHARED_METRICS = ('losses/teacher_kl', 'charts/teacher_agreement', 'losses/approx_kl', 'losses/policy_loss',
                  'losses/value_loss', 'losses/entropy')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m triforce.lab.train', description=__doc__.split('\n\n')[0])
    parser.add_argument('--name', required=True, help="the run's set name: <models>/<name>/<run-id>")
    parser.add_argument('--families', nargs='+', required=True, metavar='KEY[=YAML]')
    parser.add_argument('--shares', nargs='*', default=[], metavar='KEY=W', help="env shares (default equal)")
    parser.add_argument('--arch', default=nets.ZELDA, choices=nets.names())
    parser.add_argument('--arch-config', default='{}', help="JSON settings of the architecture")
    parser.add_argument('--load', default=None, help="checkpoint or run dir to start from (its arch)")
    parser.add_argument('--mode', default='ppo', choices=['ppo', 'imitate'])
    parser.add_argument('--actor', default='student', choices=['student', 'teacher'],
                        help="who acts in the rollouts (imitate only; ppo always the student)")
    parser.add_argument('--teacher-coeff', type=float, default=0.0, help="KL(teacher || student) weight (0: off)")
    parser.add_argument('--teacher-value-coeff', type=float, default=0.0, help="½(V - V_teacher)² weight")
    parser.add_argument('--teacher-hold', type=int, default=0, help="iterations at the full weight")
    parser.add_argument('--teacher-fade', type=int, default=0,
                        help="iterations over which the weight then falls to 0 (0: stays)")
    parser.add_argument('--teacher', nargs='*', default=[], metavar='KEY=CKPT', help="teacher overrides")
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--steps-per-update', type=int, default=PPOConfig.target_steps)
    parser.add_argument('--learning-rate', type=float, default=PPOConfig.learning_rate)
    parser.add_argument('--ent-coeff', type=float, default=PPOConfig.ent_coeff)
    parser.add_argument('--epochs', type=int, default=PPOConfig.num_epochs)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--states-dir', default=None)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help="also keep a numbered checkpoint every N iterations (select-checkpoint reads them)")
    parser.add_argument('--checkpoint-keep', type=int, default=0, help="numbered checkpoints kept (0: all)")
    parser.add_argument('--no-curriculum', action='store_true')
    parser.add_argument('--state-route', default=None, metavar='ROUTE',
                        help="every family on this route's states alone (lab.collect.route_only: state_routes, the "
                             "start-state globs the route has, the curriculum tracks explored from its states)")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--aux', nargs='*', default=[], metavar='OUTPUT=TARGET:COEFF',
                        help="a loss on the network's auxiliary output OUTPUT against TARGET (hit, hit_or_blocked, "
                             "damaged, hearts_lost, frames; lab/auxiliary.py) weighted COEFF")
    parser.add_argument('--try-others', default=None, metavar='K:P',
                        help="on a share P of the decisions, try the policy's next K candidates in the env and keep "
                             "whether each lost health and whether a Bubble touched Link (training only; the 'hit' "
                             "and 'hit_or_blocked' targets' extra labels)")
    parser.add_argument('--bptt', type=int, default=16,
                        help="a recurrent network's sequence length (truncated back-propagation through time)")
    parser.add_argument('--sampling', default='fixed', choices=['fixed', 'need'],
                        help="rows per family: fixed (equal decisions an env) or by need (NeedSampler)")
    parser.add_argument('--need-target', default='teacher',
                        help="the success a family needs: 'teacher' (its key's success in lab/reference.json) or a "
                             "number for every family")
    parser.add_argument('--need-floor', type=float, default=0.05, help="a family's weight beside its need")
    parser.add_argument('--need-window', type=int, default=100, help="recent episodes a family's success reads")
    parser.add_argument('--need-target-for', nargs='*', default=[], metavar='KEY=X',
                        help="a family's own need target (overrides --need-target for it)")
    parser.add_argument('--teacher-weight', nargs='*', default=[], metavar='KEY=W',
                        help="weight of a family's rows in the teacher term (default 1; 0: no teacher for it)")
    parser.add_argument('--rehearse', default=None, metavar='CACHE',
                        help="a distillation cache (r36_merge_distill.py cache): KL to its teacher rows of the keys "
                             "not rolled out, every minibatch (lab.rehearse)")
    parser.add_argument('--rehearse-keys', nargs='*', default=None,
                        help="the keys rehearsed (default: every key of the cache not in --families)")
    parser.add_argument('--rehearse-rows', type=int, default=300_000, help="rows held, equal a key")
    parser.add_argument('--rehearse-coeff', type=float, default=1.0)
    parser.add_argument('--demos', nargs='+', default=None, metavar='FILE',
                        help="demonstration files (triforce.demos) behavior-cloned during PPO (--demo-coeff)")
    parser.add_argument('--demo-coeff', type=float, default=0.2)
    parser.add_argument('--stop-success', type=float, default=None,
                        help="stop once every family's start-state-balanced training success >= this for "
                             "--stop-patience iterations and an evaluation of each confirms it")
    parser.add_argument('--stop-patience', type=int, default=5)
    parser.add_argument('--logs-dir', default=None,
                        help="the root of the run's log folder (default logs/, $TRIFORCE_LOGS_DIR)")
    return parser.parse_args(argv)


def teacher_coeff(args, iteration: int) -> float:
    """The teacher term's weight at (1-based) ``iteration``: held, then fading linearly to 0."""
    if args.teacher_fade <= 0 or iteration <= args.teacher_hold:
        return args.teacher_coeff
    return args.teacher_coeff * max(0.0, 1.0 - (iteration - args.teacher_hold) / args.teacher_fade)


def initial_network(args):
    """(network, source, parent): ``parent`` the loaded checkpoint's identity (``lineage.identity``), None from
    scratch."""
    if args.load:
        network, checkpoint = registry.load_model(args.load)
        return network, str(args.load), lineage.identity(checkpoint, args.load)
    spec = obs_spec_from_space(observation_space())
    return nets.create(args.arch, spec, **json.loads(args.arch_config)), None, None


def link_families(root: Path, name: str, run_id: str, keys: list[str]):
    """``<models>/<family>/<run-id>`` -> ``../<name>/<run-id>`` for every member family."""
    for family in sorted({registry.split_policy(k)[0] for k in keys}):
        link = registry.run_dir(family, run_id, root)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path('..') / name / run_id)


def imitate(network, optimizer, batch: dict, cfg: PPOConfig, value_coeff: float, aux_losses=()) -> dict:
    """One pass of the imitation-only update over ``batch`` (``BufferSet.flatten``): KL(teacher ‖ student)
    (+ ``value_coeff`` · ½(V − V_teacher)², + the auxiliary losses) in ``cfg.minibatches`` minibatches,
    ``cfg.num_epochs`` epochs; a recurrent network's in whole sequences (``ppo.minibatch_forward``)."""
    # pylint: disable=too-many-locals
    rows = batch['masks'].shape[0]
    device = batch['masks'].device
    sequence = batch.get('chunk_states')
    length = rows // sequence.shape[0] if sequence is not None else 1
    units = rows // length

    def forward(part):
        mb = (part[:, None] * length + torch.arange(length, device=device)).reshape(-1)
        obs = {k: v[mb] for k, v in batch['obs'].items()}
        out, joint_of = minibatch_forward(network, obs, batch['masks'][mb], batch['actions'][mb],
                                          None if sequence is None else (batch['starts'][mb], sequence[part]))
        return mb, out, joint_of

    network.train()
    kl = torch.tensor(0.0)
    stats: dict = held_out(list(aux_losses), network, batch) if aux_losses and sequence is None else {}
    for _ in range(cfg.num_epochs):
        for part in torch.tensor_split(torch.randperm(units, device=device), cfg.minibatches):
            mb, out, joint_of = forward(part)
            kl = joint_kl(batch['teacher_joint'][mb], joint_of())
            loss = kl
            if value_coeff > 0:
                loss = loss + value_coeff * 0.5 * (out.value - batch['teacher_value'][mb]).pow(2).mean()
            if aux_losses:
                term, aux_stats = aux_loss(list(aux_losses), out.aux, batch, mb)
                stats.update(aux_stats)
                loss = loss + term
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
            optimizer.step()
    network.eval()
    with torch.no_grad():
        agree = []
        for part in torch.split(torch.arange(units, device=device), max(1, 4096 // length)):
            mb, _, joint_of = forward(part)
            agree.append(joint_of().argmax(-1) == batch['teacher_joint'][mb].argmax(-1))
    return {**stats, 'losses/teacher_kl': float(kl.detach()),
            'charts/teacher_agreement': float(torch.cat(agree).float().mean())}


def parse_try_others(text: Optional[str]) -> Optional[dict]:
    """``K:P`` -> ``EnvPool.rollout``'s ``tried`` settings."""
    if text is None:
        return None
    k, _, share = text.partition(':')
    try:
        tried = {'candidates': int(k), 'share': float(share or 1.0)}
    except ValueError as err:
        raise SystemExit(f"--try-others {text!r}: K:P (candidates, share of decisions)") from err
    if tried['candidates'] < 1 or not 0 < tried['share'] <= 1:
        raise SystemExit(f"--try-others {text!r}: K >= 1 and P in (0, 1]")
    return tried


class RunLog:
    """The run's log folder (``logs/<yymmdd>_<NNN>_lab-<name>``, docs/lab.md "What a lab job writes"): the plan
    ``training.json`` (kind ``KIND``, one run entry a family) and one stream a family, ``<family>.jsonl``, in
    ``triforce train``'s events (``start``, a ``progress`` an iteration, ``train_end``, ``end``): the iteration
    of N for every family alike, and in ``metrics`` the family's own numbers (``family/<key>/...`` without the
    prefix: success, balanced success, hearts, ``rows_share``, ``hits_taken_per_episode``, need) with the run's
    teacher coefficient (``teacher_coeff``) and losses (``SHARED_METRICS``, ``losses/aux_*``).  The training GUI
    draws the run as one group with a row a family."""

    def __init__(self, plan, keys: list[str], run_id: str):
        self.plan = plan
        self.keys = keys
        self.streams = [EventLog(plan.directory / f'{key}.jsonl', specialist=key, run_id=run_id) for key in keys]
        self.began = time.time()

    @staticmethod
    def runs(keys: list[str], run_id: str, out_dir: Path, total: int, iterations: int) -> list[dict]:
        """The plan's run entries."""
        return [{'specialist': key, 'run_id': run_id, 'run_dir': str(out_dir), 'events': f'{key}.jsonl',
                 'target_steps': total, 'target_iterations': iterations} for key in keys]

    def start(self, **fields):
        self.began = time.time()
        for index, stream in enumerate(self.streams):
            stream.emit('start', **fields)
            self.plan.update_run(index, status='running')

    def progress(self, iteration: int, iterations: int, steps: int, target_steps: int, metrics: dict,
                 teacher: float):
        elapsed = time.time() - self.began
        sps = steps / elapsed if elapsed > 0 else 0.0
        eta = (target_steps - steps) / sps if sps else None
        shared = {key: value for key, value in metrics.items()
                  if key in SHARED_METRICS or key.startswith('losses/aux_')}
        job_progress(steps, target_steps, 'steps')
        for key, stream in zip(self.keys, self.streams):
            prefix = f'family/{key}/'
            own = {name[len(prefix):]: value for name, value in metrics.items() if name.startswith(prefix)}
            stream.emit('progress', iteration=iteration, iterations=iterations, steps=steps,
                        target_steps=target_steps, fraction=steps / target_steps if target_steps else None,
                        elapsed=round(elapsed, 1), sps=round(sps, 1), eta=None if eta is None else round(eta, 1),
                        metrics={**own, **shared, 'teacher_coeff': teacher})

    def end(self, status: str, **fields):
        """Every family's ``train_end`` and ``end`` (``status``: completed | early_stop | interrupted | error)."""
        for index, stream in enumerate(self.streams):
            if status != 'error':
                stream.emit('train_end', status=status, **fields)
            stream.emit('end', status=status, **fields)
            self.plan.update_run(index, status=status)


class FamilyStop:
    """``triforce train``'s early stop (``train.EarlyStop``) for a set of families: each family's own, fed with its
    cohorts; proposed once every family's holds (its balanced success >= the threshold for ``patience``
    iterations), confirmed by an evaluation of each family (``CONFIRM_EPISODES`` episodes a seed start state)
    whose success must reach the threshold too. Without the hearts and bombs plateau rules."""

    def __init__(self, args, collector: MultiCollector, steps_per_env: int):
        self.args = args
        self.collector = collector
        self.stops = {f.key: EarlyStop(args.stop_success, args.stop_patience,
                                       min_iterations=math.ceil(f.config.max_steps / steps_per_env)
                                       + args.stop_patience,
                                       seed_states=pool.start_states or None)
                      for f, pool in zip(collector.families, collector.pools)}

    def update(self, network, iteration: int, metrics: dict) -> bool:
        """One iteration; adds ``family/<key>/balanced_success_rate`` to ``metrics``; True: stop."""
        proposed = True
        for key, stop in self.stops.items():
            rollout = self.collector.by_family[key]
            proposed &= stop.update([counts for _, counts in rollout.cohorts], rollout.episodes,
                                    rollout.summary().get('in_flight/over_half_budget'))
            if stop.balanced is not None:
                metrics[f'family/{key}/balanced_success_rate'] = stop.balanced
        if not proposed:
            return False
        rates = {}
        for family in self.collector.families:
            by_state = success_by_state(family.config, network.to_native(), CONFIRM_EPISODES,
                                        seed=(self.args.seed or 0) + iteration, parallel=self.args.parallel,
                                        states_dir=self.args.states_dir, act=registry.DEFAULT_ACT)
            episodes = sum(e for _, e in by_state.values())
            rates[family.key] = sum(s for s, _ in by_state.values()) / episodes if episodes else 0.0
        confirmed = all(rate >= self.args.stop_success for rate in rates.values())
        print(f"[lab] early stop check at iteration {iteration}: "
              + ' '.join(f'{k}={r:.3f}' for k, r in rates.items()) + (' -> stop' if confirmed else ' -> continue'),
              flush=True)
        if not confirmed:
            for stop in self.stops.values():
                stop.reset()
        return confirmed


def main(argv=None) -> Path:
    # pylint: disable=too-many-locals, too-many-statements
    args = parse_args(argv)
    from ..nets.serve import learner_process  # pylint: disable=import-outside-toplevel
    learner_process()           # its learner uses the GPU beside the rollouts' server: no CUDA graphs there
    if args.mode == 'imitate' and args.teacher_coeff <= 0:
        args.teacher_coeff = 1.0
    if args.actor == 'teacher' and args.mode != 'imitate':
        raise SystemExit("--actor teacher is for --mode imitate (PPO trains on the student's own samples)")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    root = registry.models_root(args.models_dir)
    run_id = args.run_id or registry.new_run_id()
    out_dir = registry.run_dir(args.name, run_id, root)
    shares = {k: float(w) for k, _, w in (s.partition('=') for s in args.shares)}
    families = parse_families(args.families, args.parallel, shares, args.state_route,
                              curriculum=not args.no_curriculum, states_dir=args.states_dir)
    keys = [f.key for f in families]
    steps_per_env = max(1, args.steps_per_update // args.parallel)
    total = args.iterations * steps_per_env * args.parallel
    announce('train', work(train_steps=total), f'lab-{args.name}')
    with logged_job(f'lab-{args.name}', KIND, RunLog.runs(keys, run_id, out_dir, total, args.iterations),
                    args.logs_dir, work=work(train_steps=total), families=keys, run_dir=str(out_dir),
                    mode=args.mode) as plan:
        log = RunLog(plan, keys, run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        network, source, parent = initial_network(args)
        aux_losses = [AuxLoss.parse(text) for text in args.aux]
        tried = parse_try_others(args.try_others)
        recurrent = getattr(network, 'recurrent', False)
        if recurrent and steps_per_env % args.bptt:
            raise SystemExit(f"steps per env ({steps_per_env}) must be a multiple of --bptt ({args.bptt})")
        if args.demos and (recurrent or args.mode != 'ppo'):
            raise SystemExit("--demos is for --mode ppo with a network without memory (demonstration rows are no "
                             "sequences)")
        sampler = None
        if args.sampling == 'need':
            targets = (NeedSampler.teacher_targets(keys, REFERENCE) if args.need_target == 'teacher'
                       else {key: float(args.need_target) for key in keys})
            for key, _, value in (item.partition('=') for item in args.need_target_for):
                if key not in targets:
                    raise SystemExit(f"--need-target-for {key}: not in --families")
                targets[key] = float(value)
            sampler = NeedSampler(targets, args.need_floor, args.need_window)
        teachers = None
        if args.teacher_coeff > 0 or args.mode == 'imitate':
            overrides = dict(item.split('=', 1) for item in args.teacher)
            weights = {k: float(w) for k, _, w in (item.partition('=') for item in args.teacher_weight)}
            unknown = sorted(set(weights) - set(keys))
            if unknown:
                raise SystemExit(f"--teacher-weight {unknown}: not in --families")
            teachers = Teachers(keys, args.device, root, overrides, weights, args.state_route)
        rehearsal = None
        if args.rehearse:
            names = json.loads((Path(args.rehearse) / 'cache.json').read_text(encoding='utf-8'))['keys']
            rehearsed = args.rehearse_keys if args.rehearse_keys is not None else [k for k in names
                                                                                   if k not in keys]
            rehearsal = Rehearsal(Path(args.rehearse), rehearsed, args.rehearse_rows, seed=args.seed or 0)
            print(f"[lab] rehearsal: {rehearsal.rows:,} rows of {len(rehearsed)} keys from {args.rehearse} "
                  f"(weight {args.rehearse_coeff})", flush=True)
        curricula = {}
        for family in families:
            spec, _ = curriculum_spec(argparse.Namespace(no_curriculum=args.no_curriculum, curriculum=None,
                                                         curriculum_fraction=None, curriculum_threshold=None,
                                                         curriculum_unrecorded=None),
                                      family.config, out_dir / 'tasks' / family.key)
            curricula[family.key] = spec
        (out_dir / 'tasks').mkdir(exist_ok=True)
        for family in families:
            (out_dir / 'tasks' / family.key).mkdir(parents=True, exist_ok=True)
        cfg = PPOConfig(target_steps=steps_per_env * args.parallel, learning_rate=args.learning_rate,
                        ent_coeff=args.ent_coeff, num_epochs=args.epochs,
                        demo_bc_coeff=args.demo_coeff if args.demos else 0.0)
        ppo = PPO(cfg, device=args.device)
        ppo.aux_losses = aux_losses
        ppo.rehearsal = rehearsal
        ppo.rehearsal_coeff = args.rehearse_coeff
        if args.demos:
            ppo.demo_batch = load_demos(args.demos)
            print(f"[lab] demonstrations: {ppo.demo_batch[2].shape[0]} rows from {', '.join(args.demos)} "
                  f"(weight {args.demo_coeff})", flush=True)
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'little')
        collector = MultiCollector(families, out_dir, steps_per_env, cfg.gamma, cfg.lam, seed=seed,
                                   states_dir=args.states_dir, curricula=curricula, teachers=teachers,
                                   actor=args.actor if args.mode == 'imitate' else 'student', tried=tried,
                                   chunk_length=args.bptt, sampler=sampler)
        stopper = FamilyStop(args, collector, steps_per_env) if args.stop_success is not None else None
        run = {'args': vars(args), 'name': args.name, 'run_id': run_id, 'families': keys,
               'envs': collector.shares, 'arch': nets.arch_of(network.config.to_dict()),
               'parameters': network.parameter_count(), 'model_config': network.config.to_dict(),
               'ppo_config': cfg.to_dict(), 'source': source, 'parent': parent,
               'teachers': None if teachers is None else teachers.paths,
               'teacher_ids': None if teachers is None else teachers.identities}
        (out_dir / 'run.json').write_text(json.dumps(run, indent=2, default=str), encoding='utf-8')
        link_families(root, args.name, run_id, keys)
        print(f"[lab] {args.name} run={run_id} dir={out_dir} arch={run['arch']} "
              f"parameters={run['parameters']:,} envs={collector.shares} mode={args.mode}", flush=True)
        logger = MetricsLogger(str(out_dir))
        state = {'iteration': 0, 'steps': 0, 'metrics': {}, 'stopped_early': False}

        def save():
            registry.save_checkpoint(args.name, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                     steps=state['steps'], metrics=state['metrics'], root=root,
                                     extra={'families': keys, 'lab': {'mode': args.mode, 'source': source}})

        def on_iteration(result: IterationResult):
            state['iteration'], state['steps'] = result.iteration, result.total_steps
            metrics = {**result.rollout.summary(), **result.stats, **collector.family_metrics()}
            stop = stopper is not None and stopper.update(network, result.iteration, metrics)
            state['metrics'] = metrics
            logger.log(result.iteration, result.total_steps, metrics)
            log.progress(result.iteration, args.iterations, result.total_steps, total, metrics, ppo.teacher_coeff)
            per_family = ' '.join(f"{k}={metrics.get(f'family/{k}/success_rate', float('nan')):.2f}" for k in keys)
            print(f"[lab] iter={result.iteration} steps={result.total_steps} {per_family} "
                  + format_summary(metrics), flush=True)
            ppo.teacher_coeff = teacher_coeff(args, result.iteration + 1)
            if result.iteration % args.save_every == 0:
                save()
            if args.checkpoint_every and result.iteration % args.checkpoint_every == 0:
                save()
                registry.snapshot_checkpoint(registry.checkpoint_path(args.name, run_id, root), result.iteration,
                                             args.checkpoint_keep)
            if stop:
                state['stopped_early'] = True
                return False
            return True

        began = time.time()
        status = 'completed'
        log.start(target_steps=total, target_iterations=args.iterations, steps_per_update=cfg.target_steps,
                  start_iteration=0, start_steps=0, parallel=args.parallel, device=args.device,
                  run_dir=str(out_dir), stop_success=args.stop_success, families=keys, mode=args.mode,
                  arch=run['arch'], parameters=run['parameters'], envs=collector.shares)
        try:
            ppo.teacher_coeff = teacher_coeff(args, 1)
            ppo.teacher_value_coeff = args.teacher_value_coeff
            if args.mode == 'ppo':
                ppo.train(network, collector, args.iterations, callback=on_iteration, seed=args.seed,
                          collector=collector)
            else:
                optimizer = ppo.setup_optimizer(network)
                for iteration in range(1, args.iterations + 1):
                    collector.start(network)
                    collector.take()
                    rollout = collector.ready(network, ppo.device)
                    stats = imitate(network, optimizer, collector.buffer.flatten(ppo.device), cfg,
                                    args.teacher_value_coeff, aux_losses)
                    ppo.total_steps += collector.buffer.rows
                    if on_iteration(IterationResult(iteration, ppo.total_steps, rollout, stats)) is False:
                        break
        except KeyboardInterrupt:
            status = 'interrupted'
        except BaseException as error:
            log.end('error', message=f"{type(error).__name__}: {error}", iterations=state['iteration'],
                    steps=state['steps'])
            raise
        finally:
            collector.close()
            logger.close()
            if state['iteration']:
                save()
        status = 'early_stop' if state['stopped_early'] else status
        summary = {'status': status, 'iterations': state['iteration'], 'steps': state['steps'],
                   'wall_seconds': round(time.time() - began, 1)}
        log.end(status, iterations=state['iteration'], steps=state['steps'], wall_seconds=summary['wall_seconds'])
        (out_dir / 'train_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"[lab] {status}: {summary}", flush=True)
    return out_dir


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
