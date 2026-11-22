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
Metrics are logged per family (``family/<key>/...``) beside the merged ones.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from .. import nets, registry
from ..events import logged_job
from ..jobs import add_job_arguments, begin_job, work
from ..metrics import MetricsLogger, format_summary
from ..model import obs_spec_from_space
from ..observation import observation_space
from ..ppo import PPO, IterationResult, PPOConfig, joint_kl
from ..train import curriculum_spec
from .collect import MultiCollector, Teachers, parse_families


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
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--no-curriculum', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    add_job_arguments(parser)
    return parser.parse_args(argv)


def teacher_coeff(args, iteration: int) -> float:
    """The teacher term's weight at (1-based) ``iteration``: held, then fading linearly to 0."""
    if args.teacher_fade <= 0 or iteration <= args.teacher_hold:
        return args.teacher_coeff
    return args.teacher_coeff * max(0.0, 1.0 - (iteration - args.teacher_hold) / args.teacher_fade)


def initial_network(args):
    """(network, source)."""
    if args.load:
        network, _ = registry.load_model(args.load)
        return network, str(args.load)
    spec = obs_spec_from_space(observation_space())
    return nets.create(args.arch, spec, **json.loads(args.arch_config)), None


def link_families(root: Path, name: str, run_id: str, keys: list[str]):
    """``<models>/<family>/<run-id>`` -> ``../<name>/<run-id>`` for every member family."""
    for family in sorted({registry.split_policy(k)[0] for k in keys}):
        link = registry.run_dir(family, run_id, root)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path('..') / name / run_id)


def imitate(network, optimizer, batch: dict, cfg: PPOConfig, value_coeff: float) -> dict:
    """One pass of the imitation-only update over ``batch`` (``BufferSet.flatten``): KL(teacher ‖ student)
    (+ ``value_coeff`` · ½(V − V_teacher)²) in ``cfg.minibatches`` minibatches, ``cfg.num_epochs`` epochs."""
    rows = batch['masks'].shape[0]
    network.train()
    kl = torch.tensor(0.0)
    for _ in range(cfg.num_epochs):
        for mb in torch.tensor_split(torch.randperm(rows, device=batch['masks'].device), cfg.minibatches):
            obs = {k: v[mb] for k, v in batch['obs'].items()}
            joint = network.joint_log_probs(obs, batch['masks'][mb])
            kl = joint_kl(batch['teacher_joint'][mb], joint)
            loss = kl
            if value_coeff > 0:
                loss = loss + value_coeff * 0.5 * (network.get_value(obs) - batch['teacher_value'][mb]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
            optimizer.step()
    network.eval()
    with torch.no_grad():
        agree = []
        for mb in torch.split(torch.arange(rows, device=batch['masks'].device), 4096):
            obs = {k: v[mb] for k, v in batch['obs'].items()}
            agree.append(network.joint_log_probs(obs, batch['masks'][mb]).argmax(-1)
                         == batch['teacher_joint'][mb].argmax(-1))
    return {'losses/teacher_kl': float(kl.detach()), 'charts/teacher_agreement': float(torch.cat(agree).float().mean())}


def main(argv=None) -> Path:
    # pylint: disable=too-many-locals, too-many-statements
    args = parse_args(argv)
    begin_job(args)
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
    families = parse_families(args.families, args.parallel, shares)
    keys = [f.key for f in families]
    steps_per_env = max(1, args.steps_per_update // args.parallel)
    total = args.iterations * steps_per_env * args.parallel
    with logged_job(f'lab-{args.name}', 'train', [], log=False, work=work(train_steps=total)):
        out_dir.mkdir(parents=True, exist_ok=True)
        network, source = initial_network(args)
        teachers = None
        if args.teacher_coeff > 0 or args.mode == 'imitate':
            overrides = dict(item.split('=', 1) for item in args.teacher)
            teachers = Teachers(keys, args.device, root, overrides)
        curricula = {}
        for family in families:
            spec, _ = curriculum_spec(argparse.Namespace(no_curriculum=args.no_curriculum, curriculum=None,
                                                         curriculum_fraction=None, curriculum_threshold=None),
                                      family.config, out_dir / 'tasks' / family.key)
            curricula[family.key] = spec
        (out_dir / 'tasks').mkdir(exist_ok=True)
        for family in families:
            (out_dir / 'tasks' / family.key).mkdir(parents=True, exist_ok=True)
        cfg = PPOConfig(target_steps=steps_per_env * args.parallel, learning_rate=args.learning_rate,
                        ent_coeff=args.ent_coeff, num_epochs=args.epochs)
        ppo = PPO(cfg, device=args.device)
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'little')
        collector = MultiCollector(families, out_dir, steps_per_env, cfg.gamma, cfg.lam, seed=seed,
                                   states_dir=args.states_dir, curricula=curricula, teachers=teachers,
                                   actor=args.actor if args.mode == 'imitate' else 'student')
        run = {'args': vars(args), 'name': args.name, 'run_id': run_id, 'families': keys,
               'envs': collector.shares, 'arch': nets.arch_of(network.config.to_dict()),
               'parameters': network.parameter_count(), 'model_config': network.config.to_dict(),
               'ppo_config': cfg.to_dict(), 'source': source,
               'teachers': None if teachers is None else teachers.paths}
        (out_dir / 'run.json').write_text(json.dumps(run, indent=2, default=str), encoding='utf-8')
        link_families(root, args.name, run_id, keys)
        print(f"[lab] {args.name} run={run_id} dir={out_dir} arch={run['arch']} "
              f"parameters={run['parameters']:,} envs={collector.shares} mode={args.mode}", flush=True)
        logger = MetricsLogger(str(out_dir))
        state = {'iteration': 0, 'steps': 0, 'metrics': {}}

        def save():
            registry.save_checkpoint(args.name, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                     steps=state['steps'], metrics=state['metrics'], root=root,
                                     extra={'families': keys, 'lab': {'mode': args.mode, 'source': source}})

        def on_iteration(result: IterationResult):
            state['iteration'], state['steps'] = result.iteration, result.total_steps
            metrics = {**result.rollout.summary(), **result.stats, **collector.family_metrics()}
            state['metrics'] = metrics
            logger.log(result.iteration, result.total_steps, metrics)
            per_family = ' '.join(f"{k}={metrics.get(f'family/{k}/success_rate', float('nan')):.2f}" for k in keys)
            print(f"[lab] iter={result.iteration} steps={result.total_steps} {per_family} "
                  + format_summary(metrics), flush=True)
            ppo.teacher_coeff = teacher_coeff(args, result.iteration + 1)
            if result.iteration % args.save_every == 0:
                save()
            if args.checkpoint_every and result.iteration % args.checkpoint_every == 0:
                save()
                registry.snapshot_checkpoint(registry.checkpoint_path(args.name, run_id, root), result.iteration)
            return True

        began = time.time()
        status = 'completed'
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
                                    args.teacher_value_coeff)
                    ppo.total_steps += collector.buffer.memory_length * collector.buffer.n_envs
                    on_iteration(IterationResult(iteration, ppo.total_steps, rollout, stats))
        except KeyboardInterrupt:
            status = 'interrupted'
        finally:
            collector.close()
            logger.close()
            if state['iteration']:
                save()
        summary = {'status': status, 'iterations': state['iteration'], 'steps': state['steps'],
                   'wall_seconds': round(time.time() - began, 1)}
        (out_dir / 'train_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"[lab] {status}: {summary}", flush=True)
    return out_dir


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
