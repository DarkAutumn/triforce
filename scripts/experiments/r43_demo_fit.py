"""Round 40, Carrier43 (round40.md "Why the Darknut teacher walks past 3x5b's bomb carriers", 2): how well a checkpoint
plays darknut@d3first's demonstrations, group by group (argmax agreement and mean log-prob of the demonstrated action),
as PPO's cloning term scores them.  A job of the queue:

    triforce job run ... --express --timeout 10m -- .venv/bin/python scripts/experiments/r43_demo_fit.py CKPT...
"""
import sys
import torch
from triforce import jobs
from triforce.demos import load_demos
from triforce.ppo import compute_demo_accuracy, demo_log_prob
from triforce.registry import load_model

DEMOS = '/home/leculver/triforce-shared/data/demos/261124-gen3/'
GROUPS = {'5b-first': ['r40-darknut-5b-low.npz', 'r40-darknut-5b-full.npz'], '59-b4': ['r40-darknut-59-b4.npz'],
          '59-b8': ['r40-darknut-59-b8.npz'], 'garden-b0': ['r40-darknut-garden-b0.npz'],
          'garden-b8': ['r40-darknut-garden-b8.npz', 'r40-darknut-garden-b8-p2.npz'],
          '69-in': ['r40-d3-69-in.npz'], '69-out': ['r40-d3-69-out.npz']}
BATCH = 4096


def main() -> int:
    jobs.require_job()
    data = {g: load_demos([DEMOS + f for f in files]) for g, files in GROUPS.items()}
    print('rows', {g: int(v[2].shape[0]) for g, v in data.items()})
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, ckpt in enumerate(sys.argv[1:]):
        net, _ = load_model(ckpt)
        net = net.to(dev).eval()
        line = []
        for g, (obs, masks, acts) in data.items():
            acc, lp, n = 0.0, 0.0, acts.shape[0]
            with torch.no_grad():
                for s in range(0, n, BATCH):
                    o = {k: v[s:s + BATCH].to(dev) for k, v in obs.items()}
                    m, a = masks[s:s + BATCH].to(dev), acts[s:s + BATCH].to(dev)
                    acc += compute_demo_accuracy(net.evaluate(o, m, deterministic=True).actions, a) * a.shape[0]
                    lp += demo_log_prob(net, o, m, a).sum().item()
            line.append(f'{g} acc {acc / n:.3f} lp {lp / n:.3f}')
        print(ckpt, ' | '.join(line), flush=True)
        jobs.progress(i + 1, len(sys.argv) - 1, 'checkpoints')
    return 0


if __name__ == '__main__':
    sys.exit(main())
