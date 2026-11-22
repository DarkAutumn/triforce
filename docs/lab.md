# The lab: how experimental networks are trained, played and judged (round 36)

The program since Nov 21 (`docs/experiments/backlog.md` Track M) tries networks of other shapes, sizes and
observations, trained over several families at once with today's 51 networks as teachers. This is the machinery
they all run on: `triforce/nets/` (network classes), `triforce/nets/serve.py` (how a torch network acts in the
pool), `triforce/lab/` (multi-family training, teachers, datasets, the yardstick). Today's networks and their Rust
actor are untouched: the Rust actor stays the production path, and a winner is ported to it later.

## How an experimental network acts

Today the pool's envs act with `crates/policy`, a hand-written Rust copy of the one architecture
(`docs/rust-actor.md`). A new architecture must not need a port to be tried. Three ways were weighed, measured on
this machine (Nov 21; the machine shared, load average about 30; one `ZeldaPolicy`, zol's run, 866k parameters):

| Path | What it costs | Measured |
|---|---|---|
| (a) torch in Python computes actions for the pool's envs in batches | one rendezvous per decision; Python's GIL; batches as large as the envs waiting at once | torch CPU, one thread: 357 rows/s at batch 1, 2,064 / 2,548 / 4,767 at batch 8 / 16 / 64; torch CUDA: 438 at batch 1, 3,007 / 4,445 / 10,149 at batch 8 / 16 / 64 (copy back included) |
| (b) a generic exported graph run from Rust | TorchScript: libtorch (2 GB, its ABI tied to the torch version) linked into the module through `tch`; ONNX Runtime: the `ort` crate and its shared library, and every architecture exportable (today's isn't: `EntityEncoder`'s in-place padding mask, `aten::__iand_`, fails `torch.onnx.export`) | TorchScript on CPU, one thread, batch 1: 539 rows/s (frozen, optimised); ORT not measured (the export fails) |
| (c) the Rust copy for the standard shape, a slower generic path for the rest | nothing new for today's networks; (a) or (b) for the others | the Rust copy: 1,288 rows/s, one thread, batch 1 (0.31 ms on an idle machine, `rust-actor.md`) |

In the pool, playing a key's bench rooms on 10 envs (`python -m triforce.lab.bench speed`, the same states and seeds for
each path):

| Acting path | zol (4,126 decisions) | darknut (5,272 decisions) | Against the Rust actor |
|---|---:|---:|---:|
| Rust actor (`crates/policy`) | 885 decisions/s | 832 decisions/s | 1 |
| served, torch on the GPU | 147 (before the batching fix) | 235 | 3.5× slower |
| served, torch on the CPU (4 threads) | 4 (torch's default threads, before the fix) | 31 | 27× slower |

(Wall time from the pool's start, so the Rust actor's figure is mostly start-up: its true rate in play is higher.
Each path played the same decisions to the same successes: the served network computes the Rust network's
choices within float tolerance.) **The cost of the generic path is about 3.5× the Rust actor's time on the GPU**:
a 10-minute chain batch becomes 35 minutes, a 1,000-run segment chain a few hours; fine for the cheap tier and
training, a reason to port a winner before its full tier. Serve on the GPU; the CPU path is a fallback for tests.

**Chosen: (c) with (a) as the generic path.** A network the Rust actor knows (`ZeldaPolicy`) plays as before, bit
for bit; any other `torch.nn.Module` with the interface below is served by torch from the Python process that owns
the pool. (b) was not built: it costs a heavy native dependency, every architecture must export (recurrent state
and attention masks are where exports break), and a single row through TorchScript is still slower than the batch
of 8–16 that (a) gets from the waiting envs. The Rust copy stays the reference: a winner is ported.

**How (a) works.** A worker that decides for a served network posts its env's observation to a queue
(`pool::zelda::external::ExternalQueue`) and waits; a server thread (`nets.serve.Server`) takes every pending row at
once, runs each network on its rows and answers with the **raw head logits** and the value. The worker then picks
the action itself in Rust from those logits (`policy::logits::Heads::act`): the masks, the forced-kind rule, heads
that don't apply, the act modes (`temperature:0.5+relax:16`, `sample`, `argmax`, `floor`, `topp`, `escape`, `low`,
...) and the keyed Gumbel noise of (job, decision) are the Rust actor's code, so a served network plays exactly like
the Rust actor given the same logits (`crates/policy/tests/policy.rs`
`acting_from_logits_is_acting_with_the_policy`; `tests/test_lab_nets.py`: serving the Rust network's own logits
plays the same runs as the Rust actor, run for run, in three modes). A served torch network computes its logits
in another order than Rust, so it plays the same seeds within float tolerance, not bit for bit (replays record
inputs and verify regardless). Batches: one env per worker thread (pools with a served network use as many threads
as envs; a waiting thread costs no CPU), the server gathers what is pending. `EnvPool.play` and `EnvPool.rollout`
take Rust and served policies in one list (`HybridActor`), so a chain can play an experimental network for some
families and today's for the rest. Not for served networks: foresight (`guard`, `plan`), search, recorded
distributions and action replays (refused when the pool starts).

**Recurrent state, per env.** Each env's actor scratch has a slot; the server keeps a recurrent network's state per
(network, slot), starts it over at a job's decision 0 or a new job, and a bootstrap value request (a truncation)
reads it without advancing it (`tests/test_lab_nets.py`). Training a recurrent network is not built: the learner
recomputes rows independently from the initial state (`LabNet._heads`); sequences need the rollout's order (the
buffer keeps `jobs` and `steps`) and a burn-in.

Devices: `$TRIFORCE_SERVE_DEVICE` (`cpu` | `cuda`; default cuda when present). During training the served copy
shares the learner's GPU.

## Adding a network

A network class lives in `triforce/nets/<name>.py`, subclasses `nets.base.LabNet`, and registers itself:

```python
from . import register
from .base import Heads, LabNet

@register
class MyNet(LabNet):
    arch = 'mynet'                                  # the name checkpoints carry (model_config['arch'])
    defaults = {'hidden_dim': 512}                  # every setting it takes, with its default

    def __init__(self, config: dict):
        super().__init__(config)                    # self.config: defaults + given settings + arch + obs_spec
        ...                                         # layers; self.config.obs_spec is the observation layout

    def heads(self, obs: dict, state=None):         # obs: key -> tensor [B, ...] (observation.py), on self's device
        ...
        return Heads(kind=...,       # [B, 11] logits
                     direction=...,  # [B, 11, 8]: row k = the direction head given kind k
                     duration=...,   # [B, 11, 4]: row k = the duration head given kind k
                     value=...,      # [B]
                     aux={}), None   # aux: anything for its own losses; None: the new state (no memory)

    def initial_state(self, batch):  # only a network with memory: [batch, ...]
        return None
```

and `from . import mynet` at the end of `triforce/nets/__init__.py`. That is all: masks, sampling, act modes,
log-probs, entropies, the joint distribution, `evaluate`, `get_value`, `joint_log_probs`, checkpoints
(`registry.save_checkpoint` / `load_model`), evaluation, chains, training and the bench come from `LabNet` and
the logits. Heads whose direction and duration don't depend on the kind repeat one row 11 times. The observation is
`observation.py`'s (schema 6) for now; a network that wants another one waits for the observation round.
`triforce/nets/wide.py` is the worked example (today's encoders, wider towers). Test it like
`tests/test_lab_nets.py` `test_architectures_side_by_side_save_load_and_play`.

`ZeldaPolicy` also answers `heads` (so it can be served; `tests/test_lab_nets.py` checks that its logits give
its own `evaluate` and `joint_log_probs`). A checkpoint without `model_config['arch']` is a `ZeldaPolicy`; old and
new checkpoints load side by side (no COMPAT change).

## Several families in one run

```
python -m triforce.lab.train --name zol+skg --families zol stalfos-keese-gel --parallel 8 --iterations 80 \
    --load models/zol/r79b-zol --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 60 --seed 36
```

- `--families KEY[=YAML]`: a family (`zol`) or a route's own run (`zol@d3first`, the wood-sword half) with its task
  yaml when it isn't the family's (`zol@d3first=path/zol.yaml`). Each yaml stays its task: rooms, start states,
  hp and item randomisation, masks, rewards, `max_steps`, curriculum (its progress in `<run>/tasks/<key>/`).
- Every family has its own `EnvPool` and collector; `--parallel` envs are split by `--shares KEY=W` (default
  equal), at least one a family. The learner trains on all rows at once (`lab.collect.BufferSet`).
  `MultiCollector.shares` is the place prioritised sampling across families goes (`plr.py`'s scores per family).
- Metrics: the merged ones plus `family/<key>/<metric>` for every rollout metric (`metrics.jsonl`; the log line
  prints each family's success).
- `--arch NAME --arch-config JSON` for a fresh network of another class; `--load CKPT` to start from a checkpoint.
- The run is `<models>/<name>/<run-id>`; every member family gets a link `<models>/<family>/<run-id>` to it. So
  `triforce promote zol <run>` (and `--route d3first` for a route part), `triforce evaluate` per family,
  `eval-chain`, `select-checkpoint`, `triforce list` and the version stamp work unchanged, and `runner.Policies`
  loads a run that several keys name once.
- Not supported in a lab run (refused or absent): PLR, search, best-of-N, demonstrations, SIL, PPG, the early stop.

## Teachers

The teacher of a row is the network that plays its family's rooms today (`registry.policy_checkpoint(key)`: the
family's `promoted`, or `promoted.d3first` for `family@d3first`; `--teacher KEY=CKPT` overrides), on the GPU, run
once per rollout over its family's rows in batches of 4096 (`lab.collect.Teachers`).

- **With PPO** (`--mode ppo`): `loss += c · KL(teacher ‖ student)` over the 352 joint actions (the exact factored
  KL: kind, direction and duration together), plus `--teacher-value-coeff` · ½(V − V_teacher)² (off by default:
  teachers' values are on their own reward scales). The weight `c` is `--teacher-coeff` for `--teacher-hold`
  iterations, then falls linearly to 0 over `--teacher-fade` (0: stays). Logged: `losses/teacher_kl`,
  `charts/teacher_coeff`.
- **Imitation alone** (`--mode imitate`): no PPO, the student minimises the KL (and the value term) over each
  rollout for `--epochs` epochs; logged `charts/teacher_agreement` (the student's most likely joint action is the
  teacher's). `--actor student` (default) collects on the student's own states: DAgger, the distribution it will
  have to play, so its errors are the ones that compound; `--actor teacher` collects on the teachers' states: a
  fixed distribution, which is what a capacity test wants (does a network of this size represent these policies?)
  but which over-states how well the student plays.
- **Datasets** for offline work: `python -m triforce.lab.record --families ... --actor teacher|student [--student
  CKPT] --rollouts N --out DIR` writes `DIR/shard-NNNN.npz` (every observation key as `obs/<key>`,
  `action_mask`, `action`, `teacher_joint` `[n, 352]` (`-inf` illegal), `teacher_value`, `family`) and
  `DIR/dataset.json` (families, actor, teachers, envs); `lab.record.load(DIR)` reads it back.

## The yardstick

`python -m triforce.lab.bench` (`triforce/lab/bench.py`; the cheap tier's manifest `triforce/lab/bench.json`):

- **Cheap tier.** Every policy key of today's team (51: 34 families, 17 `family@d3first`), each on 24 room entries
  as the chains bring them: the harvested pre-entry states (`triforce/states/harvested/<family>/`, untracked: copy
  them from main), `d3first_*` ones where a family has them, spread evenly over the family's rooms by a fixed hash
  order; 5 episodes each on fixed seeds (360000 + i), `temperature:0.5+relax:16`, clips on: 5,735 episodes. A
  route part plays its family's states (the harvest does not tell the two swords apart). **Coverage missing**
  (no harvested entries; their seed start states stand in): boss-gohma, ganon, ow-boulder, ow-ghini; and lanmola,
  patra, wizzrobe and ow-lynel have only older routes' entries.
- Per key and in total: success, hearts lost per episode, hits taken per room visit, decisions per episode, bombs
  used per episode. Every episode is kept (state, seed), so `compare` pairs two networks episode by episode and gives
  each difference with its standard error.
- `run --team --out team.json` (today's networks), `run --network CKPT --keys zol stalfos-keese-gel --out
  student.json` (one network for some keys), `run --checkpoint KEY=CKPT ...`, `compare team.json student.json`.
- **Full tier** (`bench.json` `full`): the route's segment chains as the landings run them, `triforce eval-batch
  TAG --chains TAG --route d3first --runs 1000 --act model`, a candidate's networks given with `--checkpoint`; the
  `full` segment is the route from the start. It is what decides; the cheap tier only shortlists.

**The reference** (`triforce/lab/reference.json`: today's team, 2.0.0.3's 51 networks, on `bench.json`; played
Nov 21 in 19.5 minutes of play on 10 envs): 5,735 episodes, success **91.9%**, hearts lost **0.241** an episode,
hits taken **0.697** a room visit, **36.2** decisions and **0.524** bombs used an episode; per key in the file.
Every experiment is compared with it: `python -m triforce.lab.bench compare triforce/lab/reference.json NEW.json`.

## The first experiment through the pipeline (smoke, Nov 21)

One `ZeldaPolicy` for zol and stalfos-keese-gel, started from zol's run (`r79b-zol`), the families' default runs as
teachers (KL weight 1.0 for 20 iterations, fading to 0 over 60), 80 iterations on 8 envs (4 a family), 327,680 steps
in 13.5 minutes (`models/zol+skg/r36lab-smoke`; the command above). Training success at the end: zol 0.97,
stalfos-keese-gel 1.00. On the yardstick (240 paired episodes, the bench's seeds, used for nothing else), against
its teachers: success **+1.7 ± 1.6** points, hearts lost **−0.09 ± 0.03** an episode, hits **−0.16 ± 0.06** an
episode (zol +5.0 ± 2.6, hearts −0.16 ± 0.05; stalfos-keese-gel −1.7 ± 1.7). Against the networks that play those
rooms on `d3first` today (the `@d3first` runs, which the bench's `d3first_*` entries belong to), it is worse:
success −3.3 ± 1.3, hearts lost +0.17 ± 0.03. One run, one seed: it shows the pipeline runs and reports, and that
the teacher of a d3first room should be the `@d3first` run (`--families zol@d3first=...`), not the default one.
