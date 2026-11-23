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

**By envs, precision and size** (Nov 22, `bench speed --paths rust cuda cuda-half --parallel N`, the zol entry of the
bench; the machine shared, load average 40-45 on 20 threads, so the figures are noisy and wall time includes the
pool's start-up; decisions a second):

| Network | Path | 10 envs | 32 envs | 64 envs |
|---|---|---:|---:|---:|
| today's (zol's run, 866k parameters; 4,126 decisions) | Rust actor | 580 | 1,191 | 732 |
| | served, GPU, float32 | 217 | 419 | 469 |
| | served, GPU, float16 (`cuda-half`) | 200 | 307 | 275 |
| 4× (`ZeldaPolicy` hidden 704, 3.44M parameters, random weights; 15,091 decisions) | Rust actor | 764 | 1,045 | 1,096 |
| | served, GPU, float32 | 265 | 434 | 467 |
| | served, GPU, float16 | 246 | 317 | 474 |

More envs help the served path most (bigger batches: 217 → 469 a second); it takes 1.6-2.8× the Rust actor's time
at 32-64 envs and 2.7-2.9× at 10 (the Rust figures move with the load). A network four times the size costs no
more a decision on either path at these sizes: emulation, observation and the rendezvous dominate. **Half precision
never pays** (equal or slower: autocast's casts cost more than float16 saves on batches this small) and changes
some decisions (4,400 against 4,126 on the same seeds): leave `TRIFORCE_SERVE_HALF` off.

**CUDA graphs (Nov 22, on by default on the GPU; `TRIFORCE_SERVE_GRAPHS=0` turns them off).** The eager forward of
a served network costs about 4.5 ms whatever the batch (1 to 32 rows): Python and kernel launches, not arithmetic.
A chain that served four `hitshift` networks in Dungeon 3 kept its server thread busy 99.5% of the time and its
emulators waiting, at 3-4 of its 10 cores. The server now captures each network's forward once per batch-size
bucket (1, 2, 4, ..., 64; rows padded by repeating the last) and replays it: 0.67 ms at batch 1, 0.96 at 10, 1.36
at 32 (a `hitshift` of 2.4M parameters). The answers agree with the eager forward within 2e-5 (3,000 real batches
of four networks, no top-action change); a 20-run chain logging every decision matched the eager chain decision
for decision. A graph's first three replays are checked against the eager forward; a capture that fails or a
replay that disagrees serves that network eagerly from then on, with a warning. Recurrent networks stay eager.
With graphs the batch window drops from 5 ms to 1 ms (`GRAPH_GATHER`; `TRIFORCE_SERVE_GATHER` overrides): a
forward is cheap, so answering sooner beats a fuller batch. The 200-run `d3first` chain from `3_7cs_0` with four
`hitshift` networks (106,936 decisions, every run identical across the three):

| Serving | Wall | CPU (cores) | Decisions a second |
|---|---:|---:|---:|
| eager, 5 ms window (before) | 537 s | 20.7 (idle threads spinning) | 199 |
| graphs, 5 ms window | 134 s | 4.2 | 798 |
| graphs, 1 ms window (default) | 100 s | 11.0 | 1,068 |

**One index a served network (Nov 22).** `load_model(path)[0].to_native()` in a loop (`runner.Policies`) served
each copy under the id of a temporary that died at once; the next network often had the same id and replaced the
earlier one at its index, so a chain serving two or more torch networks could play one family's rooms with
another's network (2 processes of 3 when loading four; which family, random per process). A dead network's key is
now forgotten and indices are never reused (`tests/test_lab_nets.py`). Unaffected: one served network per process
(also loaded one after another, each arm replacing the last), networks the Rust actor plays.

**Chosen: (c) with (a) as the generic path.** A network the Rust actor knows (`ZeldaPolicy`) plays as before, bit
for bit (100 chain runs of `d3first` from `3_7cs_0`, seeds 0-99, today's networks: every run's record identical,
field for field, between main and this branch; again at 2.0.0.5 on the corrected route, between the route's branch
before and after main's lab was merged); any other `torch.nn.Module` with the interface below is served by torch from the Python process that owns
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
reads it without advancing it (`tests/test_lab_nets.py`). Training it: "Memory" below.

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
`observation.py`'s (schema 7: schema 6's keys plus the per-entity `entity_rel` and `entity_contact`, which only the
networks that ask for them read; `docs/architecture.md` "Observation").
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
  yaml when it isn't the family's (`zol@d3first=path/zol.yaml`). Without one, a route part reads
  `specialists/<route>/<family>.yaml` when it exists (`lab.collect.task_yaml`): the seventeen `specialists/d3first/`
  files are the wood-sword runs' tasks rebuilt from round 34's recipe (the family's yaml with `state_routes:
  [d3first]`, `ram_randomization.chain_hp: 1.0`, `harvested_fraction: 0.3`; the files each run trained on lived in
  worktrees that are gone, and some runs also narrowed the rooms). Each yaml stays its task: rooms, start states,
  hp and item randomisation, masks, rewards, `max_steps`, curriculum (its progress in `<run>/tasks/<key>/`),
  harvested starts (`harvested_fraction`: drawn as in `triforce train`, `tests/test_lab_memory.py`).
- Every family has its own `EnvPool` and collector; `--parallel` envs are split by `--shares KEY=W` (default
  equal), at least one a family, fixed for the run. The learner trains on all rows at once (`lab.collect.BufferSet`).
- Metrics: the merged ones plus `family/<key>/<metric>` for every rollout metric, `family/<key>/rows_share` and
  `family/<key>/hits_taken_per_episode` (the decisions on which Link lost health over the episodes the rollout
  ended: the bench's hits a visit, in training) (`metrics.jsonl`; the log line prints each family's success).
- `--arch NAME --arch-config JSON` for a fresh network of another class; `--load CKPT` to start from a checkpoint.
- The run is `<models>/<name>/<run-id>`; every member family gets a link `<models>/<family>/<run-id>` to it. So
  `triforce promote zol <run>` (and `--route d3first` for a route part), `triforce evaluate` per family,
  `eval-chain`, `select-checkpoint`, `triforce list` and the version stamp work unchanged, and `runner.Policies`
  loads a run that several keys name once.
- As in `triforce train`: `--checkpoint-every N [--checkpoint-keep K]` (numbered checkpoints `<run>/checkpoints/
  it<N>/`, `select-checkpoint FAMILY RUN` reads them through the family's link), `--demos FILE... --demo-coeff C`
  (PPO with a network without memory), `--stop-success X --stop-patience N` (every family's own `EarlyStop` must
  hold, then an evaluation of each family, `CONFIRM_EPISODES` episodes a seed state, must reach X; logged
  `family/<key>/balanced_success_rate`; without `triforce train`'s hearts and bombs plateau rules).
- Not supported in a lab run (absent): PLR, search, best-of-N, SIL, PPG.

### What a lab job writes (and the training GUI)

**A job the registry knows is never invisible** (the owner watches `training-gui.py` to see what the machine
does; Nov 22 two lab trainings ran while it showed one queued batch). Every lab job enters the job queue
(`triforce.jobs`) with a log folder and a plan, like `triforce train` and the evaluations, so the GUI draws its
progress; a job without one is still listed, from its registry entry alone, as "no progress data" (lab jobs of
checkouts older than this, `--no-log` evaluations, a script that registers itself).

- `triforce.lab.train`: `logs/<yymmdd>_<NNN>_lab-<name>/` (`--logs-dir`, `$TRIFORCE_LOGS_DIR`): `training.json`,
  kind `lab-train`, with the families, the run dir and the mode, and one run entry a family (`specialist`,
  `events`, `target_steps`, `target_iterations`; status pending, `running`, then `completed`, `early_stop`,
  `interrupted` or `error`); one stream a family, `<family>.jsonl`, in `triforce train`'s events (`lab.train.RunLog`):
  `start` (targets, steps an update, envs, arch), a `progress` an iteration (the run's iteration of N and steps,
  the same for every family, `sps`, `eta`, and in `metrics` the family's own `family/<key>/...` numbers without
  the prefix, success, balanced success, hearts, `rows_share`, `hits_taken_per_episode`, `need`, with the run's
  `teacher_coeff` (the weight the iteration trained with), `losses/teacher_kl`, `losses/aux_*` and PPO's losses),
  then `train_end` and `end`. The GUI shows the run as one group with a row a family: the iteration of N, the
  family's success and hearts, "rows 31% · hits 0.70/ep · teacher 0.80 · aux hit 0.120"; the group's time left
  is the families' (they train together).
- `triforce.lab.bench run` / `speed` and `triforce.lab.record`: `logs/<yymmdd>_<NNN>_lab-bench` (`-speed`,
  `-record`; `--logs-dir`), kind `eval`, one row (`events.progress_job`): the bench's episodes played of its total
  with the success so far after each key, the recorder's decisions recorded of its total.
- A script that enters the queue itself (an experiment) uses `events.progress_job(what, kind, total, unit=...)`
  and calls `update(done, success_rate)` as it goes: a folder and a one-line progress
  (`scripts/experiments/r36_merge_record.py`: decisions recorded).

### Sampling across families by need

`--sampling need` (`lab.collect.NeedSampler`): before each rollout, a family's **need** is how far its success over
its last `--need-window` (100) episodes of this run is below its target (`--need-target teacher`: its key's success
on the yardstick in `lab/reference.json`, "below its teacher"; or a number for every family; no episode yet: the
whole target), and its share of the rows is `(floor + need) / Σ (floor + need)` (`--need-floor` 0.05). Envs a family
stay fixed (each family's pool runs on its own threads); its **decisions an env** change instead, so the rows per
rollout stay `--steps-per-update`. A family with a large share takes longer to roll out than the others, which
then idle: the CPU time is the same, the wall time follows the largest share. Logged per iteration:
`family/<key>/need`, `family/<key>/recent_success`, `family/<key>/rows_share`.
`--need-target-for KEY=X` gives one family its own target (a part whose teacher is weak: a target above the
teacher's yardstick success).

### A family's teacher term, and rehearsal of the parts not rolled out

`--teacher-weight KEY=W` scales that family's rows in the teacher term (`ppo.joint_kl` weighted per row, divided by
all rows as before; default 1): `0` takes the teacher out for a part whose teacher is weak while the others keep it.

A lab run rolls out at least one env a family, so a run of `--parallel 8` rolls out at most 8 parts. `--rehearse
CACHE` keeps the others in the loss: a distillation cache (`scripts/experiments/r36_merge_distill.py cache`: the
teachers' rows of every key, `r36_merge_record.py`) is read once (`--rehearse-rows`, 300,000 by default, the same
number a key; `--rehearse-keys`, by default every key of the cache not in `--families`), and every PPO minibatch adds
`--rehearse-coeff` (1.0) × KL(teacher ‖ student) on as many of its rows as the minibatch has (`lab/rehearse.py`;
logged `losses/rehearse_kl`). The rehearsal does not fade.

`--state-route ROUTE` puts every family on one route's states alone (`lab.collect.route_only`: `state_routes:
[ROUTE]` for its start-state globs and harvested starts, the globs the route has no state for dropped, the curriculum
on the tracks explored from the route's states, linked under `target/route-tracks/`, or off when there are none).
The one network of the corrected route is trained with `--state-route d3first` (the owner, Nov 22: never on
`d2first`'s states); `r36_merge_record.py --state-route d3first` records its teachers' rows the same way.

## Memory: training a recurrent network

A `LabNet` with memory returns a state from `initial_state(batch)` (`[batch, ...]`) and takes and returns it in
`heads(obs, state)`; `LabNet.recurrent` is then true and the lab trains it on sequences:

- **Acting** keeps the state per env in the server (above).
- **Readying a rollout** (`RolloutBuffer.recompute_sequence`, from `ppo.Collector.ready`): the learner's network runs
  over the rollout in order, every env at once, from the state the env's last rollout ended with (`Collector.carry`),
  the state starting over where a row starts an episode; it replaces the acting log-probs and values as for any
  network, and keeps the state at the start of every `--bptt` decisions (`chunk_states`, default 16).
- **The update** (`ppo.PPO.optimize`, and the imitation update): minibatches of whole sequences; each is unrolled from
  its stored start state (`LabNet.unroll`, resets at episode starts inside it), the gradient flowing within the
  sequence (truncated back-propagation through time). Steps per env must be a multiple of `--bptt`.
- The reference class is `nets/gru.py` (`gru`: today's encoders, a projection, a GRU cell, towers, today's heads;
  state `[B, core_dim]`). The head entropies logged after an update are computed row by row from the initial state
  (diagnostics only). Not with `--demos` (demonstration rows are no sequences).

```
python -m triforce.lab.train --name mem --families zol stalfos-keese-gel --parallel 8 --iterations 200 \
    --arch gru --arch-config '{"hidden_dim": 256, "core_dim": 256}' --bptt 16 --seed 36
```

## Auxiliary targets

**What the rollout carries.** Per decision, besides the reward and the frames: `health_lost` (hearts the action's
step lost, net of what it gained), `hits` (monster hits it dealt, as the game showed them, no look-ahead) and
`blocked` (round 38: a Bubble's touch landed during the action, `zelda::Change::bubble_touched`: the sword blocked
for 151-160 frames, Link shoved, no health lost) (`RolloutBuffer`, written by the pool from `StepInfo`). With
`--try-others K:P` (training only), on a share P of
the decisions (drawn by the decision's key) the policy's next K candidates after the action taken (the Rust actor's
ranking, `Actor::alternatives`; a served network's from its logits, `policy::logits::Heads::ranked`: a sample
without replacement from the policy) are each tried in the env (`PoolEnv::trial`, the game's own future, back to
exactly where it stood: the play is unchanged, decision for decision, `tests/test_lab_aux.py`); `tried_actions`,
`tried_lost` (1 lost health, 0 not, -1 untried), `tried_blocked` (1 a Bubble's touch landed, read from the trial's
RAM at its end; -1 untried) and `tried_frames` keep them. Cost: each trial emulates its action
from a checkpoint and back (no observation, no network), K·P trials a decision. Logged per family:
`family/<key>/frames_per_decision`, `family/<key>/tried_frames_per_decision`, `family/<key>/hits_taken_per_episode`
and `family/<key>/bubble_touches_per_episode`. Measured (Nov 22) on goriya's
`d3first` rooms (`goriya@d3first`, its route run `r32d1-goriya` training, 8 envs, 10 iterations of 4,096 decisions,
iterations 2-10; the machine at load 40-45, so the rates are noisy):

| `--try-others` | rollout decisions a second | frames a decision played | trial frames a decision | frames emulated a decision |
|---|---:|---:|---:|---:|
| none | 399 | 17.7 | 0 | 17.7 |
| `4:0.25` | 425 | 17.7 | 14.7 | 32.4 (1.8×) |
| `8:1.0` | 114 | 17.9 | 117.0 | 134.9 (7.6×) |

`4:0.25` costs nothing measurable at this load (one trial a decision on average); `8:1.0` (eight trials a decision,
and on the Rust actor a network pass a candidate to rank them) makes the rollout 3.5× slower.

**The losses.** A network returns named auxiliary outputs in `Heads.aux`; `--aux OUTPUT=TARGET:COEFF` adds
`COEFF` × a loss on that output (`lab/auxiliary.py`), in PPO and in imitation:

| target | label (per decision) | loss |
|---|---|---|
| `hit` | the action lost health | BCE on logits; a per-action output also on every tried candidate's verdict |
| `hit_or_blocked` | the action lost health, or a Bubble's touch landed (round 38: a head that learns to avoid Bubbles as it avoids damage) | as `hit`, the tried verdicts `tried_lost` or `tried_blocked` |
| `damaged` | the action dealt a monster a hit | BCE on logits |
| `hearts_lost` | hearts lost (net, ≥ 0) | ½ squared error |
| `frames` | the action's frames / 16 | ½ squared error |

An output `[B]` predicts for the action taken; `[B, 352]` one value per joint action (`kind * 32 + direction * 4 +
duration`), read at the action taken (and the tried ones). Logged: `losses/aux_<output>`,
`charts/aux_<output>_labels` (labels a row); for a per-action `hit` or `hit_or_blocked` output (against its own
labels), before each update trains on the rollout
(so held out, on the policy's own states; `lab.auxiliary.held_out`): `charts/aux_<output>_pr_taken` (precision =
recall on the actions taken, their real outcome: experiment 2's "model's own draws"), `charts/aux_<output>_pr_judged`
(with the tried candidates) and `charts/aux_<output>_lost_taken` / `_lost_judged` (the positive labels behind each,
to pool iterations by; the precision is NaN, so not logged, in a rollout without one).

**A hit head feeding the action choice** (`nets/hithead.py`, `hithead`): today's body plus a hit tower and a
per-action hit head (`aux['hit']`, 352 logits) trained only by its own loss; its probabilities, gradient stopped, are
an extra input to the policy tower (`hit_input`; off: an auxiliary target only; `detach_body`: the hit loss leaves
the encoders alone too). At play it is served like any network: it plays from the observation alone, with no
trial, and an act mode with foresight is refused for it as for any served network (`tests/test_lab_aux.py`).

```
python -m triforce.lab.train --name hit --families darknut --parallel 8 --iterations 200 --arch hithead \
    --aux hit=hit:0.5 --try-others 4:0.25 --seed 36
```

**The hit network** (`nets/hitnet.py`, `hitnet`; round36.md "Observation v7 and the hit network"): experiment 2's
shape on observation schema 7. Link's `link` block is a token attending to the entity tokens (the type embedding,
`entity_features`, `entity_rel`, `entity_contact`; `tokens` picks the groups), 2 layers of width 128, his final token
joining the room embedding, the grid convs and the float vectors; today's towers and heads; the hit head on its own
tower (`hit`: `joint` 352 logits, `kind_direction` 88 shared by the durations, `none`), `feed` its stopped
probabilities into the policy tower, `detach_body`, `room` (the room embedding on or off).

```
python -m triforce.lab.train --name hitnet --families goriya@d3first --parallel 8 --iterations 400 --arch hitnet \
    --arch-config '{"feed": true}' --aux hit=hit:0.5 --try-others 4:0.25 --teacher-coeff 1.0 --teacher-hold 100 \
    --teacher-fade 200 --seed 36
```


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
  order; 5 episodes each on fixed seeds (360000 + i), `temperature:0.5+relax:16`, clips on: 6,120 episodes. A
  route part plays its family's states (the harvest does not tell the two swords apart). Since 2.0.0.5 the entries
  are the corrected route's: those harvested before Dungeon 1's triforce (where both orders are one) are kept, every
  later one is from chains of the corrected route (100 runs a segment from `post-d1` to `d9`, seeds 370000 on,
  `--harvest`); the first order's later entries (3 pieces or more: its legs, and Dungeons 5, 2 and 7 at its
  hearts) are left out. Every key now has harvested entries (no seed start states stand in).
- Per key and in total: success, hearts lost per episode, hits taken per room visit, Bubble touches per room visit
  (`bubble_touches`, round 38: a touch blocks the sword and costs no health, so a hit never counts it), decisions
  per episode, bombs used per episode. Every episode is kept (state, seed), so `compare` pairs two networks episode
  by episode and gives each difference with its standard error; a metric one side lacks (a result written before
  round 38, as `reference.json`, has no `bubble_touches`) is `missing`, the rest pair as before.
- `run --team --out team.json` (today's networks), `run --network CKPT --keys zol stalfos-keese-gel --out
  student.json` (one network for some keys), `run --checkpoint KEY=CKPT ...`, `compare team.json student.json`;
  `run --act MODE` plays another act mode than the manifest's (the planner beside a result:
  `temperature:0.5+relax:16+plan:16:1`, for networks the Rust actor plays).
- **Full tier** (`bench.json` `full`): the route's segment chains as the landings run them, `triforce eval-batch
  TAG --chains TAG --route d3first --runs 1000 --act model`, a candidate's networks given with `--checkpoint`; the
  `full` segment is the route from the start. It is what decides; the cheap tier only shortlists.
- **Rooms tier** (round 38, `bench-rooms.json`; `rooms --harvest DIR --chains ...` builds it): the cheap tier was
  blind to what killed a student's chains. Its entries are chain handoffs only (a room entry where the family
  changes), so a room entered from a room of the same family is never on it: Dungeon 8's Darknut rooms
  0x4e-0x3e-0x3f, the Olive Garden's fight (5:0x05 from 0x06; the cheap tier's two 5:0x05 entries are the cleared
  revisit from 0x04, 3 decisions), Dungeon 7's Goriya rooms (0x1b, 0x19, 0x18, 0x49), 4:0x31 from 0x30. On the
  51's own 1,000-run chains 69% (D5), 78% (D7) and 82% (D8) of the deaths are in such entries. It plays its entries
  with the family's `ram_randomization` (hearts and bombs drawn at reset), not what the chain brings. And its total
  weighs every key alike: a round-1 student 17.5 ± 3.5 points down on `darknut@d3first` (3:0x59: 13% against the
  team's 80%) still read +1.2 in total. The rooms tier takes every room entry where the team loses hearts, from an
  every-room harvest of the team's chains (`eval-chain --harvest DIR --harvest-every-room`), 12 states each, played
  as they arrived (`arrived`: no randomization), one row per room entry; `compare` lists the **losing** rows (down
  5 points or more beyond two standard errors) under the table, on both tiers. A candidate reaches the chains only
  with no losing row.

**The reference** (`triforce/lab/reference.json`: today's team, 2.0.0.3's 51 networks, on `bench.json`; played
Nov 22 on 10 envs on the corrected `d3first`'s entries): 6,120 episodes, success **92.9%**, hearts lost **0.248** an
episode, hits taken **0.675** a room visit, **33.2** decisions and **0.357** bombs used an episode; per key in the
file. In 2.0.0.7 the 20 episodes of `ow-zora` and `ow-zora@d3first` on `d3first_0_2dw_h370001_9` and
`_h370075_9` were played again (0:0x2d is crossed W → S since its cave left the route; `bench run --checkpoint`
with the two keys' networks, every other episode of the two keys identical): ow-zora 87.5% → 88.3%,
ow-zora@d3first 93.3% → 92.5%, the total's success unchanged. It replaced the first reference (Nov 21, on the first
order's entries: 5,735 episodes, success 91.9%, hearts lost 0.241, hits 0.697 a visit, 36.2 decisions, 0.524 bombs;
four keys on seed start states). What moved
most, per key on its new rooms (120 episodes): ow-lynel 97.5% → 64.2% (now the corrected route's 0x50, 0x60, 0x22,
0x32 and 0x07, among them the legs the baseline loses), boss-gleeok@d3first 67.5% → 44.2% and boss-gleeok 12.5% → 56.7% (8:0x3c
joins 4:0x13), boss-manhandla 65.8% → 81.7% (8:0x6e joins 3:0x4d), like-like 80.0% → 92.5%, ow-zora@d3first
99.2% → 93.3% (the docks' screens 0x55 and 0x3f). The two references are not paired (other entries): compare a
network with the reference of the manifest it was played on.
Every experiment is compared with it: `python -m triforce.lab.bench compare triforce/lab/reference.json NEW.json`.

**The rooms tier's reference** (`triforce/lab/reference-rooms.json`: the 51 on `bench-rooms.json`, 2.0.0.7, Nov 22):
the tier built from an every-room harvest of the 51's chains, 100 runs of each dungeon segment d1-d9 from seed
380000 (`eval-chain --route d3first --clips on --act model --from 3_7cs_0 --stop-at step:24 --runs 100
--first-seed 380000 --harvest DIR --harvest-every-room`, and so on with each segment's start and stop), then
`python -m triforce.lab.bench rooms --harvest DIR --chains <those outputs>`: 100 room entries, 12 states each, 5
episodes a state, 5,980 episodes; success **92.3%**, hearts lost **0.678** an episode, hits taken **1.003** a room
visit, **48.0** decisions and **0.417** bombs an episode. Its states are untracked like `harvested/`:
`bench rooms` copies the chosen ones to `triforce/states/harvested-rooms/<family>/`. On the first 2.0.0.5 harvest
(97 entries) it ranked four students as their chains did (round 37's -1.4 points, then -2.3, -3.6, and -8.1 for
the round-1 student the cheap tier had at +1.2 over the team). Compare a network with it:
`python -m triforce.lab.bench run --network CKPT --manifest triforce/lab/bench-rooms.json --out NEW.json`, then
`python -m triforce.lab.bench compare triforce/lab/reference-rooms.json NEW.json`.

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
