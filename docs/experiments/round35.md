# Round 35

## A learned guard: can a model learn the planner's veto? (goriya, Dungeon 7; an experiment, branch `r35-guard`)

Round 34 found one cause in every wood-sword dungeon: the honest planner (`temperature:0.5+relax:16+plan:16:1`)
stands, swings and bombs like the models and is hit a third to a half as often, by replacing a handful of decisions
a room at the moments the emulator's look-ahead shows a hit; six goriya runs trained on its rows did not pick it up
(round34.md §3, §8). The question here: can a network that sees only the policy's observation learn the look-ahead's
verdict ("this action gets Link hit"), and does vetoing with it, the way `plan:16:1` vetoes with the emulator, play
like the planner? Such a guard uses no look-ahead at play: it would be a model's own act mode. Family: goriya
(`r32d1-goriya`, `d3first`'s route run) on Dungeon 7's rooms, the clearest case (crossings, boomerangs).

### 1. The design

**Where the planner decides.** `plan:K:S` (`crates/pool/src/pool.rs` `PlayWork::foresee`, docs/rust-actor.md
"Foresight"): before a decision is taken the env tries the model's draw in the emulator on S redrawn futures
(`PoolEnv::trial`: the action executed in simulation mode and the session restored); if Link loses health in any,
the next action of the mode's own ranking (ln q + the decision's Gumbel noise) is tried, at most K in all, and the
first that loses none is taken. One trial costs the action's frames again: `plan:16:1` is about 2x a run's frames.
The rows `r27_plan_demos.py` recorded (round 34) are the observation, the mask and the action taken, with the
replaced decisions repeated; the refused draw and the verdicts of the other candidates were not kept.

**Where a guard can live.** Two ways were weighed:
- Python-side (a Python loop that steps the env and asks a torch guard): no such play path exists for chains (every
  evaluation and chain plays in the Rust pool, `EnvPool.play`, the policy in Rust), so it would mean a second chain
  driver: more machinery, and slower.
- In the Rust actor (taken): the guard is **the policy's own network** (`ZeldaPolicy`: the same observation, trunk
  and towers) with the value head widened to **one hit logit per joint action** (352). The Rust `Policy` already
  loads every weight of that architecture; the only change is a value head of any width (`Policy::values`). The act
  mode **`veto:K:P`** (`crates/policy/src/act.rs`) is a third kind of foresight beside `guard:K` and `plan:K:S`: no
  trial in the env; the guard scores the decision once (one forward, all 352 actions) and refuses a candidate whose
  hit probability is above P; candidates and ranking as `plan`. The guard file is **`guard.pt` beside the run's
  `ckpt.pt`** (`triforce/guard.py`; `runner.Policies` attaches it, the chain's `models` header records its hash). A
  run without one plays `veto` as the plain mode, and no other mode reads a guard: a new file beside the checkpoint,
  not a change of the checkpoint or of any model's play.

**The data's new parts** (`EnvPool.poll_plays` `tried`, `TRIFORCE_PLAN_AUDIT`): every candidate a decision's
foresight judged, with the future it was tried on and whether Link lost health (`pool::rollout::Tried`); and audit
futures, each judged candidate also tried on A more redrawn futures that never decide, for the ceiling.
`scripts/experiments/r35_guard_data.py` writes them; `scripts/experiments/r35_guard_train.py` trains and measures.

**The owner's condition** (via the lead): a learned guard counts as the models' own play only if the emulator's
look-ahead is used in training (to make the labels) and never when the game is played. Built and tested:
- `crates/pool/tests/pool.rs` `a_learned_veto_plays_without_the_emulators_look_ahead`: a run in `veto:K:P` makes no
  `PoolEnv::trial` (the only place a state is saved and restored, an action tried or the random numbers redrawn;
  the same counter reads > 0 for `plan:4:1`); the guard is handed exactly the decision's observation and mask; with
  the guard's refusals off the run is the plain mode's, frame for frame; with them on it replaces decisions with no
  trial frame.
- `tests/test_guard.py`: the same on the Zelda env and chains (a never-refusing guard plays as the plain mode, room
  record for room record; a refusing one replaces decisions with `trial_frames` 0, the chain's `models` header names
  the `veto` mode and each guard's hash, and every recorded run verifies with the emulator alone).
- `PolicyActor::with_guards` refuses a guard that reads anything but its policy's inputs; an act mode with `veto` in
  one health part and `guard`/`plan` in either is refused (`act.rs` spellings); play-zelda labels `veto` modes
  "learned guard: no look-ahead" and `plan`/`guard` modes "planner: the emulator's look-ahead".

### 2. The data (goriya, `r32d1-goriya`, Dungeon 7's goriya rooms on `d3first`)

`r35_guard_data.py` on the 13 goriya stems of the `d7` segment (`7_79s`, `69s`, `59s`, `49s`, `38e`, `28s`, `18s`,
`19w`, `1aw`, `1ac`, `1bw`, `29c`, `2bw`; chains 0-2 each), at the hearts the route's chains bring (round 34's
goriya config copy with `chain_hp: 1.0`, all sixteen goriya rooms): 200 episodes a state with the planner
(`plan:16:1`) and 200 with the model alone, its own draw judged and never replaced (`plan:1:1`), seeds from 35000;
one audit future (`--audit 1`). 27 minutes of the job queue at 10 envs.

| | Planner episodes | Model episodes | All |
|---|---:|---:|---:|
| Episodes | 7,800 | 7,800 | 15,600 |
| Decisions | 265,899 | 247,115 | **513,014** |
| The model's own draw lost health (future 0) | 20,387 (7.7%) | 7,873 (3.2%) | 28,260 |
| Episodes won / hearts lost an episode | 99.3% / -0.07 | 97.0% / 0.51 | |

Every judged candidate (the draw and, where refused, the alternatives the planner tried): 618,908 on future 0,
116,082 lost (base rate **18.8%**). By room the draw is lost most in 0x59 (19.0% of the planner's decisions, 10.0% of
the model's), 0x1b (18.0 / 5.8%), 0x18 (14.6 / 3.8%), 0x69 (10.9 / 6.4%); never in 0x79, 0x28 and 0x2b.

**The ceiling: the verdict is not luck.** The same candidate tried on a second redrawn future agrees with the first
on 99.85% of 618,908 pairs; as a predictor of the first, the second has precision 0.997 and recall 0.995 (the
held-out episodes: 88,479 pairs, the same). Within one decision's frames the redrawn random numbers barely matter:
the verdict is a function of the emulator's state and the action. What a guard can miss is only what the
observation does not show of that state.

### 3. The guard: held-out numbers

`r35_guard_train.py`: goriya's network with the hit head, all weights trained, binary cross-entropy on the 530,429
future-0 verdicts of the training episodes (base rate 18.7%), 5 epochs, batch 1024, AdamW 4e-4 one-cycle; 37.5
minutes on 8 threads. Held out: every 7th episode (2,229 episodes, 88,479 judged candidates). For all judged
candidates: ROC AUC **0.968**, log loss 0.177 against the base rate's 0.487. For the decision's own draw (what a
veto keeps or refuses):

| Held out | Draws | Lost | AUC | Vetoing as often as the look-ahead (P = R) | P > 0.2: vetoes, precision, recall | P > 0.5 |
|---|---:|---:|---:|---|---|---|
| planner's episodes | 37,325 | 2,933 | 0.948 | threshold 0.51: **0.60** | 5,697, 0.43, **0.84** | 2,990, 0.59, 0.60 |
| model's episodes | 35,791 | 1,140 | 0.951 | threshold 0.46: **0.48** | 2,485, 0.32, **0.71** | 1,006, 0.50, 0.44 |
| the ceiling (a 2nd redraw) | | | | 0.997 / 0.995 | | |

By epoch (planner's episodes, P = R): 0.50, 0.56, 0.57, 0.59, 0.60: still rising slowly at 5.

**A new checkpoint of the policy** (measured once): the same guard on labels from another goriya run,
`r34d7b-gor-s1` (40 + 40 episodes a state, seeds from 45000, 111,581 decisions, judged candidates' base rate 11.6%):
all candidates AUC 0.951; the draw AUC 0.924 (planner's episodes) and 0.937 (model's), P = R 0.44 and 0.40 (its
own policy's held-out: 0.60 and 0.48). The verdict is the game's, not the policy's, so the guard carries over, but
the precision drops where the new policy reaches other states and draws other actions: a new checkpoint would want
its guard fine-tuned on labels of its own play.

**Cost of one decision.** The guard is one more forward of a network of the policy's size: single-threaded, 411 µs a
row against the policy's 453 µs (`_native.Policy.values` / `evaluate`, a batch of 286 rows, the machine loaded); no
emulated frame (the planner's `plan:16:1` emulates every decision's action again, about 2x a run's frames). Rust
and torch agree within 2.4e-7 on the hit logits.

### 4. Not measured (the owner stopped the experiment before the chains)

- **No `d7` chain with the trained guard.** The threshold screen (`veto:16:0.2`, `0.35`, `0.5` and the plain mode,
  seeds 7000-7499) was queued and stopped before it started; the measurement on seeds 0-999 against the record
  (`r34land-1k-d7`, 38.7%), the fresh set and the planner beside never ran. Whether the veto moves the segment, and
  the per-room hits a visit and boomerangs parried, are unknown.
- The only chain played in `veto` mode: 6 runs of `d7` (seeds 0-5) with a pilot guard trained on 156 episodes: it
  attached to goriya's route run only (the `models` header carried its hash), judged 2,318 goriya decisions with 0
  trial frames and refused none (its probabilities never passed 0.5); 1 of 6 won. A smoke test, not a number.
- The wall time of a `veto` chain against the plain one (cost at play beyond the forward pass).
- The whole-game plan the owner asked for (per family: label collection at the rates above, about 1.1M decisions an
  hour at 10 envs with the audit, 37 minutes of training for 0.5M decisions; the guard file beside the checkpoint;
  how `select-checkpoint`, `promote`, `triforce list`, the version stamp and the act files would carry it) is not
  written beyond these rates. The order of families by what the planner gains over the models, computed from the
  record's segment chains (`r34land-1k`, models) against the planner's full route (`r34land-plan-full`, deaths per
  1000 visits, the planner arriving richer in hearts): boss-gleeok (route run) 736 vs 38, boss-manhandla (route
  run) 553 vs 236, darknut 134 vs 3, boss-digdogger (route run) 105 vs 8, darknut (route run) 89 vs 12, like-like
  (route run) 66 vs 0, boss-aquamentus (route run) 45 vs 0, goriya (route run) 28 vs 1, gibdo (route run) 25 vs 2.

**Versioning.** `guard.pt` is a new file beside `ckpt.pt`; a run without one plays every mode, `veto` included, as
before, and no other mode reads it, so it is not a COMPAT change of the checkpoints. The code adds a value head of
any width to the Rust network, `Play.tried`, `TRIFORCE_PLAN_AUDIT` and the `veto` act mode.

**Recommendation.** The guard learns most of the look-ahead's verdict from the observation alone (AUC 0.95 on the
draws, P = R 0.60 at the planner's veto rate), far from the ceiling (0.997) but far from luck. It is worth the
`d7` chain it did not get (screen three thresholds on seeds 7000-7499, then the chosen one on 0-999 paired with
the record and on a fresh set beside the plain model and the planner); nothing should be built for other families
before that number. The code is on `r35-guard` only.

Scripts: `scripts/experiments/r35_guard_data.py` (labels), `scripts/experiments/r35_guard_train.py` (training,
`--evaluate`); module `triforce/guard.py`.
