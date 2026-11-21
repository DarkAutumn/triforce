# Round 35

**Summary (Nov 21, the owner's checkpoint).** The round began on what round 34 found (the planner's whole edge is
hits avoided) with three pieces: the late runs' landing (2.0.0.2, below), Dungeon 3's Manhandla, and an experiment
with a learned guard. The owner then called a stop for a solid checkpoint, with two calls: **no bag of 8 bombs**
("Don't blindly add 8 bombs. The goal is a win without a cheated bomb.") and **the guard waits** ("a good idea, but
I want to dig into a bigger picture thought about model construction"). So: `d3first`'s Dungeon 5 entrance grants
nothing any more (2.0.0.3, "The checkpoint" at the end); the golden path still ends the game, three of three, and
its replay without the bag is the corpus's; for the first time a single run from the game start ends the game on
`d3first` (1 of 1000, also in the corpus); Dungeon 5 from its entrance falls from 33.0% to 7.0%, the Olive Garden
fought with no bombs; Dungeon 3's trainings and the guard's first chain were stopped unmeasured and are recorded
as they stand. Nothing runs. What is open is in backlog.md "Where the work stopped".

## 2.0.0.2: round 34's two late runs land (branch `r35-land`)

The two `d3first` route runs that qualified after 2.0.0.1's chains of record had played, darknut `r34d5-dg-s362`
(it012716, the Olive Garden) and boss-digdogger `r34d7b-dig-s1` (Digdogger at the route's hearts), promoted `--route
d3first` on the 2.0.0.1 team; `VERSION` 2.0.0.2, no code or route data changed (no identity check owed). Their causes
and the full numbers are in round34.md "2.0.0.2"; decisions 2026-11-21 "2.0.0.2".

The chains of record `r35land-1k` (seeds 0-999, models alone, clips on, paired with `r34land-1k`):

| Segment | 2.0.0.1 | 2.0.0.2 | Paired |
|---|---:|---:|---:|
| `d3` (`3_7cs_0` → 24) | 18.0% | 23.2% | **+5.2 ± 1.8** |
| `d5` (`5_76s_0` → 120) | 27.4% | 33.0% | **+5.6 ± 2.0** |
| `d7` (`7_79s_0` → 194) | 38.7% | 42.3% | +3.6 ± 2.0 |

The gate (`d3`, `d5` not worse and one better, `d7` not worse) passed. From the game start (1000): the D3 triforce
23.5% (22.7%, +0.8 ± 1.9), D4's 4.5% (2.8%), D1's 3.8% (2.3%), D5's 12 runs (3), the magic sword 2, D6's 1, the
ending 0. The planner beside (300): the ending 20.3% (25.0%, -4.7 ± 3.5), its D5 triforce 47.7% (59.0%, -11.3 ±
4.1: darknut's 5:0x37 over the new run, 21 of 171 entering lost, 7 of them stalls). The golden path from power on:
three of three in 50, 105 and 53 runs (2.0.0.1: 51, 71, 60).

```sh
# the worktree's team (main's promoted run dirs and pointers + the two runs, promoted --route d3first)
A="--route d3first --clips on"; M="--runs 1000 --act model"
triforce eval-batch r35land-1k --parallel 10 \
  --chain "d3: $A --from 3_7cs_0 --stop-at step:24 $M --output target/r56-chains/r35land-1k-d3.jsonl" \
  --chain "d5: $A --from 5_76s_0 --stop-at step:120 $M --output target/r56-chains/r35land-1k-d5.jsonl" \
  --chain "d7: $A --from 7_79s_0 --stop-at step:194 $M --output target/r56-chains/r35land-1k-d7.jsonl" \
  --chain "full: $A $M --output target/r56-chains/r35land-1k-full.jsonl" \
  --chain "plan-full: $A --runs 300 --act temperature:0.5+relax:16+plan:16:1 --output target/r56-chains/r35land-plan-full.jsonl"
python zelda-headless.py target/golden-S 1 --golden --power-on --clips on --act model --parallel 8 --first-seed S  # S = 0, 700000, 800000
```

## Dungeon 3 on `d3first`: Manhandla at the hearts runs bring (boss-manhandla, zol, stalfos-keese-gel)

**State of this section.** Stopped at the owner's checkpoint (branch `r35-d3`): the ledger (§1) and the Manhandla analysis (§2) are done; the trainings of §3 were stopped or finished untried; nothing was selected, measured on a chain, promoted or landed. The base team is main's
2.0.0.1 team with darknut `r34d5-dg-s362` (it012716, round 34's Olive Garden continuation, promoted `--route
d3first` here); darknut's rooms are held as they are. Families: boss-manhandla (`r32d3-manh`), zol
(`r34d3-zol-s341`), stalfos-keese-gel (`r34d3-skg-s341`).

### 1. The ledger on the base team

The base team on `d3` (`--from 3_7cs_0 --stop-at step:24`, 1000 runs, seeds 0-999): **23.2%** ± 1.3 (the record
`r34land-1k-d3`, with `r34d3-dark-s341`: 18.0%); the planner over the same team (`temperature:0.5+relax:16+plan:16:1`,
seeds 0-299): **60.3%** ± 2.8. Per route step (`scripts/experiments/r35d3_ledger.py`): runs entering, runs lost (per
1000 entering), hearts on first entering, hits a visit, hearts lost a visit; the last column the models' hits a visit
over the planner's:

| Step | Room | Models | Planner | Hits ratio |
|---|---|---|---|---|
| 7 | 3:0x7b E→N (zol) | 1000, 0 (0), 3.00, 0.16, 0.11 | 300, 0 (0), 3.00, 0.07, 0.04 | 2.4 |
| 8 | 3:0x6b S→N (zol) | 1000, 1 (1), 2.89, 0.54, 0.31 | 300, 0 (0), 2.96, 0.14, 0.07 | 3.8 |
| 9 | 3:0x5b S→N (darknut) | 999, 28 (28), 2.58, 0.30, 0.29 | 300, 2 (7), 2.88, 0.11, 0.11 | 2.6 |
| 10 | 3:0x4b S→W (zol) | 971, 11 (11), 2.35, 0.22, 0.13 | 298, 1 (3), 2.79, 0.09, 0.05 | 2.5 |
| 11 | 3:0x4a E→W (stalfos-keese-gel) | 960, 9 (9), 2.25, 0.34, 0.17 | 297, 0 (0), 2.74, 0.14, 0.07 | 2.4 |
| 12 | 3:0x49 E→S (zol) | 951, 41 (43), 2.10, 0.28, 0.21 | 297, 0 (0), 2.67, 0.03, 0.01 | 10.4 |
| 13 | 3:0x59 N→S (darknut) | 910, 62 (68), 1.98, 0.35, 0.01 | 297, 3 (10), 2.66, 0.07, -0.06 | 5.2 |
| 14 | 3:0x69 N→cave (darknut) | 848, 107 (126), 2.11, 0.56, 0.54 | 294, 14 (48), 2.74, 0.43, 0.42 | 1.3 |
| 15 | 3:0x0f S→N (equipment-room) | 741, 50 (67), 1.80, 0.52, 0.26 | 280, 2 (7), 2.44, 0.14, 0.07 | 3.9 |
| 18 | 3:0x5a W→E (trap-room) | 691, 43 (62), 1.65, 0.27, 0.14 | 278, 2 (7), 2.39, 0.05, 0.03 | 5.3 |
| 19 | 3:0x5b W→E (darknut) | 648, 76 (117), 1.60, 0.42, 0.08 | 276, 5 (18), 2.38, 0.16, -0.04 | 2.6 |
| 20 | 3:0x5c W→N (darknut) | 572, 50 (87), 1.73, 0.26, 0.25 | 271, 4 (15), 2.46, 0.10, 0.10 | 2.6 |
| 21 | 3:0x4c S→E (trap-room) | 522, 32 (61), 1.62, 0.19, 0.10 | 267, 2 (7), 2.40, 0.04, 0.02 | 4.2 |
| 22 | **3:0x4d W→N (boss-manhandla)** | 490, **258 (527)**, 1.62, 1.13, 0.41 | 265, 84 (317), 2.40, 1.05, 0.24 | 1.1 |

(Steps 6, 16, 17, 23 lose nothing.) **The first three losses: Manhandla (258 runs of 1000, 527 per 1000 entering
against the planner's 317), 0x69's dash (107; 126 against 48), the last pass 0x5b W→E (76; 117 against 18)**, then
0x59 N→S (62), 0x5c and 0x0f (50 each). The last pass, round 34's first room (414 per 1000 entering), is now the
third: darknut's two continuations moved the loss to Manhandla.

**Manhandla by the hearts Link brings** (first visits; bombs on entering, placed, bomb hits on his parts, his dropped
bombs picked up, hits taken; a visit):

| | Hearts in | Entered | Won | Bombs in | Placed | Bomb hits | Picked up | Hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| models | ≤ 1 | 181 | 27.1% ± 3.3 | 3.80 | 1.74 | 3.04 | 0.21 | 0.73 |
| models | 1.5-2 | 204 | 56.4% ± 3.5 | 4.16 | 2.57 | 4.52 | 0.35 | 1.24 |
| models | 2.5-3 | 105 | 64.8% ± 4.7 | 3.68 | 2.59 | 4.94 | 0.39 | 1.61 |
| models | all | 490 | 47.3% ± 2.3 | 3.92 | 2.27 | 4.07 | 0.31 | 1.13 |
| planner | ≤ 1 | 31 | 35.5% ± 8.6 | 4.06 | 1.97 | 3.42 | 0.23 | 0.65 |
| planner | 1.5-2 | 67 | 58.2% ± 6.0 | 4.03 | 2.88 | 4.84 | 0.34 | 1.06 |
| planner | 2.5-3 | 167 | 78.4% ± 3.2 | 3.83 | 2.47 | 4.95 | 0.49 | 1.12 |
| planner | all | 265 | 68.3% ± 2.9 | 3.91 | 2.51 | 4.74 | 0.42 | 1.05 |

The models bring him 1.62 hearts (37% of the runs a heart or less), the planner 2.40 (63% at 2.5-3). Both bring the
same bag (3.9 bombs) and throw about as many (2.3 against 2.5). **What the route can still deliver**: the hearts
chain of round 34 (`r34d3_ledger.py`'s kernels) with every step before his room the planner's and the start the
models': 907 of 1000 runs reach him (the models 491) with **2.41 hearts** (15% at a heart or less, 63% at 2.5-3);
with his room the models' that is `d3` 53.6%. Every room before him the planner's, one at a time: 0x59 N→S +4.2,
0x6b (zol) +3.8, 0x5b W→E +3.6, 0x49 (zol) +3.5, 0x5b S→N +3.5, 0x4a (stalfos-keese-gel) +2.6, 0x69 +2.5, 0x0f +2.4,
0x4b (zol) +2.0; by family darknut +17.8, zol +8.1, stalfos-keese-gel +2.6, Manhandla's own room +0.6 (in the chain the
planner wins at his hearts about as often as the model at the low ones). The hits ratio names every zol room (0x7b
2.4, 0x6b 3.8, 0x4b 2.5, 0x49 10.4) and stalfos-keese-gel's 0x4a (2.4): all above twice the planner's.

```sh
triforce eval-batch r35d3-ledger --parallel 10 \
  --chain "base-d3: --route d3first --clips on --from 3_7cs_0 --stop-at step:24 --runs 1000 --first-seed 0 --act model --output target/r35/base-d3.jsonl" \
  --chain "plan-d3: ... --runs 300 --first-seed 0 --act temperature:0.5+relax:16+plan:16:1 --harvest target/r35/harvest-plan --output target/r35/plan-d3.jsonl" \
  --chain "harv-d3: ... --runs 1000 --first-seed 10000 --act model --harvest target/r35/harvest-models --output target/r35/harv-d3.jsonl"
python scripts/experiments/r35d3_ledger.py target/r35/base-d3.jsonl target/r35/plan-d3.jsonl
python scripts/experiments/r34d3_ledger.py target/r35/base-d3.jsonl target/r35/plan-d3.jsonl --last 23
```

### 2. Manhandla at the hearts runs bring: what the planner does that the model does not

The same entries both ways: 64 of the 478 entries into 3:0x4d of the models' harvest (seeds 10000-10999, their own
hearts and bombs, no RAM randomization; hearts in 1.52, bombs 4.17), 4 episodes each, `r32d3-manh` as the chain
plays it (`temperature:0.5+relax:16`) and the planner over it, every episode recorded with its decisions and played
frame by frame (`scripts/experiments/r35d3_manhandla.py`; Manhandla is one $3C object in five slots, a part's death
or health drop a bomb's when one of Link's bombs was exploding in the 24 frames before it; "fight" while a part
lives):

| 256 visits | won | hits a visit | hits a 1000 fight frames | fight frames a visit | parts killed (by bombs) | bombs placed / parts struck | within 16 / 32 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r32d3-manh` | 51.6% | 0.95 | 3.15 | 303 | 3.53 (3.53) | 2.38 / 4.50 | 4.1% / 23.4% |
| planner | 62.9% | 0.77 | 1.74 | 440 | 3.96 (3.94) | 2.61 / 4.83 | 4.4% / 22.9% |

| By hearts in (108 / 112 / 36 visits) | ≤ 1 | 1.5-2 | 2.5-3 |
|---|---:|---:|---:|
| won, model / planner | 38.0% / **55.6%** | 57.1% / 65.2% | 75.0% / 77.8% |
| hits a 1000 fight frames | 2.50 / 1.14 | 3.59 / 2.06 | 3.20 / 2.24 |
| fight frames a visit | 244 / 391 | 326 / 473 | 407 / 484 |
| first hit, frame median | 161 / 270 | 131 / 163 | 147 / 204 |

Both kill him with bombs (the sword kills no part; it only chips: 0.15 against 0.33 health drops a visit), place
the first bomb at the same moment (frame 90 against 93), throw as many before the first part dies (1.06 / 1.12) and
strike as many parts, stand as close for as long (within 32 px 23% of the fight). **What differs is the hits while
it does so, and the patience they buy**: the planner is hit at half the rate (1.74 a 1000 fight frames against 3.15),
so its fights last longer (440 frames against 303) and it outlasts him at a heart or less (55.6% against 38.0%). By the
decision executing at the hit: BOMB 58 of the model's 244 hits against 25 of the planner's 196 (Link standing to place
a bomb while a hand or a shot reaches him; 141 against 77 hits within 48 frames of a bomb placed), MOVE 164 / 154,
SWORD 19 / 14; by the nearest object: Manhandla's parts ($3C) 144 / 135, his shots ($56, a monster shot) 91 / 55. Where Link
spends the fight: the middle 50% against 40%, the planner more by the W/E walls (14% / 8%) and corners (7% / 3%).
Not the route's (no reserve: the bag is his, `drops_first`), not a reward habit (a hit costs its hearts; the
fight's objective is his death): foresight while bombing, which the planner's fights teach as demonstrations.

```sh
python scripts/experiments/r35d3_specs.py eval boss-manhandla 3:0x4d target/r35/manh-fight 64 target/r35/harvest-models/boss-manhandla --seed 35
export TRIFORCE_STATES_DIR=$PWD/target/r35/manh-fight/states F=$PWD/target/r35/manh-fight
ev() { echo "$1: --specialist $F/spec.yaml --checkpoint models/boss-manhandla/r32d3-manh/ckpt.pt --episodes-per-state 4 --seed 0 --clips on --act $2 --output $F/$1.json --replays $F/rep-$1 --streams decisions"; }
triforce eval-batch r35d3-manh-fight --parallel 10 --evaluate "$(ev model temperature:0.5+relax:16)" --evaluate "$(ev plan temperature:0.5+relax:16+plan:16:1)"
python scripts/experiments/r35d3_manhandla.py model=$F/rep-model:$F/model.json plan=$F/rep-plan:$F/plan.json
```

### 3. The trainings (stopped at the owner's checkpoint: nothing selected or measured on the chain)

Every training 244 iterations planned, `--parallel 8 --checkpoint-every 30 --demo-coeff 0.05 --sil-coeff 0.1
--stop-success 1.01`, from the family's `d3first` route run; harvested starts from the base team's `d3` harvest
(seeds 10000-10999) at the chains' hearts and bombs (`scripts/experiments/r35d3_specs.py`).

- **boss-manhandla** (`r35d3-manh-s351..353`, from `r32d3-manh`): every reset a harvested entry into 3:0x4d, half
  the models' (478) and half the planner's (265, twice each; `harvested_fraction` 1.0); demonstrations: the planner
  over `r32d3-manh` from 150 models' and 150 planner's entries, 2 episodes each (400 of 600 won, 17,474 rows).
  **Stopped**: s351 at iteration 58 of 244 (one checkpoint, it003694), s353 at 72 (it003694, it003724); s352 never
  started (it waited in the queue).
- **zol** (`r35d3-zol-s351`, the plain recipe) and the **hit-price arm** (`r35d3-zolhp-s351`: the zol yaml's
  existing `reward.heart_value.potential`, the price of a change of hearts in the run's odds V(step, hearts), raised
  from 20 to 40; nothing else differs): from `r34d3-zol-s341`, `harvested_fraction` 0.25 (the base team's handoffs
  into 0x4b, 0x49, 0x3d and main's zol handoffs of Dungeons 4 and 5); demonstrations: the planner from `3_6bs`,
  `3_49e`, `3_4bs`, `3_7be` (20 each) and 120 handoffs into 0x49 and 0x4b (2 each), 20,662 rows. **Both finished**
  (244 iterations, checkpoints it001730-it001940 and `last`); their selections (to 0x59's door, `--stop-at
  step:13`, seeds 2000-2999 and 3000-3999) were stopped before a chain played.
- **stalfos-keese-gel** (`r35d3-skg-s351`, from `r34d3-skg-s341`): `harvested_fraction` 0.25 (962 handoffs into 0x4a
  and main's Dungeon 1 and 4 handoffs); demonstrations: the planner from `3_4ae` (30 each, 90 of 90 won) and 90
  handoffs (2 each), 11,870 rows, with round 32's `stalfos-keese-gel.npz`. **Finished** (244 iterations,
  it002299-it002509 and `last`); its selection was stopped before a chain played.

No verdict: no run of this section was chosen or compared with its route run on a chain; `d3` and the first
milestone stand at the base team's (`d3` 23.2% on seeds 0-999), the hit-price arm has no result. Where to go on:
the Manhandla trainings from the start (the recipe above), then the shortlist in his room (every checkpoint on 160
harvested entries, 2 episodes each) and `select-checkpoint` on `d3` (`--from d3first_3_7cs_0 --stop-at step:24`);
the three finished runs' selections as queued (screen 2000-2999, confirmation 3000-3999, arrivals and hearts into
0x59), then `d3`, `d4`, `d1` and the game start to the D3 triforce on seeds 0-999 and a fresh set.

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

## The checkpoint: no bag of bombs (2.0.0.3, the owner, Nov 21)

**The change.** routes.yaml: `d3first`'s Dungeon 5 entrance (step 97) is `d2first`'s again, `bomb_reserve: 8` and no
`grant: {bombs: 8}`; no step of any route grants anything. The `grant:` key, the ledger's route tally and the
replay's `route_granted` field stay (recorded replays carry them; backlog). Link opens 5:0x66's and 5:0x65's walls
with the bombs he brings and fights the Olive Garden with what is left (`crates/env/tests/d3first_d5_bombs.rs`: from
the segment's start state, 2 bombs, he enters the Garden with none and nothing is cheated). The older rule stands:
a wall the route must bomb when Link holds no bomb is given one, counted.

**Dungeon 5 without the bag** (`r35nobag-1k`, seeds 0-999, models alone, clips on, the 2.0.0.2 team; paired with
`r35land-1k-d5`):

| | With the bag (2.0.0.2) | Without (2.0.0.3) | Paired |
|---|---:|---:|---:|
| `d5` (`5_76s_0` → 120) | 33.0% | **7.0% ± 0.8** | −26.0 ± 1.7 |
| The Olive Garden 5:0x05, lost per 1000 entering | 343 | 811 (6.57 hearts in) | |
| Cheated a run | 5.9 bag + 2.2 wall bombs | none | |
| The planner beside (300 runs) | 82.0% (other seeds) | 58.0% ± 2.8 | not paired |

The planner without the bag loses 287 per 1000 entering the Garden and 10 of 300 runs stall in 5:0x37. Darknut's
route run was trained for a Garden with the bag (its harvested entries and demonstrations hold 8 bombs): a Garden
with none is a room it has hardly seen, and the walks' wood-sword rule cleared it 9 of 20 without bombs.

**The full route from the game start** (1000 runs, paired with `r35land-1k-full`): identical run for run up to
Dungeon 5's entrance (the D3 triforce 23.5%, D4's 4.5%, D1's 3.8%; 38 runs enter Level 5 with 2.03 bombs, 15 of them
with none); the D5 triforce 6 runs (12 with the bag), the D7 triforce and the magic sword 3 (2), the D6 triforce 1,
**the ending 1 run of 1000** (0): seed 410, 146,057 frames from the start state, the first single run of the models
alone to end the game on this route (8 wall bombs and 16 rupees cheated). Cheated a run: 0.69 wall bombs (3:0x5b's
last pass 0.51), 0.48 rupees, no key, nothing from a bag.

**The golden path without the bag** (from console power on, models alone, the default route): three of three end
the game, 161 runs together; every replay verifies with the emulator alone; each takes the triforces in the
route's order (3, 4, 1, 5, 2, 7, 6, 8), the magic sword at 0:0x21 (step 207) and ends in game mode `$13` at 9:0x32.

| Path (first seed) | D3 | D4 | D1 | D5 | D2 | D7 | D6 | D8 | to the end | Frames | Cheated bombs | Cheated rupees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 4 | 1 | 7 | 1 | 6 | 14 | 4 | 6 | 174,599 | 11 | 8 |
| 2880 | 10 | 7 | 3 | 11 | 1 | 6 | 7 | 6 | 6 | 166,381 | 8 | 15 |
| 5760 | 5 | 4 | 1 | 13 | 1 | 5 | 8 | 7 | 9 | 172,188 | 6 | 2 |

Dungeon 5's segment took 7, 11 and 13 runs (with the bag 14, 14 and 9): the race's first winner needs no more tries
without it, though the segment chain says a single try wins 7% against 33%; the race plays the leg and the dungeon
from the last winner's state, whose bombs and hearts are one run's and not the segment state's.

**No golden path is free of cheated bombs.** Six walls are cheated in all three: 3:0x5b's last pass, 1:0x53,
0:0x2c, 0:0x2d, 5:0x66 and 5:0x65; one or two paths also cheat 2:0x0e, 7:0x69, 7:0x19, 7:0x1a, 6:0x28 and 9:0x17.
The candle shop 0:0x0c takes 2-15 cheated rupees in every path. The race prefers the win with the fewest cheated
items among those it has (`triforce/golden.py`) and accepts any; a win without a cheated bomb needs the route's
bomb economy changed (backlog "A win without a cheated bomb").

**The corpus.** `corpus/replays/v2-golden-ending.tfreplay` is now the third path (172,188 frames, about 48 minutes
of game, `power-on LINK 29119`, the models in control from frame 29,783, 25 edits, 6 cheated bombs and 2 rupees,
none from a bag); it replaces 2.0.0.0's, which used the bag. New: `corpus/replays/v2-single-ending.tfreplay`, the
single run of seed 410 from the start state to the ending (146,057 frames, 44 edits).

**What was stopped and where it is.** Dungeon 3's trainings ("Dungeon 3", §3): run dirs in `models/`, not promoted,
nothing measured on a chain. The learned guard: branch `r35-guard` (code, tests, scripts; not on `main`), its
labels and trained guard under `target/worktrees/r35-guard/` (untracked), `d7` never measured. Every worktree is
removed; their target data and logs are under `target/worktrees/<name>/`.

```sh
A="--route d3first --clips on"
triforce eval-batch r35nobag-1k --parallel 10 \
  --chain "d5: $A --from 5_76s_0 --stop-at step:120 --runs 1000 --act model --output target/r56-chains/r35nobag-1k-d5.jsonl" \
  --chain "plan-d5: $A --from 5_76s_0 --stop-at step:120 --runs 300 --act temperature:0.5+relax:16+plan:16:1 --output target/r56-chains/r35nobag-plan-d5.jsonl" \
  --chain "full: $A --runs 1000 --act model --output target/r56-chains/r35nobag-1k-full.jsonl"
python zelda-headless.py target/r35-wrap/golden 3 --golden --power-on --clips on --act model --parallel 8
triforce eval-chain $A --runs 1 --first-seed 410 --act model --parallel 1 --replays target/r35-wrap/single   # the single run
python scripts/experiments/r32team_golden.py REPLAY     # triforces, sword, end RAM, cheats by room
```
