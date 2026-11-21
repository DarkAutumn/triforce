# Round 33: the families phase IX left owed

## Wizzrobe and vire: Dungeon 6's 0x78, 0x29, 0x39 and 0x2d

**State of this section.** Complete (the joint `d6` check: +7.7 ± 2.1). Promoted in this branch's checkout: wizzrobe `r33w-s93` (it003264), vire `r33-vire-s95` (it002342).

wizzrobe plays Dungeon 6's 0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c and level 9's 0x15, 0x17, 0x07, 0x05, 0x20, 0x10, 0x31,
0x30, 0x04 (0x65 has left the route); vire plays Dungeon 4's eight rooms and Dungeon 6's 0x39 and 0x2d. The
route's runs are `r30w-wj-s89` (round 30, a joint run for Dungeons 6 and 9) and `r77-b` (round 27). Phase IX
(round31.md) changed three of their rooms: 6:0x78 is a ringleader room with `drops: first` (step 262), 0x29's key
is taken (step 268), 0x39 and 0x2d are `bombs: tenth` (steps 269, 273).

### The cause

From the chains of record (`r31land-1k-d6`, 1000 runs from `6_79s_0`, seeds 0-999; the old route's
`r29land-1k-d6` beside), and two chains of the landed team with handoffs and replays: `h6` from 0:0x42 after
Dungeon 7 (600 runs, seeds 5000-5599: the route's hearts, Dungeon 6 entered with 10.83 of 12; its triforce 23.2%)
and `h6b` from `6_79s_0` (400 runs, seeds 6000-6399; 33.8%).

| room (step) | new route: entered, hearts in, lost a visit, deaths | old route | below 4 hearts in: deaths, new / old |
|---|---|---|---|
| 0x78 (262) | 1000, 12.00, 0.01, 0 (0 kills, 304 frames) | 1000, 0.00, 0 | |
| 0x38 (266) | 1000, 10.81, 3.94, 89 | 1000, 4.47, 137 | |
| 0x29 (268, the key) | 901, 6.58, **1.52**, **74** | 839, 0.54, 27 | 66 of 188 / 27 of 194 |
| 0x39 (269, vire) | 829, 5.47, −0.13, 36 | 814, −0.17, 26 | 36 of 262 / 26 of 217 |
| 0x3a (270) | 797, 5.78, 2.31, 354 | 801, 2.76, 377 | |
| 0x2d (273, vire) | 437, 4.43, 0.04, **29** | 406, −0.25, 6 | **29 of 202 / 6 of 183** |
| 0x2c (274) | 402, 4.63, 1.23, 89 | 399, 1.03, 56 | |

- **0x78:** the model walks through: no swing in 1000 visits. `drops: first` marks the drops to take; nothing makes
  them.
- **0x29:** every visit that survives takes the key, and no run leaves by a wrong exit (main's 1.9.0.0 wizzrobe on
  the new route: 73 wrong exits, the 9:0x63 cheated key's cause). What the key costs is hearts: 1.52 a visit
  against 0.54 on the old path, so 47 more deaths, nearly all entering below 4 hearts. At the route's hearts the
  room is cheaper (`h6`: 0.85 a visit, 27 deaths of 477), from the segment's start not (`h6b`: 1.6, 23 of 359). The
  room is the same: the walk's door-row way played by a rule over the env's actions from 150 states cut from each
  chain's replays, at 8 hearts, loses 1.17 and 1.12 hearts, 150 of 150 keys, no death. The states differ in what
  the model sees of the rupees (37-40 against 43-44, the observed `rupees_spare`) and the drop cycle; the model
  plays the same room dearer from the segment's start. Training takes 0x29's states from both chains.
- **0x39:** at equal hearts as before (36 of 262 deaths below 4 hearts against 26 of 217); more runs enter it low
  because 0x29 costs more. Not a vire loss.
- **0x2d:** **the Vires cost runs since they are not bombed**: 29 deaths against 6, all entering below 4 hearts
  (14% of those visits against 3%), 0.04 hearts a visit against a gain of 0.25 (the bombs' drops). About 20 runs of
  1000 on `d6`: vire is owed its round.
- 0x2c (wizzrobe) loses 33 more than with the older wizzrobe (round 30 saw it: 88 against 50 on its confirmation
  chains); 0x3a is still the dungeon's killer.

### The arm

The recipe of decisions 2026-11-19 ("One team for the whole route"), as round 30 applied it:

- **Starts at the chain's hearts.** wizzrobe's harvested set (2601 states, `harvested_fraction` 0.33, two thirds
  of them Dungeon 6's): level 9's from round 30 (0x05 250, 0x10 250, 0x17 150, 0x30 200); Dungeon 6's taken again
  on the new route from `h6` and `h6b`: 0x78 200, 0x38 400, 0x3a 400, 0x2c 351 (handoffs), 0x28 100 and 0x29 300
  (cut from the two chains' replays, `scripts/experiments/r29_zora_states.py`: they follow a wizzrobe room). vire's
  first harvested set (822: 0x39 450, 0x2d 372), `harvested_fraction` 0.2 (with its seed states Dungeon 6 is about
  a third of the resets, Dungeon 4 the rest), `chain_hp: 0.5` (Dungeon 6's rooms are profiled) and, as every family
  since the owner's Nov 18 rule, `clips: {enabled: true, off_share: 0}`.
- **0x78 by the walk's rule** (`scripts/experiments/r33_d6_demos.py ring-78`, over wizzrobe's own actions): W onto
  the doorstep; with full hearts (BEAMS offered) Link waits there facing W for 44 frames or two swings, swinging at
  a Red that fades in or stands on the row ahead; below full hearts a swing only at a Red within 24 px ahead in
  passing; then today's path, a bomb drop within 64 px of (`$D0`,`$7D`) or (`$78`,`$7D`) waited out there and
  taken. 150 runs from `6_78e_0-2`:

  | play | through | hearts lost | frames (median) | runs with a kill | leader (all five) | bomb drop | bombs gained (runs) |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | today's path (no swing) | 150 | 0.00 | 301 | 0 | 0 | 0 | 0 |
  | the rule, full hearts | 150 | 0.44 | 337 | 144 | 27 | 34 | 27 (23) |
  | the rule, drops left, full hearts | 150 | 0.10 | 337 | 144 | 27 | 34 | 0 |
  | the rule, 10 hearts | 150 | 0.00 | 302 | 57 | 12 | 16 | 15 (5) |

  Taking a drop costs: it falls on the door row the Reds shoot along. The demonstration rows (`ring78.npz`, 1200
  runs at the family's hearts draws, every one through, 0.09 hearts, 32,933 rows) are listed three times beside
  the others (about 8% of the pooled rows). `drops: first` then makes a fallen pack the objective.
- **The planner at the route's hearts** (`scripts/experiments/r27_plan_demos.py`, `temperature:0.5+relax:16+plan:16:1`,
  two episodes a harvested state, the family yaml with `ram_randomization: {}` so a state keeps its hearts):

  | file (run played) | state: won of episodes, hearts lost in the wins | rows |
  |---|---|---:|
  | `plan33-d6` (`r30w-wj-s89`) | 6:0x29 519 of 600, 0.74 (the models at the route's hearts lose 6% of their visits, 0.85-1.6 hearts a visit); 0x2c 647 of 702, 0.44 (the models 80-84% through, 1.3); 0x38 92 of 100, 2.33; 0x3a 273 of 400, 1.09 | 75,472 |
  | `plan33-vire` (`r77-b`) | 6:0x39 894 of 900, −0.57 (a gain: drops); 0x2d 725 of 744, −0.44 | 82,346 |

- **Three seeds and one**, `--sil-coeff 0.1 --checkpoint-every 30 --stop-success 1.01`, `--demo-coeff 0.05` over the
  lineage's files and the new ones: `r33w-s91`, `-s92`, `-s93` from `r30w-wj-s89`, 488 iterations (`bomb28`,
  `plan38-3a`, `wizzrobe-05`, `-10`, `-30`, `clip-07`, `clip-20n`, `plan9-wizz`, `plan30-d6-05`, `plan30-10-30`,
  `plan33-d6`, `ring78` ×3); `r33-vire-s95` from `r77-b`, 244 iterations (`plan33-vire`; the lineage had none). No
  curriculum track of either family is unfinished (wizzrobe's Dungeon 6 tracks, the rebuilt 0x29's included, all
  succeed; vire's 0x39 and 0x2d too), so none is taken out.

### The runs

- `r33-vire-s95` (last), single rooms on equal terms (100 episodes a state, this branch's yaml; `r77-b` in
  brackets): 99.5% of 3000 episodes, +0.10 hearts (98.7%, +0.04); 6:0x2d 98.3% (94.0%), 0x39 99.0% (97.0%); Dungeon
  4's seven rooms within a point or better. Its choice on `d6` (screen 2000-2999, confirmation 3000-3999, `r77-b`
  pinned) and the pairing on `d4` are queued; single rooms do not decide.
- `r33w-s91`, `-s92`, `-s93`: queued (the machine's line). `r30w-wj-s89` under this yaml: 92.8% of 5700, 6:0x29
  92.7%, 0x2c 94.0%, 0x38 78.3%, 0x3a 63.0%, 0x78 100%.

- `r33-vire-s95` chosen on `d6` (`select-checkpoint vire r33-vire-s95 --segment d6 --stop-at step:277 --runs 1000
  --first-seed 2000 --confirm 1000 --confirm-first-seed 3000 --confirm-top 1 --only it002312,it002342,it002372,last,promoted`):
  screen it002342 33.7%, `r77-b` 31.5%; **confirmation (3000-3999) it002342 33.2% against 31.7%, +1.5 ± 1.8 paired:
  not beyond noise.** In its own rooms the gain is plain: 0x2d 8 deaths against 30 (every one entering below 4
  hearts), 0x39 19 against 28, 0x2d's visits a gain of 0.24 hearts again (drops) against a loss of 0.03; the runs it
  saves mostly die later in 0x3a and 0x2c. A further 1000 seeds (4000-4999) are queued to pool with the confirmation.
- The further 1000 seeds (4000-4999, `--only it002342,promoted --confirm 0`): it002342 31.9% against `r77-b` 28.3%
  (+3.6 ± 1.7; 0x2d 13 deaths against 30, 0x39 20 against 35). **Pooled with the confirmation, 2000 paired runs on
  seeds not used to choose it: +2.55 ± 1.23, beyond noise (2.1 standard errors).** Its other segment, `d4`, is queued
  (1000 paired runs, seeds 3000-3999); the promotion waits on it.
- `r33w-s92` on `d6` (screen 2000-2999, `--no-promoted`): it003294 33.7%, last 32.3%, it003264 31.1%, it003324 29.1%;
  confirmation (3000-3999) it003294 32.9%, **last 33.5%, chosen**. The pairing against `r30w-wj-s89` on the same
  seeds waits for s91's selection, which measures it. In its 1000 confirmation runs: **0x78 swung at: a kill in 334
  runs (432 kills), 49 bomb drops, 26 picked up, 0.06 hearts and 335 frames a visit (the record: no kill, 0.01 hearts,
  304 frames); hearts on entering 0x38 10.87, 0x3a 6.34** (the record on seeds 0-999: 10.81 and 5.78); **0x29 29
  deaths of 893 visits** (the record 74 of 901), 0.34-0.63 hearts a visit.
- `r33w-s91` on `d6` (`r30w-wj-s89` pinned): screen it003324 34.8%, last 31.4%, `r30w-wj-s89` 31.5%; confirmation
  it003324 32.7% (chosen), `r30w-wj-s89` 31.7%. **Paired on seeds 3000-3999 against `r30w-wj-s89`: s91 +1.0 ± 2.1,
  s92 +1.8 ± 2.1; neither beyond noise yet.** On the same seeds (`r30w-wj-s89` / s92 / s91): 0x29 61 / 29 / - deaths;
  0x78 kills in 0 / 334 / 237 runs; hearts on entering 0x38 10.81 / 10.87 / 10.64, 0x3a 5.70 / 6.34 / 6.17; 0x3a
  330 / 368 deaths. **The 0x29 key's cost is won back (32 fewer deaths) and 0x78 is swung at with no measurable cost
  in the hearts 0x38 is entered with; the runs saved die in 0x3a**, whose fight is unchanged. 1000 more seeds of s92
  (4000-4999) are queued to pool, as for vire.
- s92 (last) and `r30w-wj-s89` on 1000 further seeds (4000-4999): 32.2% against 28.3% (+3.9 ± 2.0; 0x78 kills in 319
  runs, hearts on entering 0x38 10.80 against 10.82). **Pooled over 3000-4999 (2000 paired runs): +2.85 ± 1.46, 1.95
  standard errors: just short of the bar.** A third block (7000-7999; 6000-6399 were `h6b`'s, whose states trained it)
  and s92's `d9` and `post-d7` pairings are queued; the verdict is read on the 3000 pooled runs, which after a near
  miss carries a small optional-stopping bias.
- The third block (7000-7999): s92 32.8% against 29.4% (+3.4 ± 2.1; 0x78 kills in 317 runs; hearts on entering 0x38
  10.80 against 10.75, 0x3a 6.34 against 5.89). **Pooled over 3000 paired runs on seeds not used to choose it:
  +3.03 ± 1.19 (2.5 standard errors): `r33w-s92` is better on `d6` beyond noise.** Its `d9` and `post-d7` pairings
  decide the promotion.
- `r33w-s93` on `d6` (`--no-promoted`): screen it003264 35.7%, it003294 33.8%, it003324 29.3%, last 28.5%;
  confirmation it003264 36.5% (chosen), last 29.5%. **Paired on 3000-3999 against `r30w-wj-s89`: +4.8 ± 2.1**
  (against s92 +3.0 ± 2.1). It gains elsewhere than s92: 0x3a 302 deaths (`r30w-wj-s89` 330, s92 368), 0x2c 50 (88,
  78), but 0x29 95 (61, 29); 0x78 kills in 263 runs, 0x38 entered with 10.75 hearts. The same two further blocks
  (4000-4999, 7000-7999, paired against `r30w-wj-s89`'s chains there) and its `d9` are queued; the choice between s92
  and s93 is made on the 3000 pooled seeds of each.
- `r33-vire-s95` it002342 on `d4` (1000 paired runs, seeds 3000-3999): 94.2% against `r77-b`'s 90.8% (5.34 hearts
  out against 5.10), **+3.4 ± 1.2: better beyond noise there too.** **vire qualifies**: `d6` +2.55 ± 1.23 (2000
  runs), `d4` +3.4 ± 1.2; it002342 applied as the run's `ckpt.pt`, its evaluation queued, the promotion after the
  wizzrobe selections that play vire's promoted run.
- `r33w-s93` (it003264) on 4000-4999: 35.1% against `r30w-wj-s89`'s 28.3% (+6.8 ± 2.1; against s92 +2.9 ± 2.1).
  **Pooled over 3000-4999: +5.80 ± 1.46** (s92 +2.85 ± 1.46 on the same seeds).
- `r33w-s93` (it003264) on 7000-7999: 36.1% against 29.4% (+6.7 ± 2.1). **Over the 3000 pooled seeds: +6.10 ± 1.19
  against `r30w-wj-s89`, +3.07 ± 1.21 against s92: s93 is the wizzrobe run chosen** (applied as its `ckpt.pt`, its
  evaluation and `d9` pairing pending). Deaths over the 3000 runs, `r30w-wj-s89` / s92 / s93:

  | room | 0x38 | 0x28 | 0x29 | 0x39 | 0x3a | 0x2d | 0x2c | 0x1c (Gohma) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | deaths | 338 / 309 / 289 | 52 / 48 / 42 | 188 / 105 / **241** | 93 / 63 / 82 | 1001 / 1099 / **949** | 86 / 73 / 78 | 243 / 217 / **180** | 84 / 86 / **36** |

  s93 wins in the dungeon's back half (0x3a, 0x2c, and runs reach Gohma with more hearts) and loses more in 0x29
  than either; s92 has the 0x29 lesson. Both swing at 0x78 (s93 a kill in 233-263 runs of 1000) without changing the
  hearts 0x38 is entered with (10.75-10.81 against 10.75-10.82).
- `r33w-s93` (it003264) on `d9` (1000 paired runs, seeds 3000-3999): 15.5% against `r30w-wj-s89`'s 13.6%, **+1.9 ±
  1.3: not worse** (s92 +0.9 ± 1.2). `post-d7` is run for run (wizzrobe plays no room before 6:0x79: s92 and
  `r30w-wj-s89` 99.5% both, 10.67 hearts). Single rooms (100 episodes a state): 93.4% of 5700, −0.95 hearts
  (`r30w-wj-s89` 92.8%, −1.02); every state within a point or better but 9:0x17 (94.0% against 95.3%).

### The verdict

| family | run (checkpoint) | replaces | target, fresh seeds, paired | other segments |
|---|---|---|---|---|
| wizzrobe | `r33w-s93` (it003264) | `r30w-wj-s89` | `d6` **+6.10 ± 1.19** (3000 runs) | `d9` +1.9 ± 1.3, `post-d7` equal |
| vire | `r33-vire-s95` (it002342) | `r77-b` | `d6` **+2.55 ± 1.23** (2000 runs) | `d4` **+3.4 ± 1.2** |

Both qualify and are promoted in this branch's checkout. Each was judged with the other family's old run; together on
`d6` (seeds 8000-8999) +7.7 ± 2.1 against both old.

**0x78, plainly (for the owner).** A model can be made to swing there without paying for it in 0x38: every trained
seed swings, a kill in 23-33% of the visits, the leader and its pack sometimes; 0.06-0.2 hearts a visit, and 0x38 is
entered with the same hearts as before (10.75-10.87 against 10.75-10.82 on the same seeds). What it brings is small:
about 0.1 bombs a run carried out of the room (a pack falls in about 5% of the visits and is taken in half of
those; the rule at full hearts gets one in about 16% of the runs, at 0.4 hearts). The swing is cheap and the gain is
nearly nil: Dungeon 6's runs are won and lost elsewhere (0x3a).

**0x29.** The key's cost is the model's, not the room's: s92 learned it (105 deaths in 3000 runs against 188), s93,
the run promoted for the rest of the dungeon, did not (241). From the segment's start the room costs more than at the
route's hearts (the observed rupees differ); a run that has both lessons is the next step.

```sh
# the harvest (W = this worktree, R = W/target/r33d6)
triforce eval-batch r33d6-harvest --chain "h6: --from 0_42c_0 --stop-at step:277 --runs 600 --first-seed 5000 --act model --clips on --harvest R/harvest-d6 --replays R/replays-d6 --output R/chains/h6.jsonl"
triforce eval-batch r33d6-harvest-b --chain "h6b: --from 6_79s_0 --stop-at step:277 --runs 400 --first-seed 6000 ... --harvest R/harvest-d6b --replays R/replays-d6b"
python scripts/experiments/r29_zora_states.py R/replays-d6b R/cut-d6b --step 267:6_28s --step 268:6_29w --count 160
python scripts/experiments/r33_d6_demos.py ring-78 --runs 400 --hearts family --out R/demos/ring78.npz
python scripts/experiments/r27_plan_demos.py models/wizzrobe/r30w-wj-s89 R/demos/plan33-d6.npz --family R/wizzrobe-keep.yaml --states --state-files <6_29w, 6_2ce, 6_38s, 6_3aw harvested> --runs 2 --seed 7400
python scripts/experiments/r27_plan_demos.py models/vire/r77-b R/demos/plan33-vire.npz --family R/vire-keep.yaml --states --state-files <vire harvested> --runs 2 --seed 7300
scripts/train_round.py --round r33w-sN --specialists wizzrobe --parallel 8 --checkpoint-every 30 --train-args "--load models/wizzrobe/r30w-wj-s89 --iterations 488 --seed N --stop-success 1.01 --sil-coeff 0.1 --demos <lineage files> plan33-d6.npz ring78.npz ring78.npz ring78.npz --demo-coeff 0.05"   # N = 91, 92, 93
scripts/train_round.py --round r33-vire-s95 --specialists vire ... --train-args "--load models/vire/r77-b --iterations 244 --seed 95 ... --demos plan33-vire.npz --demo-coeff 0.05"
triforce select-checkpoint wizzrobe r33w-sN --segment d6 --stop-at step:277 --runs 1000 --first-seed 2000 --confirm 1000 --confirm-first-seed 3000 --confirm-top 1 --only <last three>,last[,promoted]
triforce select-checkpoint wizzrobe r33w-s93 --segment d6 --stop-at step:277 --first-seed 4000|7000 --confirm 0 --only it003264 --no-promoted   # and d9
```

**The pair together** (`eval-batch r33d6-joint`: `d6`, 1000 runs, seeds 8000-8999, both new against both old):
37.7% against 30.0%, **+7.7 ± 2.1**. Deaths, new / old: 0x38 91 / 104, 0x29 76 / 59, 0x39 10 / 56, 0x3a 329 / 327,
0x2d 10 / 37, 0x2c 66 / 72, Gohma 16 / 23; 0x78 a kill in 214 runs, 0x38 entered with 10.74 hearts (10.81), 0x3a with
6.30 (5.82). The two gains add up.

## like-like at 9:0x56, lanmola at 9:0x55 from E (Dungeon 9)

The owner's phase IX (round31.md) sends Dungeon 9 from 0x66 N into 0x56 (2 Bubbles, 2 Zols, 2 Like-Likes), across
it to its W bomb wall (step 310, `bombs: required`, the room `requires_bombs`) and into 0x55 from E (step 311, the
Lanmolas' room, crossed by the block clip `CLIP N` from (`$A0`,`$9D`)). Both rooms are new to their families:
like-like `r79c-like` never trained on 0x56, lanmola `r79b-lanmola` trained 0x55 from S.

### The cause

From the chains of record (`r31land-1k-d9`, 1000 runs from `9_76s_0`, seeds 0-999):

- **"stuck" is the bounce limit.** All 616 runs that end in 0x56 end `bounce-limit: like-like left into (level 9,
  room 0x66), the 5th return`. like-like walks straight back out of the S door it came in by: 1635 visits end
  `exit-wrong` after 7.6 decisions and 0.02 hearts (878 of the 1000 first visits), stalfos-keese-gel sends Link N
  again from 0x66, five times. 438 visits leave W as the route says (382 bombs laid, none cheated; 1250 frames,
  0.72 hearts a visit).
- **The route data and the engine are right.** At the S entry of `9_56s_0..2` (like-like's environment) the objective
  is `MOVE` with the next room W; the mask offers `MOVE N` and `MOVE S` (Link stands in the doorway). The walk's way
  played as a rule over the environment's own actions (below) is offered `BOMB W` at (`$20`,`$8D`) in 300 of 300
  runs with no masked choice. Nothing is missing in the engine: the model has never seen the room.
- **0x55 from E**: 438 visits: 368 cross by the clip (37 decisions, 0.18 hearts; the walk's rule takes 15 and none),
  54 wrong exits, 7 stuck, 9 deaths (16 runs lost).

### The arm (the recipe of decisions 2026-11-19)

- **The walk's rules as teachers** (`scripts/experiments/r33_d9_demos.py`, round 29's `r29_demos` loop over the
  family's environment with its own hp and bombs draws, `--hearts family`, 100 runs a chain):

  | Rule (family) | Way | Won | Hearts lost in the wins | Decisions | Rows |
  |---|---|---:|---:|---:|---:|
  | `wall-56` (like-like) | up to the bottom row Y `$BD`, W to X `$20`, up over the water to (`$20`,`$8D`), `BOMB W`, W (a held Link stabs) | 300 of 300 | 0.06 | 32.6 | 21,832 |
  | `clip-55e` (lanmola) | from the E door down to Y `$9D`, W to X `$A0`, `CLIP N`, the stairs | 300 of 300 | 0 | 15 | 4,500 |

  Each rule's first three won runs of each chain are curriculum tracks (`9_56s_<c>~rule100N`, `9_55e_<c>~rule100N`).
  Every track of both families finishes (like-like 48, lanmola 21): none is taken out.
- **Handoffs at the route's hearts** (`eval-chain --from 0_6dc_0 --runs 500 --first-seed 5000 --harvest DIR`, the
  landed team: Dungeon 9 entered as the route enters it): 9:0x56 is entered 1680 times with 11.41 hearts (the
  bounces included), 0x55 from E 629 times with 10.15. Drawn at `harvested_fraction: 0.25`: like-like 0x56 400, 0x14
  130, 0x51 66, 0x03 34; lanmola 0x55 from E 300 (`triforce/states/harvested/<family>`, not in git).
- **The runs**: like-like three seeds from `r79c-like` (`r33-like-s81`, `-s82`, `-s83`), lanmola one (`r33-lanmola-s80`
  from `r79b-lanmola`); 244 iterations, `--stop-success 1.01 --sil-coeff 0.1 --checkpoint-every 30`, clips
  `{enabled: true, off_share: 0}` (both yamls already), `--demo-coeff 0.05` over the lineage's files and the new one
  (like-like: `like-like-14`, `clip-03`, `wall-56`; lanmola: `clip-55`, `clip-55-300`, `clip-55e`).

### The selections on `d9`

`select-checkpoint like-like r33-like-sNN --segment d9 --runs 1000 --first-seed 2000 --confirm 1000
--confirm-first-seed 3000 --confirm-top 1 --only it003295,it003325,it003355,last[,promoted] --parallel 10`
(`r79c-like` pinned in s83's batch, `--no-promoted` in s81's and s82's), the rest of the team main's:

| run | it003295 | it003325 | it003355 | last | confirm (3000-3999) |
|---|---:|---:|---:|---:|---|
| `r33-like-s81` | 45.2% | 46.1% | **47.0%** | 45.2% | it003355 48.6% (3.77 hearts at the end), last 47.6% |
| `r33-like-s82` | **47.3%** | 44.9% | 46.7% | 47.3% | it003295 47.8% (3.76 hearts), last 45.5% |
| `r33-like-s83` | 45.1% | 44.9% | **47.7%** | 44.6% | **it003355 49.4% (3.80 hearts)**, last 43.7% |
| `r79c-like` (promoted) | | | | 13.0% | 13.6% (3.45 hearts) |

**On seeds not used to choose it, `r33-like-s83` at it003355 takes `d9` in 49.4% against 13.6%: +35.8 ± 1.9
points paired** (s81's choice 48.6%, s82's 47.8%: the three seeds agree within noise). 9:0x56 is crossed in every
visit (s81 it003295: 1004 of 1004 visits leave W; main's record: 438 of 2689). Main's record of the segment,
13.7% on seeds 0-999, matches the reference here.

**The other segments** like-like plays, paired with `r79c-like` on seeds 0-999 (`select-checkpoint like-like
r33-like-s83 --segment SEG --runs 1000 --confirm 0 --only it003355,promoted --output ...`):

| segment | `r33-like-s83` it003355 | `r79c-like` | paired |
|---|---:|---:|---:|
| `d9` (seeds 3000-3999, not used to choose it) | **49.4%** (3.80 hearts) | 13.6% (3.45) | **+35.8 ± 1.9** |
| `d6` (6:0x1d) | 29.5% (5.97) | 28.5% (6.07) | +1.0 ± 0.7 |
| `d4` (4:0x32) | 91.3% (4.91) | 91.7% (5.01) | −0.4 ± 1.1 |

**Verdict: `r33-like-s83` (it003355, ckpt sha1 6db044a4) qualifies** (the target better beyond noise, no other
segment worse beyond noise) and is promoted in this branch's checkout (applied by `select-checkpoint --apply`; its
canonical `eval.json` is not written yet).

**lanmola on `d9`** (`select-checkpoint lanmola r33-lanmola-s80 --segment d9 --runs 1000 --first-seed 2000 --confirm
1000 --confirm-first-seed 3000 --confirm-top 1 --only it001644,it001674,it001704,last,promoted`, like-like
`r33-like-s83` promoted in the team): screen 47.0-48.0% against `r79b-lanmola` 47.7%; confirm it001704 46.6%, last
46.7% against **49.4%: −2.8 ± 2.3** (hearts at the end 3.61 against 3.80). The room itself is learned: on the
confirmation seeds `r33-lanmola-s80` leaves 0x55 from E by the clip in 1000 of 1000 visits, `r79b-lanmola` in 984
(12 deaths, 16 wrong exits, 4 stuck; 12 runs end there). The runs it saves there are lost later in the dungeon at the
same rate, so the segment does not move beyond noise. **Verdict: not better on its segment; `r79b-lanmola` stays.**

### Backlog

- like-like: the explorer's 9_56s tracks wander (a `BOMB S` in each) beside the rule's; set them aside if a later
  round sees the model lay bombs off the wall.
- like-like is trained on two lines now (this one for `d2first`'s 9:0x56, `r32d4-like2` for `d3first`'s Dungeon 4):
  the one team needs a joint continuation or a choice on both routes' segments.
- lanmola `r33-lanmola-s80` crosses 9:0x55 from E in 1000 of 1000 visits (`r79b-lanmola` loses 1.2% of runs
  there) without moving `d9` beyond noise (−2.8 ± 2.3); a second seed or 2000 more paired runs would tell whether
  the room's gain shows in the segment.
