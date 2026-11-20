# Round 33: the families phase IX left owed

## Wizzrobe and vire: Dungeon 6's 0x78, 0x29, 0x39 and 0x2d

**State of this section.** The cause and the arm; the runs and the choice follow.

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

### The runs (in progress)

- `r33-vire-s95` (last), single rooms on equal terms (100 episodes a state, this branch's yaml; `r77-b` in
  brackets): 99.5% of 3000 episodes, +0.10 hearts (98.7%, +0.04); 6:0x2d 98.3% (94.0%), 0x39 99.0% (97.0%); Dungeon
  4's seven rooms within a point or better. Its choice on `d6` (screen 2000-2999, confirmation 3000-3999, `r77-b`
  pinned) and the pairing on `d4` are queued; single rooms do not decide.
- `r33w-s91`, `-s92`, `-s93`: queued (the machine's line). `r30w-wj-s89` under this yaml: 92.8% of 5700, 6:0x29
  92.7%, 0x2c 94.0%, 0x38 78.3%, 0x3a 63.0%, 0x78 100%.

Not decided yet: no run of either family is promoted, and nothing about 0x78's swing is measured on a chain.
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
