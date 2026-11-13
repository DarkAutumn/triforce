# Round 20: Dungeon 4 (the route's fourth dungeon)

Lee, Nov 12: finish Dungeon 4. Merge it into the one route (`d2first`, delete `d2first-d4`), take the keys of 0x51
and 0x40, let Vires be bombed (0x31 and 0x12; walk past them in 0x61), keep the scouted families (`vire`,
`like-like`, `ow-peahat`, `boss-gleeok`; the rupee room 0x11 stays in `trap-room`). Train the new families,
continue the old ones on their Dungeon 4 rooms, measure in single rooms and in chains, promote. Branch `d4`, run ids
`r70-*`; the version goes to 1.4.0.0 with the merge (four dungeons). Decisions: `docs/decisions.md` 2026-11-12.

## Summary

Dungeon 4 is part of `d2first` (steps 98–122; the route ends on the D4 triforce) and every family on it is trained
and promoted. Three engine gaps found in zelda-asm had to be closed first: the ladder (0x30's water row stands
before the route's N door), Gleeok (his heads weren't observable and his body paid boss hits), and kills the game
doesn't count (a Vire's Red Keese) not resetting the stall penalty. The explorer then found a fourth: the executor
could walk Link through a wall by turning him sideways in a doorway (§2). Its fix, a game-faithful action mask,
costs the old D2 models: the D2 segment fell from 82.1% to 68.6% (§5). **Lee (Nov 12): a complete retrain follows
from another agent; the regression is documented here and in backlog.md, not fixed in this round.**

1000 runs per segment (`r70-1k`): D4 segment (4:0x71 → D4 triforce) **21.1%**, D3 → out of D3 38.6%, after D3 → D4
99.8%; from the start to the D4 triforce **1.8%** (to D3 26.7%, round 19 36.3%). Most D4 losses are Gleeok (199
deaths, 188 stalls per 1000): Link arrives with 2.1 hearts on average, below the health Gleeok was trained at (§4).

## 1. Route merge and engine fixes (`3caf6c5`)

- **Route.** The 25 steps after the D3 triforce; `treasure: skip` dropped on 0x51 and 0x40 (the keys are taken: 7
  keys into 0x30, 5 out); `bomb_reserve: 2` from 4:0x71 (the walls of 0x21 and 0x11). The 75 states moved from
  `triforce/states/d4/` into `triforce/states/` (manifest route `d2first`; audit: 370 states, 0 problems).
  `r56_chains.sh` gained the `post-d3` (0_74c_0 → 4:0x71) and `d4` (4_71s_0 → the route's end) segments.
- **The ladder** (`Room::ladder_axes`): with the ladder in an underworld room, water exactly one square long between
  walkable tiles is passable along its axis; perpendicular moves and two-square strips stay walls. Tests: synthetic
  rooms, 4_30e_0 (crossing with the ladder, stopped at Y = $85 without it), 4_31e_0 (the four-square shortcut is
  shorter than the land path).
- **Gleeok** (`zelda::game::gleeok_heads`): one synthetic enemy per living attached head (id $42, slot 2 + neck) at
  its neck-array position with slot 5's shared HP; the neck base and the flying heads ($46) are hazards; a hit on the
  shared HP is credited once. Tests: RAM images (one hit for a shared-HP drop, nothing for the base or a flying head,
  one hit and one kill for a cut head) and 4_13w_0 in the emulator (two heads, the base a hazard, the objective a
  Fight on the heads, the observation's killable flags).
- **Uncounted kills** (`Change.observed.enemies_killed`) restart the room-stall count.

## 2. Curricula and the doorway exploit (`c222323`)

`triforce.explore` for vire (30 states), like-like (6), boss-gleeok (3), ow-peahat (3), and the D4 states of zol,
equipment-room, trap-room, ow-moblin and ow-zora: all solved (4_60cave_2 with `--seed 1`). vire solved 4_30e_0..2
(the ladder crossing); boss-gleeok needed 79–110 thousand env steps per state against the scouting build's 189–314
thousand.

trap-room's 4_11s states were "solved" in 2 steps: `MOVE S 8`, `BOMB E`. From the S doorway the executor turned Link
east by writing ObjDir; the game then moves Link along ObjDir without a tile test, and he walked through the wall
into 0x12 (215 frames). The game itself never turns Link sideways in a doorway: a perpendicular press does nothing
(1:0x44's W doorway, 4:0x30's S doorway) or first walks him back into the room (4:0x11). Fix: the mask offers only
attacks along the doorway's axis there (`GameView::attack_directions`; N006's bounds were measured with the facing
poked). The underworld D4 curricula were explored again after the fix (all solved; 0x11 now bombs its way in 33
steps).

## 3. Training (round 20)

Two streams of `--parallel 8`: new families from scratch (8M steps, `--stop-success 0.99 --sil-coeff 0.1`), old
families continued (`--load`, 4M). Single-room evaluations (100 episodes per state, seed 0), in the promoted act mode:

| family | run | act | success | hearts | D4 rooms | linger worst |
|---|---|---|---:|---:|---|---:|
| vire | r70-d4-new | sample | 98.8% | −0.13 | 0x03 100, 0x12 97.0, 0x20 98.3, 0x30 99.7, 0x31 97.5, 0x50 98.3, 0x61 100, 0x71 100 | 475 |
| like-like | r70-d4-new | argmax | 98.0% | −0.50 | 0x32 98.0 | 599 |
| boss-gleeok | r70-gleeok-hp | argmax | 94.7% | −0.66 | 0x13 94.7 | 638 |
| ow-peahat | r70-d4-new | argmax | 100% | 0.00 | 0:0x45 100 | 1 |
| zol | r70-d4-cont | sample | 99.6% | +0.12 | 0x40 99.3 | 77 |
| stalfos-keese-gel | r70-skg-cont | argmax | 99.6% | +0.03 | 0x21 98.0, 0x51 100 | 598 |
| equipment-room | r70-d4-cont | argmax | 99.4% | −0.16 | cellar 0x60 99.7 | 34 |
| trap-room | r70-d4-cont | argmax | 99.9% | −0.21 | 0x11 100 | 13 |
| ow-tektite | r70-d4-cont | argmax | 99.4% | +0.06 | 0:0x74 100 | 14 |
| ow-moblin | r70-d4-cont | argmax | 98.4% | −0.10 | 0:0x63 100, 0:0x73 100 | 41 |
| ow-octorok | r70-d4-cont | argmax | 99.0% | −0.09 | 0:0x64 99.2 | 22 |
| ow-zora | r70-d4-cont2 | sample | 99.3% | −0.20 | 0:0x55 99.5, 0:0x65 98.7 | 25 |

Continuations were compared room by room with the old promoted run re-evaluated under the current yaml
(`eval-r70.json`): promoted when every pre-D4 room kept success within 1 point and hearts lost within 0.05. All but
two passed at once. ow-zora's argmax lost 0.07 hearts in 0:0x56 and was continued once more (`r70-d4-cont2`, which
passes in sample mode, tied with argmax overall). stalfos-keese-gel needed a reset (below).

- **stalfos-keese-gel and 4:0x21** (Lee). The continued run (`r70-d4-cont`) won 0x21 (W → bomb N) in 0% of 300
  episodes: Link wandered to the bottom-left corner and waited until the quiet tail ended the episode (the explorer
  solves the state in 96 steps). Lee had the run thrown away and the family trained from scratch
  (`r70-skg-scratch`: 99.4%, 0x21 97.3%), then continued once (`r70-skg-cont`) for the D1 rooms' hearts.
- **vire in sample mode:** argmax and sample are within noise overall (98.4 vs 98.8%), but argmax wins 0x12 in 91.3%
  against 97.0% (300 episodes each).
- **boss-gleeok** (Lee). The first run (`r70-d4-new`, killed at 3M steps, set aside as `r70-d4-new.interrupted`)
  reached 22% in training. 120 episodes of it by start health: full health 54% (6.8 boss hits per episode, all
  beams), partial health 0 of 59 (0.1 hits per episode; it hid or died), although melee works (poked under a head at
  3 hearts, SWORD N cut a head in 9 swings for 2 hearts). PBRS was ±22 per episode, twice the other bosses' rate per
  step, from dodging and knockback. Lee chose start health `[0.5, max]` (3.5–7 hearts) and `pbrs: false` for this
  family; from scratch it reached 52% at 4M steps and 94.7% at 8M (argmax; sample 89.7%).
- **Linger:** like-like, boss-gleeok and stalfos-keese-gel each have one episode lingering 598–638 steps (Lee's
  500-step rule); not investigated before promotion (backlog).

## 4. Chains, 1000 runs per segment

`RUNS=1000 ACT=model TAG=r70-1k PARALLEL=20 scripts/experiments/r56_chains.sh run` (47 minutes), arrivals:

| segment | r69-1k (round 19) | r70-1k |
|---|---:|---:|
| D2 | 82.1% | 68.6% |
| after D2 | 85.9% | 83.5% |
| D1 | 98.4% | 96.7% |
| after D1 | 100.0% | 100.0% |
| D3 (to 0:0x74 since round 20; before, the D3 triforce) | 39.5% | 38.6% |
| after D3 (0:0x74 → 4:0x71) | — | 99.8% |
| D4 (4:0x71 → the D4 triforce) | — | 21.1% |
| full → D2 | 97.7% | 98.1% |
| full → after D2 | 67.6% | 56.3% |
| full → D1 | 55.2% | 45.8% |
| full → D3 | 36.3% | 26.7% |
| full → after D3 | — | 8.0% |
| full route (D4 triforce) | — | 1.8% |

D4 segment per room (1000 runs):

| room | visits | deaths | stuck | hearts lost/visit | visits with a cheated bomb | bombs placed |
|---|---:|---:|---:|---:|---:|---:|
| 4:0x71 | 1000 | 0 | 0 | 0.00 | 0 | 0 |
| 4:0x61 | 1000 | 0 | 0 | 0.18 | 0 | 0 |
| 4:0x51 | 1000 | 1 | 0 | 0.11 | 0 | 0 |
| 4:0x50 | 999 | 0 | 0 | 0.73 | 0 | 0 |
| 4:0x40 | 999 | 5 | 3 | 0.50 | 0 | 0 |
| 4:0x30 | 1871 | 119 | 0 | 0.39 | 0 | 0 |
| 4:0x31 | 1944 | 78 | 0 | 0.75 | 0 | 0 |
| 4:0x32 | 1939 | 23 | 0 | 0.65 | 0 | 0 |
| 4:0x60 cellar | 963 | 10 | 0 | 0.17 | 0 | 0 |
| 4:0x20 | 761 | 8 | 0 | 0.04 | 0 | 0 |
| 4:0x21 | 753 | 4 | 10 | 0.03 | 753 | 1557 |
| 4:0x11 | 740 | 0 | 0 | 0.00 | 739 | 739 |
| 4:0x12 | 740 | 141 | 0 | 0.75 | 0 | 0 |
| 4:0x13 | 598 | 199 | 188 | 0.93 | 0 | 0 |
| 4:0x03 | 211 | 0 | 0 | 0.00 | 0 | 0 |

- The ladder works in the chain: all 761 runs that reached 4:0x20 came from 0x30 across the water. No cheated key
  anywhere (the keys of 0x51 and 0x40 are taken).
- Every run reaches 0x21 and 0x11 without bombs (the D3 segment's Manhandla spends them), so both walls use a
  cheated bomb. Vires were never bombed (no bombs above the reserve of 2), so Lee's 0x31/0x12 bombing and the 0x61
  walk-past are untested in the chain.
- **Gleeok.** Arrival health averages 2.1 hearts. Outcomes by arrival health: 6 hearts or more 12 of 12 won; 3.5–6
  85% won; under 3.5 hearts (492 of 598 visits) 24% won, 185 deaths and all 188 stalls (2000-step room budget):
  below the trained health range Link hides.
- 0x12 (141 deaths) and 0x30 (119) are the other big D4 losses.

## 5. The doorway fix costs the old models (not fixed; Lee: a full retrain follows)

The D2 segment fell from 82.1% to 68.6% with no D2 family retrained. Same checkpoint (goriya `r68-clear5e`, sample,
seed 0), two builds: without `c222323` its rooms evaluate exactly as its old `eval.json` (2:0x1e 92.0%, −0.38
hearts); with it, 2:0x1e 85.3%, −0.63 (2:0x5e 86.3 → 86.0%, 1:0x44 99.3 → 98.7%). The old models used sideways
attacks from inside a doorway, which the game can't do (§2). D1 (96.7 vs 98.4%) and D3 (38.6 vs 39.5%) barely moved.
Any model trained before `c222323` should be retrained under the fixed mask; the D2 families (goriya, d2-rope,
moldorm, boss-dodongo) are the ones the chains show.

## 6. Promoted

vire, like-like, ow-peahat `r70-d4-new`; boss-gleeok `r70-gleeok-hp`; zol, equipment-room, trap-room, ow-tektite,
ow-moblin, ow-octorok `r70-d4-cont`; ow-zora `r70-d4-cont2`; stalfos-keese-gel `r70-skg-cont` (act modes in §3).
Everything else as round 19.
