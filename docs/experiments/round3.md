# Round 3: pre-entry start states, re-baseline, curriculum where PPO fails

Goal: replace the start-state set (most states were saved after room entry, with monsters already killed or
moved, and with wrong hearts), re-baseline every specialist on the new states with plain PPO, then use Go-Explore
+ backward curriculum (round 2) for the specialists that plain PPO can't solve.

All runs: 16 workers, CUDA, 4,096 steps per update, 2M-step budget, early stop at a balanced 95% for 5 iterations,
`triforce.evaluate` with 20 sampled-action episodes per seed state (seed 0), 4 eval processes on a background
thread while the next specialist trains. One training seed per row.

## 1. Why the old states had to go

Audit of the 84 seed states the round 1/2 configs used:

- **75 were post-entry:** saved after the room had loaded, so the spawn layout was fixed per state and any
  monsters killed on the way in stayed dead.
- **30 were short on monsters, 12 had none.** Kill history (overworld room flags `$67F+room` bits 0–2, dungeon
  flags `$6FF+room` bits 6–7 plus `LevelKillCounts $560+room`) carried over from the play session that made them.
- **15 had 16 hearts** (max health), far above the route's 3–4. Aquamentus and the D1 states were the worst.
- **4 were misnamed** (file name room ≠ RAM room).

Spawn positions are not RNG-driven per load: `AssignObjSpawnPositions` (Z_05:1879) uses `SpawnCycle $524`
(0–8), set at room entry; overworld edge-spawn rooms use `CurEdgeSpawnCell $525`. The RNG bytes ($18–$23) were
already re-randomized on every load. So a post-entry state can never vary its layout; only a state saved *before*
the room loads can.

## 2. New state set

- **Pre-entry states.** A state is saved on the transition into the room (`OptionExecutor.capture_pre_entry`).
  `GameSession.load` detects them (`room_entry.is_pre_entry`) and, before the room loads, randomizes
  `SpawnCycle` and the edge cell and clears the room's kill history (`fresh_rooms: true`, the default). Every
  episode gets a full, randomly placed set of monsters.
- **Route-generated (`triforce/entry_states.py build`).** Go-Explore walks the milestone-1 route step by step
  and saves the entry into each route room. Inventory and max hearts are whatever the route gives: 3 hearts
  through Dungeon 1, 4 after Aquamentus. Chains 0 and 1 reach `1_35s` (both failed the Aquamentus crossing after
  3 attempts); chain 2 reaches `2_2es`. 87 states, named `<level>_<room><entry dir>_<chain>`.
- **Converted library (`entry_states.py convert`).** 204 older states were converted (walk out the entry door
  and back in, set route hearts, clear kill history) and kept for other paths; 37 bad ones were deleted. The
  originals are in the old triforce repo. Configs only use the route-generated `<room>*_[0-9]` states.
- **Training mixes health.** `ram_randomization: {hp: [0.5, max], full_hp: 0.5}`: half of the resets start at
  full health (sword beams), the rest at 50–100% (no beams). boss-aquamentus uses `[0.75, max]`.
- **Bombs off** (`items: {bombs: off}`) for milestone 1 (`todo.md`, Items).
- Gaps: no state for 0x1e or 0x0e (Dodongo), so d2-goriya trains on `2_5es_2` only.

`entry_states.py audit`: all 88 configured states load in the right room with the route's hearts.

## 3. Old models on the new states

Promoted (round 1) and round 2c models, unchanged, evaluated on the new states:

| specialist | eval on old states | eval on new states |
|---|---|---|
| ow-start r1 | 98% | 62% |
| ow-octorok r1 / r2c | 94% / 96% | 98% / 91% |
| ow-zora r1 / r2c | 37% / 97% | 8% / 68% |
| d1-melee r1 | 79% | 70% |
| d1-wallmaster r1 / r2c | 18% / 87% | 0% / 62% |
| boss-aquamentus r1 | 100% | 3% |
| ow-leever-peahat r1 | 99% | 74% |
| ow-moblin r1 | 99% | 95% |
| d2-rope r1 | 98% | 44% |
| d2-goriya r1 | 100% | 95% |

The old numbers were inflated by empty rooms, fixed layouts and 16 hearts. Aquamentus collapses (100% → 3%).

## 4. Round 3 baseline (plain PPO, no curriculum)

`scripts/train_round.py --round r3-baseline --no-promote` (4.5 h).

| specialist | seed states | r1 eval | r2c eval | r3 steps | stop | r3 eval | worst room (rate) |
|---|---|---|---|---|---|---|---|
| ow-start | 7 | 98% | 98% | 221k | early | 99% | 0:0x77 (98%) |
| ow-octorok | 18 | 94% | 96% | 594k | early | 94% | 0:0x4c (65%) |
| ow-zora | 9 | 37% | 97% | 2.0M | full | **22%** | 0:0x2d, 0:0x48 (0%) |
| d1-melee | 31 | 79% | 44% | 2.0M | full | 52% | 1:0x72 (15%) |
| d1-wallmaster | 6 | 18% | 87% | 2.0M | full | **0%** | 1:0x44, 1:0x45 (0%) |
| boss-aquamentus | 3 | 100% | 100% | 2.0M | full | **0%** | 1:0x35 (0%) |
| ow-leever-peahat | 4 | 99% | 95% | 582k | early | 98% | 0:0x29 (90%) |
| ow-moblin | 2 | 99% | 96% | 377k | early | 98% | 0:0x3d (95%) |
| d2-rope | 7 | 98% | 98% | 2.0M | full | 83% | 2:0x3e, 2:0x4e (50%) |
| d2-goriya | 1 | 100% | 100% | 1.94M | early | 100% | 2:0x5e (100%) |

- **The overworld transit specialists hold** (ow-start, ow-octorok, ow-leever-peahat, ow-moblin: 94–99%, early
  stop). `0_4ce` is still ow-octorok's weak state (65%, deaths).
- **ow-zora is back to the round 1 picture:** 0x38 from the south 0–5%, 0x48 0% (death), 0x2d 0% (exit-wrong /
  death). Only 0x28 and the 0x38 W entry pass.
- **d1-wallmaster 0%:** 0x44 dies to the Goriyas or runs out the clock; 0x45 always runs to `max-steps`.
- **boss-aquamentus 0%** (58/60 `max-steps`). Trace of the r3 model in `1_35s_0`: Link walks N then E toward
  Aquamentus, never lands a hit, and spends 2,000 steps oscillating for PBRS movement reward (+30 / −29 per
  episode) while `penalty-room-stalling` adds up to −22 to −24. 17–27 attack actions per 2,000 steps, all misses.
- **d1-melee 52%** (round 1: 79% on the old states). Its numbers are per *start* room of a multi-room episode:
  0x72 15%, 0x74 25%, 0x73 31%, 0x63 47%, mostly deaths to Stalfos/Keese. Round 4 changes this to one room per
  episode.
- **d2-rope 83%:** 0x3e and 0x4e 50%, deaths. These rooms had no savestates before round 3.

Chain (milestone1, 20 runs, r3 models): 0/20, every run ends in ow-zora (0x38 death ×12, 0x38 stuck ×6, 0x2d
death ×2).

## 5. Go-Explore + curriculum on the new states (r3c)

Go-Explore was re-run on the new states (`trajectories/<specialist>` cleared first), then
`train_round.py --round r3c-curriculum --train-args "--curriculum trajectories/{name}"`.

### Exploration

| specialist | states solved | explore env steps | notes |
|---|---|---|---|
| ow-zora | 9/9 | 64k | |
| d1-wallmaster | 5/6 (1_44w_2 not) | 321k | |
| boss-aquamentus (round 2 cell) | 0/3 | 318k | ~290 cells per state |
| boss-aquamentus (enemy health in cell) | 3/3 | 47k | 900–1,100 cells per state |

**Explore fix: enemy health in the cell.** The round 2 cell (room, tile // 2, hearts, living enemies, keys,
boomerang) only changes when an enemy dies. A multi-hit boss leaves no trail of cells toward the kill, so random
25-action extensions had to land every hit in one go. The cell now includes the total health of the living
enemies, and each damaging hit counts as a progress event (`explore.enemy_health`). ow-zora and d1-wallmaster
were explored before this change.

### Training

| specialist | r3 baseline | r3c steps | stop | r3c eval | per room |
|---|---|---|---|---|---|
| ow-zora | 22% | 348k | early | **98%** | 0x2d 95%, 0x48 97%, 0x28 100%, 0x38 100% |
| d1-wallmaster | 0% | 2.0M | full | **77%** | 0x44 65%, 0x45 88% |
| boss-aquamentus | 0% | 2.0M | full | **0%** | 0x35 0% (deaths and `max-steps`) |

- Every ow-zora and d1-wallmaster trajectory completed the curriculum (start point back to the room entry).
- d1-wallmaster's remaining failures are in 0x44 (deaths to the Goriyas, 1_44w_2 worst at 50%: it had no
  trajectory) and Wallmaster grabs in 0x45.
- Compared with the old models on the new states (section 3): ow-zora r3c 98% vs r2c 68%, d1-wallmaster r3c 77%
  vs r2c 62%. Round 4 promotes the r3c models.

### boss-aquamentus: the curriculum stalls

- **Curriculum progress:** the first 2–3 stages (after the kill: walk to the heart container) passed with 87%
  success. Every trajectory then stuck at 58–60% of its length (points 77/110, 45/77, 44/76), where Aquamentus
  still has 5–6 hits left. Curriculum success there was 0% for the remaining ~450 iterations.
- **Few episodes:** 1–8 finished episodes per 4,096-step iteration (episodes run 500–2,000 steps), so each
  curriculum stage gets few samples.
- **What the explored path does** (replayed step by step through `ZeldaEnv`): Link goes N above Aquamentus and
  hits it with `SWORD S` from tile (20, 5) six times, +0.25 `reward-hit` each; Aquamentus dies with no extra
  kill reward; the heart container ends the episode (+17.5 terminal at 3/4 hearts). It takes 1–1.5 hearts of
  damage on the way.
- **What the r3c model does** from trajectory step 48 (5 hits left): runs W away from Aquamentus, waits in the
  NW corner, attacks 3–11 times per episode (all misses), and dies to fireballs after 65–1,355 steps. PBRS
  movement nets ≈ 0 (+22 / −22): it oscillates toward and away from the fight targets.
- **Why this regressed from round 1 (100%):** the round 1 states had 16 hearts, so walking into contact and
  fireball damage was free. With the route's 3 hearts, approaching costs 0.25–0.5 per hit taken, more than the
  0.25 per hit dealt, and the policy learns to avoid the boss. Go-Explore and the curriculum fix exploration;
  this looks like a reward balance problem (hit reward vs damage taken near a boss). Retried after round 4
  with the other failing specialists.

## Conclusion

The new state set is much harder and more honest: fixed layouts, empty rooms and 16 hearts had inflated several
round 1/2 results, Aquamentus most of all. Plain PPO still solves the transit specialists, d2-goriya and most of
d2-rope. ow-zora and d1-wallmaster fail with plain PPO and are solved again by Go-Explore + backward curriculum
(98%, 77%). Aquamentus stays at 0% with both: exploration now finds the kill, but the policy won't engage the
boss at 3 hearts. Next blockers: d1-melee (round 4: one-room episodes, 0x43 fix), Aquamentus, and d2-rope's
0x3e/0x4e.

Raw data: `models/r3-baseline-summary.json`, `models/<specialist>/{r3-baseline,r3c-curriculum}/eval.json`,
`models/reeval-new-states/*.json`, event logs in `logs/261102_002_r3-baseline`, `logs/261102_003_r3c-curriculum`
(ow-zora, d1-wallmaster) and `logs/261102_004_r3c-curriculum` (boss-aquamentus; its run overwrote
`models/r3c-curriculum-summary.json`, so the ow-zora / d1-wallmaster table above is from the 003 log).
