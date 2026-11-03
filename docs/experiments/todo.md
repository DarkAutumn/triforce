# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them.

## Next up

Goal until further notice: a set of specialists that carries the milestone-1 chain to the Dungeon 2 boss (Dodongo).

- **Rounds 3–5 done** (`round3.md`, `round4.md`, `round5.md`). Every specialist works in isolation now (Aquamentus
  93%, d2-goriya with 0x1e 88%); the full chain reaches Aquamentus in 2–3 of 20 runs and has left Dungeon 1 twice;
  from the D2 entrance 2 of 20 runs reach the Dodongo room. What kills chains is **health drain**: Link loses
  0.2–1.4 hearts per room and never heals.
- **Chain-matched hp.** Sample each specialist's reset hp from the hp the chain actually arrives with (per room,
  from the chain jsonl `health_in`), instead of a uniform range. r11 tries a cruder version for d1-wallmaster.
- **Promote by damage too.** A model that wins 3% more but takes twice the damage is worse in a chain
  (ow-leever-peahat r9a vs r3). Evaluate at chain-like hp and rank by success and `mean_health_change`.
- **Damage-avoidance reward.** Terminal success already scales with hp at exit; per-hit penalties are 0.25 per
  half heart. Try a larger lost-health penalty, or a terminal bonus on hp kept relative to entry (the new
  `health_change`), for the transit rooms where fighting isn't required.
- **Skip the optional boomerang in 0x44.** `docs/route.md`: the 0x44 boomerang is optional and E is open. The chain
  arrives with ~1 heart and dies to the 3 Red Goriyas in 8 of 10 runs. A route variant that walks E needs the later
  specialists (0x45, Aquamentus, D2) to cope without the boomerang (train some resets with it removed).
- **d2-goriya one-room + hp floor**: 0x5e costs 1.36 hearts in the D2 chain.
- **d1-wallmaster `1_44w_2` has no trajectory** (even with the enemy-health cell, 166k explore steps).
- **Last resort only:** split Dungeon 1 into enemy-based specialists (Stalfos 0x53/0x63/0x74, Keese 0x52/0x72,
  Gel 0x42/0x43). One-room episodes made per-room d1-melee 94–95%, so this is not needed now.

## Items

- **Bombs are off for milestone 1 (`items: {bombs: off}`, round 3 on).** Bomb use paid more than it cost: −0.25 to
  place, +0.5 per hit plus +0.25 hit reward. d2-goriya threw bombs on 22% of its actions in round 1, and no
  canonical milestone-1 room needs bombs (`docs/route.md`).
- **Bomb skill: with the Dodongo work, after the route reaches the Dungeon 2 boss room.**
  - Grant bombs at reset (`ram_randomization: {bombs: [n, 8]}`).
  - Add a bomb-into-Dodongo reward and a scarcity cost for bombs spent on other enemies.
  - Then decide which specialists get bombs back (bomb walls in later dungeons).
- **Boomerang stays on.** Route states carry it from 0x44 onward: d1-wallmaster, Aquamentus, the triforce room and
  everything after Dungeon 1. It stuns Wallmasters, Goriyas and Keese. Watch `kind/BOOMERANG` in training metrics.

## Training methods (from "Training with Savestates and Branching", Nov 2, 2026)

- **Idea 2: branched rollouts for advantage estimation (`branch:K`, VinePPO/GRPO).** Deferred in round 2. Round 2
  tests Idea 1 first, because the failing rooms are exploration problems (the policy never reaches the exit), not
  credit-assignment problems. First comparison when picked up: boss-aquamentus, GAE vs branch:K at equal wall clock.
- **Idea 3: search distilled into the network (expert iteration).** Deferred. Reserve for a specialist where Ideas 1
  and 2 both fall short (Dodongo is the expected candidate).
- **Evaluation grid (entry × health × items × RNG seed).** Deferred. `triforce.evaluate` already breaks results
  down per start state; adding health/item axes is a reset-option change plus a report.
- **Go-Explore cell definition for puzzle rooms (block push, bomb walls).** Round 2's cell is (room, tile // 2,
  hearts, living enemies, keys, boomerang). Puzzle progress (block pushed, wall bombed, door opened) isn't in it.
  Add it when a puzzle specialist is explored.
- **Curriculum threshold: fixed vs adaptive.** Round 2 uses a fixed 0.6 over windows of 10 episodes.
- **Curriculum progress doesn't survive `--load` resume.** `<run>/curriculum.json` restarts with every training
  invocation.
- **d1-melee regresses under the curriculum (79% → 44%, round 2).** The control without curriculum gets 78%.
  Candidates to test, one at a time:
  - keep a 10–20% curriculum share after every trajectory completes, starting anywhere on the whole trajectory,
    instead of switching off abruptly;
  - run the curriculum only for specialists that plain PPO can't solve (make it a per-specialist config field);
  - keep periodic checkpoints and pick by a held-out eval, so a late collapse can be seen and avoided.
- **Explore misses progress that isn't in the cell.** It doesn't capture the sword pickup (`ow-start` `start` and
  `0_77c` unsolved) or long multi-room paths (d1-melee 13/26 solved, 1.17M explore steps). Add sword, triforce
  and the items in `progress_events` to the cell. Consider exploring room by room instead of per family.
  Round 3 added enemy health to the cell (Aquamentus 0/3 → 3/3); ow-zora and d1-wallmaster were explored before
  that, and d1-wallmaster `1_44w_2` is still unexplored: re-run it with the new cell.
- **Training-time `balanced_success_rate` lags** when a specialist finishes few room-entry episodes per iteration
  (d1-melee: 26 states, 5–17 episodes per iteration). Consider a time-based window.
- **Long episodes starve the curriculum** (round 3; fixed for boss-aquamentus in round 5 with `max_steps: 400`,
  0% → 93%). `max_steps` sets the episode rate; set it from the explored path length for any curriculum specialist
  whose failures are slow (fleeing, waiting) rather than quick deaths.
- **Single seed per configuration.** d1-wallmaster r2b (96%) vs r2c (87%) shows the run-to-run spread; repeat key
  comparisons over 2–3 seeds before acting on a few-point difference.

## Driver

- **Re-running a subset with the same `--round` overwrites `models/<round>-summary.json`** (round 3: the
  boss-aquamentus r3c run replaced the ow-zora / d1-wallmaster summary). Merge into the existing summary, or
  refuse to reuse a round ID.
- **Mapper runs log JSONL and get their own GUI view (Lee, Nov 3; later, not urgent).** `entry_states build/path`
  and `triforce.explore` print text only. Write event logs (`logs/<yymmdd>_<NNN>_<what>/`, like training):
  per room/state attempts, iterations, cells, env steps, best progress, success/failure. Then teach
  `training-gui.py` to recognize a mapper log folder and show it sensibly (likely a different view that opens for
  that kind of folder: route steps as rows, per-attempt progress).

## Start states

- Round 3 replaced the state set (`round3.md`): pre-entry, route-generated, route-true hearts and inventory. That
  resolved the round 2 items (1-step states, triforce 0 on post-D1 visits, 16-heart states, missing D2 states).
- **No state for 0x0e** (the Dodongo room; not needed for milestone 1). `2_1es_2` (0x1e) was added in round 5.
- **Only chain 2 crosses Aquamentus**, so post-D1 rooms have one state each (`*_2`). More chains past 0x35 would
  give them spawn/health variety from different paths.
- **`0_4ce` (ow-octorok, the 0x4d → 0x4c route entry) is still the weak state:** 65% in round 3 (deaths).

## Runner / objectives

- Round 4 fixed 0x43 → 0x33 (route-only exits, pathfinding through tagged rooms only, route order on revisited
  rooms; `round4.md`).
- **The chain records no keys or route step per room.** Adding them to `RoomRecord` would have found the 0x72/0x33
  key problem without a throwaway script.

## Rewards

- **Damage still resets the room-stalling counter** (v1 behaviour, `critics.critique_movement`). In rooms with
  frequent ranged damage (Zora fireballs) this delays the stalling penalty. Not changed in round 2.
