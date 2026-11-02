# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them.

## Next up

- **Round 3: re-baseline every specialist on the new start states.** Pre-entry states, random spawns, fresh rooms,
  full/partial hp, route-true inventory and max hearts, bombs off (below).
- **Round 4: d1-melee, cheap route fixes then one-room episodes.**
  - Make the curriculum a per-specialist config field: on for ow-zora and d1-wallmaster, off elsewhere.
  - Promote ow-zora and d1-wallmaster (round 2 or round 3, whichever evaluates better).
  - Fix 0x43 → 0x33 (see Runner / objectives below).
  - One-room episodes: every episode ends at its first room exit. Today a d1-melee episode runs until Link leaves
    the 9-room family. Traced "0x73" episodes are 4–9 rooms, with the deaths in 0x74/0x63 (Stalfos), and 90% per
    room over 9 rooms is ~39%. Compare per room against round 3's d1-melee.
  - **Last resort only:** split Dungeon 1 into enemy-based specialists (Stalfos 0x53/0x63/0x74, Keese
    0x52/0x72, Gel 0x42/0x43). Do this only if one-room episodes and the other fixes fail.

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
- **Training-time `balanced_success_rate` lags** when a specialist finishes few room-entry episodes per iteration
  (d1-melee: 26 states, 5–17 episodes per iteration). Consider a time-based window.
- **Single seed per configuration.** d1-wallmaster r2b (96%) vs r2c (87%) shows the run-to-run spread; repeat key
  comparisons over 2–3 seeds before acting on a few-point difference.

## Start states (round 2 diagnosis)

- **Seed states that end in one step.** `0_38w`, `0_2ds` (ow-zora) and `1_45n` (d1-wallmaster) start at an edge
  whose exit is also the objective's exit, so they succeed in 1–2 steps and inflate eval success. They are
  entries from the opposite direction of the route.
- **Wrong objective for post-D1 overworld visits.** Every `0_28*` and `0_38*` state has 0 triforce pieces, so the
  objective sends Link toward D1: 0x28 → S, 0x38 → W. On the route, 0x28 (S → E) and the second 0x38 visit
  (W → N) happen after D1, so those room/exit pairs are never trained. Needs savestates (or a RAM override) with
  triforce 1.
- **d1-wallmaster states with 16 hearts.** `1_44e`, `1_44w` and `1_45w` have max health 16, not the route's 3.
- **No savestates for D2 rooms 0x4e/0x3e/0x2e/0x1e** (from round 1).
- **`0_4ce` (ow-octorok, the 0x4d → 0x4c route entry) learns slowly.** 90% in round 1 with 2M steps, 10–35%
  with early-stopped runs. It is also the start state with 0.5 hearts in the savestate.

## Runner / objectives

- **d1-melee leaves 0x43 north into untagged 0x33 and the objective calls it correct** (8 of 20 chain runs in
  r2b ended `error`). The route wants 0x43 → E. Either restrict `next_rooms` to the route or tag 0x33.

## Rewards

- **Damage still resets the room-stalling counter** (v1 behaviour, `critics.critique_movement`). In rooms with
  frequent ranged damage (Zora fireballs) this delays the stalling penalty. Not changed in round 2.
