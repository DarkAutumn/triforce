# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them.

## Next up

Goal until further notice: a set of specialists that carries the milestone-1 chain to the Dungeon 2 boss (Dodongo).

- **Round 6 done** (`round6.md`). Best chains (40 runs): full chain → Dodongo room 1/40 (milestone1, r23) and
  2–3/40 (milestone1-fairy); **fairy post-D1 → Dodongo room 8/40**; D2 entrance → Dodongo room 9/40.
- **Promoted now:** ow-zora / ow-octorok / ow-start r24-fairy, ow-moblin r26c-moblin, ow-leever-peahat /
  boss-aquamentus r23-more, stalfos-keese-gel r25-skg-plain, goriya r21-goriya-chainhp, d2-rope r28-d2,
  wallmaster r10 copy. Overworld families train on both routes' rooms; D2 rooms have 3 start states each.
- **Next: the D1 Goriya room (0x44).** It kills 15 of the 28 full-chain runs that reach it (Link arrives with
  ~1.1 hearts), and only 8 of 40 runs leave D1 alive. goriya r28 (2M more steps on 3× D2 states) did not beat r21
  in the chain. Ideas below ("Fight Goriyas without trading hits"); also 0x53 (5 deaths).
- ~~Chain-matched hp~~ tested on goriya (within noise) and stalfos-keese-gel (worse); off everywhere. The option
  (`ram_randomization.chain_hp`) stays for a later retry.
- ~~Pass-through damage penalty~~ tested twice in round 6 (overworld, d2-rope): no chain gain; dropped (code
  removed). A damage-avoidance idea for later would need a different form.
- **Promote by damage too.** A model that wins 3% more but takes twice the damage is worse in a chain
  (ow-leever-peahat r9a vs r3). Evaluate at chain-like hp and rank by success and `mean_health_change`.
- ~~Refill hearts on the way~~ **Done (Lee's design): second route `milestone1-fairy`** after D1 (south road +
  fairy pond 0x39; `docs/route.md`, `docs/decisions.md`). Train the overworld families on both routes and evaluate
  both chains. Not done: a fairy stop before D1 (none on the way), D2 has none.
- ~~Heart objective~~ (`collect_hearts`) tested on stalfos-keese-gel and d2-rope (round 6 §10): more pickups on
  d2-rope but no chain gain; dropped (code removed). Hearts left behind are still measured by the heart tracker.
- **The last D2 Goriya room (0x1e) and the first (0x5e)** now end most D2 runs that get past the rope rooms.
- **Fight Goriyas without trading hits.** At 1 heart the goriya model turns passive (33% WAIT) and dies to contact
  (round 6 trace). Ideas: a demo/curriculum from damage-free Go-Explore paths (the explored paths lose 1.5–2.5
  hearts; select by health with more patience), shield blocking of boomerangs, a contact-damage penalty that grows
  as hp falls.
- **Bow route** (later): 0x43 N → 0x33 (key) → 0x23 (key) → 0x22 (push block, stairs to the bow cellar 0x7F).
  Needs a route/objective for the bow and a specialist for 0x22 and the cellar.
- ~~Skip the optional boomerang in 0x44~~ **Rejected by Lee:** keep the boomerang.
- ~~Split Dungeon 1 into one model per enemy type~~ **Rejected by Lee:** group models by the enemies in the room,
  but don't boil it down to one model per enemy type or per room (beyond what Lee asks for, e.g. `wallmaster`).

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
