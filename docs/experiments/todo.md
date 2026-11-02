# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them.

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

## Rewards

- **Damage still resets the room-stalling counter** (v1 behaviour, `critics.critique_movement`). In rooms with
  frequent ranged damage (Zora fireballs) this delays the stalling penalty. Not changed in round 2.
