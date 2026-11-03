# Round 5: toward the Dungeon 2 boss (overnight, Nov 2–3)

Goal: specialists that carry the milestone-1 chain to the Dodongo room (2, 0x0e); the milestone ends on entering
it. Starting point: round 4's r4c chain follows the whole Dungeon 1 route but dies of health drain; Aquamentus is
at 0%; there was no route state for 0x1e.

Setup as in rounds 3–4 unless a row says otherwise (2M-step budget, balanced 95% early stop, 20 sampled episodes
per seed state for eval, 20-run chain). "Reach" columns count chain runs (of 20) that entered a room.

## Promotions before the experiments

On the round 3 states: ow-start, ow-leever-peahat, ow-moblin, d2-rope, d2-goriya → r3-baseline (better than the
round 1 models, `round3.md` section 3). ow-octorok stays round 1. ow-zora, d1-wallmaster → r3c. d1-melee → r4c.

## Where the chain loses health (r4c)

Mean hearts lost per room on the route: 0x48 0.23, 0x38 0.35, 0x37 0.29 (Link enters Dungeon 1 with 1–3 of 3
hearts), then 0x72 0.29, 0x74 0.68, 0x63 0.62, 0x53 0.50, 0x44 1.5. Training started every room at 1.5–3 hearts,
so no specialist practised the 0.5–1.5 hearts the chain arrives with.

## Experiments

### r5: d1-melee hp floor, Aquamentus boss hit reward

- **d1-melee `hp: [0.15, max]`** (was `[0.5, max]`): eval 95% (r4c 94%) with less damage per room (−0.16 vs
  −0.31 hearts) and shorter episodes (34 vs 47 steps); full 2M steps (no early stop). 0x53 is the weakest room
  (77%). Promoted.
- **boss-aquamentus: `reward-boss-hit`** (0.75 per hit on a boss, no combat decay; was 0.25 `reward-hit`) +
  curriculum: still 0%. The curriculum got one stage further (points 66/110, 37/77, 36/76; r3c 77, 45, 44) and boss
  hits show up late in training, but 1–8 episodes finish per 4,096-step iteration: Link flees and dodges for up to
  2,000 steps, so the stalled stage gets almost no samples.

| chain | 0x73 | 0x63 | 0x52 | 0x44 | 0x45 | 0x35 | where runs end |
|---|---|---|---|---|---|---|---|
| r4c | 19 | 13 | 6 | 4 | 1 | 1 | 0x74 ×6, 0x53 ×5, 0x44 ×3, 0x63 ×2, 0x43 ×2 |
| r5 | 19 | 18 | 9 | 6 | 1 | 1 | 0x53 ×6, 0x44 ×5, 0x63 ×3, 0x42 ×2, 0x74 ×1 |

### D2 route state for 0x1e

`entry_states build --chains 2 --start-step 41 --source 2_2es_2`: 0x2e → 0x1e found in 7.7k env steps
(`2_1es_2`); 0x1e → 0x0e failed 3 times (175k env steps). 0x1e's N shutter opens only when all 5 Red Goriyas are
dead. The milestone ends on entering 0x0e, so no Dodongo state is needed. `triforce.explore --specialist d2-goriya`
then solved 0x1e in 77k steps (158-step path), so d2-goriya gets the curriculum (r6).

d1-wallmaster `1_44w_2` is still unexplored with the enemy-health cell (8,000 iterations, 166k env steps, 6,340
cells).

### r6: d2-goriya with the curriculum (0x5e + 0x1e)

88% (0x1e 85%, 0x5e 90%), full 2M steps at 8 workers. The r3 model on the same states: 50% (0x1e 0%: never
trained there). Promoted.

### r7: d1-wallmaster hp floor; Aquamentus with `max_steps: 400`

- **boss-aquamentus: 0% → 93%** (56/60; 4 deaths; +0.12 hearts per episode because the heart container ends it).
  Same boss hit reward and curriculum as r5; the only change is `max_steps: 400` (explored kills take 76–110
  steps). Episodes that flee now end at 400 steps, so 4–17 episodes finish per iteration instead of 1–8, and
  every trajectory finished the curriculum. Entry success stayed 0 until iteration ~380, then rose to 74% by 436.
  Promoted. The lesson generalizes: a curriculum stage needs episodes, and `max_steps` sets the episode rate.
- **d1-wallmaster `hp: [0.15, max]`**: 78% (0x44 60%, 0x45 95%) vs the r3c model at the same low-hp distribution
  68% (0x44 50%, 0x45 87%), and less damage (−0.15 vs −0.28 hearts). Promoted. 0x44 (3 Goriyas + boomerang)
  remains the weak room.

### Bug: walking back down the Dungeon 1 stairs counted as success

A chain started after Dungeon 1 (`eval-chain --from 0_37c_2`, new option) went back into Dungeon 1 in all 20
runs: ow-octorok lists `enter-room:1:0x73` as success, and `RoomExit` returned enter-room endings before checking
the objective. In r3's eval, 13 of 20 `0_37c_2` episodes "succeeded" that way. Now an enter-room ending needs the
objective to want the room (test `test_enter_room_success_needs_the_objective`).

### r8: ow-octorok one-room episodes + hp floor (with the fix)

95% (0x4c 75%, the rest ≥ 95%), early stop at 1.14M steps (8 workers); `0_37c_2` 20/20 exit E. Promoted
(replaces round 1).

Post-D1 chain from `0_37c_2` (20 runs): reaches 0x28 20/20, 0x2d 18/20, D2 (0x7d) 2/20. Link leaves Dungeon 1
with 4 hearts and loses them on the way: 0x38 0.75, 0x2a/0x2b 0.47/0.76, 0x2d 0.47, 0x3d 0.83, 0x4c 0.83; deaths
0x4c ×8, 0x2d ×3, ow-moblin ×3, ow-leever-peahat ×2. → r9: hp floor + one-room for the overworld specialists and
d2-rope.

### Full chain with every promoted model (after r8)

| chain | 0x73 | 0x63 | 0x52 | 0x44 | 0x45 | 0x35 | 0x36 | 0x28 | 0x2d | where runs end |
|---|---|---|---|---|---|---|---|---|---|---|
| r5 | 19 | 18 | 9 | 6 | 1 | 1 | 0 | 0 | 0 | 0x53 ×6, 0x44 ×5, 0x63 ×3 |
| after r8 | 19 | 19 | 14 | 10 | 3 | 3 | 1 | 1 | 1 | 0x44 ×7, 0x53 ×4, 0x35 stuck ×2, 0x43 ×2 |
| after r10 | 19 | 19 | 14 | 10 | 2 | 2 | 2 | 2 | 2 | 0x44 ×8, 0x53 ×4, 0x43 ×2 |

The first runs kill Aquamentus, take the triforce and leave Dungeon 1. In the chain, boss-aquamentus's
`max_steps: 400` is also its room budget (`stuck` ×2 after r8). Link reaches 0x44 with ~1.15 of 3 hearts and loses
0.85–0.95 there; 0x53 (Stalfos) is the other death room.

### r9a: ow-zora with hp floor + one-room

91% (0x38 89%, 0x48 90%) vs the r3c model under the same config 96% (0x38 92%, 0x48 97%, less damage: −0.34 vs
−0.46 hearts). Not promoted: retraining with the curriculum at the lower hp didn't beat the r3c policy.

### r10: d1-wallmaster one-room episodes (continued from r7)

One-room eval: 88% (0x44 77%, 0x45 98%) vs r7 78% (0x44 62%, 0x45 95%). Promoted. The chain didn't move: 0x44
still kills 8 of the 10 runs that reach it, at ~1 heart (the eval draws partial hp uniformly over 0.5–3 hearts).
→ r11: partial resets drawn from 0.5–1.5 hearts (`hp: [0.15, 0.5]`, half the resets still full).

### r9a/r9b: ow-leever-peahat, ow-moblin, d2-rope with hp floor + one-room

Each compared with its promoted round 3 model evaluated under the same (new) config:

| specialist | r3 model | r9 model | promoted |
|---|---|---|---|
| ow-leever-peahat | 96%, −0.19 hearts, 16 steps | 99%, −0.39 hearts, 31 steps | r3 (half the damage; damage is what kills chains) |
| ow-moblin | 92%, −0.38 hearts | 98%, −0.33 hearts | r9a |
| d2-rope | 81% (0x3e 55%, 0x4e 70%), −0.64 hearts | 89% (0x3e 70%, 0x4e 85%), −0.52 hearts | r9b |

### Runner room budget

The runner's per-room budget is now `max(max_steps, 2000)`: boss-aquamentus's training cap of 400 had ended 2 of
its 3 chain fights as `stuck`.

### Dungeon 2 chain (`--from 2_7ds_2`, 4 hearts at the D2 entrance)

| models | reach 0x0e (Dodongo room) | where runs end |
|---|---|---|
| d2-rope r3, d2-goriya r6 | 1/20 | 0x3e ×9, 0x4e ×7, 0x5e ×2, 0x1e ×1 |
| d2-rope r9b, d2-goriya r6 | **2/20** | 0x4e ×6, 0x5e ×6, 0x3e ×3, 0x6e ×2, 0x2e ×1 |

Same health-drain pattern: 0x6d −0.90, 0x6e −0.85, 0x5e (d2-goriya) −1.36, 0x4e −0.62, 0x3e −1.42 hearts.

### r11: d1-wallmaster partial hp 0.5–1.5 hearts (continued from r10) — regressed

At the r11 eval distribution (`hp: [0.15, 0.5]`, half full): r11 32% (0x44 28%, 0x45 35%, mostly `wallmastered`)
vs r10 72% (0x44 45%, 0x45 100%). 2M more steps at the narrower low-hp range broke 0x45 entirely. Not promoted;
the config is back to `[0.15, max]`. r10 at chain-like hp is 45% in 0x44, which matches the chain's 0x44 deaths.

### State at 5 am (all promoted models)

Promoted: ow-start r3, ow-octorok r8, ow-zora r3c, d1-melee r5, d1-wallmaster r10, boss-aquamentus r7,
ow-leever-peahat r3, ow-moblin r9a, d2-rope r9b, d2-goriya r6.

| chain (20 runs) | reaches | where runs end |
|---|---|---|
| full, from `start` | 0x63 19, 0x52 14, 0x44 10, Aquamentus 2, triforce 2, 0x2d 2 | 0x44 ×8, 0x53 ×4, 0x43 ×2, 0x4c ×2 |
| after D1, from `0_37c_2` (4 hearts) | 0x28 20, 0x2d 18, **D2 5** | 0x4c ×8, 0x2d ×3, 0x6d ×3, 0x3d ×2 |
| D2, from `2_7ds_2` (4 hearts) | **Dodongo room 2** | 0x4e ×6, 0x5e ×6, 0x3e ×3 |

Every segment works in some runs; no run does all of them yet. Rough per-segment rates: start → leave D1 10%,
post-D1 → D2 25%, D2 → Dodongo 10%. The deaths are almost all health drain into a hard room (0x44, 0x53, 0x4c,
0x4e/0x5e).

### r12: d2-goriya hp floor (continued from r6)

At the new hp distribution: r12 82% (0x5e 75%, 0x1e 90%, −0.85 hearts) vs r6 80% (0x5e 75%, 0x1e 85%, −1.05
hearts). Promoted on less damage (40 episodes; the success difference is noise). D2 chain from `2_7ds_2` with r12:
still 2/20 reach the Dodongo room, but no run dies in a d2-goriya room any more (0x5e ×6 before); the deaths moved
to d2-rope's 0x4e ×10, 0x3e ×6, 0x6e ×2.
