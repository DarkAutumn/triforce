# Round 12: beat Dodongo (Dungeon 2 boss, room 2:0x0e)

Lee, Nov 8: train a model that kills Dodongo the good way. Set a bomb in his path but *just next to him*, so he
doesn't eat it and the smoke stuns him; then hit him with the sword while he is stunned. The model gets bombs as it
needs them (rules below). The curriculum over his movement patterns matters most.

## Summary

- **Dodongo is beaten: r45-dodongo kills him in 300/300 evaluation fights** (100 per start state), losing 0.13
  hearts per fight (90% of fights without a hit), in 32 actions. Promoted.
- 86% of kills are the bomb-stun-and-sword kill Lee asked for (67% with no swallow at all); 14% are two-swallow
  kills. 52% of fights take one bomb (the perfect kill); 2.5 bombs per fight on average (§4).
- 2M more steps (r46-more) or a doubled swallow penalty (r46-eat1) didn't move these (§5).
- Mechanics verified by probes (§1). Env support, rewards and Lee's bomb rules (`f2141cf`, §2). Curriculum:
  Go-Explore over 3 start states × 10 explore seeds, swallow kills removed: 25 stun-and-sword tracks (§3).
- Chain check: milestone1 ends on entering 0x0e; the d2-first route (next) plays through the kill and the
  triforce, so its chains measure Dodongo in a run.

## 1. Mechanics (probes, `target/dodongo/`; disassembly `asm:Dodongo_Update`)

- Dodongo2 (`$32`) is in object slot 1: X `$71`, Y `$85`, dir `$99`; ObjState `$AD`: 0 walk, 1 bloated (ate a
  bomb), 2 stunned; stun timer `$3E`; metastate `$406` ≥ `$10` = dying. In `2_0es_0` he spawns at (160, 157)
  facing E, 28 frames after load.
- Stun: the bomb in slot `$10` must be exploding (dust, a 36-frame window after the 48-frame fuse) within
  [−12, +11] px of his hotspot while he walks. He *eats* a lit bomb sitting in his mouth window. A second bomb lands
  in slot `$11`, which he ignores.
- Geometry probe (Dodongo at (156, 141) facing W, Link on an 8-px lattice ±48 px, 4 bomb directions): 45 stuns,
  20 eats, 611 nothing. Stun lasts about 309 frames. A sword stab from any side kills a stunned Dodongo; beams
  never hurt him; a stunned Dodongo can't hurt Link.
- After the kill the heart container (`0x1a`) appears 7 frames later at (128, 144); taking it raises max hearts,
  so success is `boss-dead-and-heart`.

## 2. Env, rewards, bombs (`f2141cf`)

- `specialists/boss-dodongo.yaml` is trainable: start states `2_0e*_[0-9]` (3), success `boss-dead-and-heart`,
  pbrs + item targets, objective/target observation, hp `[0.15, max]` with `full_hp: 0.5`, bombs on, curriculum,
  `one_room`, `max_steps: 600`.
- Critic `critique_dodongo`: `reward-boss-stunned` (+0.75) when his state goes walk → stunned,
  `penalty-boss-ate-bomb` (−0.5) on walk → bloated. The kill pays `reward-boss-hit`. Every bomb still costs
  `penalty-bomb-used` (−0.25): Lee wants bombs per kill minimized (1 = a perfect kill).
- Bombs (Lee's rule, replaces a short-lived "infinite bombs" refill): counts are natural; a `requires_bombs` room
  (game.yaml; only 2:0x0e) tops 0 up to one *cheated* bomb, and leaving with it unused takes it back. Training,
  eval visits and chains report `bombs/used` and `bombs/cheated`; the watch player's footer shows cheated bombs.

## 3. Curriculum (`a09d9bd`)

- Explore seeds also draw the reset RNG and spawn, so each seed is another Dodongo pattern. `explore --variant`
  writes `<state>~s<seed>` beside the seed state's own track and the curriculum keeps each directory as a track.
- Seeds 0–9 × 3 states: 30/30 solved (21–116 steps). Kill kind read from the saved states (Dodongo ObjState and
  BombHits `$437+1`): 25 stun-and-sword, 5 with bombs swallowed (moved to `target/dodongo/swallow-tracks/`).
- Explored kills spend 1–13 bombs (mostly random placement); the curriculum starts near the kill and walks back.

## 4. r45-dodongo (training)

`train_round.py --round r45-dodongo --specialists boss-dodongo --steps 3000000 --parallel 16 --no-promote
--chain-runs 0 --train-args "--stop-success 0.99"`, 25 tracks. Early (iteration 27, 110k steps): success
0.33–0.53, health change about −1.5, ~22 bombs used per episode (18 of them cheated).

Finished at 3M steps (733 iterations, 26 min at 16 envs); the curriculum had completed every track by about
iteration 110 (episodes start at the room entry from then on). Eval (100 per state, sampled actions, under the
current code: stunned Dodongo not `hurts_on_touch` (`70a7e26`), and the fixed counters (`106fa25`)):

| | 2_0es_0 | 2_0es_1 | 2_0es_2 | all |
|---|---|---|---|---|
| success | 100% | 100% | 100% | 100% |
| bombs used / cheated per fight | 2.32 / 0.07 | 2.50 / 0.49 | 2.67 / 0.67 | 2.50 / 0.41 |
| stun kill rate (≥ 1 stun, no swallow) | 66% | 64% | 72% | 67% |

Kill kinds over 300 (stuns, swallows): (1, 0) 202, (1, 1) 55 (a swallow, then a stun and the sword), (0, 2) 32
and (1, 2) 9 (two-swallow kills), 2 odd ones (a stun that ended and restarted inside one step is the likely miss).
Bombs per fight: 1 in 157, 2 in 56, 3 in 24, 4+ in 63. Hearts lost 0.127 per fight, 89.7% no-hit, 32.3 steps
(about 300 frames). Deaths 0.

The first counters (0→2 stun, 0→1 swallow between step ends) showed 51 kills with one swallow and no stun, which
the ASM rules out (a stun only starts from state 0; the sword only kills in state 2): one step can hold the swallow,
the bloat and the stun. Counting entries into state 2 and BombHits increments fixed it (`106fa25`).

## 5. r46: more steps vs a doubled swallow penalty

Both continue from r45 for 2M steps at 8 envs each, seed 46, side by side: `r46-more` (same rewards) and
`r46-eat1` (`reward.boss_ate_bomb_penalty: 1.0`, `specialists/arms/r46-eat1/`, knob `5f0bef4`). Same eval:

| run | success | stun kill rate | bombs / fight | 1-bomb fights | swallows / fight | two-swallow kills | hearts lost | no-hit | steps |
|---|---|---|---|---|---|---|---|---|---|
| r45-dodongo | 100% | 67.3% | 2.50 | 157 | 0.46 | 41 | 0.127 | 89.7% | 32.3 |
| r46-more | 99.3% | 67.8% | 2.25 | 159 | 0.42 | 30 | 0.115 | 90.3% | 29.0 |
| r46-eat1 | 98.7% | 65.9% | 2.16 | 153 | 0.45 | 32 | 0.128 | 89.0% | 30.4 |

Within noise of each other; r46's failures are deaths at 0.5–1.5 starting hearts. Deterministic actions
(r46-more) changed nothing (2.14 bombs, 159 one-bomb fights). r45 stays promoted (no deaths).

Why it may plateau near half one-bomb kills: the fuse is 48 frames, in which Dodongo walks 24 px and passes a
direction decision (every 16 px, partly random), so a placement is partly a bet on his next turn. The observation
has his position, facing and the bomb's active/exploding flags but not the bomb's position. Ideas, not yet tried
(backlog): curriculum tracks of one-bomb stun kills only (explore more seeds, keep 1-bomb tracks); the bomb's
position and fuse in the observation (a schema change); a larger per-bomb penalty for this room.
