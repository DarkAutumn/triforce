# Round 12: beat Dodongo (Dungeon 2 boss, room 2:0x0e)

Lee, Nov 8: train a model that kills Dodongo the good way. Set a bomb in his path but *just next to him*, so he
doesn't eat it and the smoke stuns him; then hit him with the sword while he is stunned. The model gets bombs as it
needs them (rules below). The curriculum over his movement patterns matters most.

## Summary (in progress)

- Mechanics verified by probes (§1). Env support, rewards and Lee's bomb rules are in (`f2141cf`, §2).
- Curriculum: Go-Explore over 3 start states × 10 explore seeds, swallow kills removed: 25 stun-and-sword
  tracks (§3, `a09d9bd`).
- r45-dodongo is training (§4).

## 1. Mechanics (probes, `target/dodongo/`; disassembly `Z_04`)

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
