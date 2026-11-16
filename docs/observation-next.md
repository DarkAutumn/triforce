# The observation for the rest of the game: one change of the models' inputs (design, Nov 16, round 26)

The owner's decision 6 (Nov 16): "A change that breaks trained models is allowed when a family cannot be trained
without it: convert where the models stay identical, else retrain; reported with the COMPAT bump. All input changes
are to be made in ONE break." This document designs that one change for Dungeon 7 through Ganon (rounds 26-29), from
`docs/routes/rest-of-game.md` "Everything the engine lacks" (a), (b), (d), `docs/routes/late-monsters.md` §2, §3, §6,
the code (`crates/env/src/observation.rs`, `crates/zelda/src/game.rs`, `enums.rs`, `triforce/model.py`,
`crates/policy/src/lib.rs`, `triforce/convert.py`) and probes in the running game
(`scripts/experiments/r26_obs_probes.py`, `scripts/experiments/r26_entity_widen_poc.py`).

Marks as in the route documents: **(walked)** seen in the running game by the probe named, with frames; **(walked,
written)** a phase entered by writing the RAM byte the game's own code would write, then watched; **(code)** read
from the disassembly through the `asm:` label; **(repo)** read in this project's code; **(inferred)** reasoning.

## Summary

- **Schema 6, version 1.5.1.0** (COMPAT +1). Added: 8 entity features (33 → 41), 6 `link` features (87 → 93), 4
  `room_state` features (85 → 89). Nothing else changes size; `history`, `objective`, `target`, `grid`, `room_id`,
  `entity_types` and `action_mask` stay as they are.
- **No existing column changes meaning.** The corrected touch and target rules the dossier asks for (G5, G6) are new
  columns (`touch_harms`, `weapon_target`); `hurts_on_touch`, `killable`, `active`, `hidden` keep their schema 5
  computation to the bit (they are frozen, documented as such). So no trained model sees a different value in any
  column it reads.
- **Every promoted model converts bit-identically**: `triforce convert-model` extended to `entity_features` (zero
  columns of `entity_encoder.input_proj.weight` by feature name) and the Rust policy's all-zero-column drop extended
  to `input_proj` (22 lines). Proof of concept on 7 promoted runs (goriya, boss-manhandla, wallmaster, boss-gleeok,
  darknut, ow-zora, vire): 5,488 decisions with 8 extra entity columns holding random values, plus 2,105 with 3
  (goriya, boss-gleeok): the joint log-probs over all 352 actions, the value and the sampled action identical to the
  bit in every one (§5.3). Without the drop, 1 extra column changed the last bits in 224 of 240 decisions.
- **Recommendation: plan A** (convert all 28 families, retrain none for the break; the families the route extends
  continue from their converted runs as each round needs). Plan B (retrain all 28 from scratch) costs about 178M env
  steps, 66 stream-hours (about 33 hours in the machine's two training places) before evaluations and chains, and
  risks the team's chain numbers, which took rounds 15-25 to reach (§5.4).
- **Hidden positions** (Ganon, a Red Wizzrobe between appearances): a `visible` column for every entity and a
  per-family switch `obs.hidden_positions: ram | last_seen | zero` on the same layout, default `ram` (today). The
  owner's later choice is a yaml value, not a second break (§3).
- **The entity encoder stays** (type embedding, projection, one transformer layer, mean pooling): the cases that
  seemed to need slot identity are answered by per-entity features (`weapon_target`, `ringleader`, the type ids);
  attention pooling is the convertible fallback if Patra's or Lanmola's crowded rooms show pooling is the
  bottleneck (§4).

## 1. Today's layout (schema 5, version 1.5.0.0; `crates/env/src/observation.rs`)

`OBS_SCHEMA_VERSION` 5, action schema 3. Ten keys, in `OBS_SPEC` order. Scalars are computed in f64 and stored as
f32; entity positions and velocities in f32 (repo).

| Key | dtype, shape | Content |
|---|---|---|
| `room_id` | int64 [1] | (level min 9, in_cave, room & $7F) → `(level·2 + in_cave)·128 + room`, vocabulary 2560 |
| `room_state` | float32 [85] | below |
| `link` | float32 [87] | below |
| `entity_types` | int64 [12] | index i < 11: ObjType of NES slot i+1 ($34F+i+1), 0 empty, capped at $7F; a dropped item ($60) is $80 + its item id; Gleeok's attached heads are synthesized in indices 1-4 (NES slots 2-5) as type $42; index 11 is the room item (treasure), $80 + its id. Vocabulary 164 |
| `entity_features` | float32 [12, 33] | below; feature 0 = presence |
| `grid` | int64 [22, 32] | tile class per 8-px tile (vocabulary 15, `zelda::tiles::build_grid`) |
| `history` | float32 [38] | below |
| `objective` | float32 [10] | below (all 0 with `obs.objective` off) |
| `target` | float32 [8] | below (all 0 with `obs.target` off) |
| `action_mask` | bool [143] | `ActionMask::to_flat()`: 11 kinds × (1 + 8 directions + 4 durations); no network input |

The network (`triforce/model.py`, `crates/policy`): `room_id` → embedding 16; entities → type embedding 16,
`input_proj` Linear(33 + 16 → 64) per slot, one transformer encoder layer (4 heads, ff 128) with a padding mask from
presence (`feature 0 <= 0` is empty; an all-empty row unmasks slot 0), **mean over present slots**, Linear(64 → 64),
ReLU; grid → tile embedding 4, three 3×3 convs (16, 32, 32), Linear → 128; the float vector keys concatenated in
sorted order (`history`, `link`, `objective`, `room_state`, `target`); policy and value towers 256-256-256; heads
kind (11), direction (8), duration (4) conditioned on the kind. No slot or positional input anywhere in the entity
path: the encoder is a function of the set of entities.

### 1.1 `entity_features` (33), per slot; all 0 for an empty slot

Computed per NES slot 1-$B from ObjType $34F, ObjX $70, ObjY $84, ObjDir $98, ObjState $AC, ObjStunTimer $3D,
ObjMetastate $405, ObjHP $485, item lifetime $3A8, ObjInvincibilityTimer $4F0, ObjInvincibilityMask $4B2, ObjAttr
$4BF (each at +slot). `class` = `zelda::enums::ObjectClass::of(type)`; `monster` = class Monster ($01-$48 but
hazards and the Other types $19, $29, $2E, $2F, $35-$37); `meta` = ObjMetastate; `active` = state 3 for Leevers
$0F/$10, state 2-4 for a Zora $11, state 1 for a Wallmaster $27, else `meta == 0`; `clock` = `room_state`
`clock_active`; `flashing` = invincibility timer > 0 and ObjAttr bit 5 clear; `gleeok_base` = index 0 holding
$42-$45.

| # | Name | Value |
|---:|---|---|
| 0 | `presence` | 1 |
| 1, 2 | `x`, `y` | ObjX/255; (ObjY − 64)/176, clamped to [0, 1] |
| 3, 4 | `dx`, `dy` | (ObjX − Link X)/255, (ObjY − Link Y)/176, clamped to [−1, 1] |
| 5, 6 | `vx`, `vy` | position change since the last observation ÷ (frames × 4), clamped; 0 on a room's first observation or when the slot's type changed |
| 7 | `hp` | ObjHP high nibble / 15 for a monster (not the Gleeok base), else 0 |
| 8 | `stun` | 1 under the clock for a monster, else min(ObjStunTimer/16, 1) |
| 9 | `invincibility` | min(ObjInvincibilityTimer/32, 1) |
| 10-13 | `facing_N/S/W/E` | ObjDir bits 8, 4, 2, 1 |
| 14 | `hurts_on_touch` | can_hurt(type) ∧ active ∧ ¬dying ∧ ¬clock ∧ ¬(Dodongo $31/$32 in state 2) ∧ ¬(monster ∧ (stun > 0 ∨ flashing)) |
| 15 | `killable` | monster ∧ ¬gleeok_base ∧ type ∉ {$38, $39 big Digdogger} ∧ type ≠ $3F (guard fire) |
| 16 | `projectile` | class Shot ($53-$5C) |
| 17 | `blockable` | type ∈ {$53, $5B, $5C} |
| 18 | `spawning` | meta 1-4 |
| 19 | `dying` | meta $10-$13 |
| 20 | `active` | active ∧ ¬dying |
| 21 | `is_item` | type $60 (and the treasure) |
| 22 | `item_lifetime` | $3A8/255 for a dropped item |
| 23 | `is_treasure` | the room item (index 11) |
| 24 | `hidden` | the room item is not visible yet (index 11 only) |
| 25 | `carried` | the room item is carried by a monster (index 11 only) |
| 26 | `is_tile_object` | type $61-$68 |
| 27-32 | `immune_sword/boomerang/arrow/bomb/rod/fire` | ObjInvincibilityMask bits 0-5 |

Gleeok's attached heads (indices 1-4, one per living neck, at its head): presence, `hp` from slot 5's HP,
`invincibility` from slot 5's timer, `hurts_on_touch` = timer 0 ∧ ¬clock, `killable` 1, `active` 1, positions and
velocities as for any slot. The treasure (index 11, when the room has an item): presence, `is_item`, `is_treasure`,
`active` = visible, `hidden` = not visible, `carried`, position.

### 1.2 `link` (87)

| # | Name | Value |
|---:|---|---|
| 0, 1 | `x`, `y` | Link ObjX/255, (ObjY − 64)/176 |
| 2 | `grid_offset` | signed $394 / 8, clamped |
| 3, 4 | `sub_x`, `sub_y` | (X & 7)/8, ((Y − $3D) & 7)/8 |
| 5-8 | `facing_N/S/W/E` | ObjDir bits |
| 9-13 | `hp_fraction`, `hp`, `max_hp`, `hp_low`, `hp_full` | hearts/max, hearts/16, max/16, hearts ≤ 1, full |
| 14 | `invincibility` | Link's ObjInvincibilityTimer / 32 |
| 15-18 | `shove_N/S/W/E` | shove direction bits ($C0) |
| 19 | `shove_distance` | $D3/32 |
| 20 | `paralyzed` | IsLinkParalyzed $512 ≠ 0 (a Like-Like holds him) |
| 21 | `halted` | Link ObjState & $C0 = $40 (text, item held up, scenes) |
| 22-25 | `doorway_N/S/W/E` | $53 bits |
| 26-29 | `sword_0..3` | sword grade one-hot |
| 30 | `beams_ready` | `GameView::are_beams_available` |
| 31-37 | `sword_active`, `beam_active`, `bomb_active`, `bomb_exploding`, `fire_active`, `boomerang_active`, `arrow_active` | weapon slots $B9-$BE |
| 38-42 | `bombs`, `has_bombs`, `keys`, `has_key`, `rupees` | bombs/bag, > 0, keys/8, > 0 or magic key, rupees/255 |
| 43-50 | `triforce_1..8` | InvTriforce bits |
| 51-71 | `arrow_wood`, `arrow_silver`, `bow`, `candle_blue`, `candle_red`, `whistle`, `food`, `potion_blue`, `potion_red`, `rod`, `raft`, `book`, `ring_blue`, `ring_red`, `ladder`, `magic_key`, `bracelet`, `letter`, `boomerang`, `magic_boomerang`, `magic_shield` | inventory |
| 72-79 | `bomb_usable`, `bomb_reserve`, `bombs_spare`, `help_count`, `next_kill_arms`, `drop_armed`, `drop_is_bombs`, `fairy_next` | the mask's BOMB bit; route reserve/bag; bombs above it/bag; $50/10; the 10th-kill drop state; $627 = 15 |
| 80-83 | `clip_usable_N/S/W/E` | the mask allows CLIP that way |
| 84-86 | `recorder_usable`, `food_usable`, `candle_usable` | the mask allows the kind |

### 1.3 `room_state` (85)

| # | Name |
|---:|---|
| 0-3 | `in_cave`, `in_cellar`, `overworld`, `dark_room` |
| 4-27 | `door_<N/S/W/E>_<none/open/key/bomb/shutter/is_open>` |
| 28-30 | `door_animating`, `shutter_triggered`, `visited` |
| 31-34 | `kill_state_0..3` (underworld only) |
| 35, 36 | `ow_kill_count` (/7), `ow_secret_found` |
| 37-49 | `item_taken`, `has_item`, `item_visible`, `item_hidden_until_clear`, `item_carried`, `item_x`, `item_y`, `item_is_key/map/compass/heart_container/triforce/other` |
| 50-56 | `cave_ware_0..2`, `cave_price_0..2` (/255), `cave_pay` |
| 57-64 | `trigger_none/all_dead/ringleader/last_boss/block_door/block_stairs/money_or_life/all_dead_item` |
| 65-72 | `push_block_present`, `push_block_<state>` (4), `block_push_complete_0..2` |
| 73-78 | `enemies_remaining`, `enemies_unspawned`, `enemies_dying` (/8), `all_dead`, `room_obj_count` (/8), `clock_active` |
| 79-84 | `has_map`, `has_compass`, `is_start_room`, `is_boss_room`, `is_triforce_room`, `maze_step` |

### 1.4 `history` (38), `objective` (10), `target` (8)

`history`: `has_last_action`, `kind_<MOVE..CANDLE>` (11), `dir_<N,S,W,E,NW,NE,SW,SE>`, `duration_<4,8,16,32>`,
`frames` (/64), `interrupt_<damage,spawn,enemy-near,transition,item,door,wall,game-over>`, `room_changed`,
`entered_from_N/S/W/E`. `objective`: `next_room_N/S/W/E`, `objective_<NONE,MOVE,CAVE,TREASURE,FIGHT,ITEM>`.
`target`: `target_N/S/W/E` (the descent's first step), `target_distance` (/32), `target_dx` (/256), `target_dy`
(/176), `target_is_item`.

**These two blocks are inputs too.** The mask is no network input, but `objective`, `target` and the `*_usable`
bits are: a rule change that moves an existing family's FIGHT targets or usable bits in its own rooms changes what
its model reads (§5.5).

## 2. What to add

### 2.1 Rules for the change

1. **No column changes meaning.** A correction is a new column; the old one is frozen at its schema 5 computation
   (a golden test pins it, §7). The cost is four redundant columns (`hurts_on_touch`, `killable`, `active`,
   `hidden` beside `touch_harms`, `weapon_target`, `visible`); the gain is that every trained model reads exactly
   what it was trained on, so plan A needs no forced retrain.
2. **Every new column is computed for every object type**, old families' types included. A converted model's
   weights on it start at zero; a family that trains on reads it from there.
3. **Values in [0, 1]** (no new signed feature), so the bounds of `OBS_SPEC` stay (`entity_features` −1..1,
   `link` −1..1, `room_state` 0..1).
4. **Three-valued where the game alternates frames:** a monster tested or drawn only on every other frame (a
   fading Red Wizzrobe, a teleporting Blue one, brown Ganon's last ~96 frames) gets 0.5, not the parity of the frame
   the decision happens to fall on (a coin flip to the model, `late-monsters.md` §3.7).
5. **One function owns the per-type rules** (`zelda`, beside `ObjectClass`), so `GameView`, the objective and the
   critic can use the same answers later (§6). Each row of §2.3 is a unit test on a RAM fixture.

### 2.2 `entity_features` 33-40 (new)

| # | Name | Definition | RAM | Needed by (dossier) | First round | Evidence |
|---:|---|---|---|---|---|---|
| 33 | `touch_harms` | 1 if the object harms Link when they overlap this frame (Link's own invincibility, halt and Like-Like hold are his, not the object's: `link` shows them); 0.5 when it is tested on alternate frames; per type, §2.3 | ObjType, ObjState, ObjDir, ObjMetastate, timers; asm:Link_CheckMonsterTouch, asm:Link_CheckMonsterTouch_NoReset | G5: Wizzrobes flashing still hurt; hidden Red, teleporting Blue, returning boomerangs, unemerged Lanmola segments, Patra children before the last appears, dying Ganon do not (`late-monsters.md` §2.2) | 26 (the Blue Goriyas' boomerangs coming home, 7:0x69 … 0x29) | shots: walked (`goriya-boomerang`: $5C out in ObjState $10-$17, home in $28-$57); Wizzrobes, Ganon, Patra, Lanmola: rows below |
| 34 | `weapon_target` | 1 if some weapon can damage it in its current phase (its 32-frame flash after a hit is `invincibility`, not this); 0.5 on alternate frames; which weapon stays the `immune_*` bits | as above; asm:Collision_CheckMonsterVsWeaponsAndLink | G6: Patra's body while children live, Manhandla's base, flying Ghini, Ganon by phase, big Digdogger, Gohma's eye (§3.10, §3.13, §3.16, §3.17, §3.22) | 27 (Red Wizzrobes, Gohma, flying Ghinis at 0x21) | rows below |
| 35 | `touch_damage` | hearts a touch costs after the ring, /4, clamped to 1: `b` = asm:Damage_ObjTypeDamageToLink[type] (low nibble whole hearts, high nibble sixteenths: $80 = ½), shifted right once per ring grade (InvRing 1 blue, 2 red), hearts/4; synthesized Gleeok heads use the type-0 entry $60 (3/8 heart, asm:Gleeok_CheckSegmentCollisions); 0 for types ≥ $5D, items, the treasure | ObjType, InvRing $662 | G7: the red ring quarters every damage mid-Dungeon 9; one model must remap every type (`late-monsters.md` §4's table) | 26 (Blue Goriya 1 heart against the Red's ½; small Digdoggers 2) | (code) the table; no probe needed (a constant per type) |
| 36 | `visible` | 1 if the game draws it every frame, 0.5 on alternate frames, 0 not at all; per type, §2.3 | as above | the hidden-position encodings (§3); Wallmasters in the wall show stale floor positions today | 26 (7:0x0d's hands in the wall) | walked (`wallmaster`: hands in the wall at (128,157), (160,125), (160,157), (208,141), (80,93) with ObjState 0) |
| 37 | `phase` | the type's state, normalised, §2.3; 0 for every type not listed | ObjState and the per-type bytes | G2 (§2.2) | 26 (a Goriya waiting for its boomerang: the moment to hit it, §3.1) | rows below |
| 38 | `phase_time` | frames until the type's next event, /512, clamped; §2.3; 0 where not listed | per-type timers | G2 | 27 | rows below |
| 39 | `in_line` | 1 if Link stands in the object's line of fire: for a shooter (Lynels $01/$02, Moblins $03/$04, Goriyas $05/$06, Octoroks $07-$0A, Wizzrobes $23/$24) facing cardinal `d` (ObjDir), and for a flying straight shot ($53, $57, $58, $59, $5B, $5C in ObjState $10-$1F) moving `d`: Link's 16×16 box overlaps the 16-px lane from the object along `d` (perpendicular offset < 16 px, Link ahead along `d`); 0 when `visible` is 0 or for anything else | ObjX/ObjY/ObjDir of both | §2.2 G3/G10, §3.7 "Link is on this Wizzrobe's shot line", §3.22 Lynels; the approach shaping walks into shot lines (`late-monsters.md` §5.8) | 26 (Blue Goriyas throw along their facing within $51 px) | (code, geometry); the Blue Wizzrobe's own test is narrower (Y high nibbles equal, or Link's X equal to its X & $F0, asm:Wizzrobe_Blue_TryShooting): the lane covers it |
| 40 | `ringleader` | 1 for a living monster in NES slot 1 of a room whose secret trigger is 2 (killing it kills every monster) | `room_state` trigger, slot 1 | G1: 6:0x78's five Red Wizzrobes (asm:Secret_Trigger2Ringleader) | 27 | (code) |

### 2.3 The per-type rules of `touch_harms`, `weapon_target`, `visible`, `phase`, `phase_time`

`legacy_touch` = schema 5's `hurts_on_touch` rule; `legacy_target` = `killable` ∧ `active` ∧ ¬dying ∧ ¬spawning.
Rows are checked top to bottom; the first that matches decides. `s` = ObjState.

| Types | `touch_harms` | `weapon_target` | `visible` | `phase` | `phase_time` (frames, /512) | Evidence |
|---|---|---|---|---|---|---|
| Red Wizzrobe $24, clock held | 0 | 1 | 1 | s/255 | 0 | (code) asm:Wizzrobe_Red_Update (the clock path draws and tests it every frame; Link is not touched under the clock, asm:Link_CheckMonsterTouch) |
| Red Wizzrobe $24 | solid ($80-$BF) 1; fading ($40-$4F, $C0-$FE) 0.5; hidden ($00-$3F) or choosing ($FF) 0 | the same | the same | s/255 | frames to its next shot: s − $B0 if s ≥ $B0; else (s − $2F if s ≥ $80, s + 1 if s ≤ $4F) + 79. A rejected spot repeats state $00 a frame (asm:Wizzrobe_Red_State3Appear@Retry): the value is a lower bound | **walked** (`red-wizzrobe`, 6:0x78, slot 1): $FF at frame 4 (place chosen, (112,109), facing up toward Link at (120,77)), $BF at 68, the $59 shot at $B0, frame 83 (79 after $FF), $7F never stored: $80 → $4F at frame 132, $3F at 148, $00 at 211, a retry at 212 (the rejected spot (144,77) stands in ObjX/ObjY for that frame), $FF at 213; next shot at 292 (209 frames, one retry); its position stays the last spot through the hidden frames. Touch while fading and hidden: (code) asm:Wizzrobe_DrawAndCheckCollisionsIntermittently, asm:Wizzrobe_Red_State0Hidden |
| Blue Wizzrobe $23 | teleporting (ObjRemDistance $394 ≠ 0) 0.5, else 1 (a flash does not stop its touch, asm:Wizzrobe_DrawAndCheckCollisions@CheckLink); 0 under the clock | teleporting 0.5, else 1 | teleporting 0.5, else 1 | teleporting 1, standing (ObjTimer < $10) 0.5, walking 0 | teleporting: ObjRemDistance (1 px a frame); else ObjTimer (it stands at $10, teleports at 0) | **walked** (`blue-wizzrobe`, 9:0x15): walk timers reloaded to 117, 254, 244, 252, 120 (Random OR $70), stand 16 frames, teleport 32 frames (ObjRemDistance 32 → 0, diagonal 1 px a frame), $58 shots only on FrameCounter & $1F = 0 (frames 177, 209, 241, 433, 497, 529) |
| Gohma $33/$34 | `legacy_touch` | Gohma_ObjEyeState $46B = 3 (only an arrow flying up into the eye then, asm:Gohma_HandleWeaponCollision) | 1 | eye state / 3 | eye open (Gohma_ObjEyeOpenTimer $444 ≠ 0): that timer; closed: 2 × Gohma_ObjNextOpenEyeCounter $42C | **walked** (`gohma`, 6:0x1c): openings at frames 3 and 501; open timer $80 at opening, eye 2 for 17 frames, eye **3 for 96 frames** (open timer 110 → 14), eye 2 until 0; the next-open counter −1 every second frame (250 → 0); Gohma paces X 98-160 around $80 |
| Ganon $3E, `Ganon_ScenePhase` $445 < 2 | 0 | 0 | 1 | 0 | 0 | **walked** (`ganon`, 9:0x42, room left dark): scene 0 frames 1-80, scene 1 81-272 ($C0), Link halted ($40) until the fight |
| Ganon, dying (Ganon_ObjDyingCount $42C ≠ 0) | 0 | 0 | count < $50: 1, else 0 (ashes) | 1 | $A0 − count (the Triforce of Power appears at $A0) | (code) asm:Ganon_UpdateDying |
| Ganon, brown (s ≠ 0) | 1 | 1 (silver arrow only; asm:Ganon_CheckCollisions) | s ≥ $30: 1, else 0.5 | 0.75 | 2·s (frames left brown) | **walked, written** (ObjState := $FF at frame 500): −1 every second frame, $FF → 0 in 508 frames, standing at (193,160), no fireballs, then blue at (176,160) |
| Ganon, blue, shown after a hit (ObjTimer $28+slot ≠ 0) | 1 | 0 | 1 | 0.5 | ObjTimer (he jumps and vanishes at 1) | **walked, written** (ObjTimer := $40 at frame 420): stood at (68,109) 64 frames, then at Y $A0, X $B0, moving |
| Ganon, blue, invisible | 1 (tested every fight frame, asm:Ganon_ScenePhase2_Fight) | 1 (sword only) | 0 | 0.25 | 0 | **walked**: ObjX/ObjY move 1 px a frame while he is not drawn ((48,173) → (48,109) → (68,109)); fireballs at FrameCounter 0, 64, 128 (& $3F = 0, asm:Ganon_MoveAndShoot) |
| Patra $47/$48 | `legacy_touch` | 0 while any slot $02-$09 holds $25/$26 (asm:Patra_Update@LoopChildren), else `legacy_target` | 1 | s (0 children appearing, 1 all out) | 0 | **walked** (`patra`, 9:0x61): children made at frame 2, out one by one ~9 frames apart, Patra's state 1 at frame 67 |
| Patra children $25/$26 | Patra's s = 1 ? `legacy_touch` : 0 (asm:Patra_UpdateChild) | Patra's s = 1 ? `legacy_target` : 0 | Patra's s = 1 ? 1 : 0 | own s (0 waiting, 1 orbiting) | 0 | walked states; "not drawn, not tested before" (code) |
| Lanmola $3A/$3B with ObjDir 0 | 0 | 0 | 0 | 0 | 0 | **walked** (`lanmola`, 9:0x55): heads (slots 5, $A) out at frame 2, a segment pair every 8 frames, all 10 by frame 34; (code) asm:Lanmola_Update skips direction-0 segments |
| Wallmaster $27 | `legacy_touch` (state 1) | s = 1 | s = 1 | s (0 in the wall, 1 on its trip) | 0 (the shared emergence timer is `room_state` `wallmaster_cooldown`) | **walked** (`wallmaster`, 7:0x0d, Link still at ($20,$8D)): hands in slots 4-8, out at +2, +98, +194 frames |
| Manhandla $3C in NES slot 5 (the base) | `legacy_touch` | 0 (takes hits, never dies; asm:Manhandla_CheckCollisions) | 1 | 0 | 0 | **walked** (`manhandla`, 8:0x6e): slots 1-4 the hands N, S, W, E of slot 5 at the middle, all HP 4 |
| Gleeok neck base (index 0, $42-$45) | `legacy_touch` | 0 | 1 | 0 | 0 | (repo) as `killable` |
| big Digdogger $38/$39 | `legacy_touch` | 0 | 1 | 0 | 0 | (repo) as `killable` |
| flying Ghini $22 | `legacy_touch` | 0 (immune to everything, asm:Ghini_UpdateFlying) | 1 | 0 | 0 | (code) |
| Goriya $05/$06 | `legacy_touch` | `legacy_target` | 1 | 1 while s bit 7 (it stands until it catches its boomerang, asm:Goriya_Update) | 0 | **walked** (`goriya-boomerang`, 7:0x1b): ObjState $80 from the throw to the catch |
| Leever $0F/$10, Zora $11 | `legacy_touch` | `legacy_target` | `active` (surfaced) | 0 | 0 | (repo) |
| Shots $53-$5C | (s & $F0) = $10 ∧ ¬clock (asm:Link_CheckMonsterTouch_NoReset) | 0 | 1 | 0 | 0 | **walked** for $5C (above) |
| Bubbles $2B-$2D, traps, boulders, standing fire | `legacy_touch` | 0 | 1 | 0 | 0 | (repo) |
| synthesized Gleeok heads | `legacy_touch` | 1 | 1 | 0 | 0 | (repo) |
| guard fire $3F | `legacy_touch` | `active` ∧ ¬dying ∧ ¬spawning (`killable` says 0, but it dies to any damaging weapon: HP $10, asm:GuardFire_Update) | 1 | 0 | 0 | (code) |
| every other monster | `legacy_touch` | `legacy_target` | 1 | 0 | 0 | (repo) |
| items, the treasure, persons, Other, tile objects | 0 | 0 | 1 (the treasure: its `active`) | 0 | 0 | (repo) |

### 2.4 `link` 87-92 (new)

| # | Name | Definition | RAM | Needed by | First round | Evidence |
|---:|---|---|---|---|---|---|
| 87 | `sword_blocked` | SwordBlockedLongTimer ≠ 0 or IsSwordBlocked ≠ 0 (`GameView::is_sword_blocked`, which already masks SWORD/BEAMS) | SwordBlockedLongTimer $4C, IsSwordBlocked $52E | G4: every Bubble room (D7 0x49, 0x18, 0x0d; D6, D8, D9) | 26 | **walked** (`bubble`, 7:0x49) |
| 88 | `sword_blocked_time` | SwordBlockedLongTimer / 16 (10-frame units; 1.0 = 151-160 frames) | $4C, LongTimerCycle $26 | the same: how long to fight with bombs or wait | 26 | **walked**: $4C = $10 on the touch, −1 every 10 frames (asm:Bubble_Update, asm:Nmi_Isr), reloaded to $10 by a second touch 47 frames later; $52E stayed 0 (no Q2 Bubbles on the route) |
| 89 | `arrow_usable` | the mask's ARROW bit (as `bomb_usable`) | — | G9; Gohma, Ganon; the mask is no input | 26 (arrows held from 0x44) | (repo) |
| 90 | `rupee_reserve` | the route step's rupee reserve / 16, clamped (0 until routes.yaml carries one; read as `bomb_reserve` is, `GameCompletion`) | — | Gohma's and Ganon's arrows (`rest-of-game.md` (b)) | 26 | (repo) |
| 91 | `rupees_spare` | max(0, rupees − reserve) / 16, clamped | InvRupees $66D | the same | 26 | (repo) |
| 92 | `rupees_few` | min(rupees, 16) / 16 (rupees/255 can't tell 1 from 0) | InvRupees | G9 | 26 | (repo) |

### 2.5 `room_state` 85-88 (new)

| # | Name | Definition | RAM | Needed by | First round | Evidence |
|---:|---|---|---|---|---|---|
| 85 | `frame_phase_32` | (FrameCounter & $1F) / 32 | FrameCounter $15 | G3: Blue Wizzrobes shoot only on & $1F = 0 | 27 | **walked** (`blue-wizzrobe`) |
| 86 | `frame_phase_64` | (FrameCounter & $3F) / 64 | FrameCounter | Ganon's fireball every & $3F = 0 | 29 | **walked** (`ganon`) |
| 87 | `wallmaster_cooldown` | in a room holding a $27: min(ObjTimer of slot 1 / $60, 1) (one hand per $60 frames, the timer shared whatever slot 1 holds); else 0 | ObjTimer $29 | 7:0x0d's lure (§3.3) | 26 | **walked**: slot 1's timer (a Bubble's slot) jumps to $60 at each emergence |
| 88 | `wallmaster_lure` | a $27 in state 0 is in the room and Link stands where asm:Wallmaster_Update@CheckLinkAtWall lets one out (X $20 or $D0 with Y in $6D..$B4, or Y $5D or $BD with X in $29..$C7; take the bounds from the code when building) | Link ObjX/ObjY | the same: the lure is pixel-exact | 26 | walked at ($20,$8D): hands came out |

### 2.6 Not added, and what each would have cost

With the converter extended (§5.2) a later entity, `link` or `room_state` column is a conversion, not a retrain, so
pruning risks little: anything a walk or a round proves missing can be added the same way.

| Candidate (source) | Why not | Cost if added later |
|---|---|---|
| `size_x`, `size_y` (G8: Ganon 32×32, Gohma 48 px, big Digdogger, Dodongo sideways) | constant per type (Dodongo: per type and facing, both observed): the type embedding learns it; Ganon's touch middle at +$10 is the same per type | 2 entity columns, a conversion |
| `damage_scale` in `link` (ring grade) | `touch_damage` carries the ring per entity; `ring_blue`/`ring_red` exist | 1 `link` column |
| slot one-hots or `slot_one` (G1) | `ringleader` names the one slot convention a policy needs; the others are answered by features (§4) | 12 or 1 entity columns |
| `segment_role` head/body/base (G1: Lanmola heads, Manhandla base) | the base is `weapon_target` 0 with `touch_harms` 1, like Gleeok's; any Lanmola segment kill takes that worm's tail and the head dies last, so which segment is the head changes nothing about what to hit, and its velocity shows where the worm goes | 1 entity column |
| `person_fireballs` (IsPersonFireballsEnabled $4CC) | the masks keep A off persons and the Grumble (§6); the fireballs are entities | 1 `room_state` column |
| `ringleader_alive` (`room_state`) | `ringleader` per entity | — |
| the whirlwind's index $523, summoned $508, carrying $522 | the lead's answer 3: the mask offers RECORDER only where and facing as the step needs; the model presses, the count lives in the mask | 3 `link` columns |
| trap state (home, charging, returning; `d9.md` §8) | velocity tells charging (1.75 px a frame) from creeping back (0.5) from home (0); trap-room trains without it today | 1 entity column (the `phase` row) |
| fireball wait before flight (G10) | ~15 frames, shorter than most decisions; the aim is in its facing bits | the `phase` row |
| Patra's orbit pattern (Patra_ObjManeuverIndex $45E) | positions and velocities of the children show the orbit; the flip is every ~292 frames | the `phase_time` row |
| Gohma at home X | `x` | — |
| overworld tile objects, flying Ghinis as tiles | flying Ghinis are objects ($22) already; gravestones and armos statues are tiles in `grid` | — |
| shop wares | `room_state` `cave_ware_*`, `cave_price_*` exist | — |
| Triforce of Power ($672) | 9:0x42's room item is the treasure entity and `room_state` `item_taken` | 1 `link` column |
| the old man's text halt | `halted` (Link ObjState $40) covers it | — |

## 3. Hidden positions

The game keeps every object's position in ObjX/ObjY whether it is drawn or not, and today's observation copies it:
an invisible Ganon's position moves under the model's eyes (walked: 1 px a frame), a hidden Red Wizzrobe shows its
last spot (walked: unchanged through the 64 hidden frames; the next spot is written on the frame it is chosen, and
a rejected candidate stands there for a frame). A player sees neither. The owner decides later (round 27, before
the Wizzrobes train) whether the model may; the layout must not depend on the answer.

**Design.** `visible` (§2.2, index 36) is in the layout for every entity. A family yaml key
`obs: {hidden_positions: ram | last_seen | zero}` (default `ram`, today's values) chooses what positions show for an
entity of a **hidden-position type** ({$24 Red Wizzrobe, $3E Ganon}: the two the owner's question names; others
keep RAM positions in every mode, so no existing family's inputs depend on the switch) whose `visible` is 0:

| Encoding | `x`, `y` | `dx`, `dy` | `vx`, `vy` | `in_line` | `phase`, `phase_time`, `touch_harms`, `weapon_target`, `visible` | What the model knows |
|---|---|---|---|---|---|---|
| `ram` (default) | ObjX/ObjY | from ObjX/ObjY | as today | 0 (§2.2) | as §2.3 | the true position: more than a player |
| `last_seen` | where it was at the last observation with `visible` > 0 in this room (0 if never) | that spot relative to Link now | 0 while hidden; on the first visible observation after, 0 | 0 | as §2.3 | what a player remembers |
| `zero` | 0 | 0 | 0, also on the first visible observation after | 0 | as §2.3 | that it exists and its phase, nowhere |

`presence` stays 1 in every encoding (the entity keeps its place in the pooling; its phase still counts: a player can
count a Red Wizzrobe's hidden frames or Ganon's fireball rhythm too). The `ObservationBuilder` keeps
`last_seen: [[f32; 2]; 12]` beside `prev_xy`, reset with it (room change, slot type change). A converted model never
sees these types, so the switch is free of compatibility questions; the wizzrobe and ganon families should be
trained under the encoding the owner picks (switching later is a distribution shift for those two families only:
a continuation, not a break).

Ganon under `zero` or `last_seen`: the sword hits him only while he is invisible (asm:Ganon_CheckCollisions), so the
model would have to find him from his fireballs' origin, as a player does (inferred: much harder; the reason this is
the owner's call and not a default).

## 4. The entity encoder

The question: does mean pooling over a set (no slot identity) suffice for the late fights?

| Case | What the policy needs | Answered by |
|---|---|---|
| ringleader (6:0x78, five Red Wizzrobes; slot 1 kills all) | which one | `ringleader` (40) |
| Manhandla's base (slot 5, same type $3C as the hands) | don't aim at it, avoid it | `weapon_target` 0, `touch_harms` 1 (as Gleeok's base) |
| Lanmola heads (slots 5, $A) against tails | nothing to tell: any segment kill removes that worm's tail, the head dies last | — |
| Patra's body against its children | body is untargetable while children live | different type ids ($47/$48 against $25/$26) and `weapon_target` |
| big monsters' size | where the box is | type id (constant size per type) |
| crowded rooms (Patra 9, Lanmola 10, Gleeok with heads and flying heads, Wizzrobe rooms with shots) | the dangerous one among many | one attention layer before the pooling routes it; what trains today is smaller (boss-gleeok: base, two heads, a flying head and fireballs; darknut's Olive Garden: six Blue Darknuts), so pooling under 9-11 entities is untested (inferred risk, watched in the patra and lanmola rounds) |

Options and their cost:

| Option | Compatibility | Training | Verdict |
|---|---|---|---|
| (a) per-entity role and phase features (this design) | new columns, converted bit-identically | none extra | **chosen**: smallest, semantic, works across rooms (slot assignments differ by room) |
| (b) attention pooling: a learned query, softmax over present slots | an architecture change; converts exactly only if the Rust policy keeps the mean path while the query is zero (softmax of equal scores is 1/n, but Σ t·(1/n) and (Σ t)/n round differently) and torch accepts the same tiny gap in the learner | changes every family's dynamics once it trains | fallback if patra or lanmola stall on crowds; convertible later with that special case |
| (c) slot embedding (12 one-hot columns, or a learned per-slot vector added to tokens) | one-hots convert like any column; an added vector converts if zero-initialised | must learn game-internal slot conventions per room | not needed: (a) covers the named cases |

## 5. How the trained models survive

### 5.1 Every change, and whether it converts

| Change | Converts bit-identically? | Families that must retrain because of it |
|---|---|---|
| 8 entity columns (33-40) | yes: zero `input_proj` columns by name + the Rust entity drop (§5.2, §5.3) | none |
| 6 `link`, 4 `room_state` columns | yes: as round 24's widening (tower first layers, zero columns by name, the existing drop) | none |
| legacy columns frozen | nothing changes | none |
| `obs.hidden_positions` (default `ram`) | affects only $24, $3E, which no family meets | none |
| *if* `hurts_on_touch` were corrected in place instead | no | goriya (Red Goriyas' boomerangs in 1:0x44, 1:0x23, 2:0x5e, 2:0x1e: harmless on the way home), and any family with Wizzrobes, Lanmolas, Patra, Ganon (none yet) |
| *if* `killable` were corrected in place | no | boss-manhandla (3:0x4c's base, slot 5) |
| *if* `active`/positions changed for Wallmasters in the wall | no | wallmaster (1:0x45) |

So the in-place alternative would force exactly goriya, boss-manhandla and wallmaster to retrain; the design avoids
it. The legacy columns can be dropped at a later full retrain if the owner wants a clean layout (§8).

### 5.2 The converter and the policy

- **`triforce/convert.py`** becomes the 1.5.0 → 1.5.1 converter (clean cutover: every 1.4.1 run was converted in
  round 24): `SOURCE_VERSION` 1.5.0 (`registry.CONVERTIBLE_VERSION`), `SOURCE_OBS_SCHEMA` 5, `SOURCE_ACTION_SCHEMA`
  3, `SOURCE_KINDS` the current 11 (no kind rows change; the kind-widening path goes). `ADDED` gains `entity` (the
  8 names) next to `link` (6) and `room_state` (4). `entity_encoder.input_proj.weight` [64, 33 + 16] → [64, 41 +
  16]: column of old feature name `n` → new index of `n`; the 16 type-embedding columns → 41..56; new feature
  columns zero (`model.EntityEncoder` concatenates features then embedding: that order holds). The tower first
  layers as today. Adam moments padded with zeros along dim 1 for `input_proj.weight` too. `check_source` compares
  `entity_features` against [12, len(old entity names)].
- **`model.widen_policy`** (used by `registry.load_model` with a caller's `obs_spec`) refuses an `entity_features`
  change with a message pointing at `convert-model` (it widens only new vector keys).
- **`crates/policy`**: `input_proj`'s all-zero input columns are dropped as the towers' are (`Policy::live`):
  `entity_live`, `keep_inputs`, a gather into a `token_live` scratch (Appendix A, 22 lines). A zero column adds
  nothing, but it shifts later columns into other lanes of `kernels::dot`'s 8-lane sum and changes the last bits.

### 5.3 The proof of concept (this branch)

`scripts/experiments/r26_entity_widen_poc.py FAMILY --extra N`: the promoted checkpoint (A) and a copy (B) with N
zero `input_proj` columns after the 33 features and `obs_spec` `entity_features` [12, 33 + N]; one `_native.Env` of
the family plays B's action each decision; at every decision A gets the env's observation and B the same with N
columns of uniform random values in [−1, 1] in every slot (present or not); compared bit for bit: the 352 joint
log-probs, the value and the action each draws from the same Gumbel noise. Seed 0, 20 episodes, up to 400 decisions.

| Build | N | Family (run) | Decisions | Differing |
|---|---:|---|---:|---:|
| without the entity drop | 1 | goriya (r75-f-a), 3 episodes | 240 | 224 (log-probs within 6e-6, values in the 6th digit) |
| without | 8 | goriya, 3 episodes | 296 | 0 (8 = the lane count: the shift keeps every lane; not a general guarantee) |
| with the drop | 1 | goriya, 3 episodes | 240 | 0 |
| with | 8 | goriya (r75-f-a) | 1,105 | 0 |
| with | 8 | boss-manhandla (r75-mh-fight) | 754 | 0 |
| with | 8 | wallmaster (r71-e) | 423 | 0 |
| with | 8 | boss-gleeok (r71-d) | 1,222 | 0 |
| with | 8 | darknut (r75-d537-c5) | 1,073 | 0 |
| with | 8 | ow-zora (r74-leg) | 339 | 0 |
| with | 8 | vire (r75-f-a) | 572 | 0 |
| with | 3 | goriya, boss-gleeok | 906 + 1,199 | 0 |

The drop was a scratch patch for this run, not committed (Appendix A). What the builder's proof adds (§7, commit 4):
the real converter (by name, both blocks at once), the real observation (computed new columns, not random ones),
and the round 24 style evaluation: every promoted run, 20 episodes per state, seed 0, original on 1.5.0 against
converted on 1.5.1, `temperature:0.5+relax:16` and argmax, identical per episode.

### 5.4 Plan A against plan B

| | (A) extend the converter, keep the 28 models | (B) retrain all 28 once |
|---|---|---|
| Build | converter + Rust drop (the PoC's patch, tests): about a day of one builder; the observation itself is the same work in both plans | the same observation work |
| Training for the break | none | 178.2M env steps: the summed lineage of the 28 promoted runs (`train_summary.json` `steps`: darknut 41.2M, ow-moblin 20.0M, goriya 13.0M, boss-manhandla 10.0M, moldorm 9.0M, like-like 8.8M, ow-lynel 8.0M, boss-gleeok 8.0M, vire 7.7M, d2-rope 7.0M, …, dodongo-crossing 0.35M) at each run's logged rate (600-1,400 steps/s at `--parallel 8`): **66 stream-hours, ~33 hours** in the two training places. Round 24's from-scratch runs (gibdo 2.6M 1.12 h, pols-voice 1.64M 0.62 h, boss-digdogger 2.96M 1.18 h, dodongo-crossing 0.35M 0.16 h) fit that rate; darknut alone needed five 4M passes and a checkpoint selection for the Olive Garden |
| Measurement | the proof: 28 runs × 20 episodes per state × 2 acts (round 24 did 24 runs the same way) | 28 canonical evaluations, act checks, `select-checkpoint` for the chain-scored families (1000-run chains per checkpoint), then all chain segments 1000 runs clips on and off (round 22: 45 + 42 minutes for 8 segments; now 10) |
| Training later | the families each round extends continue from converted runs, as they would anyway: a 244-iteration (1M step) continuation took 14-28 minutes in round 25 (goriya r75-f-a 846 s, gibdo r75-f-a 1,654 s), round 24's continuations 0.09-2.08 h | the same, from the retrained runs |
| Risk | a converter bug (caught by the proof); the frozen legacy columns mislead new training a little (`killable` 1 on Patra's body beside `weapon_target` 0; the network learns which to trust) | the team took rounds 15-25 to reach D5 46.4% clips on; a retrain from scratch is not known to get back there (round 22: ow-moblin from scratch with 8M was no better than continuing; darknut's Olive Garden needed 20M of continuation); every model's act mode and chain behaviour must be measured again; round 26 waits days |
| Layout | 4 redundant legacy columns | clean (`hurts_on_touch`, `killable` corrected in place; 4 columns fewer) |

**Recommendation: A.** Nothing in the rest of the game needs B: every new input is a new column, and no existing
column needs a new meaning. B buys a cleaner layout for two to three days of machine time and the risk of losing the
current team. If the owner wants the clean layout, do it at the one optimisation round after the first complete
run (decision 2), when every family is retrained anyway.

### 5.5 Input changes outside the observation block (for the builders of rounds 26-29)

These change `objective`, `target` or a `*_usable` bit, so they reach the network even though they are "rules":

| Rule change | Families whose inputs change in their own rooms | Handle |
|---|---|---|
| FIGHT targets filtered by `weapon_target` (`rest-of-game.md` (c)) | boss-manhandla (the base), boss-digdogger (the big one before the split), boss-gleeok (the base is already a hazard: none) | gate by an explicit list of post-D5 types until those families retrain (round 28 for 8:0x6e); Steps26 told |
| Manhandla's base as a hazard in `GameView` ((d)) | boss-manhandla: objective targets and rewards | round 28, with the 8:0x6e retrain |
| `recorder_wanted` per route step ((b)) | boss-digdogger: unchanged in 5:0x24 (big one, not heard) | keep 5:0x24's rule bit-identical |
| ARROW masked by `arrows: false` / `rupee_reserve` | none: no state so far held arrows | — |
| FOOD, CANDLE masks per step | none: masked for every family today | — |

## 6. Masks and reward rules that need no input change (where they go)

Pointers only; each round's builder designs them (`rest-of-game.md` (b), (d); `late-monsters.md` §5, §6.4).

| Rule | Where | Round |
|---|---|---|
| ARROW off: `items: {arrows: false}` in every family but boss-gohma, ganon (pols-voice: owner) | `specialists/*.yaml`, `ItemsConfig` (`crates/env/src/config.rs`) | 26 |
| route `rupee_reserve` (as `bomb_reserve`), ARROW masked at or below it and at 0 rupees (asm:Arrow_Wield) | `triforce/routes.yaml`, `GameCompletion`, `zelda::actions::compute_action_mask` | 26 |
| arrow target masks (Gohma: eye state 3, Link in the X window facing N; Ganon: brown, his row or column) | `zelda::actions` beside the BOMB wall mask | 27, 29 |
| RECORDER per step (7:0x1c yes, 7:0x39 never, the whirlwind steps facing as the count needs, 0x42's pond once) | `actions::recorder_wanted` | 26 |
| FOOD only at 7:0x28 after the text; CANDLE only at 0x6d's and 0x47's trees | the mask, by route step | 26, 28 |
| A never on persons and the Grumble | the mask | 26 |
| BOMB modes per room (never at Patra, Pols Voice, Gohma, Ganon; required at 7:0x0c, 0:0x05) | yamls, the BOMB mask | 26-29 |
| `penalty-lost-health` capped at 1.0: a 4-heart Red shot costs what a 2-heart hit does; scale with hearts (`touch_damage` gives the critic the same number) | `crates/env/src/critic.rs` | 27 |
| the wallmaster wall shaping punishes 7:0x0d's lure: off there | `critic.rs` | 26 |
| Ganon's kill (Ganon_ObjDyingCount, the Triforce of Power), his 4th hit, a living-monster rule without a dead Ganon | `critic.rs`, `room_state.rs` | 29 |
| `PATRA2` ($48) in the boss list or not; dead dummies $5D and the stall penalty | `zelda::enums::enemy::BOSS_KINDS`, `end_conditions.rs` | 29 |
| rooms where waiting is right (Gohma, Patra, Ganon's scene, traps, the Grumble, lures, a Red's hidden frames) against room-stalling, `no_damage_steps`, `no_progress_steps` | `end_conditions.rs`, per family | 26-29 |
| the approach shaping into shot lines (`in_line` gives the critic the test) | PBRS (`docs/architecture.md`) | 27 |

## 7. Order of commits and tests

1. **Policy: the entity zero-column drop** (`crates/policy/src/lib.rs`, Appendix A). Tests: a Rust unit test that
   builds a random small policy, inserts k ∈ {1, 3, 8} zero columns into `input_proj` at feature positions, and
   compares `joint_log_probs` and `value` bit for bit on inputs whose new columns hold random values; a test that a
   policy without zero columns builds `entity_live: None` (today's path untouched).
2. **Converter: entity features by name** (`triforce/convert.py`, `registry.CONVERTIBLE_VERSION`, `model.widen_policy`
   refusal). Tests (`tests/test_convert.py`): a synthetic 1.5.0 checkpoint converts; the converted Rust policy's
   joint log-probs equal the source's bit for bit with random values in every new column of every block; Adam
   moments padded (shapes, zeros in the new entries, old entries unchanged); a checkpoint of another layout is
   refused.
3. **The observation** (`crates/env/src/observation.rs`, the per-type rules in `zelda` beside `ObjectClass`,
   `env::config::ObsConfig.hidden_positions`, `triforce/observation.py` tuples, `OBS_SCHEMA_VERSION` 6). Tests:
   - **legacy identity**: before the change, record the 33 + 87 + 85 + 38 + 10 + 8 schema 5 columns over a fixed set
     of states and steps (every family's start states, 50 seeded steps each); after it, the old columns equal them
     bit for bit (the golden file a fixture);
   - **one RAM fixture per row of §2.3** (write the bytes into a `GameView`, assert the five values), plus §2.2's
     `touch_damage` against the table with no ring, blue, red, `in_line` lanes in four directions and a hidden
     shooter, `ringleader`;
   - **walked traces as tests** (probe-built, like `tests/test_d5.py`'s): 6:0x78 step to $B0 → slot 1 `visible` 1,
     `phase_time` 0, a $59 shot appears; 9:0x42 scene 2 → Ganon `visible` 0, `touch_harms` 1; 7:0x49 Bubble touch →
     `sword_blocked_time` 1.0 then 0.9 ten frames later;
   - **the hidden-position modes** on the Red Wizzrobe fixture: `ram` (today), `last_seen` (frozen at the last
     visible spot, velocity 0), `zero`; other types unchanged in every mode.
4. **Version 1.5.1.0** (COMPAT +1, REV 0; `VERSION`, `pyproject.toml`, the Cargo workspace 1.5.1;
   `tests/test_version.py`), `triforce convert-model --promoted --out <dir>` for the 28 families, and **the proof**:
   every promoted run, 20 episodes per state, seed 0, original on 1.5.0.0 against converted on 1.5.1.0, in
   `temperature:0.5+relax:16` and argmax, identical per episode (outcome, steps, frames, hearts, records). Then
   `docs/architecture.md` (the observation table, schema 6), `docs/decisions.md` (the entry), AGENTS.md's
   versioning and "Where things stand". Replays are unaffected (`REPLAY_VERSION` stays: replays hold inputs and
   states, not observations).

Commits 1 and 2 can land first on their own (no layout change, no version bump): they are inert until commit 3.

## 8. Open points

For the owner:
1. **Hidden positions:** `ram`, `last_seen` or `zero` for Red Wizzrobes and Ganon (§3); asked when round 27 starts,
   before the wizzrobe family trains.
2. **Plan A or B** (§5.4); and whether the legacy columns go at the optimisation round after the first complete run.

For walks (none is needed before commit 3 can be built; each can change a row of §2.3, not the layout):
3. A Wizzrobe's touch while it flashes after a hit, while fading, while teleporting (code only; `late-monsters.md`
   §11 #4).
4. Ganon with real hits: the visible and brown phases were entered by writing ObjTimer and ObjState; the dying
   sequence and the critic's reading of it are code only (`late-monsters.md` §11 #1).
5. Patra's body had ObjMetastate 1-3 (a spawn cloud) for its first 14 frames in the probe (`late-monsters.md` §1
   says Patra has none); whether that is the probe's room entry or the game. Children "not drawn before the last
   appears": code only.
6. The Red Wizzrobe's appear retries lengthen the cycle (the walk saw one retry per cycle with Link in the N
   doorway): `phase_time` is a lower bound; measure how often in the D6/D9 rooms.
7. `in_line`'s 16-px lane against the shooters' own tests (the Blue Wizzrobe's 16-px row by high nibbles, Lynels
   at grid points within $51): the lane is wider than some; a planner walk in a Wizzrobe room would show whether
   the model needs the narrower per-type test.
8. 7:0x0d: standing still at ($20,$8D) with hearts held, no capture in 300 frames while three hands came out
   (level-7.md: captured after 161 frames); told to Walks26.
9. `rupee_reserve` values per step come from the route builder (Steps26); the observation reads whatever the route
   carries.

## 9. As built (round 26, 1.5.1.0, branch `r26-obsbuild`)

Built as §2-§3 describe; the reference is now `docs/architecture.md` "Observation" (the tables as built) and the
proof `docs/decisions.md` 2026-11-16 "Schema 6". Deviations from §2.3:

- **Patra's children:** `visible` is the child's own state 1 (asm:Patra_UpdateChild_State1Orbit draws it from its
  state 1 on; only the collisions wait for Patra's state 1), not "Patra's s = 1".
- **The boulder spawner `$1F`:** `visible` 0 (an invisible generator, asm:Boulder_UpdateSpawner).
- **A Blue Wizzrobe under the clock:** `weapon_target` 1, `visible` 1 (drawn and hittable every frame), `touch_harms` 0.
- **The VERSION bump** went into the observation commit (AGENTS.md: a bump goes in the commit that causes it).
- Open point 5 (Patra's spawn cloud) is the game's: asm:Patra_Update notes Patra's spawn cloud ends on frame 13.

Walked on the 1.5.1.0 build (`scripts/experiments/r26_obs_probes.py`, Nov 16; Link held on the object every frame,
his invincibility timer and knockback cleared before each frame and his hearts held; "harmed" = the game started his
knockback that frame, "drawn" = sprites of the OAM buffer in the object's box; the observation is read before the
frame, so a phase change inside a frame shows as one harmed frame of the phase before):

| Probe | Result | Verdict |
|---|---|---|
| `red-wizzrobe-touch` (6:0x78, slot 1) | hidden 0 of 132 frames harmed, 0 drawn; fading in 64/126 harmed and drawn; solid 127/128; fading out 15/32; with the clock 0/119 harmed, 119/119 drawn, observed (0, 1, 1) | the rows confirmed (0.5 on fading) |
| `blue-wizzrobe-touch` (9:0x15, its invincibility timer held at `$10`) | walking 320/320 harmed, standing 18/20, teleporting 80/160 harmed and drawn | a flash does not stop its touch; 0.5 teleporting confirmed |
| `ganon-dying` (dying count WRITTEN 1) | no harm through `$C0`; observed `visible` 1 below `$50`, 0 from `$50` (sprites drawn there are the ashes) | `touch_harms` 0 confirmed |
| `ringleader` (6:0x78) | `ringleader` 1 on slot 1 only; slot 1 emptied: every other Red Wizzrobe's death (metastate `$10`) began the next frame | confirmed |
| `bubble-block` (7:0x49) | `sword_blocked_time` 1.0 at the touch, 0.9375 two frames later, then −1/16 every 10 frames; a second touch at +47 reloads 1.0 | confirmed |
| `patra-touch` (9:0x61; a waiting child moved, WRITTEN, 40 px from Patra and the orbiting children) | slot 9 waiting 0/65 harmed, 0 drawn, then out with Patra's state 1: 95/95; slot 2 waiting 0/2, out before Patra's state 1 0/63 harmed (the 1 harmed frame is the one Patra's state turned 1), 63/63 drawn, after 95/95 | confirmed: a child is drawn from its own state 1 and harms only from Patra's |
| `lanmola-touch` (9:0x55, the tail, slot 1, moved WRITTEN 40 px from the moving segments while its ObjDir is 0) | ObjDir 0: 0 of 33 frames harmed (the 34th is the frame its direction arrives), 0 drawn; moving 56/56 | confirmed |
| `wallmaster-lure` (7:0x0d from W) | standing at (`$20`,`$8D`): lure 1, a hand came out each time the shared timer ran out; walked one pixel on (Right 2 frames: (`$21`,`$8D`), grid offset 1): lure 0, 116 frames with the timer at 0 and no hand. (A written X without the grid offset `$394` does not hold: the game moves Link back to `$20` the next frame, which is why the first probe saw hands at "`$21`") | confirmed |

Still code only (a later round walks them before relying on them): Gohma's eye row with real arrows (round 27);
Ganon's brown and shown phases with real hits, and his dying sequence entered by a real silver arrow (round 29);
`in_line` against each shooter's own test (open point 7); `phase_time` of a Red Wizzrobe with retries (6);
`touch_damage` of the late types (a constant table; a ring walk would close it).

## Appendix A: the entity zero-column drop (the PoC's patch to `crates/policy/src/lib.rs`)

```rust
// struct Scratch
    token_in: Vec<f32>,
    /// The token inputs `input_proj` reads ([`Policy::entity_live`]), when some are dropped.
    token_live: Vec<f32>,

// struct Policy
    input_proj: Linear,
    /// The entity token columns (features, then the type embedding) `input_proj` reads, when some are read by no
    /// weight: the same drop as [`Policy::live`], for entity features added with zero weights.
    entity_live: Option<Vec<usize>>,

// Policy::new
        let mut input_proj = w.linear("entity_encoder.input_proj", d, entity_features + config.entity_embed_dim)?;
        let entity_live: Vec<usize> = (0..input_proj.inp).filter(|&i| input_proj.reads(i)).collect();
        let entity_live = (entity_live.len() < input_proj.inp).then(|| {
            input_proj.keep_inputs(&entity_live);
            entity_live
        });

// Policy::scratch
            token_live: vec![0.0; self.entity_live.as_ref().map_or(0, Vec::len)],

// Policy::entities, per slot, after token_in is filled
            let token = if let Some(live) = &self.entity_live {
                for (o, &k) in s.token_live.iter_mut().zip(live) {
                    *o = s.token_in[k];
                }
                &s.token_live
            } else {
                &s.token_in
            };
            self.input_proj.apply(token, &mut s.tokens[i * d..(i + 1) * d]);
```

## Appendix B: the probes

`.venv/bin/python scripts/experiments/r26_obs_probes.py [red-wizzrobe blue-wizzrobe gohma bubble patra ganon lanmola
goriya-boomerang manhandla wallmaster]`. Each enters its room with `triforce.probe.Probe` (Link walked in through a
door with the room id and level block written; Ganon's room left dark so its entry scene can end), holds Link's
hearts each frame and prints the bytes when they change. Frame numbers in §2 are the probe's (frame 1 = the first
frame after entry); the walks were Nov 16 on 1.5.0.0. `scripts/experiments/r26_entity_widen_poc.py FAMILY --extra N
[--episodes E] [--steps S] [--seed SEED]` needs Appendix A's patch to pass for N not a multiple of 8.
