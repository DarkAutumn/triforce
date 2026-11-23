# The late monsters: what each demands of observation, actions, masks and rewards (rest-of-game research, Nov 15-16)

The monsters of the rest of the game (Dungeon 7, the magic sword, Dungeons 6, 8, 9, Ganon, Zelda), one by one:
what the game does with each, what the observation shows of it today, which actions and masks it needs and what the
reward and objective misjudge. It was mined from the disassembly, the level documents and the project's code before
any of it is built; `docs/routes/rest-of-game.md` is the synthesis over it and its five sibling documents. Sources:
`zelda-asm/enemies.md` (every section of a monster on the draft route after Dungeon 5), `zelda-asm/mechanics.md`
("Monster spawning...", "Combat...", "Hearts, rings and death"), `zelda-asm/items.md` (weapon slots, sword and beam,
bombs, arrows, boomerang, candle, recorder, food, ring, shield), `zelda-asm/levels/level-6.md` to `level-9.md`
("Monsters and boss", "Route", "Hazards"), `zelda-asm/data/levelblock-ow.md`, `docs/routes/first-quest.md` (the
draft), `docs/routes/d5.md`, `docs/architecture.md`, `docs/rust-actor.md`, the project's code
(`crates/env/src/observation.rs`, `critic.rs`, `config.rs`, `end_conditions.rs`; `crates/zelda/src/actions.rs`,
`game.rs`, `enums.rs`, `change.rs`, `objectives.rs`, `room_state.rs`, `attack.rs`; `triforce/model.py`,
`triforce/convert.py`) and `specialists/*.yaml`.

Marks: **(trace)** observed in the running game per the level document named; **(code)** read from the disassembly
(through the bullet's `asm:` label); **(data)** ROM tables (`zelda-asm/data/`); **(draft)** `docs/routes/first-quest.md`;
**(inferred)** reasoning, not checked; **(repo)** read from this project's code. Times are frames (60 per second). HP and
weapon damage in game units: `$10` = one wooden sword hit; damage to Link in hearts, before rings unless said.

---

## 0. What Link carries where (the frame for every number below)

| Stretch | Max hearts | Sword (damage) | Ring | Keys in → out | Bombs (no drops, draft) | Rupees |
|---|---|---|---|---|---|---|
| D5 → D7 leg | 11 (draft) | white `$20` | none | 5 | 3 in (draft table, after 0x65's +4) | 42-54 (+100 at 0x0f, -80 arrows at 0x44, -60 food at 0x34 → **2-14**) (draft) |
| Dungeon 7 | 11, 12 after Aquamentus (draft) | white `$20` | none | 5 → 2 (draft) | 3 → 0 → 4 → 2 (walls 0x69, 0x19, 0x1a, +4 at 0x1b, walls 0x0c, 0x29) (draft) | 2-14 |
| D7 → sword → D6 | 12 | white, **magic `$40` at 0x21** (needs 12 containers, asm:Cave_State2TalkOrShopOrDoorCharge) | none | 2 | 2 | 2-14 |
| Dungeon 6 | 12, 13 after Gohma | magic | none | 2 → 2 (spends 2, takes 0x58's and 0x2d's) (draft) | 2 → 1 (0x28 wall) | 2-14, **-1 per arrow** (Gohma) |
| D6 → D8 | 13 | magic | none | 2 | 1 | ≥1 needed later |
| Dungeon 8 | 13, 14 after Gleeok | magic | none | 2 → 3 (draft) | 1 → 0, **1 cheated** (0x4c) | |
| D8 → D9 | 14, 15 after 0x47's take-any heart container | magic | none | 3 | 0, 1 cheated (0x05 rock wall) | |
| Dungeon 9 to the red ring | 15 | magic | none | 3 → 1 | 0 → 4 → 0, **4 cheated before 0x16's Patra** | ≥1 for Ganon's arrow |
| Dungeon 9 after cellar 0x00 | 15 | magic | **red: damage ÷ 4** (asm:Link_BeHarmed) | | | |

- No magic shield, no blue ring, no potions (the owner's decisions of Nov 16, `docs/routes/rest-of-game.md` "What this
  is"). So every type `$55`, `$57`, `$58`, `$59` shot is **unblockable** on this route; only `$53`, `$54`, `$5B`,
  `$5C` (rock, monster arrow, **Goriya boomerang**) are parried, by the small shield (code:
  asm:Link_CheckMonsterTouch_NoReset@FacingOpposite).
- Hearts refill only at a triforce piece (asm:Mode12EndLevel_StartFillingHearts) and from drops/fairies. A heart
  container adds one container and one full heart, no refill (asm:Item_Take_CheckClass1). Dungeon 9 has no piece:
  15 hearts must last 32-42 room entries and ~10 forced fights (level-9.md "Route").
- The sword beam needs full health (up to `$7F`/256 of a heart missing) (code: asm:SwordShot_Make): after any hit
  the beam is gone until a heart/fairy refills.

---

## 1. Rules every late fight obeys (code unless marked)

| Rule | Value | Label |
|---|---|---|
| Death test on a monster | dies when HP < damage or HP − damage = 0 (HP 0 dies to the boomerang's 0) | asm:Damage_DealToMonster |
| Monster hit, not killed | shoved `$40` px the weapon's way, `ObjInvincibilityTimer` `$10` ≈ 32 frames: no weapon test **and its touch does not hurt** (generic pass) | asm:Shove_Begin, asm:Collision_CheckMonsterVsWeaponsAndLink |
| …exceptions | Wizzrobes: touch still tested while flashing (asm:Wizzrobe_DrawAndCheckCollisions@CheckLink); Manhandla base: timer cleared every frame (asm:Manhandla_CheckCollisions); Ganon has his own test (asm:Ganon_CheckCollisions) | |
| Link hit | knockback `$20` px, invincibility `$18` ≈ 48 frames; ring does not shorten it | asm:Shove_Begin, asm:Link_HarmByObjType |
| Every harm (Bubble's 0 and whirlwind's 0 included) | zeroes `WorldKillCount`, `HelpDropCount`, `IsHelpDropBomb` | asm:Link_BeHarmed@ResetHelp |
| Link dies | whole-heart damage > full hearts (the last heart is the partial one), or a borrow with no full heart: **a hit of D hearts kills at ≤ D hearts shown** (inferred from the rule) | asm:Link_BeHarmed@HandleDied |
| Boomerang | damage 0, stun `$10` long-timer units = 151-160 frames; a stunned monster cannot touch Link | asm:Collision_CheckWeaponPointVsMonster, asm:Link_CheckMonsterTouch |
| Shield | only in Link state `$00`-`$0F` (not swinging, not using an item, not catching): off frames 0-12 of a swing, 0-11 of an item use | asm:Link_CheckMonsterTouch_NoReset |
| Spawn cloud | no update, no test, no harm in metastate 1-3; Wizzrobes, Patra children, Gleeok, Gohma, Lanmola, Manhandla, Ganon, Wallmaster and shots have none; traps do | asm:Obj_Update, mechanics.md "Spawn cloud" |
| Sword hit window | sword state 2 only, frames 4-11 of a 16-frame swing | items.md "Sword swing frame table" (trace) |
| Bomb | blast state `$13` only (frames 48-71), `$40`, box < `$18`; walls tested once when the blast ends | asm:Bomb_CheckBombOrFireVsMonster, asm:Bomb_TryOpenWall |
| Arrow | 1 rupee per shot (refused at 0 rupees), `$20` wooden / `$40` silver, stops on the first monster it overlaps (not a Pols Voice) | asm:Arrow_Wield, asm:Collision_CheckShotVsMonster |
| Help drop | 10th damage-free kill → 5 rupees, or 4 bombs if that 10th kill was a bomb kill; 16th → fairy | asm:Drop_SetUpItem, asm:Damage_KillMonster |

---

## 2. The observation and action space today (repo), and the generic gaps

### 2.1 What each entity carries (`crates/env/src/observation.rs:405-440` names, `:758-896` values)

`entity_types[12]` (ObjType of slots 1-`$B`, the room item in the 12th; a dropped item as `$80` + its id) and
33 features: presence, x, y, dx, dy, vx, vy (position delta between decisions ÷ frames ÷ 4), hp (high nibble/15),
stun (/16; 1 under the clock), invincibility (timer/32), facing N/S/W/E, hurts_on_touch, killable, projectile,
blockable (`$53`, `$5B`, `$5C` only), spawning, dying, active, is_item, item_lifetime, is_treasure, hidden, carried,
is_tile_object, immune_{sword, boomerang, arrow, bomb, rod, fire} (the invincibility-mask bits).
Gleeok's attached heads are synthesized (`:849-865`). `link` (87) has position, facing, hp, max hp, invincibility,
shove, `paralyzed` (Like-Like hold, `$512`), `halted`, sword grade, beams_ready, weapon-slot activity bits, bombs,
keys, rupees (/255), the inventory, bomb/drop features, `clip_usable_*`, `recorder_usable`, `food_usable`,
`candle_usable` (`:1123-1230`). `room_state` (85) has the secret trigger one-hot, all_dead, enemies_remaining, etc.

### 2.2 Generic gaps (each one matters to several monsters below)

| # | Gap | Evidence | Who needs it |
|---|---|---|---|
| G1 | **No slot identity.** The entity encoder is type embedding + projection + transformer with a presence mask, **mean-pooled, no positional/slot input** (`triforce/model.py:184-193`): which entity sits in slot 1 is not knowable. | repo | ringleader rooms (D6 0x78: kill slot 1 and all die, asm:Secret_Trigger2Ringleader); room-item carriers in slot 1 (Gibdo, Like-Like; the treasure entity sits on them, which covers it); **Manhandla's base (slot 5, same type `$3C` as the hands)**; Lanmola heads (slots 5, `$A`) vs tails |
| G2 | **No per-type state byte.** `ObjState` reaches the features only through `active` (Leever, Zora, Wallmaster special-cased) and the Dodongo stun test. | repo `:787-791` | Red Wizzrobe's cycle, Blue Wizzrobe's teleport, Gohma's eye, Ganon blue/visible/brown, Patra's "orbit started", Goriya waiting for its boomerang, the monster boomerang out vs back |
| G3 | **No frame-counter phase.** | repo | Blue Wizzrobe shoots only on `FrameCounter AND $1F = 0` (asm:Wizzrobe_Blue_TryShooting); Ganon's fireball every `AND $3F = 0` (asm:Ganon_MoveAndShoot); Pols Voice moves on even frames; Gohma's eye counter on odd frames |
| G4 | **Sword-off is invisible.** `GameView::is_sword_blocked` (`$4C`, `$52E`) removes SWORD/BEAMS from the mask (`crates/zelda/src/game.rs:1048-1067`), but the mask is no network input (architecture.md "Item kinds"), and `link` has no `sword_blocked` bit or timer. | repo | every Bubble room (D7 0x0d, 0x18, 0x49; D6 0x38, 0x3a, 0x1d, 0x2d; D8 0x7c, 0x3f, 0x4e, 0x7f; D9 0x65, 0x10, 0x07, 0x17, 0x31, 0x03 ...) |
| G5 | **`hurts_on_touch` is wrong for several late types** (`:806-819`): it is `harmful && active && !dying && !clock && !stunned_dodongo && !(stun or flashing)`; `active` is `metastate == 0` for almost all. | repo vs code | Wizzrobes flashing still hurt (asm:Wizzrobe_DrawAndCheckCollisions@CheckLink); a hidden Red Wizzrobe and an odd-frame teleporting Blue one do not (asm:Wizzrobe_DrawAndCheckCollisionsIntermittently, asm:Wizzrobe_Red_State0Hidden); a returning monster boomerang does not (asm:Link_CheckMonsterTouch_NoReset); unemerged Lanmola segments (direction 0) do not (asm:Lanmola_Update); Patra children before the last appears do not (asm:Patra_UpdateChild); dying Ganon (dying count, metastate 0) does not (asm:Ganon_UpdateDying) |
| G6 | **`killable` is static and wrong for some.** `monster && !gleeok_base && !big_digdogger && !GUARD_FIRE` (`:829-830`). | repo vs code | Patra's body is untargetable while any child lives (asm:Patra_Update@LoopChildren); Manhandla's base never dies; flying Ghini (`$22`) never; Ganon only in phases; guard fires *are* killable (HP `$10`, asm:GuardFire_Update) |
| G7 | **No damage-to-Link feature.** The type embedding must learn each type's cost, and the **red ring quarters all of it mid-Dungeon 9**: one model must remap every type at once through the `ring_red` bit. | repo, asm:Damage_ObjTypeDamageToLink | Dungeon 9 families (they play both sides of the ring) |
| G8 | **Size is a point.** Positions are ObjX/ObjY; `Enemy::dims` knows Aquamentus 3x3 and the big Digdogger 4x4 tiles, the rest 2x2 (`crates/zelda/src/game.rs:150-160`). | repo | Ganon 32x32, middle at +`$10` (asm:Ganon_DrawBody); Gohma's 5 parts across 48 px (asm:Gohma_CheckCollisions); Dodongo 32 wide sideways |
| G9 | **Rupees as arrows are a hair-thin signal.** `rupees` = count/255; with 2-14 rupees and 1 rupee per arrow that is 0.008-0.05. No `arrow_usable` bit. | repo `:1179` | Gohma, Ganon, any family that may fire ARROW |
| G10 | **Velocity hides intent.** Fireballs (`$55`, `$56`) sit still `$0F` frames, already harmful, before flying (asm:Fireball_Update): vx = vy = 0 then; their facing bits are set at aim time, so the direction is there. Monster magic shots fly 3 px/frame over tiles. | code | all statue rooms, Gohma, Manhandla, Gleeok, Ganon |

### 2.3 Actions and masks today (repo)

Kinds (`crates/zelda/src/enums.rs:11-27`, `crates/zelda/src/actions.rs`): MOVE (4 dirs × 4, 8, 16, 32 frames),
WAIT, SWORD, BEAMS, BOMB, BOOMERANG (8 dirs), ARROW, CLIP, RECORDER (no direction), FOOD, CANDLE.
`compute_action_mask` (`actions.rs:313-374`): an attack is legal where the game would start it
(`GameView::weapon_ready`) and the buttons can make it (`attack::plan`: in a doorway no sideways turn); SWORD,
BEAMS, BOOMERANG, ARROW need an active enemy (or a dropped item for the pickup kinds); BOMB an enemy or a bomb wall,
then the family's `items: {bombs: MODE}`; RECORDER only where `recorder_wanted` (`actions.rs:382-388`: a dungeon
room in play with a big Digdogger that has not heard the tune). `ItemsConfig` (`crates/env/src/config.rs:352-383`):
`boomerang` and **`arrows` default true**, `recorder`, `food`, `candle` default false; **no family yaml sets
`arrows`** (grep of `specialists/`). ARROW is available with arrows, the bow and rupees > 0
(`game.rs:999-1000`). Link has had the bow since Dungeon 1's detour and **buys arrows at 0x44 before Dungeon 7**
(draft): from then on ARROW is unmasked in every family that meets an enemy, at 1 rupee a shot.

---

## 3. Monster by monster

Each entry: **Code** (what the game does), **Eyes** (observation today, and what to add), **Hands** (actions and
masks), **Reward/objective** (what is misjudged), **Where** (rooms on the draft route).

### 3.1 Blue Goriya (`$05`) and its boomerang (`$5C`) — Dungeon 7

- **Code.** HP `$50`: **3 white**, 2 magic, 2 bombs (asm:Obj_TypeHpPairs). Touch 1 heart; boomerang 1 heart
  (asm:Damage_ObjTypeDamageToLink). Walks the grid at 0.5 px/frame; on each grid point, if the longer axis distance
  to the chase target is < `$51` it faces the target along that axis and **wants to throw** (asm:Goriya_Update). The
  blue one throws **whenever** its timer is 0 and it wants to (the red one needs `Random` = `$23` or `$77`, 2/256 per
  frame) (asm:Walker_SetInputDirAndTryThrowBoomerang). Boomerang: 2.5 px/frame out to `$51` px, tiles ignored, room
  bounds turn it back; harmful **only while flying out**, tested every other frame; any shield parries it facing
  against it while idle; touching Link (hurt or parried) sparks it home (asm:Boomerang_DrawAndCheckCollision). The
  Goriya **stands still (state `$80`) until it catches it**, then waits `$30`/`$50`/`$70` frames
  (asm:Boomerang_UpdateSlowAndReturn@MonsterCatches). Killing it while the boomerang is out removes the boomerang
  but leaks one of the room's 4 counted shots (asm:Collision_CheckMonsterVsWeaponsAndLink). Boomerang stuns it;
  food attracts it (`$05` is an attracting first type, asm:Boomerang_UpdateOrFood@CheckMonsterType). Drop group 3
  (no bombs from blue Goriyas; red ones are group 2). Counts for "all dead".
- **Eyes.** Type, position, facing (its throw line), velocity (0 while waiting) and HP are there. Missing: the
  wait state (G2) — the moment to hit it (level-7.md "Monsters and boss"); the monster boomerang's out/back phase
  (G5: `hurts_on_touch` stays 1 on the way home). `blockable` is 1 for `$5C`: correct with the small shield.
- **Hands.** Sword, bombs (goriya is `free`), boomerang (stun), and **parry by standing idle facing it**: there is no
  "hold the shield" action; WAIT while facing works (the shield is passive). ARROW is unmasked after 0x44 (2.3):
  each shot is a rupee the route needs for Gohma and Ganon.
- **Reward.** `reward-block-projectile` pays a confirmed parry (architecture.md "Experimental reward..."). The 4
  room bombs of 0x1b are fought for "when the bag isn't full" (draft): a `treasure:` rule reading the bag
  (`TreasureRule::wants(bombs)`, objectives.rs:1222-1224) exists.
- **Where.** 0x69 (5, bomb wall: Link stands at the N wall ~81 frames; trace: 2-3 hearts lost there and in 0x19,
  level-7.md "Walked with three keys"), 0x59 (3 blue + 3 red, crossed), **0x49 (3 + 3 Keese + 2 Bubbles, forced; one
  Goriya spawned N of the water row and never crosses it, trace)**, 0x38 (3 + 3, crossed), 0x18 (3, dark, crossed),
  0x19 (6, bomb wall), 0x1a (3 + 3, forced for the candle push, or the block clip), 0x1b (6, fought for bombs), 0x29
  (3 + 3, bomb wall). Blind crossings cost ~11 hearts over the dungeon (trace, level-7.md).

### 3.2 Bubble (`$2B`, flashing) — Dungeons 6-9

- **Code.** Cannot be hurt, stunned or shoved: its update runs only the touch test (asm:Bubble_Update). Wanderer at
  1 px/frame, turn rate `$40`. Touch: 0 hearts but the **whole harm path**: knockback `$20` px, 48 invincible
  frames, hurt sound, help counters zeroed (asm:Link_HarmByObjType, asm:Link_BeHarmed@ResetHelp), and
  `SwordBlockedLongTimer` = `$10` → **A button dead 151-160 frames**, reloaded by each new touch (asm:Bubble_Update,
  asm:Nmi_Isr@UpdateTimers). B items still work. `$2C`/`$2D` (cure/curse) are Q2 only: **no permanent sword loss
  on this route** (data: `data/levelblock-uw-*.md`). Never counts for all-dead (asm:Secret_UpdateRoomAllDead).
- **Eyes.** Type `$2B` (hazard class, not killable; `hurts_on_touch` 1 though it costs no heart). **Missing: Link's
  own sword-off timer (G4).** A model cannot tell why SWORD vanished from its options nor for how long.
- **Hands.** The mask already removes SWORD/BEAMS while blocked (repo). What's left: bombs (scarce), boomerang
  (cannot hurt Wizzrobes, Darknuts, Pols Voices, Patra), candle (masked; `$10` damage, hurts Link).
- **Reward.** Its touch costs no health penalty (no health lost) but resets the 10th-kill bomb streak
  (`help_count` feature shows it). In Wizzrobe rooms (Wizzrobes take only sword, beam, bomb) a Bubble touch leaves
  Link 160 frames with only bombs (level-9.md "Monsters and boss").
- **Where.** D7 0x49 (2), 0x18 (2), **0x0d (3, with the Wallmasters)**; D6 0x38, 0x3a (1 each, forced fights), 0x19,
  0x1d, 0x2d (2); D8 0x7f, 0x4e, 0x3f; D9 0x65 (3), 0x10 (3, forced), 0x07 (3, forced, ring detour), 0x17, 0x31,
  0x03 (2, forced) (level docs' monster tables).

### 3.3 Wallmaster (`$27`) in 7:0x0d, with a push block and 3 Bubbles

- **Code.** HP `$20`: 1 white, 1 magic. Inside the wall (state 0): not drawn, not tested. Emerges only when the
  shared slot-1 timer is 0, Link not halted, and Link stands **exactly** at X `$20`/`$D0` with Y `$6D`-`$B4`, or Y
  `$5D`/`$BD` with X `$29`-`$C7`; at most one per `$60` frames (asm:Wallmaster_Update@CheckLinkAtWall). Path fixed
  at emergence: 7 squares at 0.375 px/frame (~299 frames): 2 out, 3 along toward where Link was, 2 back in
  (asm:Wallmaster_State1Trip). A harming touch (½ heart) **captures**: Link halted, carried; at the trip's end mode 3:
  **Link restarts in the level's start room 0x79**, room history cleared (asm:Mode3Unfurl_Init_Sub1). No capture
  while Link is invincible, the clock is held, the hand is stunned or flashing. Carrying hand untestable.
- **0x0d specifics (trace, level-7.md).** Entering through the bombed W wall puts Link at (`$20`,`$8D`) — **already
  against the wall**: the first hand is out 2 frames later. Standing still: captured after 161 frames. Stepping to X
  `$30` and back brings hands out one by one (47-66 frames at the wall each). Two killed from X `$2E` facing the
  wall; **a Bubble's touch knocked Link along the wall and the fourth hand took him**. All 5 must die before the
  block at (10,3) moves; stairs at (`$D0`,`$60`) (asm:Secret_Trigger5BlockStairs). The way back from 0x79 to 0x0d is
  13-15 room entries (needs the ladder; level-7.md "Route").
- **Eyes.** `active` = state 1 (repo `:790`); in the wall its position is stale. Missing: the shared emerge timer
  (`ObjTimer+1`), "Link is on a wall line" (derivable from Link X/Y but at pixel exactness), the Bubble timer (G4),
  and capture state (Link `halted` is set; the hand's capture timer is not).
- **Hands.** Sword toward the wall, boomerang stun (prevents capture). The critic has wall-tile shaping
  (`penalty-fighting-wallmaster`, `penalty-moved-onto-wallmaster`, `reward-moved-off-wallmaster`, critic.rs:396-430)
  that **punishes standing on the wall tiles** — but in 0x0d standing at the wall is how the hands are lured
  (inferred: the shaping fights the strategy).
- **Reward/objective.** `wallmastered` is a hard failure in training (end_conditions.rs:217-224); in a chain the
  drag sends the run 15 rooms back (inferred: the chain's handling of a drag mid-dungeon is not documented; walk it).
  The push block is the exit (stairs), FIGHT forced by `push_requires_combat` (objectives.rs:1274-1281).
- **Where.** 7:0x0d only (5 hands). No Wallmaster in levels 6, 8, 9 (level docs).

### 3.4 Digdogger in Dungeon 7: big `$38` (3 children) in 0x1c and 0x39; small `$18`

- **Code.** Big: attribute bit `$20`, **no weapon touches it** (asm:Collision_CheckMonsterVsWeaponsAndLink), 2 hearts
  a touch at four 16x16 corners, 0.25-0.5 px/frame 8-way, no tiles. The recorder in its room: 152-frame freeze,
  `$40`-frame shrink inside it, split on the first play frame (B+152, trace in D5), three small ones at slot 1's
  position, random 8-way directions, each with a spawn cloud first (asm:Digdogger_HandleFlute@LoopMakeChild). Small:
  HP `$80` = **4 white** (2 magic), 2 hearts a touch, speed ramps to **1.25-1.5 px/frame** over `$141` frames, 8-way,
  no tiles, turns 45° every `$10` frames (asm:Digdogger_SpeedUp, asm:Digdogger_ChangeSpeedAndTurn). No knockback.
  Boomerang stuns. The first small kill saves the room as "all dead" for good (non-recurring,
  asm:RoomHistory_SaveKillCountUW). In 0x1c four statues shoot meanwhile (layout `$24`).
- **0x1c (trace, level-7.md):** from the W door B is not read in the doorway; after 5 frames of Right the tune
  starts; the three small ones exist 217 frames after the press; the N shutter opens when the third dies (they were
  put into the death state, not fought).
- **0x39:** a big one beside the door row; crossing S→W touched Link in 17/30 tries straight, 1/30 by row 1
  (trace).
- **Eyes.** Big vs small by type; `killable` 0 for the big (repo `:828-830`). Size 32x32 for the big is known to the
  objective (`dims`), not to the network (G8). Three fast 8-way 2-heart chasers: velocity is there.
- **Hands.** `recorder_wanted` offers RECORDER in **both** 0x1c and 0x39 (any big Digdogger, not heard yet) to a
  family with `recorder: true` (repo). In 0x39 splitting it makes three fast 2-heart monsters in a crossing room
  (inferred: must not). boss-digdogger trained on D5's single child (`$39` → 1 child, boss-digdogger.yaml).
- **Reward.** Small ones are bosses (`DIGDOGGER_CHILD` in the boss list, enums.rs:333-352): every hit
  `reward-boss-hit` 0.75; the split pays `reward-recorder-split` (critic.rs:572).
- **Where.** 0x1c (forced), 0x39 (crossed). 0x6c off the route.

### 3.5 Grumble Goriya (`$36`) — 7:0x28

- **Code.** Link halted while "GRUMBLE,GRUMBLE..." types (121-123 frames, trace); then it waits for **any food in
  slot `$0F`** (state bit 7), anywhere in the room (asm:Grumble_State1WaitForFood); Link halted `$40` frames,
  `InvFood` = 0, the Goriya gone for good (room flag `$10`) (asm:Grumble_State3TakeFood). While it stands, Link cannot
  move up past Y `$8D` (asm:Person_BlockLinkMovingUp). **Hitting it turns on two fireball sources** at (`$48`,`$82`)
  and (`$A8`,`$86`) until Link leaves (asm:Person_CheckCollisions). B is not read in the S doorway facing out:
  walk 6 frames up first (trace). A boomerang out blocks FOOD (same slot) (asm:Food_Wield).
- **Eyes.** Type `$36` is class Other (no entity flags but presence). `halted` shows the text phase;
  `food_usable` the mask.
- **Hands.** FOOD (round 24) — off in every family; needs a family or a route step that turns it on only here.
  Sword toward it: the mask allows attacks only with an active enemy or item (`actions.rs:348-356`); 0x28 has no
  enemy, so SWORD is masked (repo) — safe unless a drop lies in the room.
- **Reward.** `reward-gained-food` exists; nothing pays giving it. The objective needs a step "give food" (draft
  "What is not built").

### 3.6 Aquamentus in 7:0x2a (white sword)

Existing family (boss-aquamentus, D1; `d2first` takes the white sword before Dungeon 1, so the grade is the same).
Code: HP `$60` (**3 white**), only its 16x16 head box at (X+8, Y+8) is hit or touches; boomerang and fire masked;
three `$55` fireballs every `$70`-`$7F`/`$F0`-`$FF` frames, ½ heart, unblockable here (asm:Aquamentus_Shoot). 7:0x2a
has blocks on the east around its row; Link enters from the bombed W wall at (`$20`,`$8D`) with the open west half
between them (trace, level-7.md). Not fought in the walk (death state).

### 3.7 Wizzrobes: Blue (`$23`), Red (`$24`), magic shots `$58`, `$59` — Dungeons 6 and 9

The dominant monster of the late game: D6 has them in 12 rooms, D9 in 20 (level docs).

- **Weapons (code).** Their own collision routine tests only slots `$0E` (beam/magic shot), `$10`/`$11`
  (bomb/fire), `$0D` (sword); mask `$F6` → **only sword, beam, bomb blast hurt**; arrows, boomerang, rod pass
  through, no stun; no knockback (asm:Wizzrobe_DrawAndCheckCollisions). Blue HP `$A0` = **3 magic**, 3 bombs; Red
  `$40` = **1 magic**, 1 bomb. After a hit ~32 frames untestable by weapons **but their touch still hurts**.
- **Blue (code).** Walk phase `Random OR $70` frames at 0.5 px/frame; stands 14 frames; teleports to a random
  diagonal spot `$20` px off on both axes if walkable, `$20` frames, ignoring tiles, **neither drawn nor tested on
  odd-distance frames** (asm:Wizzrobe_Blue_WalkOrTeleport, asm:Wizzrobe_DrawAndCheckCollisionsIntermittently).
  Walking into a block or water teleports it through. Turns toward Link every `$40` steps alternating axes. Shoots
  `$58` (2 hearts, 3 px/frame, over tiles) only when not teleporting, on `FrameCounter AND $1F = 0`, **already
  facing Link**, Link on the same 16-px row (Y high nibbles equal) or with Link's X = its X rounded down to `$10`
  (asm:Wizzrobe_Blue_TryShooting, asm:Wizzrobe_Blue_TryShooting_CheckColumn). Touch 2 hearts.
- **Red (code).** 208-frame cycle, `ObjState` counting down from `$FF`: 1 frame choosing, 63 fading in, **64 solid**,
  16 fading out, 64 hidden (asm:Wizzrobe_Red_Update). Appears 2-5 squares from Link in a cardinal direction,
  **facing him**, square-aligned (Y `$5D`-`$C3`, walkable tile; retried every frame) (asm:Wizzrobe_Red_State3Appear).
  **One `$59` shot per cycle at state `$B0`, the 16th solid frame, along its facing: 4 hearts**
  (asm:Wizzrobe_Red_State2Solid). Hittable on the 64 solid frames and every other fading frame; hidden: nothing.
  Its first update already chooses a place near Link: its list position is unused. Touch 1 heart.
- **Shots** need a free slot and < 4 counted shots; the sound plays even when refused (asm:Wizzrobe_ShootMagicShot).
  Link's weapons cannot touch them. A Like-Like's hold makes Link untouchable by them (asm:Link_CheckMonsterTouch_NoReset).
- **Ringleader.** D6 0x78 (5 Red): killing the one in slot 1 kills all (asm:Secret_Trigger2Ringleader); the route
  only crosses it, and five Red Wizzrobes appear on the door row west of Link within 40 frames (trace, level-6.md).
- **Eyes today.** Type, position (also of a **hidden** Red one: its last spot), facing (the shot line), velocity (a
  teleport is 1 px/frame diagonal), HP, immune bits (boomerang, arrow, magic shot, fire set: correct). Wrong:
  `active` and `hurts_on_touch` 1 while a Red one is hidden or a Blue one is on an odd teleport frame, and 0 while one
  flashes after a hit though its touch hurts (G5). Missing: **the Red one's cycle position** (frames to its shot,
  solid or not: `ObjState`), the Blue one's teleport flag, the 32-frame shot window phase (G3), "Link is on this
  Wizzrobe's shot line now" (pixel-exact row/column tests the network sees only as floats /176, /255), the
  ringleader (G1), the sword-off timer (G4: Bubbles share most Wizzrobe rooms).
- **Hands.** SWORD, BEAMS (full hearts only), BOMB (`$40`; scarce: 7 cheated on the route). Boomerang and arrows
  are useless: **ARROW is legal and wastes a rupee** (2.3). Dodging needs off-line positioning: MOVE 4-32 frames on
  the 8-px grid is fine.
- **Reward/objective.** FIGHT targets are the enemy tiles (objectives.rs:1233-1248); in a fight room PBRS is an
  approach reward toward moving enemies (architecture.md "PBRS") — **approaching a Wizzrobe along its row or column
  is exactly where its shot lands**, and a Red one re-places itself relative to Link each cycle (inferred). Damage
  penalty `penalty-lost-health` = 0.25 per half heart, capped at 1.0 (critic.rs:385-391): a 4-heart Red shot costs
  the cap, the same as a 2-heart Blue one (inferred: the critic under-weights the worst hit of the game 2:1).
- **Where (draft).** Forced: D6 0x38, 0x3a (2 Blue + 2 Red + 3 Like-Likes + 1 Bubble each); D9 0x05, 0x20, 0x10
  (3 Blue + 2 Red), 0x30 (2 + 2 among 4 traps), 0x07 (ring detour, 3 + 2, 3 Bubbles). Crossed: D6 0x78 (5 Red),
  0x28 (bomb wall: Link stands at (`$D0`,`$9D`) facing up for the bomb, ~`$50` frames), 0x29 (dark water ring, key),
  0x2c (with 4 traps); D9 0x65 (3 + 2 + 3 Bubbles, bomb N with the ladder), 0x15 (3 Blue), 0x17, 0x31 (bomb walls),
  0x04 (traps, bomb wall). First-quest.md: "the fights and the Wizzrobe rooms are where a run will be lost"; blind
  crossings in D6 cost 12-22 hearts (draft).

### 3.8 Like-Like (`$17`) — Dungeons 6 and 9

- **Code.** HP `$90` = **3 magic**, 3 bombs. Wanderer 0.5 px/frame; knocked back before a capture. Capture on a
  harming touch (1 heart): it jumps onto Link, `IsLinkParalyzed` = 1 (asm:LikeLike_Update). Held: **direction input
  discarded, A and B still work**; **no monster or shot can touch Link** (asm:Link_CheckMonsterTouch_NoReset). The
  magic shield is taken after 95 frames (no shield on this route: moot). Released when it dies or the room has no
  counting monster left (asm:Secret_UpdateRoomAllDead). Boomerang stuns it (a stunned one cannot capture). No drop;
  counts for all-dead. In slot 1 it carries the room item.
- **Trace.** D6 0x19 walked straight: caught in the middle, five magic-sword swings to kill it (level-6.md); D9 0x17
  held Link on the square before the N wall while his bomb burned; D9 0x41 caught him in the middle lane
  (level-9.md).
- **Eyes.** `paralyzed` bit in `link`; the attack planner knows the held facing (attack.rs:240-256). Missing: the
  capture timer (not needed without the shield).
- **Hands.** Sword/beam/bomb, boomerang stun. Being held is a safe spot from Wizzrobe shots (inferred from the code:
  worth knowing, not worth seeking).
- **Where.** D6 0x38, 0x3a (forced), 0x19, 0x1d (crossed); D9 0x14 (5, inside a block spiral), 0x51 (6), 0x41 (4
  + traps, one-way), 0x31 (3), 0x03 (2, forced), 0x17 (3, ring detour).

### 3.9 Vire (`$12`) and its red Keese (`$1C`) — D6 0x39, 0x2d

- **Code.** HP `$40`; **the magic sword (or beam, bomb, silver arrow) kills it outright, no split** (asm:Vire_CheckCollisions).
  Weaker hits split it into two red Keese after a 4-px/frame slide. Red Keese die to anything (HP 0), add no
  `RoomKillCount`, so a split room is never banked as cleared (asm:Obj_EndMetastate). Touch 1 heart (Keese ½).
- With the magic sword from D6 on, the vire family's split-handling stops mattering; its states were all white
  sword (inferred). Dark rooms change only the picture (level-6.md). Forced: 0x39 (5, the ladder over the water
  column at X `$A0`); 0x2d (3 + 2 Bubbles, the key level 9 needs). Trace: one Vire killed with a magic swing, no split.

### 3.10 Gohma (`$34` red in 6:0x1c; `$33` blue off-route in D8)

- **Code.** Mask `$FB`: **only an arrow, flying up (direction 8), overlapping part 3 or 4 (the eye), while eye state
  = 3** damages (asm:Gohma_HandleWeaponCollision). Red HP `$20`: **one arrow** (wooden or silver). Five parts 8 px
  apart from ObjX−`$10` to ObjX+`$10`, tested left to right: an arrow's hit point must be ObjX−1..ObjX+`$0E`; Link
  facing up shoots from his X+3, so **Link X = ObjX−`$0A`..ObjX+5**; on the 8-px turn grid only X = ObjX−8 and ObjX
  (trace: kills from X `$78` and `$80` with Gohma at `$80`). Any part touched uses the arrow up. Eye: closed
  (states 0/1), `$11` frames half (2), **`$60` frames open (3)**, `$0F` half; openings start 384-510 frames apart,
  the first on the first update (asm:Gohma_Update@AnimateEye, asm:Gohma_Update@DecNextOpenEyeCounter). Paces `$20`
  px left/right/down and back from (`$80`,`$70`) at 0.5 px/frame, always returning. Fireball `$56` every 65 frames,
  1 heart, unblockable; touch 2 hearts. No clock test. Trace: eye open ~16-110 frames after Link can move, next
  ~frame 470; shot when Gohma is back at X `$80` (level-6.md).
- **Eyes.** Missing: **eye state and its timers** (`Gohma_ObjEyeState`, the open timer, the next-open counter) (G2),
  its 48-px width (G8), whether Gohma is at its home X. The arrow's flight is only `arrow_active` (a bit).
- **Hands.** ARROW N (1 rupee). Sword, beam, bomb, boomerang: parry sound only. Every miss is a rupee from the
  2-14 that must also pay Ganon (draft ledger) and ≥ 384 frames.
- **Reward/objective.** HP `$20` → dies → a credited boss hit + kill (change.rs: HP drop or death sparkle) (repo).
  Waiting is right here (up to ~500 frames between openings): room-stalling penalty after 150 steps without
  progress (CLAUDE.md "Game mechanics"), `no_damage_steps` 600 actions (architecture.md) (inferred: tolerable, but
  a wasted arrow is not recoverable on this rupee budget).
- **Where.** 6:0x1c (boss, forced). D8's 0x1e/0x4d Gohmas are off the draft route (no magic key).

### 3.11 Blue Darknut (`$0C`) with the magic sword — Dungeon 8

Existing family (darknut: D3 red, D5 blue, white sword). Code: HP `$80` = **2 magic**, 2 bombs; touch 2 hearts;
0.625 px/frame; mask `$F6` (sword class and bomb only); **a hit whose direction is exactly opposite its facing is
parried** — no damage, no knockback, no invincibility (asm:Damage_HandleWeaponHit@CheckDarknut); a bomb placed
facing it head-on is parried too (stored direction). Never stunned. Facing is in the observation (repo): that is the
shield side. D8 forced: 0x5e (5 blue, 4 corner statues), 0x3e (6 blue, 2 statues on the door row); crossed: 0x4e
(1 blue, 2 red, 3 Gibdos, 2 Bubbles), 0x3f (one-way, 1 blue, 2 red, 3 Gibdos, 2 Bubbles: 1-2 touches per crossing,
trace). Banked kills: a Darknut room can be fought in parts through the door Link came by (trace, level-8.md) — a
strategy a one-room episode never sees (inferred).

### 3.12 Gibdo (`$30`) and Pols Voice (`$16`) — Dungeon 8

Existing families (gibdo, pols-voice, D5). With the magic sword: Gibdo `$70` = 2 hits, Pols Voice `$A0` = 3 hits,
both 2 hearts a touch. **Any arrow kills a Pols Voice outright, past the mask, and flies on to the next**
(asm:Collision_CheckShotVsMonster) — an arrow tempts the model in 8:0x4c (8 Pols Voices, a 180-210-frame walk plus
a bomb at the N wall, trace) and 0x1f; each costs a rupee. Bombs do nothing to Pols Voices. A sword hit makes a Pols
Voice harmless ~32 frames (the family's learned trick, pols-voice.yaml).

### 3.13 Manhandla crossed (`$3C`) — 8:0x6e

- **Code.** Four hands (slots 1-4) and a base (slot 5), all type `$3C`; hand HP `$40` = 1 magic; the base takes
  hits but cannot die; each dead hand +0.5 px/frame for all (asm:Manhandla_Update@LoopSegment); 8-way, no tiles;
  fireballs `$56` 1 heart unblockable, hand touch 1 heart (asm:Manhandla_Update). A bomb within 8 px of the base on
  both axes kills all four hands at once.
- **0x6e (trace):** the route bombs the N wall with it alive; a bomb laid at the N wall took three hands off once
  (level-8.md) — which leaves the survivor at 2 px/frame (inferred).
- **Eyes.** **The base and the hands share one type and the network has no slot identity (G1)**: `killable` is 1 for
  the base. Hits on the base: the first one drops its HP nibble `$4`→0 and is credited as a **boss hit** (Manhandla is
  in the boss list); later ones change nothing visible (inferred from change.rs's HP-drop rule and
  asm:Manhandla_CheckCollisions).
- **Where.** Only crossed on the draft route (0x6e); 0x7d and 0x2e are off it (no book, no magic key).

### 3.14 Gleeok with four heads (`$45`) — 8:0x3c

- **Code.** Mask `$FE`: sword, beam, rod only (asm:Gleeok_Init). One HP byte for all attached heads: first head
  falls at `$A0` total, each later at `$60`: **magic 3 + 2 + 2 + 2 = 9 hits**, ≥ 32 frames apart (one shared
  invincibility timer) (asm:Gleeok_CheckSegmentCollisions; trace: 9 hits, heads at hits 3, 5, 7, 9). A cut head
  becomes a **flying head `$46`**: chases (208/256 decisions), never slows, cannot die, 1 heart a touch, shoots; **up
  to three fly at once**. One Gleeok fireball at a time (slot `$0B`), 1 heart, unblockable. Attached head touch
  3/8 heart; neck base 1 heart. Necks move once every 4 frames, heads 1 px per move within `$0B` px per segment.
  S and W are bombable walls, not shutters: Link can leave (level-8.md).
- **Eyes.** Heads are synthesized in slots 2-5 with the shared HP and timer (repo `:849-865`); base and flying heads
  are hazards (game.rs:372-376). Four necks fit the 12 slots (heads idx 1-4, flying 6-9, fireball 10) (repo).
- **Hands.** boss-gleeok is `bombs: never` (correct). Beam at full hearts.
- **Where.** 8:0x3c (forced boss). The family trained on 2 heads, white sword, 1 flying head max (D4) (repo).

### 3.15 Lanmola, red (`$3A`) — 9:0x55

- **Code.** Two five-segment worms in slots 1-`$A` (heads `$05`, `$0A`); each segment HP `$20` (1 magic), touch 2
  hearts, **1 px/frame** (blue `$3B`, off-route, 2 px). Any segment kill removes that worm's tail; the head dies
  last: **10 segment kills** (asm:Lanmola_Update@FindTail). No knockback, hits never turn it. Heads choose every 16
  px: `Random < $80` toward Link — the horizontal way **only if it equals the way Link faces**, else vertical even
  if reversing (asm:Lanmola_UpdateHead@TurnTowardLink). Segments with direction 0 (not yet emerged from (`$40`,`$8D`))
  are not moved, drawn or tested. Boomerang stuns a segment (it keeps moving, but its touch is harmless). With 10
  kills `HelpDropCount` reaches 10 at the second head: forced drop.
- **0x55 route step (level-9.md):** clear both worms, stand at (`$60`,`$9D`), hold Up `$10` frames, the block slides,
  stairs at (`$80`,`$8D`).
- **Eyes.** Missing: emerged or not (G5: all ten start "active" at one point); head vs tail (G1); the head's turn
  rule depends on Link's facing (the facing is in `link`).
- **Disagreement to walk:** level-9.md says the worms are back on every later entry ("never saved as cleared") and
  after a death. Mechanics: the room's list count is 1 (ids `$32`-`$61` count 1, asm:Room_EnterCommon) and
  `RoomObjCount` is set to 8, while kills bank as 2 in `LevelKillCounts`; the entry rule "count minus banked kills"
  would give none while the room is in the history, and CONTINUE keeps `LevelKillCounts` (mechanics.md "Death")
  (inferred). One of the two readings is wrong. **Walked (round 29, `d9-fights-walks.md` §1): the mechanics
  reading is right.** One worm killed, out by the S wall and back: the room is empty and `IsRoomAllDead` is set (20
  of 20); cleared and written out of the room history: still empty (20 of 20); three segments killed, no head:
  ten segments again. After a death it was not walked.

### 3.16 Patra (`$47`/`$48`) and its eight children (`$25`/`$26`) — 9:0x16, 0x61, 0x52 (0x27 crossed)

- **Code.** Mask `$FE` on body and children: **sword and beam only** (arrows, boomerang, bombs, fire do nothing)
  (asm:Patra_Init). Children appear one at a time straight above Patra, each when slot 2's angle reaches its entry;
  **no child tests anything (weapons or Link) and none is drawn until the last one has appeared** and set Patra's
  state to 1 (asm:Patra_UpdateChild). **While any child lives, Patra's body is tested against Link only**
  (asm:Patra_Update@LoopChildren). Child HP `$60` = **2 magic**; body `$B0` = **3 magic**: 16 + 3 = 19 hits, each
  target with ~32 untestable frames after a hit. Touch of body or child 2 hearts. Body flies to 0.25 px/frame
  (chase 3/4), children orbit at radius `$2C` (`$47`) or `$18` (`$48`), ~73/85 frames per orbit, the pattern
  (circle/ellipse, small/wide) flips at most every 255 frames (first flip 63/74 frames after init, trace). No clock
  test. Children become dead dummies (no drop, but `WorldKillCount`/`HelpDropCount` +1 each).
- **Where.** 0x61 (`$47`, forced, key in the open first), 0x52 (`$47`, forced: its N shutter into Ganon's room),
  0x16 (`$48`, **killed for its 4 bombs** — the owner, draft), 0x27 (`$48`, crossed with a bomb at the N wall, ring
  detour). Not fought in any walk (level-9.md).
- **Eyes.** Missing: whether children are live (Patra's state) (G5/G2); body `killable` 1 while children live (G6);
  orbit phase and pattern index (positions and velocities show the orbit; the next flip is hidden).
- **Walked (round 29, `d9-fights-walks.md` §2).** Children live from frame 66 (`$47`) / 77 (`$48`); the second
  pattern is eight flat ellipses passing through the body (`$47`) or a ring that swells 21 → 85 px every orbit
  (`$48`), and both fights begin in it; **the child in slot 3 dead: no flip ever again**; arrows, bombs, boomerang 0
  of 780; the beam hurts the children. By rule: 0x52 20 of 20 at 0.70 hearts, 0x61 20 of 20 at 2.50, 0x16 17 of 20 at
  8.65 without the ring and 20 of 20 at 2.33 with it. Every model family: no kill.
- **Reward.** `PATRA` (`$47`) is in the boss list, **`PATRA2` (`$48`) is not** (enums.rs:333-352): 0x16's body hits
  pay `reward-hit` 0.25 where 0x61's pay `reward-boss-hit` 0.75 (repo). Children are never bosses.

### 3.17 Ganon (`$3E`) — 9:0x42

- **Code.** Entry scene: Link halted `$40` + `$C0` frames (~270, trace); Ganon then at (`$50`,`$5D`) (walked, round 29: on the room's
  spawn spot, one of seven seen in 20 entries; `d9-fights-walks.md` §4.1); both shutters
  shut until the Triforce of Power is taken (asm:Secret_Trigger3PowerTriforce). **Blue phase:** not drawn, moves 1
  px/frame with the Blue Wizzrobe's step routine (passes blocks), fireball `$56` every 64 frames (1 heart,
  unblockable). Only the sword (slot `$0D`, state 2) and an arrow (slot `$12`) are tested — **beam, bombs, rod,
  boomerang never** (asm:Ganon_CheckCollisions). The sword hits only while he is invisible; each hit shows him `$40`
  frames, unhurtable, then he jumps to Y `$A0`, X `$30` or `$B0` and vanishes. **4 magic hits** (HP `$F0` as a
  counter). Touch **4 hearts** (middle at X+`$10`,Y+`$10`, within 9 px), tested every fight frame, invisible too
  (asm:Ganon_ScenePhase2_Fight). **Brown phase:** after the 4th hit, ~510 frames standing still, no shots,
  flickering the last ~96; only an arrow in flight with `InvArrow` = 2 (silver) kills; a wooden arrow passes; if the
  time runs out he is blue again with full count (asm:Ganon_UpdateBrownState). Trace: hits possible every 71 frames
  (Link 8 px left of Ganon's X and 8 below his Y, facing right); one silver arrow along his row from `$30` px killed
  him 8 frames later (level-9.md).
- **Death bookkeeping (code).** The arrow starts `Ganon_ObjDyingCount`; his object is never destroyed, metastate
  stays 0, HP stays `$F0`; at `$A0` the room item (Triforce of Power) appears at the ashes and `RoomKillCount` +1
  (asm:Ganon_UpdateDying). The 4th sword "kill" sets the metastate and is undone in the same frame, HP back to `$F0`
  (asm:Ganon_CheckCollisions@UndoDeath).
- **Eyes.** **His invisible position is in ObjX/ObjY and so in the observation** (repo: positions are read raw) — the
  model sees what a player cannot (inferred: an advantage). Missing: blue/visible/brown phase and the brown time left
  (G2), his 32x32 size and touch middle (G8), the frame phase for his fireball (G3); the silver arrow is
  `arrow_silver` in `link` (there).
- **Hands.** SWORD while invisible; ARROW (silver, 1 rupee) in brown. BEAMS do nothing (harmless waste). BOMB does
  nothing (would waste a bomb; the bag is empty by then on the draft's ledger).
- **Reward/objective (inferred from change.rs's rules).** Sword hits 1-3 drop HP and pay `reward-boss-hit`; the 4th
  is reset to `$F0` within the step and is **not** credited; **the killing arrow changes no HP and no metastate:
  no hit, no kill is observed**; `room_state` keeps counting Ganon as a living monster after death
  (`is_living_monster`, room_state.rs:256-259), and `hurts_on_touch` stays 1 on the ashes. FIGHT targets his 2x2 at
  ObjX — the approach reward pulls Link into the 4-heart touch box. The success has to come from the Triforce of
  Power (item `$0E`, not a level piece: what the critic pays for it is unverified) and the N shutter.
- **Where.** 0x42; a dead end without bow, silver arrow and a rupee (level-9.md).

### 3.18 Zelda (`$37`) and the guard fires (`$3F`) — 9:0x32

Code: the game ends when Link's X is `$70`-`$80` and Y **exactly `$95`**, fires dead or not; next frame mode `$13`
(asm:Zelda_Update, asm:Zelda_State1WaitForEnding). Fires at (`$70`,`$9D`), (`$80`,`$9D`) on the middle path, ½ heart,
HP `$10`, any damaging weapon. Trace: one sword stroke each from below, or walk up between them for one fire touch.
Observation: guard fires are `killable` 0 and excluded from living monsters (repo) — fine, they need not die.
Needs an "end of game" step (draft).

### 3.19 Dungeon persons, standing fires, the man of 9:0x66

Persons (`$4B`-`$52`) halt Link while text types (6 frames per character), block moving up past Y `$8D`; hitting one
starts fireballs from (`$48`,`$82`)/(`$A8`,`$86`) until Link leaves (asm:Person_CheckCollisions). Standing fires `$40`
(hazard class, repo): ½ heart, no shield, harmless while Link is halted. 9:0x66: with all 8 pieces the man is gone
on every entry, his fires stay (trace). On the route: 9:0x06 (talks; bomb its W wall from below the fire, Y `$9D`).

### 3.20 Traps (`$49` from `$49`/`$4A` generators)

Code: no weapon tested; 1 heart; charge when Link is within `$0E` of its row (else column) and the direction is one
its slot allows, at 1.75 px/frame toward the middle, creep back at 0.5, sense nothing until home
(asm:Trap_Update_Full). Trace: D6 0x48 crossed at X `$78` holding Up in 110 frames unhurt; 30 frames standing on the
door row cost 2 hearts, 90 frames caught between a pair 10-12 hearts in 600 frames; D6 0x2c: walk west off the
doorway at once; D9 0x30: the stairs appear on the top-right trap's home square (lure it, step on as it creeps
back). Eyes: position and velocity show charging/returning; home square implicit. Family trap-room exists (D1, D3,
D4). Rooms: D6 0x48 (6), 0x2c (4 + Wizzrobes); D9 0x41 (4 + Like-Likes, one-way), 0x30 (4 + Wizzrobes, forced),
0x04 (4 + Wizzrobes).

### 3.21 Statue fireballs (`$55`)

Code: layout `$24` (4 sources) or `$23` (2); per source a random reload of `$50`/`$80`/`$F0`/`$60` frames; **no shot
within `$18` px of a source**; none while the highest free slot is below `$06` (asm:Statue_Update). ½ heart,
unblockable here. Aims once at Link, waits `$0F` frames harmful, 1.25-1.75 px/frame. On the route: D7 0x69 (2), 0x39
(4), 0x1c (4), 0x0c (2); D6 0x58 (4, forced Keese room); D8 0x5e (4), 0x3e (2); D9 none on the route (0x37, 0x53 off)
(level docs).

### 3.22 Overworld monsters of the late legs (data: `levelblock-ow.md`)

| Room | Monsters | Leg |
|---|---|---|
| 0x0b | Red Leever ×1 | D5→D7 start |
| 0x45 | Peahat ×1 (whirlwind drop) | D5→D7 |
| 0x55, 0x54, 0x44, 0x3c, 0x4c, 0x37, 0x38 | Octoroks (0x44 shop: arrows 80) | D5→D7, D6→D8, D8→D9 |
| 0x34 | Blue Leever ×4; **food shop under the Armos at X `$40`** | D5→D7 |
| 0x53, 0x52, 0x51, 0x4d, 0x5d, 0x6d | Moblins (0x6d tree, level 8) | D5→D7, D7→D6, D6→D8 |
| 0x42 | red Moblin ×1, the pond (recorder) | D5→D7 end |
| **0x50** | **Red Lynel ×5** | D7→D6 (after the 0x51 wrap) |
| **0x40, 0x41, 0x31, 0x21** | **Ghini ×1 each; gravestones; 0x21 the magic sword** | D7→D6 |
| **0x32** | 2 Peahats, **2 Red + 2 Blue Lynels** | D7→D6 |
| 0x22 | Red Lynel ×1 (level 6) | D7→D6 end, D6→D8 start |
| 0x48, 0x47 | Red Leevers; 0x47 tree → take-any heart container | D8→D9 |
| 0x28, 0x27 | Peahats ×6, Zora | D8→D9 |
| 0x17 | Boulders + Zora (ladder over the river) | D8→D9 |
| **0x07, 0x06, 0x05** | **Red Lynel ×1; 2 Peahats + 2 Red + 2 Blue Lynels; 2 Blue + 2 Red Lynels + 2 Leevers**; 0x05 rock wall (bomb) | D8→D9 end |

- **Lynels (code).** Blue `$60` (3 white, 2 magic), touch 2; red `$40` (2 white, 1 magic), touch 1; sword shot `$57`
  **2 hearts, unblockable without the magic shield**, 3 px/frame over tiles (asm:MonsterShot_Update). Blue re-arms
  at every grid point within `$51` px and shoots every square; red only on `Random ≥ $F8` frames
  (asm:Shooter_RunShootTimer). Being hit cancels a pending shot. ow-lynel family exists (0x0a, 0x50, 0x60).
- **Ghini (code).** HP `$90` (5 white, 3 magic), 1 heart; wanderer 0.5 px/frame. **Touching a gravestone square
  (tiles `$BC`-`$BF`) at a grid point with a direction held releases a flying Ghini** (`$22`) on it: immune to
  everything, 1 heart (also while fading in on drawn frames), flies 0.625 px/frame 8-way, no limit but free slots
  (asm:Spawn_CheckTouchedArmosOrGravestone, asm:Ghini_UpdateFlying). **Killing the room's one Ghini kills every flying
  one** (asm:Ghini_Update). The magic sword's stone at tile (18,10) moves up or down (overworld.md); a push is made
  by walking into it (asm:Rock_Update: X equal, vertical input), so it likely releases a flying Ghini at each try
  (inferred; "Ghinis kept knocking Link off it", overworld.md). Observation: flying Ghini type `$22` is `killable` 1
  (G6: wrong). No family.
- **Armos (code).** Woken by walking into an armos square; fades `$3F` frames (touch hurts on drawn frames, not
  hittable), then HP `$30` (2 white), 1 heart, speed **0.5 or 1.5 px/frame by one random byte**
  (asm:Armos_InitOrFlyingGhiniFadeIn@UseChosenTile). 0x34's stairs (food shop) are under the Armos at X `$40`, Y
  `$80` (asm:Armos_SecretXs); they are not saved and appear each time it is woken. No family, no step type.

---

## 4. Damage to Link per touch and shot, against the hearts Link has there

Damage (asm:Damage_ObjTypeDamageToLink) and, in brackets, **touches from full health to death**, ⌈H/d⌉ with H the
max hearts of that stretch (inferred from asm:Link_BeHarmed@HandleDied; ignores drops and the 1/256 borrow).

| Source | No ring | Red ring (÷4) | D7 (H 11) | Legs (12-14) | D6 (12) | D8 (13) | D9 before ring (15) | D9 with ring (15) |
|---|---|---|---|---|---|---|---|---|
| Ganon touch `$3E` | 4 | 1 | | | | | | 1 [15] |
| Red Wizzrobe shot `$59` | 4 | 1 | | | 4 [3] | | 4 [4] | 1 [15] |
| Blue Wizzrobe touch / shot `$58` | 2 | ½ | | | 2 [6] | | 2 [8] | ½ [30] |
| Blue Lynel touch, Lynel sword shot `$57` | 2 | ½ | | 2 [6-7] | | | | |
| Blue Darknut, Gibdo, Pols Voice | 2 | ½ | | | | 2 [7] | | |
| Digdogger (big, small) | 2 | ½ | 2 [6] | | | | | |
| Gohma touch | 2 | ½ | | | 2 [6] | | | |
| Lanmola segment, Patra and children | 2 | ½ | | | | | 2 [8] | ½ [30] |
| Red Wizzrobe touch, Like-Like (then held), Vire, Zol | 1 | ¼ | | | 1 [12] | | 1 [15] | ¼ [60] |
| Blue Goriya touch, monster boomerang `$5C` (parryable) | 1 | ¼ | 1 [11] | | | | | |
| Red Darknut, Manhandla, Gleeok base and flying head | 1 | ¼ | | | | 1 [13] | | |
| Fireball `$56` (Gohma, Manhandla, Gleeok, Ganon) | 1 | ¼ | | | 1 [12] | 1 [13] | | ¼ [60] |
| Trap, Dodongo, Aquamentus touch, Ghini, flying Ghini, Armos, red Lynel, Blue Leever | 1 | ¼ | 1 [11] | 1 [12-14] | 1 [12] | | 1 [15] | ¼ [60] |
| Gleeok attached head | 3/8 | 3/32 | | | | 3/8 [35] | | |
| Red Goriya, Keese, red Keese, Wallmaster (+ capture), statue/Aquamentus/Zora fireball `$55`, Gel, guard and standing fire, Moblin and arrow, Octorok and rock, Peahat, Red Leever, Zora, Boulder, Tektite, Stalfos, own candle fire | ½ | ⅛ | ½ [22] | ½ [24-28] | ½ [24] | ½ [26] | ½ [30] | ⅛ [120] |
| Bubble | 0 (sword off 151-160 frames, knockback, help reset) | 0 | | | | | | |

- The red ring shifts the 16-bit damage right twice; half a heart becomes `$20` (asm:Link_BeHarmed, items.md "Ring").
- The single worst spot of the game: **D9 before the red ring** (0x65, 0x15, 0x17, 0x07: Red Wizzrobe shots at 4
  of 15 hearts, no refill since the D8 piece) (inferred from the table and the route order).
- D6 at 12 hearts: **three Red Wizzrobe shots kill from full**; 0x38/0x3a hold 2 Red + 2 Blue each (code + draft).

### Hits to kill, by the sword Link holds there

| Monster | HP | Where | Sword | Hits | Other ways |
|---|---|---|---|---|---|
| Blue Goriya | `$50` | D7 | white | 3 | bomb 2 |
| Red Goriya | `$30` | D7 | white | 2 | bomb 1 |
| Wallmaster | `$20` | D7 | white | 1 | |
| Small Digdogger | `$80` | D7 0x1c ×3 | white | 4 each (12) | bomb 2 |
| Aquamentus | `$60` | D7 | white | 3 | bomb 2 |
| Red / Blue Lynel | `$40` / `$60` | legs | white → magic | 2/3 → 1/2 | |
| Ghini | `$90` | 0x40-0x21 | white (magic at 0x21) | 5 (3) | |
| Armos | `$30` | 0x34 | white | 2 | |
| Blue / Red Wizzrobe | `$A0` / `$40` | D6, D9 | magic | 3 / 1 | bomb 3 / 1 |
| Like-Like | `$90` | D6, D9 | magic | 3 | bomb 3 |
| Vire | `$40` | D6 | magic | 1 (no split) | bomb 1 |
| Zol | `$20` | D6, D9 | magic | 1 (no split) | |
| Red Gohma | `$20` | D6 0x1c | — | — | 1 arrow up into the open eye |
| Blue / Red Darknut | `$80` / `$40` | D8 | magic | 2 / 1 (not from the front) | bomb 2 / 1 |
| Gibdo | `$70` | D8 | magic | 2 | bomb 2 |
| Pols Voice | `$A0` | D8 | magic | 3 | 1 arrow (any) |
| Manhandla hand | `$40` | D8 0x6e | magic | 1 each | one bomb for all four |
| Gleeok, 4 heads | `$A0`+3×`$60` | D8 0x3c | magic | 3+2+2+2 = 9 | beam, rod only |
| Lanmola segment | `$20` | D9 0x55 | magic | 1 each, 10 | bomb, arrow, fire |
| Patra children / body | `$60` / `$B0` | D9 | magic | 2 each (16) + 3 | beam only besides |
| Ganon | `$F0` counter | D9 0x42 | magic | 4 (invisible only) | then 1 silver arrow |
| Guard fire | `$10` | D9 0x32 | magic | 1 | any weapon |

---

## 5. What reward and objective misjudge (consolidated)

1. **Arrows spend the route's last rupees.** ARROW is legal for every family after 0x44 (default `arrows: true`,
   repo); 2-14 rupees must pay Gohma's 1 and Ganon's ≥1. No rupee reserve exists (bombs have `bomb_reserve`).
   `penalty-attack-miss` is 0.01 (critic.rs:503). A wasted last rupee turns Ganon's room into a death trap
   (level-9.md "Dead end").
2. **Ganon's kill is invisible to the critic** (3.17): no hit, no kill, the room still "has a monster" (inferred
   from change.rs and asm:Ganon_UpdateDying). The 4th sword hit is not credited either.
3. **Bases that take hits.** Manhandla's base: the first hit is a credited boss hit and `killable` says 1 (3.13).
4. **Bosses by type id.** `PATRA2` (`$48`) is not a boss; Lanmola and Patra children are not; the small Digdoggers of
   0x1c are — check each is intended (enums.rs:333-352).
5. **Kills that must not be made / hits that must not land.** Persons and the Grumble Goriya (fireballs until Link
   leaves); 0x39's big Digdogger must not be split (RECORDER offered there by `recorder_wanted`); 7:0x1b's Goriyas
   only for bombs when the bag isn't full (draft); 9:0x16's Patra must die (the owner) but 0x27's need not.
6. **Monsters that never count.** Bubbles, traps, statues (correct today: hazard class); red Keese count for the
   shutter but not for `RoomKillCount` (a split room refills); Lanmola rooms never bank as cleared; flying Gleeok
   heads are in the game's all-dead type range but vanish with the Gleeok.
7. **Damage that only pays on one part.** Gohma's eye (arrow, up, open, X window); Ganon (sword invisible, arrow
   brown); Aquamentus's head box; Manhandla's hands; Patra's body only after all children; Darknuts from side/back.
   FIGHT targets every active enemy's tiles (objectives.rs:1233-1248) — Patra's untargetable body, Ganon's touch
   middle, a hidden Red Wizzrobe's stale spot are all targets.
8. **The approach reward walks into shot lines.** PBRS in fight rooms rewards closing on moving enemies
   (architecture.md "PBRS"): on a Wizzrobe's row or column, in front of a Blue Lynel (sword shot every square), into
   Ganon's middle (inferred).
9. **Damage penalty is capped.** `penalty-lost-health` = 0.25 per half heart, max 1.0 (critic.rs:385-391): 2-heart
   and 4-heart hits cost the same (inferred: under-weights Red Wizzrobes and Ganon).
10. **Rooms where waiting is right.** Gohma (~400-500 frames between openings, and Gohma back at X `$80`), Patra
    (children untargetable until all eight are out, ~1 orbit), Ganon's entry scene (~270 frames halted), traps
    (wait for the creep back), the Grumble's text, Wallmaster lures (stand at the wall — punished by the wallmaster
    shaping, critic.rs:396-430), Red Wizzrobes' hidden 64 frames. Room-stalling (150 steps), `no_damage_steps` (600
    actions) and `no_progress_steps` may cut these short (inferred: measure per family).
11. **A drag is not the end.** `wallmastered` is a hard failure in training; on the route a 7:0x0d capture costs
    15 rooms back from 0x79 (3.3).

---

## 6. Cross-cutting: every change the rest of the game needs, grouped for ONE compatibility break

The version rule (CLAUDE.md "Versioning"): an observation layout or action-space change is a COMPAT bump unless
`triforce convert-model` widens checkpoints exactly. Today the converter widens the vector keys (`link`,
`room_state`, `history`, `objective`, `target`) by name with zero columns and new kinds with WAIT-minus-5 logits;
**it does not touch `entity_types`, `entity_features`, `grid`, `action_mask`** (`SPECIAL_KEYS`,
`triforce/convert.py:57-69`). The Rust policy drops all-zero first-layer columns of the towers (`docs/rust-actor.md`
§1); the entity projection (`EntityEncoder.input_proj`, model.py:189) is a different layer (inferred: it needs the
same treatment to stay bit-exact).

### 6.1 `link` (vector key; widenable today)

| Feature | Value | Needed by | Evidence |
|---|---|---|---|
| `sword_blocked`, `sword_blocked_time` | `$4C` ≠ 0 or `$52E` ≠ 0; `$4C`/16 | every Bubble room | asm:Bubble_Update, G4 |
| `arrow_usable` | the mask's ARROW bit | Gohma, Ganon, any family | as `bomb_usable` |
| `rupees_spare` (or `arrows_spare`) | (rupees − route `rupee_reserve`)/16 | Gohma, Ganon | asm:Arrow_Wield |
| `rupees_low` | rupee count/16 clamped (the /255 one is too coarse) | | G9 |
| `damage_scale` | 1, ½, ¼ by ring grade | D9 | asm:Link_BeHarmed |

### 6.2 `room_state` (widenable today)

| Feature | Value | Needed by |
|---|---|---|
| `frame_phase_32`, `frame_phase_64` | (`FrameCounter` AND `$1F`)/32, (AND `$3F`)/64 | Blue Wizzrobe shots, Ganon fireballs (G3) |
| `person_fireballs` | `IsPersonFireballsEnabled` | rooms with a person (asm:Person_CheckCollisions) |
| `ringleader_alive` | trigger 2 and slot 1 holds a monster | D6 0x78 (G1) |
| `wallmaster_ready` | `ObjTimer+1` = 0 in a Wallmaster room | 7:0x0d |

### 6.3 `entity_features` (needs the converter and the Rust live-column drop extended to the entity projection, or a COMPAT bump with every family retrained)

| Feature | Value | Monsters |
|---|---|---|
| `phase` | a per-type normalized state: Red Wizzrobe `ObjState`/255; Gohma eye state/3; Ganon 0 blue / ½ visible / 1 brown; Patra state; Goriya waiting (`$80`); Dodongo state; Zol/Vire slide | G2 |
| `phase_time` | frames to the next event, /256: Red Wizzrobe to its shot (`ObjState` − `$B0`), Gohma's eye open timer and next-open counter, Ganon's brown time left, Gohma's fireball timer | G2 |
| `targetable_now` | the game will test weapons on it this frame | Patra body, Red Wizzrobe hidden/fading parity, Blue teleport, Ganon by phase, Manhandla base, big Digdogger, Peahat in flight (G6) |
| corrected `hurts_on_touch` | per the code: Wizzrobes hurt while flashing; hidden/odd-frame Wizzrobes, returning boomerangs, unemerged segments, Patra children before state 1, dying Ganon do not | G5 |
| `damage_to_link` | the type's touch/shot damage after the ring, /4 | G7 |
| `slot_one` (or a slot embedding) | the object is in slot 1 | ringleaders, carriers (G1) |
| `segment_role` | head / body / base | Lanmola, Manhandla base (G1) |
| `size_x`, `size_y` | /32 | Ganon, Gohma, big Digdogger, Dodongo (G8) |
| `on_line` | Link is on the shooter's current shot line (row/column per its own test) | Wizzrobes, Lynels, Goriyas, Octoroks |

Alternative to `segment_role` for Manhandla: synthesize the base as a hazard as the Gleeok base already is
(game.rs:372-376) — no layout change (inferred: preferred). The corrected `hurts_on_touch` and `killable` are a change
of meaning of existing columns: a COMPAT bump by the version rule even without new columns.

### 6.4 Actions and masks (no new kind needed; `action_mask` unchanged in size)

| Change | Where | Evidence |
|---|---|---|
| `rupee_reserve` per route step (routes.yaml, like `bomb_reserve`) and ARROW masked below it | every family after 0x44; 2 before Gohma, 1 after | asm:Arrow_Wield, draft ledger |
| `items: {arrows: false}` for every family except Gohma's, Ganon's (and a deliberate choice for Pols Voice rooms) | specialists | 2.3 |
| ARROW target mask (`arrow_targets`, as `bomb_targets`): Gohma only with the eye at 3, Gohma at home, Link in the X window facing N; Ganon only brown and on his row/column | boss-gohma, ganon | asm:Gohma_HandleWeaponCollision, asm:Ganon_CheckCollisions@CheckArrowCollision |
| `recorder_wanted` widened by route step: 7:0x1c yes, **7:0x39 no**; overworld whirlwinds and 0x42's pond as route macros, not model actions (the destination counts every overworld play) | actions.rs:382-388 | items.md "Recorder", draft "whirlwind" |
| FOOD only at 7:0x28 after the text | step | asm:Grumble_State1WaitForFood |
| CANDLE only at 0x6d and 0x47's trees | step | asm:Tree_Update |
| SWORD/BEAMS masked toward persons and the Grumble (today safe only because those rooms have no enemy) | mask | asm:Person_CheckCollisions |
| BOMB modes: `never` for Patra, Pols Voice, Gohma, Ganon rooms; `tenth` or a reserve for Wizzrobe rooms (bombs hurt them but the route cheats 7) | yamls | draft bomb ledger |

---

## 7. Families: which take the new rooms, which are new

| Family | New rooms | Fit | Why |
|---|---|---|---|
| goriya | D7 0x69, 0x59, 0x49, 0x38, 0x18, 0x19, 0x1a, 0x1b, 0x29 (+ red ones) | **extend, retrain** | blue Goriyas throw at once and hit 1 heart; trained on red only (goriya.yaml); 0x49 has Keese and Bubbles; arrows must go off |
| wallmaster | D7 0x0d | **extend with new states** | 5 hands + 3 Bubbles + push block + the drag to 0x79; the critic's wall shaping fights the lure (3.3) |
| boss-digdogger | D7 0x1c | extend | three small ones and statues instead of one |
| (new) digdogger-crossing, or goriya with RECORDER masked | D7 0x39 | new rule | walk round the big one |
| boss-aquamentus | D7 0x2a | as is (new states) | same white sword as D1; blocks on the east |
| dodongo-crossing | D7 0x0c | as is | crossed to the E wall |
| (route step) | D7 0x28 Grumble; equipment-room for cellars 0x4a, 0x7b | step | FOOD step |
| ow-lynel | 0x50, 0x32, 0x22, 0x07, 0x06, 0x05 | extend | blue Lynels' 2-heart shots; magic sword from 0x21 on |
| **ow-ghini (new)** | 0x40, 0x41, 0x31, **0x21 (gravestone push for the sword)** | new | immune flying Ghinis from every gravestone touched; kill the Ghini to clear them |
| ow-leever + armos step | 0x34 | new step | wake the Armos at X `$40`, food shop |
| ow-moblin, ow-octorok, ow-peahat, ow-boulder, ow-zora | the rest of the legs | as is | |
| **wizzrobe (new)** | D6 0x78, 0x28, 0x29, 0x38, 0x3a, 0x2c; D9 0x65, 0x15, 0x05, 0x20, 0x10, 0x30, 0x31, 0x04, 0x17, 0x07 | new, the biggest | nothing trained has seen them; needs 6.3's phase features |
| like-like | D6 0x19, 0x1d; D9 0x14, 0x51, 0x41, 0x03 | extend | magic sword; held Link is shot-proof |
| vire | D6 0x39, 0x2d | extend | the magic sword never splits |
| zol | D6 0x68; D9 0x63 | as is | magic sword, no split |
| stalfos-keese-gel | D6 0x58 (8 Keese + 4 statues); D9 0x62 | extend | forced, statues |
| trap-room | D6 0x48; D9 0x41 (traps + Like-Likes) | extend | 0x2c, 0x30, 0x04 go to wizzrobe |
| **boss-gohma (new)** | D6 0x1c | new | arrow-only, eye phases |
| darknut | D8 0x5e, 0x3e, 0x4e, 0x3f | extend | magic sword; statues; one-way 0x3f |
| gibdo, pols-voice | D8 0x7e (empty), 0x4c (8 Pols Voices, long walk, bomb) | extend | |
| boss-manhandla | D8 0x6e (crossed, bomb N wall) | new rule or family | the fight is not wanted; the base is a fake target |
| boss-gleeok | D8 0x3c | extend | four heads, three flying, magic sword |
| **lanmola (new)** | D9 0x55 | new | |
| **patra (new)** | D9 0x16, 0x61, 0x52, 0x27 (crossed) | new | children phase, body immunity, sword only |
| **ganon (new)** | D9 0x42 | new | invisible phase, brown phase, silver arrow, rupees |
| (route step) | D9 0x32 Zelda | end-of-game step | X `$70`-`$80`, Y `$95` |

---

## 8. Where the sources disagree or are thin

- **Lanmola re-entry** (3.15): level-9.md "worms back on every later entry"; the kill-banking rule suggests an empty
  room while it is in the history (inferred). Walk it.
- **Ganon after death**: level-9.md describes the game (ashes, item, shutters on the take); the project's
  observation/critic reading of it is inferred from the code (3.17). Walk it with the critic attached.
- **The gravestone of 0x21**: overworld.md says the push was walked Nov 13 (draft) but Ghinis kept knocking Link off
  on the first try; whether the pushed stone itself releases a flying Ghini is (inferred).
- **Chase target**: CLAUDE.md says monsters chase "shadow Link" copied from the current position before the flip
  (asm:Mode5Play_UpdateWorld); enemies.md "How to read this" calls it the mirrored point (255 − X, 255 − Y) for up
  to 70 frames. Both say the chase target is not always Link; Lanmola's turn, Rope-like rushes, traps and the Red
  Wizzrobe's appearance use Link's real position (code).
- **Fights played for real** in the Nov 13 walks: the D5 small Digdogger; D6's Gohma (one arrow), one Vire and one
  Like-Like (sword); D8's Manhandla of 0x7d, Gohma of 0x1e and the Gleeok; Ganon (first-quest.md "What was walked",
  level-6.md, level-8.md, level-9.md). **No fight of Dungeon 7, and none in Dungeon 9 but Ganon, was played**;
  hearts were written back in most walks. Every number about what a fight costs here is from the code.
- Rooms not walked (level docs): D7 0x3a, 0x08, 0x09, 0x0a; D6 0x0a, 0x0b, 0x18, 0x1a, 0x1b, 0x3c, 0x6a; D8 the
  entrance tree, 0x5f, 0x4d, 0x3d, 0x0e, 0x1d, 0x2d; D9 0x01, 0x02, 0x12, 0x13, 0x25, 0x33-0x37, 0x40, 0x43-0x47,
  0x50, 0x53, 0x54, 0x56, 0x57, 0x64, 0x71, 0x73, 0x74, cellar 0x72. None of them is on the draft route.

---

## 9. Hard problems, ranked

1. **Wizzrobe rooms (D6 ×6, D9 ×10 on the route).** 4-heart shots from a monster that appears beside Link facing
   him, hidden between; Blue ones that shoot on a 32-frame clock along exact pixel rows; only sword/beam/bomb hurt;
   Bubbles take the sword away; mixed with Like-Likes and traps. Today's model cannot see the Red one's cycle, the
   clock, its own sword-off timer, or tell a hidden one from a solid one (G2-G5); the approach shaping walks into
   the shot lines. Needs 6.1-6.3 features, a wizzrobe family with staged starts (room_kills as in D5), and likely
   the planner (`plan:K:S`) as a teacher (the owner's answer 1 in the owner's decisions of Nov 16, `docs/routes/rest-of-game.md`
   "What this is").
2. **Dungeon 9's damage budget.** 15 hearts, no refill, ~10 forced fights, 4 cheated bombs before 0x16, the worst
   rooms before the red ring; then the ring changes every damage by 4x mid-dungeon. One-room families trained
   from refilled states do not see the budget; the chain's health at 0x07 decides the run (inferred). Needs
   `damage_to_link`/`damage_scale`, start states drawn at the chain's real health, and the order of D9 measured.
3. **Ganon.** Three phases, an invisible 4-heart body, the only kill by a silver arrow inside a 510-frame window,
   with a rupee that the rest of the game must not spend; the critic does not see the kill (3.17). Needs a ganon
   family with phase features, an arrow target mask, rupee reserve, and a success rule on the Triforce of Power.
4. **Rupees as ammunition, everywhere.** Every family after 0x44 can spend them today (2.3). Without `arrows:
   false` and a reserve, the game can be lost in Ganon's room long after the mistake (inferred). Cheap to fix,
   fatal if missed.
5. **Patra (×3 forced).** 19 sword hits on 9 targets that orbit, 2 hearts a touch, no target until the eighth child
   appears, the body immune until the last child dies. Slot identity and the children's phase are invisible.
6. **7:0x0d Wallmasters with Bubbles.** The lure requires standing at the wall (punished by shaping); a Bubble's
   knockback hands Link to a hand; a capture costs 15 rooms. Needs features (6.2) and the chain's drag handling.
7. **Gohma.** One arrow, one rupee, 96-frame windows 400+ frames apart, an 8-px X window; the eye state is
   invisible. An arrow target mask makes it nearly a scripted step (inferred).
8. **Gleeok with four heads** (three immune flying heads chasing at once) and **8:0x3e/0x5e's Blue Darknuts** with
   statues — the existing families' hardest rooms, now bigger.
9. **7:0x1c's three small Digdoggers** (12 white hits, 2 hearts each, 1.5 px/frame, statues).
10. **Overworld Lynel rooms** (0x50, 0x32, 0x06, 0x05): blue Lynels shoot 2-heart unblockable shots every square.

## 10. What must be built (beyond rooms, states and curricula)

1. Observation (6.1-6.3), in one break; the converter extended to entity features (or a retrain), the Rust policy's
   live-column drop extended to the entity projection.
2. `rupee_reserve` route field and ARROW masking; `items: {arrows: false}` in every family yaml.
3. Target masks for ARROW (Gohma, Ganon) on the `bomb_targets` model.
4. `recorder_wanted` per route step; whirlwind/pond as route steps; FOOD step at 7:0x28; CANDLE tree steps;
   armos step at 0x34; gravestone push at 0x21 (with the flying Ghinis).
5. Critic: a Ganon kill detector (`Ganon_ObjDyingCount` or the Triforce of Power), Ganon's 4th hit, the Manhandla
   base as a hazard, `PATRA2` in the boss list (or a decision), a damage penalty that scales with hearts at stake,
   wallmaster wall shaping off in 7:0x0d, a living-monster rule that drops a dead Ganon.
6. Objective: FIGHT targets filtered by `targetable_now` (Patra body, hidden Wizzrobes, Ganon's middle offset);
   end-of-game step (Zelda); a Wallmaster drag mid-dungeon in a chain.
7. New families: wizzrobe, boss-gohma, lanmola, patra, ganon, ow-ghini; extended: goriya (blue), wallmaster,
   boss-digdogger, ow-lynel, like-like, vire, darknut, gibdo, pols-voice, boss-gleeok, trap-room, stalfos-keese-gel.

## 11. What to walk in the game before trusting this

1. 9:0x42 Ganon with the critic and the observation recorded: hits credited, the kill, `room_state` after death,
   what the Triforce of Power pays.
2. 9:0x55 re-entry after the worms die (in the history, and after CONTINUE).
3. 0x21: does pushing the sword's gravestone release flying Ghinis; does killing the Ghini first make it safe.
4. A Wizzrobe's observed `hurts_on_touch` against real touches (flashing, hidden, teleporting).
5. 8:0x6e: Manhandla's base credited hits when a bomb or sword hits it.
6. The chain after a 7:0x0d capture: where the run goes from 0x79.
7. Every forced fight for real (none of D7's, D6's Wizzrobe rooms, D9's Wizzrobe, Like-Like, Patra, Lanmola fights
   was played): hearts lost per room with a scripted or planner player, to set families' start-health ranges.
8. The rupee count at 0x44 and after, on real chains (the 2-14 is from the draft's arithmetic).

## 12. Open questions for the owner

1. **Arrows outside Gohma and Ganon:** off everywhere (recommended here), or allowed in Pols Voice rooms (8:0x4c's
   eight, one arrow can kill a line) at the cost of a rupee each against the 2-14 budget?
2. **The observation break:** may Ganon's and the Red Wizzrobes' hidden positions (in RAM, invisible on screen) stay
   in the observation? Today they are there by default.
3. **0x39's big Digdogger:** walk round it (as the trace did) or split it? (No file decides; the draft only crosses.)
4. **A Wallmaster capture in a chain:** end the run, or let the route replay from 0x79?
