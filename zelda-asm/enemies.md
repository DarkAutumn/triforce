# Enemies: monsters, bosses, hazards and other objects

Everything the code says about each object that can hurt Link, be hurt, or stand in a room: one section per
entity, variants in the heading, so a search for `# Zora` or `# Dodongo` lands on it. Order: drop tables;
overworld monsters; dungeon monsters; bosses; hazards and projectiles; other objects. Every bullet ends with
the label that proves it. `(trace)` marks a result observed in the running game; `(unverified)` a statement not
confirmed in the code. Items and weapons are in [items.md](items.md), shared rules (combat, spawning, room
history) in [mechanics.md](mechanics.md), room contents in [data/](data/README.md).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L66 How to read this
- L85 Drop tables
- L130 Octorok (red, blue; slow, fast)
- L173 Tektite (blue, red)
- L211 Leever (blue, red)
- L249 Peahat
- L286 Moblin (blue, red)
- L321 Lynel (blue, red)
- L357 Zora
- L402 Armos
- L437 Ghini (and flying Ghini)
- L474 Boulder
- L508 Keese (blue, red, black)
- L558 Gel
- L599 Zol
- L646 Rope
- L687 Stalfos
- L732 Goriya (blue, red)
- L780 Wallmaster
- L827 Bubble (flashing, red, blue)
- L871 Darknut (red, blue)
- L914 Gibdo
- L950 Pols Voice
- L993 Like-Like
- L1036 Vire
- L1081 Wizzrobe (blue, red)
- L1135 Lanmola (red, blue)
- L1184 Aquamentus
- L1240 Dodongo
- L1308 Manhandla
- L1365 Gleeok
- L1425 Digdogger
- L1482 Gohma (blue, red)
- L1529 Moldorm
- L1574 Patra
- L1620 Ganon
- L1668 Trap
- L1699 Statue fireballs
- L1731 Fireball
- L1762 Monster shots (rock, sword shot, magic shot)
- L1798 Monster arrow
- L1823 Monster boomerang
- L1848 Standing fire
- L1873 Guard fire
- L1896 Zelda
- L1917 Old men and shopkeepers (cave and dungeon persons)
- L1975 Grumble Goriya
- L2000 Rupee stash
- L2019 Pond fairy
- L2042 Item fairy
- L2065 Dropped items
<!-- /toc -->

# How to read this

Every monster section has the same eight labelled lines in the same order. Numbers are raw values from the tables and routines linked at the end of each bullet; object type ids are hex.

- **Types**: the object type id in `ObjType` of the monster's object slot (`$01`-`$0B`); it indexes the update and init tables, whose rows name every type. [`Obj_Update_JumpTable`](src/bank7/object-update-table.asm#L172), [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177)
- **HP**: one nibble per type in the HP table (even type = high nibble, odd = low); `ObjHP` starts as nibble x `$10`. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105), [`Obj_ExtractStartHp`](src/bank4/drop-items.asm#L62)
- Weapon damage uses the same units: sword and sword beam `$10` wooden, `$20` white, `$40` magic; rod swing `$20`; magic shot `$20`; arrow `$20` wooden, `$40` silver; bomb `$40`; fire `$10`; boomerang 0. A monster dies when HP < damage or HP - damage = 0. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181), [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)
- **Damage to Link**: one byte per type, low nibble = whole hearts, `$80` = half a heart; written here as the 16-bit amount (`$0100` = one heart, `$0080` = half). Each ring grade halves it: blue ring 1/2, red ring 1/4. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200), [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- **Weapons**: damage type bits are 1 sword, rod swing and sword beam; 2 boomerang; 4 arrow; 8 bomb; `$10` magic shot; `$20` fire. A bit set in the monster's invincibility mask makes that weapon harmless. The mask is `$00` unless a type's code stores one (room entry zeroes it; no monster of this part writes it). [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127), [`ObjInvincibilityMask`](src/Variables.inc#L343), [`Util_ClearRam0300UpTo`](src/bank7/ppu-tablejump-input.asm#L162)
- Attribute byte bits used below: `$01` the type's update draws it and tests its collisions (it chooses the states in which it can be hit; without the bit the object loop does both every frame); `$80` it does not turn around after hitting Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70), [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)
- A hit that does not kill pushes the monster `$40` pixels the way the weapon faces and sets its invincibility timer to `$10` (about 32 frames), during which no weapon is tested against it and its touch does not hurt. [`Shove_Begin@SetShoveDir`](src/bank1/shove-knockback.asm#L215), [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- The boomerang does 0 damage and sets the stun timer to `$10` (10-frame units: 151-160 frames); a stunned monster's touch does not hurt Link. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82), [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- **Behaviour**: speeds are q-speeds, pixels per frame = q-speed / 64 (`$20` = 0.5); `ObjTimer` counts down 1 per frame, the "long" timers 1 per 10 frames. [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244), [`Nmi_Isr`](src/bank7/run-game-nmi.asm#L140)
- "Chase target" is `ChaseTargetX/Y`: Link's position, or for random spells of up to 70 frames the fixed point (255 - X, 255 - Y) mirrored from Link's position, or food on the ground. [`Mode5Play_UpdateWorld@CheckChaseTarget`](src/bank7/mode5-play-update.asm#L145), [`ChaseTargetX`](src/Variables.inc#L144)
- "Clock" is `InvClock` = 1, from a dropped clock until Link leaves the room: no touch or shot is tested against Link; what each monster still does is in its section. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48), [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354)
- **Spawn**: room entry writes the room's monster types to slots 1..count; positions come from fixed spot lists, or in rooms with the edge attribute monsters walk in from the screen edge one at a time. Rooms per type: `data/levelblock-ow.md`, `data/objlists.md`. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143), [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64), [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)
- During the spawn cloud (metastate 1-3) a monster's update does not run and nothing is tested against it: it cannot hurt or be hurt. [`Obj_Update`](src/bank7/object-update-table.asm#L120), [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54)
- **Drops** names the drop group explained in `# Drop tables`. To reach the code: every label starts with its subject, so `grep -P "^Zora_" index/symbols.tsv` lists a monster's routines; shared movement is under `Wanderer_`, `Walker_`, `Shooter_`, `Jumper_`, `Burrower_`, `Flyer_`. [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157), [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)

# Drop tables

When a monster's death sparkle ends, its object slot becomes a dropped item object (type `$60`) and one routine decides what appears there, if anything.

- Order of the decision: a no-drop type drops nothing; a Stalfos (`$2A`) or Gibdo (`$30`) in object slot 1 drops nothing; `WorldKillCount` = `$10` gives a fairy; `HelpDropCount` >= `$0A` gives a forced bomb or 5 rupees; otherwise the table item, if the random test passes. [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)

| Group | Rate: drop when random byte is below | Object types |
|---|---|---|
| 0 | `$50` = 80/256 (31%) | `$07` `$08` red Octorok, `$0E` red Tektite, `$04` red Moblin, `$0F` Blue Leever, `$23` Blue Wizzrobe |
| 1 | `$98` = 152/256 (59%) | `$21` Ghini, `$22` flying Ghini, `$0D` blue Tektite, `$10` Red Leever, `$13` Zol, `$28` Rope, `$2A` Stalfos, `$27` Wallmaster, `$16` Pols Voice |
| 2 | `$68` = 104/256 (41%) | `$09` `$0A` blue Octorok, `$03` blue Moblin, `$01` blue Lynel, `$12` Vire, `$06` red Goriya, `$0B` red Darknut, `$24` Red Wizzrobe, `$30` Gibdo |
| 3 | `$68` = 104/256 (41%) | every type in no list: `$02` red Lynel, `$05` blue Goriya, `$0C` blue Darknut, `$11` Zora, `$1A` Peahat, the bosses, ... |
| none | never | `$5D` dead dummy (what a killed Armos becomes), `$14` child Gel, `$15` Gel, `$1B`-`$1D` Keese, `$17` Like-Like |

- The dead monster's type is searched in the no-drop list, then the lists of groups 0, 1, 2; a type found nowhere is group 3. [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96), [`Drop_Group0Types`](src/bank4/drop-items.asm#L100), [`Drop_Group1Types`](src/bank4/drop-items.asm#L105), [`Drop_Group2Types`](src/bank4/drop-items.asm#L111), [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)

| Group / kill cycle | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | heart | rupee | heart | rupee | fairy | rupee | heart | heart | rupee | rupee |
| 1 | 5 rupees | rupee | heart | rupee | 5 rupees | heart | clock | rupee | rupee | rupee |
| 2 | heart | bomb | rupee | clock | rupee | heart | bomb | rupee | bomb | heart |
| 3 | heart | heart | fairy | rupee | heart | fairy | heart | heart | heart | rupee |

- The table item is entry group x 10 + `WorldKillCycle`; item ids `$22` heart, `$18` rupee, `$0F` 5 rupees, `$23` fairy, `$21` clock, `$00` bomb. [`Drop_ItemTable`](src/bank4/drop-items.asm#L132), [`Drop_GroupRowOffsets`](src/bank4/drop-items.asm#L117)
- Bombs come only from group 2 (columns 1, 6, 8); clocks from group 1 column 6 and group 2 column 3. [`Drop_ItemTable`](src/bank4/drop-items.asm#L132)
- `WorldKillCycle` counts 0-9 and wraps. It advances by 1 when a death sparkle ends, just before the lookup: a death that ends with the cycle reading N uses column N + 1 (9 uses 0). [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308), [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216)
- Types `$5D` (dead dummy), `$14` (child Gel) and `$1C` (red Keese) do not advance the cycle; the other no-drop types (Gel `$15`, Keese `$1B` and `$1D`, Like-Like) do. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- The cycle survives room changes, damage, death and saving; only a console reset zeroes it. [`WorldKillCycle`](src/Variables.inc#L468), [`Reset_ClearRam`](src/bank5/save-ram-clear-ram.asm#L81)
- The sparkle runs metastates `$10`-`$14`, three steps of 6 frames: the drop is decided about 19 frames after the killing hit. [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54), [`Obj_UpdateMetastate`](src/bank7/object-update-table.asm#L285)
- Random test: an unforced item appears only when `Random,X` (X = the dead monster's object slot) is below the group's rate; otherwise the slot is freed. [`Drop_SetUpItem@RandomlyCancel`](src/bank4/drop-items.asm#L253), [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- Fairy at 16 kills: `WorldKillCount` goes up by 1 the moment any killing hit lands (every type, no cap); a death that ends while it reads exactly `$10` drops a fairy with no random test, whatever the group. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216), [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216)
- The count is read when the sparkle ends, not at the hit: a 17th kill landed before the 16th's sparkle ends loses the fairy, and every death that ends while it reads `$10` gives one. [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216)
- Help drop: `HelpDropCount` goes up by 1 per kill and stops at `$0A`; a death that ends while it is `$0A` drops 5 rupees, or a bomb when `IsHelpDropBomb` is nonzero, with no random test and ignoring the table. [`Drop_SetUpItem@SetHelpItem`](src/bank4/drop-items.asm#L244), [`HelpDropCount`](src/Variables.inc#L72)
- Bomb case: `IsHelpDropBomb` is set only when the kill that brings `HelpDropCount` to exactly `$0A` is made by a bomb blast (damage type 8). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216), [`IsHelpDropBomb`](src/Variables.inc#L76)
- A stunned Dodongo killed with the sword sets both bytes to `$0A`, so the next drop decided is a forced bomb. [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296)
- The fairy and both help drops zero `HelpDropCount` and `IsHelpDropBomb` and leave `WorldKillCount` alone: without damage, and with each death ending before the next kill, the forced drops fall on kills 10, 16 (fairy), 26, 36, ... [`Drop_SetUpItem@SetHelpItem`](src/bank4/drop-items.asm#L244)
- A no-drop type frees its slot before the counters are read: a forced help drop stays pending for the next death, but a 16th kill of a no-drop type gives no fairy. [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157), [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)
- Resets: every harm to Link (a monster's touch or shot, his own candle fire, the whirlwind with 0 damage) zeroes `WorldKillCount`, `HelpDropCount` and `IsHelpDropBomb`; a touch that is not tested (Link invincible, clock) resets nothing. [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210), [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- The item uses the dead monster's slot. Its lifetime starts at `$FF` and drops 1 on odd frames: gone after about 510 frames; while it is >= `$F0` (about 32 frames) the item flashes and cannot be taken. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297), [`Item_TryTake`](src/bank1/take-item.asm#L206)
- Takers, tested in this order: slot `$12` (arrow or rod), `$0D` (sword), `$0F` (boomerang; food never), then Link. A weapon slot counts while its state is nonzero with bit 7 clear. [`Drop_ItemTakerObjSlots`](src/bank4/drop-items.asm#L280), [`Drop_UpdateItem@LoopItemTaker`](src/bank4/drop-items.asm#L331)
- Pickup range: taker X within 8 pixels of the item's X, and taker Y + 3 within 8 pixels of the item's Y. [`Item_TryTake`](src/bank1/take-item.asm#L206)
- Nothing is taken while Link is halted (state AND `$C0` = `$40`), but the item keeps aging. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)
- A dropped fairy flies (0.5, then 0.625 px/frame, random turns every `$10` frames) instead of lying still. [`Fairy_Init`](src/bank4/fairy-flyer-common.asm#L82), [`Fairy_DecideFlightState`](src/bank4/fairy-flyer-common.asm#L165)
- Effects: heart +1 full heart, fairy +3, rupee +1, 5 rupees +5, bomb +4 up to the bomb maximum, clock sets `InvClock`. [`Item_Take`](src/bank1/take-item.asm#L259), [`Item_TakeHeartsNoSound`](src/bank1/take-item.asm#L475), [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)

# Octorok (red, blue; slow, fast)

**Types**
- `$07` red slow, `$08` red fast, `$09` blue slow, `$0A` blue fast; one update routine for all four. [`Octorok_Update`](src/bank4/octorok.asm#L48), [`Obj_Update_JumpTable`](src/bank7/object-update-table.asm#L172)
- Attribute `$05`: draws itself and tests its own collisions every frame; turns around after hitting Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- Red `$10`: one hit of any sword, beam, arrow, bomb, fire or rod. Blue `$20`: 2 wooden sword hits, 1 white or magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105), [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)

**Damage to Link**
- Touch `$0080` (half a heart) for all four types. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Rock (object type `$53`) `$0080`. Any shield parries it when Link's state is `$00`-`$0F` (not swinging or using an item) and he faces against it. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)

**Weapons**
- Mask `$00`: every weapon hurts; the boomerang stuns. Tested every frame once the spawn cloud is over. [`Octorok_Update`](src/bank4/octorok.asm#L48), [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- Being hit (invincibility timer nonzero) zeroes the shoot timer: a rock about to be spat is cancelled. [`Shooter_RunShootTimer@TryShootingNow`](src/bank4/moblin-lynel-shooting.asm#L118)

**Behaviour**
- Walks the 16-pixel grid as a wanderer with turn rate `$70` (red) or `$A0` (blue). [`Octorok_Update`](src/bank4/octorok.asm#L48)
- On a grid point: if turn rate >= `Random+1,X` (113 or 161 of 256 values) and the chase target is less than 9 pixels off on one axis, it faces the target along the other axis; otherwise, when its turn timer is 0, it turns perpendicular to its facing, toward the target. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Every turn reloads the turn timer with `Random,X` frames and sets "wants to shoot"; a grid point without a turn clears the flag. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Speed: slow types q-speed `$20` (0.5 px/frame), fast types `$40` (1 px/frame). Type `$08` starts at `$30` (0.75) and gets `$40` only when its shoot gate first passes. [`Octorok_Update`](src/bank4/octorok.asm#L48), [`Octorok_InitFast`](src/bank4/zora-ow-inits.asm#L219)
- Shoot sequence: with the flag set the shoot timer starts at `$30`; the Octorok stands still (and cannot turn) for 48 frames; the rock leaves when the timer reaches `$10`, 32 frames after it stopped. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Blue types test the flag every frame; red types only on frames where `Random,X` >= `$F8` (8 of 256), so red ones shoot far less often. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Rock: starts at the Octorok's position, flies in its facing direction at q-speed `$C0` (3 px/frame); destroyed by an unwalkable tile ahead, the room bound, or on harming Link. [`MonsterShot_Update`](src/bank4/monster-shots-fireballs.asm#L102), [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82)
- A parried rock drifts 2 pixels per frame the way Link faced for `$10` frames, harmless, then vanishes. [`MonsterShot_Bounce`](src/bank4/monster-shots-fireballs.asm#L209)
- No rock when no object slot `$01`-`$0B` is free or 4 counted monster shots are already in flight. [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70)
- Clock or stun: it does not move and the rock at `$10` is skipped, but the shoot timer still counts. [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87), [`Shooter_RunShootTimer@SetShootTimer`](src/bank4/moblin-lynel-shooting.asm#L161)
- Blocked by a tile or room bound it tries a random perpendicular direction, the opposite of that, then reverse. [`Walker_GetNextAltDir_JumpTable`](src/bank7/walker-collision-screen-edge.asm#L322)

**Spawn**
- Rooms use a single type (`$07`-`$0A`) or the mixed lists `$68`-`$6C` (list `$6C` mixes blue fast Octoroks with Moblins). [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185), [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- Spawn cloud: first step (slot + 1) x `$10` frames (`$20` in slot 1 to `$C0` in slot `$0B`), then 6 + 6 frames. [`Octorok_InitFast`](src/bank4/zora-ow-inits.asm#L219), [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54)
- In a room with the edge attribute it walks in from a screen edge without a cloud. [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)
- First facing: vertically toward the chase target, or horizontally away from it when the vertical difference is not the smaller one. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113)

**Drops**
- Red `$07`, `$08`: group 0 (31%). Blue `$09`, `$0A`: group 2 (41%), the row with the bombs. [`Drop_Group0Types`](src/bank4/drop-items.asm#L100), [`Drop_Group2Types`](src/bank4/drop-items.asm#L111)

**Notes**
- The cloud delay grows with the slot: an Octorok in slot 1 starts acting about 44 frames after its first update, one in slot 6 about 124 frames. [`Octorok_InitFast`](src/bank4/zora-ow-inits.asm#L219)
- While the shoot timer runs the Octorok stands in place for 48 frames: a fixed target. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)

# Tektite (blue, red)

**Types**
- `$0D` blue, `$0E` red; both run the jumper routine shared with the Boulder. Attribute `$81`. [`Jumper_Update`](src/bank4/tektite-boulder.asm#L166), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- `$10` for both: any damaging weapon kills in one hit. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch `$0080` (half a heart) for both. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Weapons**
- Mask `$00`. Link and all weapons are tested every frame, on the ground and in the air. [`Jumper_AnimateAndCheckCollisions`](src/bank4/tektite-boulder.asm#L390), [`Obj_DrawMirroredAndCheckCollisions`](src/bank4/octorok.asm#L139)
- The boomerang (or the clock) freezes a Tektite where it is, in mid-air too; it is still drawn and still tested. [`Jumper_Update@CheckStunned`](src/bank4/tektite-boulder.asm#L179)

**Behaviour**
- Two states: 0 on the ground until `ObjTimer` is 0; 1 in the air. [`Jumper_Update`](src/bank4/tektite-boulder.asm#L166)
- At take-off it turns at most 45 degrees (toward the chase target when the turn rule allows) and, if the direction has no horizontal part, adds left or right toward the target's X: every jump is sideways or diagonal. [`Jumper_Update`](src/bank4/tektite-boulder.asm#L166), [`Dir_TurnObjTowardTarget8`](src/bank4/fairy-flyer-common.asm#L407)
- In the air X changes exactly 1 pixel per frame; tiles are ignored. [`Jumper_Update@MoveX`](src/bank4/tektite-boulder.asm#L308)
- Vertical start speed: blue -3 px/frame, red -4; falling speed capped at 2 px/frame. [`Jumper_StartSpeedsWholeY`](src/bank4/tektite-boulder.asm#L143), [`Jumper_MoveY`](src/bank4/tektite-boulder.asm#L483)
- Gravity in 1/256 px/frame per frame: blue `$40` (`$30` on up-diagonals), red `$80` (`$50` on up-diagonals). [`Jumper_YAccelerations`](src/bank4/tektite-boulder.asm#L131)
- A jump aims 0 (sideways), `$20` pixels lower (down-diagonal) or `$20` higher (up-diagonal) than it began and ends on the first frame the Tektite is not rising and is within 3 pixels of that height. [`Jumper_TargetYOffsets`](src/bank4/tektite-boulder.asm#L122), [`Jumper_Update@MoveX`](src/bank4/tektite-boulder.asm#L308)
- Jump length computed from the tables: blue 29 frames sideways (rising 24 pixels), 45 down-diagonal, 16 up-diagonal; red 20 (rising 20 pixels), 36, 13. [`Jumper_YAccelerations`](src/bank4/tektite-boulder.asm#L131), [`Jumper_MoveY`](src/bank4/tektite-boulder.asm#L483)
- Wait after landing: blue `$20`-`$FF` frames; red `$00`-`$7F` frames, or `$00`-`$0F` when `Random+1,X` >= `$A0` (3 in 8). [`Jumper_Update@Landed`](src/bank4/tektite-boulder.asm#L340)
- At a room bound the direction is reversed and a new jump starts from where it is; a bound hit on two frames in a row flips left and right instead. [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309), [`Jumper_Update@SetUpJump`](src/bank4/tektite-boulder.asm#L221)

**Spawn**
- First quest rooms: blue `$4A`, `$79`, `$7A`; red `$0C`, `$0D`, `$1A`, `$1E`, `$2C`, `$74`, `$76`. [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- Init: random direction right, left, down-right or up-left; the spawn cloud's first step lasts 4, 8, `$14` or `$28` frames (direction x 4), then 6 + 6. [`Tektite_Init`](src/bank4/zora-ow-inits.asm#L174), [`Tektite_StartDirs`](src/bank4/zora-ow-inits.asm#L149)

**Drops**
- Blue `$0D`: group 1 (59%). Red `$0E`: group 0 (31%). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105), [`Drop_Group0Types`](src/bank4/drop-items.asm#L100)

**Notes**
- The landing height is not snapped: computed from the tables a sideways jump ends 2 pixels higher than it began, so a Tektite drifts up. [`Jumper_Update@MoveX`](src/bank4/tektite-boulder.asm#L308)
- The last `$20` frames before a jump show frame image 0 without the leg animation: a visible tell. [`Jumper_AnimateAndCheckCollisions`](src/bank4/tektite-boulder.asm#L390)
- Steering uses the chase target, so Tektites periodically jump toward the mirrored point instead of Link. [`Dir_TurnObjTowardTarget8`](src/bank4/fairy-flyer-common.asm#L407)

# Leever (blue, red)

**Types**
- `$0F` Blue Leever, `$10` Red Leever; separate update routines, shared burrower drawing. Attribute `$01`. [`Leever_UpdateBlue`](src/bank4/leever.asm#L79), [`Leever_UpdateRed`](src/bank4/leever.asm#L252), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- Blue `$40`: 4 wooden sword hits, 2 white, 1 magic. Red `$20`: 2 wooden, 1 white or magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: blue `$0100` (one heart), red `$0080` (half a heart). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Weapons**
- Mask `$00`, but Link and weapons are tested only in state 3 (fully out): in every other state a Leever cannot be hit and cannot hurt. [`Burrower_AnimateDrawAndCheckCollisions`](src/bank4/leever.asm#L152)

**Behaviour**
- Six states for both: 0 under ground (not drawn), 1 mound, 2 half out, 3 out, 4 half out, 5 mound. [`Burrower_AnimateDrawAndCheckCollisions`](src/bank4/leever.asm#L152)
- Blue state lengths: `$80`, `$20`, `$0F`, `$FF`, `$10`, `$60` frames: a 542-frame cycle with 255 frames out. [`Burrower_StateTimes`](src/bank4/leever.asm#L63), [`Burrower_Update@CycleState`](src/bank4/leever.asm#L121)
- Blue moves in every state, under ground too, as a wanderer with turn rate `$A0`; q-speed per state `$08`, `$0A`, `$10`, `$20`, `$10`, `$0A` (0.125 to 0.5 px/frame). [`Leever_UpdateBlue`](src/bank4/leever.asm#L79), [`Burrower_StateQSpeeds`](src/bank4/leever.asm#L57)
- Clock or stun stops a Blue Leever's movement but not its state timers: it still sinks and comes back. [`Leever_UpdateBlue`](src/bank4/leever.asm#L79), [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87)
- Red state 0: waits until `RedLeeverLongTimer` is 0 and fewer than 2 Red Leevers are out. [`Leever_UpdateRed`](src/bank4/leever.asm#L252), [`ActiveRedLeeverCount`](src/Variables.inc#L425)
- Red surfaces `$28` pixels from Link: ahead of him in the direction he faces, or behind him when `Random+1,X` >= `$C0` (1 in 4). [`Leever_UpdateRed`](src/bank4/leever.asm#L252)
- No surfacing at Y below `$5D`, on an unwalkable tile, or when the X wrapped around the screen; the attempt repeats every frame with Link's position then. [`Leever_UpdateRed@CheckSafeToSpawn`](src/bank4/leever.asm#L374), [`Leever_UpdateRed@PlaceHorizontally`](src/bank4/leever.asm#L318)
- Red states 1 and 2: `$10` + 8 frames emerging. State 3: runs straight at q-speed `$20` (0.5 px/frame) toward where Link stood, until an unwalkable tile or the room bound is ahead. States 4 and 5: 8 + `$10` frames sinking. [`Leever_RedStateTimes`](src/bank4/leever.asm#L227), [`Leever_UpdateRed@CheckStunned`](src/bank4/leever.asm#L409)
- Each surfacing sets `RedLeeverLongTimer` to 2: the next Red Leever surfaces 11-20 frames later at the earliest. [`Leever_UpdateRed@CheckSafeToSpawn`](src/bank4/leever.asm#L374)
- Knockback, stun and the clock hold a Red Leever in state 3 (out, hittable); the other states run on with the clock, so Red Leevers still surface. [`Leever_UpdateRed@CheckOtherStates`](src/bank4/leever.asm#L395), [`Leever_UpdateRed@CheckStunned`](src/bank4/leever.asm#L409)

**Spawn**
- No spawn cloud. Every Leever init sets `RedLeeverLongTimer` to 5 (41-50 frames before the first Red Leever can surface). [`Leever_Init`](src/bank4/zora-ow-inits.asm#L194)
- A Blue Leever stays under ground (state 0) for (slot + 1) x `$10` frames before its first cycle; it starts from the room's spawn spot. [`Octorok_InitFast`](src/bank4/zora-ow-inits.asm#L219), [`Burrower_Update`](src/bank4/leever.asm#L94)
- First quest rooms: single-type rooms and the mixed lists `$66` (with Lynels) and `$67` (with Peahats). [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185), [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)

**Drops**
- Blue `$0F`: group 0 (31%). Red `$10`: group 1 (59%). [`Drop_Group0Types`](src/bank4/drop-items.asm#L100), [`Drop_Group1Types`](src/bank4/drop-items.asm#L105)

**Notes**
- The Red Leever count drops when one finishes sinking or starts dying, which lets the next one surface. [`Leever_CycleRedStateAndDraw`](src/bank4/leever.asm#L447), [`Burrower_AnimateDrawAndCheckCollisions`](src/bank4/leever.asm#L152)
- A Red Leever never turns toward Link after surfacing: stepping off its line avoids it. [`Leever_UpdateRed@CheckStunned`](src/bank4/leever.asm#L409)

# Peahat

**Types**
- `$1A`. Attribute `$01`: its update draws it and chooses which collisions to test. [`Peahat_Update`](src/bank4/flying-ghini-peahat.asm#L101), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- `$20`: 2 wooden sword hits, 1 white or magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch `$0080` (half a heart), tested in every flight state. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200), [`Peahat_Update`](src/bank4/flying-ghini-peahat.asm#L101)

**Weapons**
- Link's weapons are tested only in flight state 5 (stopped). In states 0-4 only its touch on Link is tested: nothing can hurt, stun or push it while it flies. [`Peahat_Update@DrawAndCheckCollisions`](src/bank4/flying-ghini-peahat.asm#L120)
- Mask `$00`: when stopped, every weapon works. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)

**Behaviour**
- Flight states: 0 speed up, 1 decide, 2 chase, 3 wander, 4 slow down, 5 stopped. [`Peahat_ControlFlight_JumpTable`](src/bank4/flying-ghini-peahat.asm#L147)
- Movement: speed AND `$E0` is added to a fraction each frame, 1 pixel per carry in one of 8 directions; top speed `$A0` = 0.625 px/frame; below `$20` it does not move. Tiles are ignored; a room bound reverses it. [`Flyer_Move`](src/bank4/fairy-flyer-common.asm#L250), [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309)
- State 0: speed +1 per frame from `$1F` until it reaches `$A0` (129 frames), then state 1. [`Flyer_State0SpeedUp`](src/bank4/fairy-flyer-common.asm#L210), [`Flyer_FinishInit`](src/bank4/zora-ow-inits.asm#L256)
- State 1: `Random,X` >= `$B0` chase (80/256); >= `$20` wander (144/256); else slow down (32/256). [`Peahat_DecideFlightState`](src/bank4/flying-ghini-peahat.asm#L162)
- Chase and wander: 5 turns of at most 45 degrees, `$10` frames apart (toward the chase target, or at random), then state 1 again. [`Flyer_State2Chase`](src/bank4/fairy-flyer-common.asm#L377), [`Flyer_State3Wander`](src/bank4/fairy-flyer-common.asm#L507), [`Flyer_SetFlyingStateAnd6Turns`](src/bank4/flying-ghini-peahat.asm#L179)
- Slow down: speed -1 per frame until below `$20` (129 frames from top speed), then state 5 for `$40`-`$7F` frames, then state 0. [`Flyer_State4SlowDown`](src/bank4/fairy-flyer-common.asm#L200), [`Flyer_State5Delay`](src/bank4/fairy-flyer-common.asm#L186)
- Clock or stun: the flight code is skipped, so it hangs where it is in the state it had. [`Peahat_Update`](src/bank4/flying-ghini-peahat.asm#L101)

**Spawn**
- No spawn cloud; starts facing up at speed `$1F`. [`Peahat_Init`](src/bank4/zora-ow-inits.asm#L243)
- Its init sets the room's one top-speed byte `FlyerMaxSpeedFrac` to `$A0`. [`Flyer_FinishInit`](src/bank4/zora-ow-inits.asm#L256), [`FlyerMaxSpeedFrac`](src/Variables.inc#L395)
- First quest rooms: single-type rooms and the mixed lists `$64` (with Lynels) and `$67` (with Leevers). [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185), [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)

**Drops**
- In no group list: group 3 (41%). [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)

**Notes**
- The only vulnerable window is the 64-127 frame stop, reached on 1 decision in 8. [`Peahat_DecideFlightState`](src/bank4/flying-ghini-peahat.asm#L162), [`Flyer_State4SlowDown`](src/bank4/fairy-flyer-common.asm#L200)
- A boomerang hit during the stop keeps it in state 5, still vulnerable, for the whole stun: the handler that ends the stop does not run while stunned. [`Peahat_Update`](src/bank4/flying-ghini-peahat.asm#L101), [`Flyer_State5Delay`](src/bank4/fairy-flyer-common.asm#L186)
- A clock picked up while it flies leaves it untouchable until the room is left. [`Peahat_Update@DrawAndCheckCollisions`](src/bank4/flying-ghini-peahat.asm#L120)

# Moblin (blue, red)

**Types**
- `$03` blue, `$04` red. Attribute `$00`: the object loop draws it and tests its collisions every frame; it turns around after hitting Link. [`Moblin_Update`](src/bank4/moblin-lynel-shooting.asm#L55), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- Blue `$30`: 3 wooden sword hits, 2 white, 1 magic. Red `$20`: 2 wooden, 1 white or magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch `$0080` (half a heart) for both. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Arrow (object type `$5B`) `$0080`. Any shield parries it when Link's state is `$00`-`$0F` and he faces against it. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)

**Weapons**
- Mask `$00`: every weapon hurts; the boomerang stuns. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- Being hit cancels an arrow about to be shot (shoot timer forced to 0). [`Shooter_RunShootTimer@TryShootingNow`](src/bank4/moblin-lynel-shooting.asm#L118)

**Behaviour**
- Wanderer with turn rate `$A0` (homes on the chase target when in line on 161 of 256 random values), q-speed `$20` (0.5 px/frame). [`Moblin_Update`](src/bank4/moblin-lynel-shooting.asm#L55), [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- A turn sets "wants to shoot"; then the shoot timer runs `$30` frames with the Moblin standing still, and the arrow leaves at `$10` (32 frames in). [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Blue tests the flag every frame; red only on frames where `Random,X` >= `$F8` (8 of 256). [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Arrow: from the Moblin's position in its facing direction, q-speed `$80` (2 px/frame); tiles do not stop it; it sparks for 3 frames at the room bound and is destroyed when it harms Link. [`MonsterArrow_Update`](src/bank4/moblin-lynel-shooting.asm#L236)
- A parried arrow bounces for `$10` frames, harmless, then is destroyed. [`MonsterShot_Bounce`](src/bank4/monster-shots-fireballs.asm#L209)
- Arrows count toward the limit of 4 monster shots; none is made without a free object slot. [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70)
- Clock or stun: no movement, and the arrow at `$10` is skipped. [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87), [`Shooter_RunShootTimer@SetShootTimer`](src/bank4/moblin-lynel-shooting.asm#L161)

**Spawn**
- Single-type rooms and the lists `$62`, `$63` (blue and red mixed) and `$6C` (with blue fast Octoroks). [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185), [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- Spawn cloud: first step as many frames as the slot number, then 6 + 6; in rooms with the edge attribute it walks in from a screen edge with no cloud. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122), [`Obj_Init@InitMonsterFromEdge`](src/bank7/object-init-table.asm#L79)

**Drops**
- Blue `$03`: group 2 (41%, the row with bombs). Red `$04`: group 0 (31%). [`Drop_Group2Types`](src/bank4/drop-items.asm#L111), [`Drop_Group0Types`](src/bank4/drop-items.asm#L100)

**Notes**
- A Moblin shoots only after a turn, along the direction it then faces. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224), [`Shooter_Shoot`](src/bank4/shoot-helpers-candle.asm#L94)

# Lynel (blue, red)

**Types**
- `$01` blue, `$02` red. Attribute `$00`: drawn and collision-tested by the object loop every frame; turns around after hitting Link. [`Lynel_Update`](src/bank4/moblin-lynel-shooting.asm#L69), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- Blue `$60`: 6 wooden sword hits, 3 white, 2 magic. Red `$40`: 4 wooden, 2 white, 1 magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: blue `$0200` (2 hearts), red `$0100` (1 heart). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Sword shot (object type `$57`) `$0200` from either colour. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Only the magic shield parries the sword shot, and only when Link's state is `$00`-`$0F` and he faces against it. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)

**Weapons**
- Mask `$00`: every weapon hurts; the boomerang stuns. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- Being hit cancels a sword shot about to be fired. [`Shooter_RunShootTimer@TryShootingNow`](src/bank4/moblin-lynel-shooting.asm#L118)

**Behaviour**
- Walks the grid at q-speed `$20` (0.5 px/frame) using the Goriya's movement routine. [`Lynel_Update`](src/bank4/moblin-lynel-shooting.asm#L69), [`Goriya_Update`](src/bank4/goriya.asm#L45)
- On every grid point it takes the longer of the horizontal and vertical distances to the chase target; if that is under `$51` pixels it faces the target along that axis and sets "wants to shoot", else it keeps its direction and clears the flag. [`Goriya_Update`](src/bank4/goriya.asm#L45)
- Shoot sequence as for the Octorok: `$30` frames standing still, the shot leaves at `$10`. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Blue tests the flag every frame; red only on frames where `Random,X` >= `$F8` (8 of 256). [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Sword shot: from the Lynel's position in its facing direction at q-speed `$C0` (3 px/frame); tiles do not stop it; destroyed at the room bound or on harming Link; a parried one bounces for `$10` frames. [`MonsterShot_Update`](src/bank4/monster-shots-fireballs.asm#L102), [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82)
- Sword shots count toward the limit of 4 monster shots. [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70)
- Clock or stun: no movement, and the shot at `$10` is skipped. [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87), [`Shooter_RunShootTimer@SetShootTimer`](src/bank4/moblin-lynel-shooting.asm#L161)

**Spawn**
- Single-type rooms and the lists `$64` (with Peahats), `$65` (both colours) and `$66` (with Leevers). [`Spawn_ObjLists`](src/bank5/mode-inits-obj-lists.asm#L185), [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- Spawn cloud: first step as many frames as the slot number, then 6 + 6. [`Obj_Init@NormalSpawn`](src/bank7/object-init-table.asm#L122), [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54)

**Drops**
- Blue `$01`: group 2 (41%, the row with bombs). Red `$02`: in no list, group 3 (41%). [`Drop_Group2Types`](src/bank4/drop-items.asm#L111), [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)

**Notes**
- A blue Lynel within `$51` pixels re-arms the flag at each grid point, so it can shoot again after every square walked. [`Goriya_Update`](src/bank4/goriya.asm#L45), [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)

# Zora

**Types**
- `$11`. Attribute `$81`: draws itself and tests its own collisions; does not turn after hitting Link. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- `$20`: 2 wooden sword hits, 1 white or magic; one arrow, bomb, rod hit or magic shot. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)
- HP is not kept across a dive: the object is destroyed when it submerges and a new one with `$20` is created. [`Zora_Update@DestroyIfDived`](src/bank4/zora-ow-inits.asm#L324), [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L77)

**Damage to Link**
- Touch `$0080` (half a heart), only while surfaced (states 2-4). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200), [`Burrower_AnimateDrawAndCheckCollisions`](src/bank4/leever.asm#L152)
- Fireball (object type `$55`) `$0080`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Only the magic shield blocks the fireball: Link's state must be `$00`-`$0F` and he must face against its horizontal or its vertical component. The small shield never blocks it. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- The fireball is destroyed on any touch of Link, blocked or not. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)

**Weapons**
- Mask `$00`, but Link and weapons are tested only in states 2, 3 and 4 (65 frames per surfacing); the mounds of states 1 and 5 cannot be hit. [`Burrower_AnimateDrawAndCheckCollisions`](src/bank4/leever.asm#L152)
- A hit that does not kill makes it invincible for about 32 frames; knockback never moves it (its update calls no movement routine). [`Shove_Begin@SetShoveDir`](src/bank1/shove-knockback.asm#L215), [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301)

**Behaviour**
- It never moves. It runs the Blue Leever's state cycle: state 1 mound `$20` (32) frames; state 2 surfaced `$0F` (15); state 3 surfaced 34; state 4 surfaced `$10` (16); state 5 mound `$60` (96): 193 frames, 65 of them surfaced (trace). [`Burrower_StateTimes`](src/bank4/leever.asm#L63), [`Burrower_Update`](src/bank4/leever.asm#L94)
- State 3 starts with timer `$FF`; on its third frame the Zora shoots and the timer is cut to `$20`. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301)
- One fireball per surfacing, 17 frames after it surfaces (trace). [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301), [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58)
- The fireball appears 4 pixels right of the Zora, aims once at Link's position, holds still for `$0F` frames (already harmful), then flies straight to the room bound: 1.75 px/frame along one axis, down to 1.25 on both axes for a diagonal. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286), [`Fireball_QSpeedsX`](src/bank4/monster-shots-fireballs.asm#L268), [`Fireball_QSpeedsY`](src/bank4/monster-shots-fireballs.asm#L272)
- Fireballs are not counted in the 4-shot limit; none is made when no object slot is free. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58), [`MonsterShot_Destroy`](src/bank4/monster-shots-fireballs.asm#L189)
- When the cycle wraps to state 0 the object is destroyed and `IsZoraActive` cleared; a new Zora is made in the same frame on another random water square. Mound to mound: 195 frames, 196 when the first search finds no water (trace). [`Zora_Update@DestroyIfDived`](src/bank4/zora-ow-inits.asm#L324), [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L77)
- Damage that did not kill is lost at the dive: HP lowered to `$10` during a surfacing read `$20` on the next (trace, HP set by a RAM write). [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L77), [`Obj_Init@FetchAttrs`](src/bank7/object-init-table.asm#L141)
- It shows its back for the whole surfacing when its Y >= Link's Y at the end of state 1, else its front. [`Burrower_Update`](src/bank4/leever.asm#L94)
- Clock: its update returns at once: not drawn, cannot be hit, cannot shoot, state frozen. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301)
- It has no stun test: a boomerang hit while surfaced does not stop the cycle or the fireball. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301), [`Burrower_Update`](src/bank4/leever.asm#L94)

**Spawn**
- Only in overworld rooms with bit 3 of attribute byte A; one Zora at a time; needs a free object slot (the highest free one is used). [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L77), [`LevelBlockAttrsA`](src/Variables.inc#L695)
- First quest Zora rooms: `$0A` `$17` `$18` `$19` `$1A` `$1E` `$26` `$27` `$28` `$2D` `$2E` `$35` `$36` `$38` `$3E` `$3F` `$44` `$46` `$47` `$48` `$4F` `$54` `$55` `$56` `$59` `$5A` `$5F` `$65` `$69` `$6A` `$6F` `$75` `$7B` `$7C` `$7D` `$7E` `$7F`. [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- Position: up to 13 bytes per frame, `Random+$0D` down to `Random+1`; the first, address `$25`, lies past the 13-byte register, read 0 on every frame observed (trace) and so never qualifies: 12 random candidates. High nibble = X (`$10`-`$E0`), low nibble x `$10` = Y (`$50`-`$D0`, stored as `$5D`-`$DD`); the lower left tile of the square must be water (`$8E`-`$90`, `$92`-`$98`). [`Zora_TrySpawn@TryCandidate`](src/bank4/zora-ow-inits.asm#L99), [`Tile_WalkableTilesOW`](src/bank7/tile-collision.asm#L43)
- No spawn cloud, and never from the screen edge. [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177), [`Obj_Init`](src/bank7/object-init-table.asm#L45)

**Drops**
- In no list: group 3 (41%). [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)
- A killed Zora advances `WorldKillCycle` but is not added to `RoomKillCount`. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)

**Notes**
- A killed Zora leaves `IsZoraActive` set: no Zora appears again until the room is re-entered (trace, death forced by a RAM write: none in 800 frames). [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301), [`IsZoraActive`](src/Variables.inc#L430)
- Two wooden sword hits fit in one surfacing only if the first lands in its first half: 65 frames surfaced against about 32 frames of invincibility. [`Burrower_StateTimes`](src/bank4/leever.asm#L63), [`Shove_Begin@SetShoveDir`](src/bank1/shove-knockback.asm#L215)

# Armos

**Types**
- `$1E`. Attribute `$01`: draws itself and tests its own collisions; turns around after hitting Link. [`Armos_Update`](src/bank4/ghini-armos.asm#L304), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- `$30`: 3 wooden sword hits, 2 white, 1 magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch `$0100` (one heart), also while it fades in (on the frames it is drawn). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200), [`Armos_DrawAndCheckCollisions`](src/bank4/ghini-armos.asm#L332)

**Weapons**
- While it fades in (timer nonzero) only its touch on Link is tested: no weapon can hit it. Afterwards mask `$00`: every weapon hurts. [`Armos_DrawAndCheckCollisions`](src/bank4/ghini-armos.asm#L332)

**Behaviour**
- Fade-in: `$3F` frames after the statue is touched, drawn on even timer values, facing down. [`Armos_InitOrFlyingGhiniFadeIn`](src/bank4/ghini-armos.asm#L154)
- When the fade ends the statue's square becomes floor tile `$26` (or stairs) and the walking speed is picked once: q-speed `$20` (0.5 px/frame) if `Random+1,X` < `$80`, else `$60` (1.5 px/frame). [`Armos_InitOrFlyingGhiniFadeIn@UseChosenTile`](src/bank4/ghini-armos.asm#L237)
- It walks with the Goriya's routine: on every grid point, if the longer axis distance to the chase target is under `$51` pixels it faces the target along that axis, else it keeps its direction. It never shoots. [`Armos_Update`](src/bank4/ghini-armos.asm#L304), [`Goriya_Update`](src/bank4/goriya.asm#L45)
- Clock or stun: it does not move. [`Walker_Move@CheckStunned`](src/bank7/walker-move.asm#L87)

**Spawn**
- Made when Link, on a grid point with a direction held, is blocked by an armos statue square (tiles `$C0`-`$C3`); it appears on that square. [`Spawn_CheckTouchedArmosOrGravestone`](src/bank1/armos-ghini-spawn.asm#L54)
- It takes the highest free object slot `$0B`-`$01`; with none free nothing happens; a square with an Armos still fading in makes no second one. [`Spawn_CheckTouchedArmosOrGravestone@FindObjAtLocation`](src/bank1/armos-ghini-spawn.asm#L158)
- No spawn cloud and no screen-edge entry. [`Obj_Init`](src/bank7/object-init-table.asm#L45)
- Secrets under an Armos at Y = `$80`: stairs in rooms `$0B` (X `$B0`), `$1C` (`$B0`), `$22` (`$30`), `$34` (`$40`), `$3D` (`$90`), `$4E` (`$A0`); the power bracelet in room `$24` (X `$E0`). [`Armos_SecretRoomIds`](src/bank4/ghini-armos.asm#L138), [`Armos_SecretXs`](src/bank4/ghini-armos.asm#L143)
- The stairs are not saved in the room flags and have no quest test: they appear each time that Armos is woken. [`Armos_InitOrFlyingGhiniFadeIn`](src/bank4/ghini-armos.asm#L154)

**Drops**
- None: a dying Armos is changed to type `$5D` (dead dummy), a no-drop type. [`Armos_DrawAndCheckCollisions`](src/bank4/ghini-armos.asm#L332), [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)
- Its death advances neither `WorldKillCycle` nor `RoomKillCount`. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)

**Notes**
- The kill still counts at the hit in `WorldKillCount` and `HelpDropCount`: Armos kills move the fairy and help-drop counters without moving the drop column. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216), [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- The fast Armos (q-speed `$60`, 1.5 px/frame) moves at Link's walking speed; which speed it gets is decided once, by one random byte at wake-up. [`Armos_InitOrFlyingGhiniFadeIn@UseChosenTile`](src/bank4/ghini-armos.asm#L237), [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)

# Ghini (and flying Ghini)

**Types**
- `$21` Ghini, `$22` flying Ghini (released from a gravestone). Attribute `$81` for both. [`Ghini_Update`](src/bank4/ghini-armos.asm#L56), [`Ghini_UpdateFlying`](src/bank4/flying-ghini-peahat.asm#L45), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- Ghini `$90`: 9 wooden sword hits, 5 white, 3 magic. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)
- Flying Ghini: its HP nibble (`$F`) is never used, because no weapon is tested against it. [`Ghini_DrawAndCheckLinkCollision`](src/bank4/ghini-armos.asm#L91)

**Damage to Link**
- Touch `$0100` (one heart) for both; a flying Ghini also on the drawn frames of its fade-in. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200), [`Ghini_FinishFlyingFadeIn`](src/bank4/ghini-armos.asm#L286)

**Weapons**
- Ghini: mask `$00`, every weapon hurts, the boomerang stuns. [`Ghini_Update`](src/bank4/ghini-armos.asm#L56)
- Flying Ghini: immune to everything; it cannot be hurt, stunned or pushed. It dies only when a Ghini dies. [`Ghini_UpdateFlying`](src/bank4/flying-ghini-peahat.asm#L45), [`Ghini_DrawAndCheckLinkCollision`](src/bank4/ghini-armos.asm#L91)

**Behaviour**
- Ghini: wanderer with turn rate `$FF` at q-speed `$20` (0.5 px/frame): on every grid point it faces the chase target if that is less than 9 pixels off on one axis; otherwise it turns perpendicular toward the target when its turn timer is 0. [`Ghini_Update`](src/bank4/ghini-armos.asm#L56), [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Ghini with the clock or stunned: it does not move. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185)
- When a Ghini starts dying, every flying Ghini in slots `$0B`-`$01` is set dying (metastate `$11`) in the same frame. [`Ghini_Update`](src/bank4/ghini-armos.asm#L56)
- Flying Ghini: fades in for `$3F` frames, then flies like a Peahat: speed up from `$1F` to top speed `$A0` (0.625 px/frame), 8 directions, tiles ignored. [`Armos_InitOrFlyingGhiniFadeIn`](src/bank4/ghini-armos.asm#L154), [`Flyer_FinishInit`](src/bank4/zora-ow-inits.asm#L256)
- Flying decide state: `Random,X` >= `$A0` chase (96/256); >= `$08` wander (152/256); else slow down and stop `$40`-`$7F` frames (8/256). [`Ghini_DecideFlightState`](src/bank4/flying-ghini-peahat.asm#L83)
- Flying Ghini with the clock: frozen in place; the stun timer is not tested. [`Ghini_UpdateFlying`](src/bank4/flying-ghini-peahat.asm#L45)

**Spawn**
- Ghini: one per room in first quest rooms `$20`, `$21`, `$30`, `$31`, `$40`, `$41`; spawn cloud first step (slot + 1) x `$10` frames. [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66), [`Octorok_InitSlowOrGhini`](src/bank4/zora-ow-inits.asm#L204)
- Flying Ghini: made when Link, on a grid point with a direction held, is blocked by a gravestone square (tiles `$BC`-`$BF`); it appears on that square in the highest free slot. [`Spawn_CheckTouchedArmosOrGravestone@ChooseObjType`](src/bank1/armos-ghini-spawn.asm#L177)
- No limit on their number other than free slots; the same gravestone gives another once the last one has left the square. [`Spawn_CheckTouchedArmosOrGravestone@FindObjAtLocation`](src/bank1/armos-ghini-spawn.asm#L158)

**Drops**
- Both types: group 1 (59%). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105)
- Flying Ghinis killed by the Ghini's death each go through the drop code (kill cycle +1, `RoomKillCount` +1, own random test) but are not counted in `WorldKillCount` or `HelpDropCount`. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308), [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

**Notes**
- A flying Ghini still fading in when the Ghini dies survives: its init keeps running until the fade ends and then resets the metastate. [`Obj_Update`](src/bank7/object-update-table.asm#L120), [`Ghini_FinishFlyingFadeIn`](src/bank4/ghini-armos.asm#L286)
- Killing the room's one Ghini removes every released flying Ghini at once; with it dead, later ones cannot be killed at all. [`Ghini_Update`](src/bank4/ghini-armos.asm#L56)

# Boulder

**Types**
- `$1F` boulder spawner (invisible, no collisions), `$20` Boulder. Attribute `$81`. [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62), [`Jumper_Update`](src/bank4/tektite-boulder.asm#L166), [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- HP nibble `$F` for both, never used: no weapon is tested against either. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105), [`Jumper_AnimateAndCheckCollisions@DrawBoulder`](src/bank4/tektite-boulder.asm#L423)

**Damage to Link**
- Boulder touch `$0080` (half a heart). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200), [`Obj_DrawNotMirroredAndCheckLinkCollision`](src/bank4/ghini-armos.asm#L132)

**Weapons**
- None works: a Boulder only tests its touch on Link, so it cannot be hurt, stunned or pushed. [`Jumper_AnimateAndCheckCollisions@DrawBoulder`](src/bank4/tektite-boulder.asm#L423)

**Behaviour**
- Spawner: when its timer is 0, fewer than 3 Boulders are alive and an object slot is free, it makes a Boulder at Y = `$40` with a random X in the half of the screen that holds the chase target, then waits 0-`$1F` frames. [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62)
- With 3 Boulders alive the spawner waits a random 0-`$FF` frames; with no free slot it retries every frame. [`Boulder_UpdateSpawner@RandomizeTimer`](src/bank4/tektite-boulder.asm#L111)
- A new Boulder sits at the top for 4, 8, `$14` or `$28` frames, then bounces without pause. [`Boulder_Init`](src/bank4/zora-ow-inits.asm#L164), [`Jumper_Update@Landed`](src/bank4/tektite-boulder.asm#L340)
- Each bounce: start speed -2 px/frame, gravity `$60`/256 px/frame per frame, falling speed capped at 2, X 1 pixel per frame, aimed `$20` pixels lower; computed from the tables it lasts 29 frames and rises 9 pixels first. [`Jumper_StartSpeedsWholeY`](src/bank4/tektite-boulder.asm#L143), [`Jumper_YAccelerations`](src/bank4/tektite-boulder.asm#L131), [`Jumper_TargetYOffsets`](src/bank4/tektite-boulder.asm#L122)
- Before each bounce it turns at most 45 degrees toward the chase target; its direction always keeps the down bit, so it only chooses left or right. [`Jumper_PointBoulderDownward`](src/bank4/tektite-boulder.asm#L461), [`Dir_TurnObjTowardTarget8`](src/bank4/fairy-flyer-common.asm#L407)
- Only the left and right room bounds reverse it; it is destroyed at Y >= `$F0`, which frees its place in the count of 3. [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309), [`Jumper_AnimateAndCheckCollisions@DrawBoulder`](src/bank4/tektite-boulder.asm#L423)
- Clock: Boulders hang in the air; the spawner is not stopped by the clock. [`Jumper_Update@CheckStunned`](src/bank4/tektite-boulder.asm#L179), [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62)

**Spawn**
- First quest rooms with a spawner: `$03`, `$08`, `$16`, `$17`, `$18`, `$19`. [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)
- The spawner's init zeroes the Boulder count; neither type has a spawn cloud. [`Boulder_InitSpawner`](src/bank4/zora-ow-inits.asm#L156), [`Boulder_Init`](src/bank4/zora-ow-inits.asm#L164)

**Drops**
- None: neither type can die. [`Jumper_AnimateAndCheckCollisions@DrawBoulder`](src/bank4/tektite-boulder.asm#L423)

**Notes**
- Boulders fall on the side of the screen the chase target is on: X < `$80` or X >= `$80`. [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62)
- Each Boulder takes one of the object slots `$01`-`$0B`, the pool the Zora and monster shots also need (rooms `$17`-`$19` have both a spawner and a Zora). [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169), [`Zora_TrySpawn`](src/bank4/zora-ow-inits.asm#L77)

# Keese (blue, red, black)

An 8-way flyer with no tile collision. All three colours run the same update; only the start speed differs.

**Types**
- `$1B` blue, `$1C` red, `$1D` black; one update routine for all three. [`Keese_Update`](src/bank4/bubble-keese.asm#L163) [`Obj_Update_JumpTable`](src/bank7/object-update-table.asm#L172)
- Blue uses its own init; red and black call the blue init and then raise the start speed. [`Keese_InitBlue`](src/bank4/bubble-keese.asm#L57) [`Keese_InitRedOrBlack`](src/bank4/bubble-keese.asm#L77)

**HP**
- HP nibble 0 for all three: `ObjHP` starts at `$00`. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: table byte `$80` = half a heart (`$0080`) with no ring; each ring grade halves it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Attribute byte `$81` (bit `$80`): a Keese does not reverse direction after touching Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Weapons**
- Any weapon hit that reaches the damage routine kills: the death test is "HP < damage or HP - damage = 0", true for HP 0 even with damage 0. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- The boomerang therefore kills a Keese (damage 0), on its way out or back. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Sword, sword beam, bomb blast, arrow, fire, rod and magic shot all kill in one hit. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- Invincibility mask `$00`: the mask array is zeroed on room entry and no Keese code writes it. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- Knockback is cancelled at the end of every update: a Keese is never shoved. [`Keese_Update`](src/bank4/bubble-keese.asm#L163)

**Behaviour**
- Flight states (`Flyer_ObjFlyingState`): 0 speed up, 1 decide, 2 chase, 3 wander, 4 slow down, 5 wait. [`Keese_ControlFlight_JumpTable`](src/bank4/bubble-keese.asm#L191)
- Init: one of 8 directions by `Random,X AND 7`; flight state 0; start speed `$1F` (blue) or `$7F` (red, black). [`Keese_InitBlue`](src/bank4/bubble-keese.asm#L57) [`Keese_InitRedOrBlack`](src/bank4/bubble-keese.asm#L77)
- Top speed byte `$C0` = 0.75 px/frame (a diagonal moves 1 pixel on both axes per step); it is one byte shared by every flyer in the room. [`FlyerMaxSpeedFrac`](src/Variables.inc#L395) [`Flyer_Move`](src/bank4/fairy-flyer-common.asm#L250)
- State 0: speed +1 per frame until `(speed AND $E0)` reaches the top speed: `$A1` frames for blue, `$41` for red and black, on the first acceleration. [`Flyer_State0SpeedUp`](src/bank4/fairy-flyer-common.asm#L210)
- State 1 reads `Random+1,X`: `>= $A0` (96/256) chase, `$20`-`$9F` (128/256) wander, `< $20` (32/256) slow down; it loads a turn counter of 6. [`Flyer_KeeseDecideState`](src/bank4/bubble-keese.asm#L205)
- State 2 (chase): 5 turns of at most 45 degrees toward the chase target (`ChaseTargetX/Y`, not always Link), `$10` frames apart, then state 1. [`Flyer_State2Chase`](src/bank4/fairy-flyer-common.asm#L377) [`Dir_TurnObjTowardTarget8`](src/bank4/fairy-flyer-common.asm#L407)
- State 3 (wander): 5 random turns `$10` frames apart (80/256 counter-clockwise, 80/256 clockwise, 96/256 none), then state 1. [`Flyer_State3Wander`](src/bank4/fairy-flyer-common.asm#L507) [`Dir_TurnObjRandomly8`](src/bank4/fairy-flyer-common.asm#L527)
- State 4: speed -1 per frame until below `$20` (`$A1` frames from `$C0`); then `ObjTimer` = random `$40`-`$7F` frames and state 5. [`Flyer_State4SlowDown`](src/bank4/fairy-flyer-common.asm#L200)
- State 5: the Keese rests (below speed `$20` nothing moves) until `ObjTimer` is 0, then state 0. [`Flyer_State5Delay`](src/bank4/fairy-flyer-common.asm#L186)
- After the first slow-down every Keese restarts from speed `$1F`, so red and black are faster only on their first acceleration. [`Flyer_State4SlowDown`](src/bank4/fairy-flyer-common.asm#L200)
- Only the room bounds stop it: crossing one reverses its direction (180 degrees). [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309)
- While `InvClock` or `ItemLiftTimer` is nonzero the Keese neither runs its flight state nor moves; collisions are still tested. [`Keese_Update`](src/bank4/bubble-keese.asm#L163)

**Spawn**
- Made on room entry from the room's monster list, with a spawn cloud. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- Every cellar (mode 9) gets four blue Keese in object slots `$01`-`$04`, whatever the room data says. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64) [`Mode9Cellar_KeeseXs`](src/bank5/spawn-positions.asm#L44)
- Red Keese are made by a Vire that splits (up to two, type `$1C`); no level block in `data/` lists type `$1C`. [`Vire_Update`](src/bank4/vire.asm#L51)
- Black Keese (`$1D`) appear only in the Q2 level blocks (`data/levelblock-uw-*-q2.md`). [`Keese_InitRedOrBlack`](src/bank4/bubble-keese.asm#L77)

**Drops**
- Never drop anything: `$1B`, `$1C` and `$1D` are in the no-drop list. [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)

**Notes**
- Kills: every Keese death adds 1 to `WorldKillCount` and `HelpDropCount` when the killing hit lands. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- A blue or black Keese death advances `WorldKillCycle` (the drop table column) and `RoomKillCount`; a red Keese (`$1C`) death advances neither. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- The flying Gleeok head is initialized through the blue Keese init. [`Keese_InitBlue`](src/bank4/bubble-keese.asm#L57)

# Gel

The small blob. Type `$15` is placed by rooms; type `$14` is the child a Zol splits into.

**Types**
- `$14` child Gel (from a Zol), `$15` Gel; one update routine. [`Gel_Update`](src/bank4/zol-gel.asm#L244)
- Attribute byte `$43`: drawn and collided at half width (middle point X + 4, Y + 8). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Obj_GetMiddle`](src/bank1/monster-collisions.asm#L173)

**HP**
- HP nibble 0 for both: `ObjHP` starts at `$00`. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: table byte `$80` = half a heart (`$0080`) with no ring; each ring grade halves it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- After touching Link a Gel reverses its direction (attribute bit `$80` clear). [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Weapons**
- Any weapon hit kills, including the boomerang's damage 0 (HP 0 meets "HP - damage = 0"). [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Invincibility mask `$00`: zeroed on room entry, never written by Gel or Zol code. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- A Gel tests collisions in every state, the start-up dash of a child included. [`Gel_Update`](src/bank4/zol-gel.asm#L244) [`Gel_CheckCollisions`](src/bank4/zol-gel.asm#L446)

**Behaviour**
- State 2 (wander): q-speed `$40` (1 px/frame), one grid square at a time as a wanderer with turn rate `$20`. [`Gel_Move`](src/bank4/zol-gel.asm#L281) [`Zol_WalkAndPause`](src/bank4/zol-gel.asm#L350)
- At each grid point it pauses `$08`, `$18`, `$28` or `$38` frames (`Random,X AND 3`); it moves again when 4 frames of the pause are left. [`Zol_PauseTimes`](src/bank4/zol-gel.asm#L393) [`Zol_WalkAndPause`](src/bank4/zol-gel.asm#L350)
- Child state 0 (1 frame): `ObjTimer` = 5, then state 1. [`Gel_Move`](src/bank4/zol-gel.asm#L281)
- Child state 1 (dash): q-speed `$FF` (about 4 px/frame) in its facing direction on the 4 frames the timer reads 4 to 1, unless a tile or the room boundary blocks it. [`Gel_Move`](src/bank4/zol-gel.asm#L281) [`Gel_MoveSplitting`](src/bank4/zol-gel.asm#L403)
- When the dash ends or is blocked the Gel snaps to the grid (X = (X + 8) AND `$F0`; Y = ((Y + 8) AND `$F0`) OR `$0D`) and enters state 2. [`Gel_Move`](src/bank4/zol-gel.asm#L281)
- The clock and a stun stop its walking. [`Walker_Move`](src/bank7/walker-move.asm#L64)

**Spawn**
- Type `$15` comes from the room's monster list with a spawn cloud and starts in state 2, skipping the dash. [`Gel_Init`](src/bank4/zol-gel.asm#L235) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- Type `$14` is created in pairs by a splitting Zol, at the Zol's position, in state 0. [`Zol_CreateChildGel`](src/bank4/zol-gel.asm#L147) [`Zol_State2Split`](src/bank4/zol-gel.asm#L113)
- A child Gel first shows the spawn cloud; its update and collisions do not run until the cloud ends. [`Obj_Update`](src/bank7/object-update-table.asm#L120) [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54)

**Drops**
- Never drop anything: `$14` and `$15` are in the no-drop list. [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)

**Notes**
- Every Gel death adds 1 to `WorldKillCount` and `HelpDropCount` when the hit lands. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- A child Gel (`$14`) death advances neither `WorldKillCycle` nor `RoomKillCount`; a type `$15` death advances both. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Drawn 4 pixels right of its X, sprite palette 3, image changing every 2 frames. [`Gel_Update`](src/bank4/zol-gel.asm#L244)

# Zol

The large blob. A hit that does not kill it turns it into two child Gels.

**Types**
- `$13`. [`Zol_Update`](src/bank4/zol-gel.asm#L57)

**HP**
- HP nibble 2: `ObjHP` starts at `$20`. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: table byte `$01` = one heart (`$0100`) with no ring; each ring grade halves it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- A Zol reverses its direction after touching Link (attribute byte `$01`, bit `$80` clear). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- A sliding or splitting Zol (states 1 and 2) tests no collisions: it cannot hurt Link or be hit. [`Zol_CheckCollisions`](src/bank4/zol-gel.asm#L170)

**Weapons**
- Damage `$20` or more kills it outright with no split: white sword `$20`, magic sword `$40` and their beams, rod swing `$20`, magic shot `$20`, arrow `$20`/`$40`, bomb `$40`. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- Damage `$10` splits it: the wooden sword, the wooden sword's beam and candle fire. [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Zol_CheckCollisions`](src/bank4/zol-gel.asm#L170)
- The split test: after the collision pass the Zol is alive (metastate 0) and its invincibility timer is nonzero, which only a damaging hit's knockback sets. [`Zol_CheckCollisions`](src/bank4/zol-gel.asm#L170) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The boomerang stuns it (stun timer `$10`, in 10-frame units) and does not split it: a boomerang hit starts no knockback and sets no invincibility timer. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Any weapon hit except the boomerang copies the weapon's direction into the Zol's direction before the damage. [`Damage_HandleWeaponHit@CheckZolVire`](src/bank1/weapon-damage-death.asm#L143)
- Invincibility mask `$00`: zeroed on room entry, never written by Zol code. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)

**Behaviour**
- State 0 (wander): q-speed `$18` (0.375 px/frame), one grid square at a time as a wanderer with turn rate `$20`. [`Zol_State0Wander`](src/bank4/zol-gel.asm#L87) [`Zol_WalkAndPause`](src/bank4/zol-gel.asm#L350)
- At each grid point it pauses `$18`, `$28`, `$38` or `$48` frames (`Random,X AND 3`); it moves again when 4 frames of the pause are left. [`Zol_PauseTimes`](src/bank4/zol-gel.asm#L393)
- After a splitting hit: if Link faces left or right and the Zol's (Y AND `$0F`) = `$0D`, or Link faces up or down and its (X AND `$0F`) = 0, it goes to state 1 (slide); otherwise to state 2 (split at once). [`Zol_CheckCollisions`](src/bank4/zol-gel.asm#L170)
- State 1 (slide): q-speed `$FF` (about 4 px/frame) in the weapon's direction until an unwalkable tile (tested at grid offset 0) or the room boundary blocks it; then state 2. [`Zol_State1Slide`](src/bank4/zol-gel.asm#L98) [`Gel_MoveSplitting`](src/bank4/zol-gel.asm#L403)
- State 2 (split): the Zol's slot is freed and two child Gels (type `$14`) are created at its position. [`Zol_State2Split`](src/bank4/zol-gel.asm#L113)
- The Gels face left and right when the Zol faced up or down, up and down when it faced left or right: they leave across the line of the hit. [`Zol_State2Split`](src/bank4/zol-gel.asm#L113)
- Each Gel first shows its spawn cloud, then dashes about 4 px/frame for up to 4 frames in its facing direction, snaps to the grid and wanders. [`Gel_Move`](src/bank4/zol-gel.asm#L281) [`Zol_CreateChildGel`](src/bank4/zol-gel.asm#L147)
- The clock and a stun stop its walking. [`Walker_Move`](src/bank7/walker-move.asm#L64)

**Spawn**
- Made on room entry from the room's monster list, with a spawn cloud; init is the common walker init. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- A split raises `RoomObjCount` by 1 (one Zol becomes two Gels). [`Zol_State2Split`](src/bank4/zol-gel.asm#L113)

**Drops**
- Killed outright: drop group 1 (rate `$98` = 152/256). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- Split: no drop from the Zol (its slot is freed without a death) and none from the child Gels. [`Zol_State2Split`](src/bank4/zol-gel.asm#L113) [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)

**Notes**
- A split Zol is not a kill: no counter moves for it, and its child Gels advance neither `WorldKillCycle` nor `RoomKillCount`. [`Zol_State2Split`](src/bank4/zol-gel.asm#L113) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- A Zol killed outright advances `WorldKillCycle` and `RoomKillCount` once. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- The first Gel reuses the Zol's freed slot. If no slot in `$01`-`$0B` is free for the second, the unchecked result leaves Y = 0 and Link's `ObjState`, `ObjGridOffset` and `ObjDir` are overwritten. [`Zol_CreateChildGel`](src/bank4/zol-gel.asm#L147) [`Zol_State2Split`](src/bank4/zol-gel.asm#L113)
- The Vire's state 1 is the same slide routine. [`Zol_State1Slide`](src/bank4/zol-gel.asm#L98)

# Rope

The dungeon snake: slow random walk, then a straight rush when Link lines up with it.

**Types**
- `$28`. [`Rope_Update`](src/bank4/rope.asm#L51)

**HP**
- Q1: `ObjHP` = `$10`. Q2 (`QuestNumbers[CurSaveSlot]` nonzero): `$40`. The init writes it over the table value. [`Rope_Init`](src/bank4/rope.asm#L32)

**Damage to Link**
- Touch: table byte `$80` = half a heart (`$0080`) with no ring; each ring grade halves it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Attribute byte `$89` (bit `$80`): it does not reverse direction after touching Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Weapons**
- Q1: every damaging weapon kills in one hit (lowest damage is `$10`: wooden sword, its beam, fire). [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Rope_Init`](src/bank4/rope.asm#L32)
- Q2 (HP `$40`): one hit of the magic sword, a bomb or a silver arrow (`$40`); two of the white sword, rod, magic shot or wooden arrow (`$20`); four of the wooden sword or fire (`$10`). [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Rope_Init`](src/bank4/rope.asm#L32)
- The boomerang stuns it: stun timer `$10` (10-frame units), damage 0. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Invincibility mask `$00`: zeroed on room entry, never written by Rope code. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- Tested against all of Link's weapons and Link every frame, in every state. [`Rope_Update`](src/bank4/rope.asm#L51) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)

**Behaviour**
- Walks in its facing direction at q-speed `$20` (0.5 px/frame). [`Rope_Update`](src/bank4/rope.asm#L51)
- While walking, each time `ObjTimer` reaches 0 it is reloaded with `Random,X AND $3F` (0-63 frames) and the Rope turns to an unblocked direction (only on a grid point). [`Rope_Update`](src/bank4/rope.asm#L51) [`Walker_FaceUnblockedDir`](src/bank7/walker-collision-screen-edge.asm#L403)
- Rush test, every frame at q-speed `$20` and grid offset 0: if |Link X - Rope X| < 8 it faces up (Link's Y < Rope's Y) or down; else if |Link Y - Rope Y| < 8 it faces right (Link's X >= Rope's X) or left. [`Rope_Update@CheckVertical`](src/bank4/rope.asm#L152)
- The rush sets q-speed `$60` (1.5 px/frame), three times the walk. [`Rope_Update@Rush`](src/bank4/rope.asm#L120)
- The rush test uses Link's real position (`ObjX`, `ObjY` of slot 0), not the chase target. [`Rope_Update`](src/bank4/rope.asm#L51)
- A rush has no timer: it ends when the facing direction changes during a move (a blocked Rope is turned by the tile collision code); q-speed returns to `$20`. [`Rope_Update@CheckSpeed`](src/bank4/rope.asm#L82) [`Walker_CheckTileCollision`](src/bank7/walker-collision-screen-edge.asm#L89)
- While rushing it makes no random turns. [`Rope_Update`](src/bank4/rope.asm#L51)
- With the clock or while stunned it neither moves nor turns at random, but the rush test still sets its direction and q-speed. [`Rope_Update`](src/bank4/rope.asm#L51)

**Spawn**
- Made on room entry from the room's monster list, with a spawn cloud; the init sets HP by quest and then runs the walker init. [`Rope_Init`](src/bank4/rope.asm#L32) [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113)

**Drops**
- Drop group 1 (rate `$98` = 152/256). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)

**Notes**
- Q2 Ropes flash: sprite attributes = `FrameCounter AND 3`, a new palette every frame; Q1 Ropes use attributes 2. [`Rope_Update@Animate`](src/bank4/rope.asm#L125)
- Staying 8 or more pixels off both its row and its column never triggers a rush; the test runs only at q-speed `$20` and grid offset 0. [`Rope_Update`](src/bank4/rope.asm#L51)

# Stalfos

The skeleton: a plain wanderer in Q1; in Q2 it also fires sword shots.

**Types**
- `$2A`. [`Stalfos_Update`](src/bank4/stalfos.asm#L43)

**HP**
- HP nibble 2: `ObjHP` starts at `$20`, in both quests. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: table byte `$80` = half a heart (`$0080`) with no ring; each ring grade halves it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Q2 sword shot (type `$57`): table byte `$02` = two hearts (`$0200`) with no ring. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- The sword shot is parried only by the magic shield, with Link facing against it and his state below `$10` (not swinging or using an item). [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)

**Weapons**
- One hit of damage `$20` or more kills (white or magic sword and beam, rod, magic shot, arrow, bomb); the wooden sword, its beam and fire (`$10`) need two. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- The boomerang stuns it: stun timer `$10` (10-frame units), damage 0. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Invincibility mask `$00`: zeroed on room entry, never written by Stalfos code. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- A weapon hit knocks it back and makes it untouchable (and harmless) for invincibility timer `$10`, about 32 frames. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)

**Behaviour**
- Wanderer with turn rate `$80` at the room-entry default q-speed `$20` (0.5 px/frame); Q1 code never writes its speed. [`Stalfos_Update`](src/bank4/stalfos.asm#L43) [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L265)
- On a grid point, with chance about 129/256, it turns toward the chase target if that is within 8 pixels on one axis; otherwise it turns only when its turn timer is 0, perpendicular to its facing, toward the target. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- With the clock or while stunned it does not move; collisions are still tested. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185) [`Stalfos_Update`](src/bank4/stalfos.asm#L43)
- Q1: never shoots; the update returns after drawing. [`Stalfos_Update`](src/bank4/stalfos.asm#L43)
- Q2: with the shoot timer at 0, a shot sequence starts only on a frame where `Random,X >= $F8` (8/256), its invincibility timer is 0 and "wants to shoot" is set (it turned on its last grid point). [`Stalfos_Update@PrepareToShoot`](src/bank4/stalfos.asm#L65) [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Q2 sequence: `ObjShootTimer` = `$30`, -1 per frame; the Stalfos stands still (q-speed 0) while it runs; the shot leaves when the timer reads `$10`, `$20` frames after the start. [`Stalfos_Update@SetShootTimer`](src/bank4/stalfos.asm#L88)
- The shot is a sword shot (type `$57`) from the Stalfos's position in its facing direction at q-speed `$C0` (3 px/frame). [`Stalfos_Update`](src/bank4/stalfos.asm#L43) [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82)
- No shot if the clock is held or the Stalfos is stunned on the frame the timer reads `$10`; the timer still runs out. [`Stalfos_Update@SetShootTimer`](src/bank4/stalfos.asm#L88)
- A weapon hit (invincibility timer nonzero) zeroes the shoot timer and cancels the sequence. [`Stalfos_Update@PrepareToShoot`](src/bank4/stalfos.asm#L65)
- The shot fails with no free slot in `$01`-`$0B` or with 4 counted monster shots already in flight. [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70) [`Stalfos_Update@CheckResult`](src/bank4/stalfos.asm#L120)
- The sword shot ends at the room boundary or when it harms Link; tiles do not stop it. [`MonsterShot_Update`](src/bank4/monster-shots-fireballs.asm#L102)

**Spawn**
- Made on room entry from the room's monster list, with a spawn cloud; init is the common walker init. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- A Stalfos in object slot 1 carries the room item: the item is drawn on it and moves with it. [`Item_MoveAndDrawRoomItem`](src/bank7/room-item-draw.asm#L65)

**Drops**
- Drop group 1 (rate `$98` = 152/256). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- A Stalfos in object slot 1 never drops anything, not even the forced fairy or help drop. [`Drop_SetUpItem@Found`](src/bank4/drop-items.asm#L206)

**Notes**
- A Q2 Stalfos standing still is in a shot sequence: hitting it before the timer reads `$10` cancels the shot. [`Stalfos_Update@PrepareToShoot`](src/bank4/stalfos.asm#L65)

# Goriya (blue, red)

A walker that throws a boomerang and stands still until it comes back. Lynel and Armos reuse its movement routine.

**Types**
- `$05` blue (sprite palette 1), `$06` red (sprite palette 2); one update routine. [`Goriya_Update`](src/bank4/goriya.asm#L45) [`Obj_Update_JumpTable`](src/bank7/object-update-table.asm#L172)
- The hungry Grumble Goriya is a different object (type `$36`) with its own code. [`Grumble_Update`](src/bank7/object-stubs-destroy-monster.asm#L106)

**HP**
- Blue: nibble 5, `ObjHP` `$50`. Red: nibble 3, `ObjHP` `$30`. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: blue `$01` = one heart (`$0100`); red `$80` = half a heart (`$0080`), with no ring; each ring grade halves it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- The boomerang (type `$5C`), from either colour: `$01` = one heart with no ring. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Any shield (small or magic) parries the boomerang when Link faces against its direction and his state is below `$10` (not swinging or using an item). [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- The boomerang can touch Link only while flying out (state `$1x`), and is tested against him only every other frame. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73) [`Boomerang_AnimateAndCheckCollision`](src/bank7/boomerang-return-draw.asm#L199)
- A Goriya reverses its direction after touching Link (attribute byte `$00`, bit `$80` clear). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Weapons**
- Every weapon damages it; hits to kill blue / red: wooden sword or fire (`$10`) 5 / 3; white sword, rod, magic shot, wooden arrow (`$20`) 3 / 2; magic sword, bomb, silver arrow (`$40`) 2 / 1. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)
- Link's boomerang stuns it: stun timer `$10` (10-frame units), damage 0; a stunned Goriya neither moves nor throws, and its touch does not hurt. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Invincibility mask `$00`: zeroed on room entry, never written by Goriya code. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)
- Attribute `$00`: the play-mode object loop, not the Goriya's update, draws it and runs its collision pass, so it can be hit while waiting for its boomerang. [`Mode5Play_UpdateWorld@LoopObject`](src/bank7/mode5-play-update.asm#L198)

**Behaviour**
- Walks the grid at the room-entry default q-speed `$20` (0.5 px/frame). [`Goriya_Update`](src/bank4/goriya.asm#L45) [`Room_EnterCommon@ResetObjLoop`](src/bank5/enter-room.asm#L265)
- On every grid point it takes |dx| and |dy| to the chase target; if the longer one (dy on a tie) is below `$51` pixels it faces the target along that axis and sets "wants to shoot"; otherwise it clears the flag and keeps its facing. [`Goriya_Update`](src/bank4/goriya.asm#L45)
- Blue throws whenever its `ObjTimer` is 0 and it wants to shoot; red in addition needs `Random,X` to be `$23` or `$77` on that frame (2/256). [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156)
- No throw with the clock, while stunned, without a free slot in `$01`-`$0B`, or with 4 counted monster shots in flight. [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156) [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70)
- The boomerang starts at the Goriya's position and facing: state `$10`, q-speed `$A0` (2.5 px/frame), range `$51` pixels, no spawn cloud. [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156)
- After a throw the Goriya's state is `$80` and its update returns at once every frame: it stands still until the boomerang is caught. [`Goriya_Update`](src/bank4/goriya.asm#L45)
- Boomerang flight: out to the range (only the room bounds block it, not tiles), then `$10` frames at 1 px/frame, `$10` frames returning at half speed, then full speed (2 px/frame on an axis) back to the thrower. [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192) [`Boomerang_UpdateSlowAndReturn`](src/bank7/boomerang-return-draw.asm#L52)
- A boomerang that touches Link (harmed or parried) or a room bound becomes a spark for 3 frames, then returns at full speed. [`Boomerang_DrawAndCheckCollision`](src/bank7/boomerang-return-draw.asm#L217) [`Boomerang_HandleBlockedOrArrow`](src/bank7/boomerang-arrow-food.asm#L404)
- Caught when within 8 pixels of the thrower on both axes: thrower's state = 0 and its `ObjTimer` = `$30`, `$50` or `$70` frames (boomerang slot's `Random,X` < `$30`, < `$70`, else) before the next throw. [`Boomerang_UpdateSlowAndReturn@MonsterCatches`](src/bank7/boomerang-return-draw.asm#L141)

**Spawn**
- Made on room entry from the room's monster list, with a spawn cloud; init is the common walker init. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)

**Drops**
- Red (`$06`): drop group 2 (rate `$68` = 104/256; the group with bombs). [`Drop_Group2Types`](src/bank4/drop-items.asm#L111) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- Blue (`$05`): in no list, so drop group 3 (rate `$68`). [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)

**Notes**
- A Goriya killed while its boomerang is out has the boomerang's slot emptied in the same call. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- That removal does not decrement `MonsterShotCount`, while a caught boomerang does: each such kill uses up one of the room's 4 counted shots until the next room entry zeroes the count. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`MonsterShot_DestroyCounted`](src/bank4/monster-shots-fireballs.asm#L198) [`MonsterShotCount`](src/Variables.inc#L295)
- The "waiting for the boomerang" test is skipped when the object in the next slot (X + 1) is an Armos: the code reads `ObjType+1,X`. [`Goriya_Update`](src/bank4/goriya.asm#L45)
- Lynel and Armos call the same routine for movement and aiming; only types `$05` and `$06` throw. [`Lynel_Update`](src/bank4/moblin-lynel-shooting.asm#L69) [`Armos_Update`](src/bank4/ghini-armos.asm#L304) [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156)

# Wallmaster

The hand that comes out of a dungeon wall when Link stands against it, and carries him to the level's start room.

**Types**
- `$27`. [`Wallmaster_Update`](src/bank4/wallmaster.asm#L91)

**HP**
- HP nibble 2: `ObjHP` starts at `$20`. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: table byte `$80` = half a heart (`$0080`) with no ring; each ring grade halves it. The capturing touch does this damage. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164)

**Weapons**
- State 0 (inside the wall): not drawn, no collision test, cannot be hit. [`Wallmaster_Update`](src/bank4/wallmaster.asm#L91)
- During a trip without Link: all weapons are tested; damage `$20` or more kills in one hit, `$10` (wooden sword, its beam, fire) in two. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- The boomerang stuns it (stun timer `$10`, 10-frame units); a stun or the clock freezes the trip, and a stunned Wallmaster cannot touch Link. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- While carrying Link it runs no collision test: Link's weapons cannot hit it. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231)
- Invincibility mask `$00`: zeroed on room entry, never written by Wallmaster code. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`ObjInvincibilityMask`](src/Variables.inc#L343)

**Behaviour**
- State 0 ends only when all hold: `ObjTimer+1` (slot 1's timer, shared by all Wallmasters) is 0; Link's state is not `$40`; Link's X = `$20` or `$D0` with `$6D` <= Y < `$B5`, or Link's Y = `$5D` or `$BD` with `$29` <= X < `$C8`. [`Wallmaster_Update`](src/bank4/wallmaster.asm#L91)
- Those coordinates are Link standing exactly against the left, right, top or bottom wall, outside the corner regions. [`Wallmaster_Update@CheckLinkAtWall`](src/bank4/wallmaster.asm#L111)
- Emerging sets `ObjTimer+1` to `$60`: at most one Wallmaster emerges per `$60` frames. [`Wallmaster_Update@SetUpToEmerge`](src/bank4/wallmaster.asm#L192)
- It starts inside the wall Link touches: X `$00` (left) or `$F0` (right), or Y `$3D` (top) or `$DD` (bottom). [`Wallmaster_StartXs`](src/bank4/wallmaster.asm#L75) [`Wallmaster_StartYs`](src/bank4/wallmaster.asm#L79)
- Along the wall it starts `$24` pixels from Link if he holds no direction, `$32` if he does; on the up side (side walls) or left side (top and bottom walls) if Link faces that way, otherwise on the down or right side. [`Wallmaster_CalcStartPosition`](src/bank4/wallmaster.asm#L434)
- Trip (state 1): 7 squares of `$10` pixels at q-speed `$18` (0.375 px/frame, about 299 frames): 2 squares out of the wall onto the line Link stands on, 3 along it toward where Link was, 2 back into the wall. [`Wallmaster_StepsLeft`](src/bank4/wallmaster.asm#L55) [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231)
- Started `$24` away, the 3 squares along the wall end `$0C` pixels past Link's position at the time it emerged; started `$32` away, 2 pixels short of it. [`Wallmaster_CalcStartPosition`](src/bank4/wallmaster.asm#L434) [`Wallmaster_StepsLeft`](src/bank4/wallmaster.asm#L55)
- The path is fixed when it emerges; it does not follow Link afterwards. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231)
- Capture: a touch that harms Link during the trip sets `ObjCaptureTimer`; Link's state becomes `$40` (halted), his knockback is cancelled, and each frame he is moved to the hand's position. [`Collision_CheckMonsterVsWeaponsAndLink@CheckCaptors`](src/bank1/monster-collisions.asm#L151) [`Wallmaster_State1Trip@CheckCaptureAndDraw`](src/bank4/wallmaster.asm#L302)
- No capture while Link is invincible or stunned, the clock is held, the Wallmaster is stunned, or its own invincibility timer runs (just hit by a weapon). [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- End of a trip with Link captured: `GameMode` = 3, `GameSubmode` = 0, Link's state 0. Mode 3 in a dungeon sets `RoomId` to `LevelInfo_StartRoomId`: Link restarts at the level's start room. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231) [`Mode3Unfurl_Init_Sub1`](src/bank7/init-mode-dispatch.asm#L294)
- End of a trip without a capture: back to state 0, ready to emerge again. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231)
- A knockback from a weapon hit replaces that frame's step of the trip. [`Wallmaster_State1Trip`](src/bank4/wallmaster.asm#L231)

**Spawn**
- Made on room entry from the room's monster list, without a spawn cloud; it stays hidden in state 0 and its position is set only when it emerges. [`Obj_ResetMetastateAndTimer`](src/bank7/flute-secret-pond.asm#L69) [`Wallmaster_Update`](src/bank4/wallmaster.asm#L91)

**Drops**
- Drop group 1 (rate `$98` = 152/256). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)

**Notes**
- The capturing touch goes through the normal harm path, so it also zeroes `WorldKillCount`, `HelpDropCount` and `IsHelpDropBomb`. [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Mode 3's first init step also clears the room history. [`Mode3Unfurl_Init_Sub0`](src/bank7/init-mode-dispatch.asm#L258) [`RoomHistory_Clear`](src/bank7/init-mode-dispatch.asm#L271)
- Not standing at exactly X `$20`/`$D0` or Y `$5D`/`$BD` keeps every Wallmaster in its wall. [`Wallmaster_Update@CheckLeftAndRight`](src/bank4/wallmaster.asm#L127) [`Wallmaster_Update@CheckTopAndBottom`](src/bank4/wallmaster.asm#L160)
- Its sprites go behind the background when X or X + 8 is below `$18` or at least `$E9`, which hides it inside the side walls. [`Wallmaster_PutSpriteBehindBgIfNeeded`](src/bank4/wallmaster.asm#L500)

# Bubble (flashing, red, blue)

An indestructible wanderer whose touch does no damage but takes away, or gives back, the use of the sword.

**Types**
- `$2B` flashing: cycles through the four sprite palettes, one per frame. [`Bubble_Update@Flash`](src/bank4/bubble-keese.asm#L106)
- `$2C` (sprite palette 1, the blue Goriya's palette): restores the sword. `$2D` (sprite palette 2, the red Goriya's palette): blocks the sword until cured. [`Bubble_Update@BlockOrUnblock`](src/bank4/bubble-keese.asm#L137)
- In the level data `$2C` and `$2D` appear only in the Q2 level blocks; `$2B` appears in both quests (`data/levelblock-uw-*.md`). [`Bubble_Update`](src/bank4/bubble-keese.asm#L93)

**HP**
- HP nibble `$F` (`ObjHP` `$F0`) for all three; it is never used, because no weapon test runs. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Bubble_Update`](src/bank4/bubble-keese.asm#L93)

**Damage to Link**
- Touch: table byte `$00` = no hearts lost. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- The touch still runs the whole harm path: hurt sound, knockback of `$20` pixels, Link's invincibility timer `$18` (about 48 frames). [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The touch zeroes `WorldKillCount`, `HelpDropCount` and `IsHelpDropBomb`: the 16-kill fairy and the 10-kill help drop start over. [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210)

**Weapons**
- Nothing harms, stuns or shoves it: its update calls only the touch test on Link, never the weapon tests. [`Bubble_Update`](src/bank4/bubble-keese.asm#L93) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Sword, beam, boomerang, bomb, arrow, fire, rod and magic shot all pass through it; its invincibility mask (`$00`) is never consulted. [`Bubble_Update`](src/bank4/bubble-keese.asm#L93) [`ObjInvincibilityMask`](src/Variables.inc#L343)

**Behaviour**
- Wanderer at q-speed `$40` (1 px/frame) with turn rate `$40`. [`Bubble_Init`](src/bank4/bubble-keese.asm#L46) [`Bubble_Update`](src/bank4/bubble-keese.asm#L93)
- On a grid point, with chance about 65/256, it turns toward the chase target if that is within 8 pixels on one axis; otherwise it turns only when its turn timer is 0, perpendicular to its facing, toward the target. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- The clock stops its movement. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185)
- Touching `$2B`: `SwordBlockedLongTimer` = `$10`. The timer loses 1 every 10 frames, so the sword is gone for 151 to 160 frames; each new touch reloads it. [`Bubble_Update`](src/bank4/bubble-keese.asm#L93) [`Nmi_Isr@UpdateTimers`](src/bank7/run-game-nmi.asm#L239)
- Touching `$2D`: `IsSwordBlocked` = 1, no timer: the sword stays blocked until something clears the flag. [`Bubble_Update@BlockOrUnblock`](src/bank4/bubble-keese.asm#L137)
- Touching `$2C`: `IsSwordBlocked` = 0. It does not clear `SwordBlockedLongTimer`. [`Bubble_Update@BlockOrUnblock`](src/bank4/bubble-keese.asm#L137)
- While either `SwordBlockedLongTimer` or `IsSwordBlocked` is nonzero the A button does nothing; the B item still works. [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- No touch registers while Link's invincibility timer runs, the clock is held or Link is stunned, so a Bubble cannot re-trigger during the 48 frames after any hit on Link. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Attribute byte `$89` (bit `$80`): it does not reverse direction after touching Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Spawn**
- Made on room entry from the room's monster list, with a spawn cloud; the init sets the q-speed and runs the walker init. [`Bubble_Init`](src/bank4/bubble-keese.asm#L46) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)

**Drops**
- None: a Bubble cannot die. [`Bubble_Update`](src/bank4/bubble-keese.asm#L93)

**Notes**
- Besides Bubble `$2C`, `IsSwordBlocked` is cleared when a heart refill completes and when a save slot is loaded from the file menu. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343) [`Mode1Menu_Update_Sub1Load`](src/bank2/mode1-menu-update.asm#L104)
- Heart refills are started by a potion, the pond fairy, the end-of-level triforce sequence and the Triforce of Power fanfare. [`Potion_Wield`](src/bank5/wield-items.asm#L269) [`PondFairy_Update`](src/bank4/pond-fairy.asm#L43) [`Mode12EndLevel_StartFillingHearts`](src/bank5/mode12-lay-out-room.asm#L118) [`Item_UpdatePowerTriforceFanfare`](src/bank1/tile-block-power-triforce.asm#L99)
- `IsSwordBlocked` (`$52E`) lies outside the `$0300`-`$051F` range zeroed on room entry, so the block survives room changes. [`IsSwordBlocked`](src/Variables.inc#L474) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- A Bubble never dies, so it never adds to `WorldKillCount`, `WorldKillCycle` or `RoomKillCount`. [`Bubble_Update`](src/bank4/bubble-keese.asm#L93) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

# Darknut (red, blue)

Armoured dungeon walker. Its shield is not in its own code: the shared weapon-hit code refuses any hit whose direction is exactly opposite to the way the Darknut faces.

**Types**
- `$0B` red, `$0C` blue; both use the same init and update routines. [`Darknut_Init`](src/bank4/darknut-gibdo.asm#L44) [`Darknut_Update`](src/bank4/darknut-gibdo.asm#L89)
- Attribute byte `$81`: updates, draws and collision-tests itself; does not turn around after touching Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**HP**
- Red `$40`, blue `$80` (HP nibbles 4 and 8, stored times `$10`). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)
- Unblocked hits to kill, red / blue: wooden sword (`$10`) 4 / 8, white (`$20`) 2 / 4, magic (`$40`) 1 / 2, bomb (`$40`) 1 / 2. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)

**Damage to Link**
- Touch only: red 1 heart, blue 2 hearts, before rings (blue ring halves, red ring quarters). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)

**Weapons**
- Invincibility mask `$F6`: only damage type `$01` (sword, sword beam, rod swing) and `$08` (bomb blast) can hurt it. [`Darknut_Init`](src/bank4/darknut-gibdo.asm#L44)
- Boomerang, arrow, Link's magic shot and fire are refused by the mask; boomerang, arrow and magic shot play the parry tune, fire is silent. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127) [`Damage_PlayParryTuneUnlessBombOrFire`](src/bank1/weapon-damage-death.asm#L199)
- Shield: a sword-class or bomb hit does no damage when (weapon direction OR Darknut direction) is exactly `$0C` (up + down) or `$03` (left + right), i.e. the weapon points straight at the Darknut's front. [`Damage_HandleWeaponHit@CheckDarknut`](src/bank1/weapon-damage-death.asm#L157)
- A hit from the side or from behind (weapon direction perpendicular to, or the same as, the Darknut's) goes through. [`Damage_HandleWeaponHit@CheckDarknut`](src/bank1/weapon-damage-death.asm#L157)
- The direction compared is the weapon object's: the sword's and bomb's are Link's facing when swung or placed, the beam's is its flight direction. [`Link_PlaceWeaponAhead`](src/bank1/bomb-candle-wield.asm#L154) [`Damage_HandleWeaponHit@CheckDarknut`](src/bank1/weapon-damage-death.asm#L157)
- A parried sword, beam or rod hit gives the parry tune, no knockback and no invincibility timer. [`Damage_ParryOrShove`](src/bank1/sword-bomb-arrow-collision.asm#L318)
- Bomb: `$40` damage unless parried by the same direction rule (the bomb's direction is Link's facing when he placed it); parried or not, the blast knocks the Darknut back and starts its invincibility timer (`$10`, about 32 frames). [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- A sword beam or magic shot ends on overlap even when it is parried or masked; an arrow turns into a spark. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64) [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- Never stunned: the boomerang is masked, and the stun timer is zeroed every frame anyway. [`Darknut_Update`](src/bank4/darknut-gibdo.asm#L89)

**Behaviour**
- Wanderer with turn rate `$80`: at each 16-pixel grid point it turns toward the chase target when `Random+1,X` <= `$80` (129 in 256) and the target is within 8 pixels on one axis; otherwise it turns perpendicular when its turn timer (random 0-`$FF` frames) has run out. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224) [`Darknut_Update`](src/bank4/darknut-gibdo.asm#L89)
- Speed: red q-speed `$20` (0.5 pixel per frame), blue `$28` (0.625). [`Darknut_Init`](src/bank4/darknut-gibdo.asm#L44) [`Obj_MoveByQSpeed`](src/bank7/walker-move.asm#L244)
- With the clock it does not move, but it is still drawn and collision-tested. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185)
- An unparried sword hit knocks it back `$40` pixels in the weapon's direction and makes it untouchable (no weapon test, no touch on Link) for about 32 frames. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- It never shoots. [`Darknut_Update`](src/bank4/darknut-gibdo.asm#L89)

**Spawn**
- Placed by the room's monster list; starts with the spawn cloud, then faces by the walker rule (vertically toward the chase target, or horizontally away from it). [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113) [`Obj_Update`](src/bank7/object-update-table.asm#L120)

**Drops**
- Red (`$0B`) is in drop group 2 (rate `$68`, 41%); blue (`$0C`) is in no list, so group 3 (rate `$68`, 41%). [`Drop_Group2Types`](src/bank4/drop-items.asm#L111) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)

**Notes**
- The Darknut keeps facing the way it walks after hurting Link (attribute bit `$80`), so its shield side does not flip on contact. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- A bomb placed while Link faces the Darknut head-on does no damage; place it facing any other way relative to the Darknut's facing at the moment of the blast. [`Damage_HandleWeaponHit@CheckDarknut`](src/bank1/weapon-damage-death.asm#L157)

# Gibdo

Mummy; a plain wanderer with no immunities. In object slot 1 it carries the room item.

**Types**
- `$30`; attribute byte `$81`; init is the common walker init. [`Gibdo_Update`](src/bank4/darknut-gibdo.asm#L69) [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177)

**HP**
- `$70` (nibble 7): wooden sword 7 hits, white 4, magic 2, bomb 2. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)

**Damage to Link**
- Touch: 2 hearts before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Weapons**
- Invincibility mask is never set (0): sword, beam, rod, magic shot (`$20`), arrow (`$20` wooden, `$40` silver), bomb (`$40`) and fire (`$10`) all hurt. [`Gibdo_Update`](src/bank4/darknut-gibdo.asm#L69) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- Boomerang: no damage, stun timer `$10` (counted in 10-frame units, about 160 frames); while stunned it does not move and its touch does not hurt Link. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Every damaging hit knocks it back `$40` pixels in the weapon's direction and gives about 32 frames in which nothing is tested against it. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Behaviour**
- Wanderer, turn rate `$80`, q-speed `$20` (0.5 pixel per frame, the room-entry default). [`Gibdo_Update`](src/bank4/darknut-gibdo.asm#L69) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- With the clock or while stunned the walk is skipped; it is still drawn and collision-tested. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185)
- It never shoots. [`Gibdo_Update`](src/bank4/darknut-gibdo.asm#L69)

**Spawn**
- Placed by the room's monster list with a spawn cloud; first facing from the walker rule. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113)

**Drops**
- Drop group 2 (rate `$68`, 41%). [`Drop_Group2Types`](src/bank4/drop-items.asm#L111)
- A Gibdo in object slot 1 never drops anything (it is the room-item carrier). [`Drop_SetUpItem@Found`](src/bank4/drop-items.asm#L206)

**Notes**
- Room item: when slot 1 holds a Gibdo (also Stalfos `$2A`, Like-Like `$17`), the room item's position is overwritten with the monster's every frame and the item is drawn on it; only slot 1 and only the type are tested. [`Item_MoveAndDrawRoomItem`](src/bank7/room-item-draw.asm#L65) [`Item_DrawRoomItemOnMonster`](src/bank7/room-item-draw.asm#L46)
- The carried item is still the ordinary room item in slot `$13`: Link takes it by coming within pickup range of it (under 9 pixels in X and in Y+3), carrier alive or not, unless the room item is inactive or already taken. [`Item_TryTakeRoomItem`](src/bank1/take-item.asm#L173) [`Item_TryTake`](src/bank1/take-item.asm#L206)
- After the carrier dies the item stays at the carrier's last position. [`Item_DrawRoomItemOnMonster`](src/bank7/room-item-draw.asm#L46)
- The slot-1 Gibdo's death still advances the kill cycle and the room kill count; only the drop is suppressed. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)

# Pols Voice

Hopping dungeon monster with high HP that any arrow kills outright.

**Types**
- `$16`; attribute byte `$81`; init is the common walker init. [`PolsVoice_Update`](src/bank4/pols-voice.asm#L82) [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177)

**HP**
- `$A0` (nibble `$A`): wooden sword 10 hits, white 5, magic 3. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: 2 hearts before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Weapons**
- Invincibility mask `$FE`, rewritten every frame: only damage type `$01` (sword, sword beam, rod swing) passes the normal test. [`PolsVoice_DrawAndCheckCollisions`](src/bank4/pols-voice.asm#L197)
- Boomerang (no stun), bomb, fire and Link's magic shot do nothing. [`PolsVoice_DrawAndCheckCollisions`](src/bank4/pols-voice.asm#L197) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- Arrow: on overlap the code sets `ObjHP` to 0 and deals the damage directly, past the mask, so a wooden or silver arrow kills it whatever HP it has. [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- The arrow is not turned into a spark: it flies on and can kill further Pols Voices in its path. [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- A sword hit starts its invincibility timer (`$10`, about 32 frames); during it no weapon is tested, so an arrow passes through without killing it. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- No knockback: its update never runs the knockback mover. [`PolsVoice_Update`](src/bank4/pols-voice.asm#L82)

**Behaviour**
- Moves only on frames where `FrameCounter` is even, 1 pixel per moving frame (0.5 pixel per frame); not at all with the clock or while stunned. [`PolsVoice_Update`](src/bank4/pols-voice.asm#L82) [`PolsVoice_WalkSpeedsX`](src/bank4/pols-voice.asm#L48)
- State 0 (walking): walks `$31` or `$71` moving frames (98 or 226 frames) in one direction, then starts a jump. [`PolsVoice_State1Jumping`](src/bank4/pols-voice.asm#L220) [`PolsVoice_Update@Walk`](src/bank4/pols-voice.asm#L107)
- State 1 (jumping): start speed -3 pixels per moving frame when facing left or right, -1 facing down, -4 facing up; gravity `$38`/256 per moving frame. [`PolsVoice_JumpStartSpeeds`](src/bank4/pols-voice.asm#L60) [`PolsVoice_State1Jumping`](src/bank4/pols-voice.asm#L220)
- A sideways jump lands level, a downward jump `$20` pixels lower, an upward jump `$20` pixels higher; it keeps moving horizontally during the jump. [`PolsVoice_JumpTargetYOffsets`](src/bank4/pols-voice.asm#L65) [`PolsVoice_MoveX`](src/bank4/pols-voice.asm#L353)
- Above Y `$78` it always jumps down; at Y `$A8` or below always up. [`PolsVoice_Update@StartJump`](src/bank4/pols-voice.asm#L159)
- Jump length from the table speeds: 31 moving frames (62 frames) sideways or up, 25 (50 frames) down (derived from the arithmetic, not traced). [`PolsVoice_State1Jumping`](src/bank4/pols-voice.asm#L220)
- On landing: direction = `Random,X` AND 3 (right, left, down, up), walk distance from bit 6 of the same byte, position snapped to the nearest square. [`PolsVoice_LandingDirs`](src/bank4/pols-voice.asm#L69) [`PolsVoice_State1Jumping`](src/bank4/pols-voice.asm#L220)
- Tiles: tested at (X, Y+`$0B`) and (X+`$0E`, Y+`$11`); a block tile (`$B0`-`$B3`) or a tile >= `$F4` starts a jump (ignored if already jumping); any other unwalkable tile reverses its direction, in mid-air too. [`PolsVoice_Update@CheckWalkability`](src/bank4/pols-voice.asm#L121) [`PolsVoice_IsSquareWalkable`](src/bank4/pols-voice.asm#L296)
- It does not chase Link and never shoots. [`PolsVoice_Update`](src/bank4/pols-voice.asm#L82)

**Spawn**
- Placed by the room's monster list with a spawn cloud. Its walk distance is 0 at room entry, so its first moving frame starts a jump. [`PolsVoice_Update@Walk`](src/bank4/pols-voice.asm#L107) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)

**Drops**
- Drop group 1 (rate `$98`, 59%). [`Drop_Group1Types`](src/bank4/drop-items.asm#L105) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)

**Notes**
- The arrow kill counts as an ordinary kill (kill counters, kill cycle, drop). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- Nothing in its code or in the weapon code tests the flute or any sound: only the arrow is special-cased. [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- `PolsVoice_Update` opens with a leftover test of slot 1 whose two outcomes are the same. [`PolsVoice_Update`](src/bank4/pols-voice.asm#L82)

# Like-Like

Slow wanderer that holds Link on contact and removes the magic shield.

**Types**
- `$17`; attribute byte `$81`; init is the common walker init. [`LikeLike_Update`](src/bank4/like-like.asm#L46) [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177)

**HP**
- `$90` (nibble 9): wooden sword 9 hits, white 5, magic 3, bomb 3. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- The capturing touch costs 1 heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- While held (`IsLinkParalyzed` nonzero) no monster and no shot can touch Link, so he takes no further damage. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Weapons**
- Invincibility mask is never set (0): every weapon hurts; the boomerang stuns (about 160 frames). [`LikeLike_Update`](src/bank4/like-like.asm#L46) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Before a capture a damaging hit knocks it back like any wanderer. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185)

**Behaviour**
- Free: wanderer, turn rate `$80`, q-speed `$20` (0.5 pixel per frame); stops with the clock or while stunned. [`LikeLike_Update`](src/bank4/like-like.asm#L46) [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185)
- Capture starts on the frame its touch harms Link: the shared collision pass increments `ObjCaptureTimer` for type `$17`. [`Collision_CheckMonsterVsWeaponsAndLink@CheckCaptors`](src/bank1/monster-collisions.asm#L151)
- On that frame the Like-Like jumps onto Link's exact position, Link's knockback, object timer and metastate are cleared, and `IsLinkParalyzed` becomes 1. [`LikeLike_Update`](src/bank4/like-like.asm#L46)
- A touch that does not harm Link (Link invincible, clock, Like-Like stunned) does not capture. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- While holding, the Like-Like does not move; the capture timer counts 1 per frame. [`LikeLike_Update@HandleCaptured`](src/bank4/like-like.asm#L100)
- Magic shield: when the capture timer reaches `$60`, 95 frames after the capture frame, `InvMagicShield` is set to 0; the timer is then held at `$C0` and the shield is zeroed again every frame. [`LikeLike_Update@IncCaptureTime`](src/bank4/like-like.asm#L111)
- Neither the stun timer nor the clock pauses the capture timer. [`LikeLike_Update@HandleCaptured`](src/bank4/like-like.asm#L100)
- Held Link: his direction input is discarded (he cannot walk or turn), but A and B still reach the input handler, so sword and items still work. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- Release: when this Like-Like's metastate becomes nonzero (it is dying) `IsLinkParalyzed` is cleared. [`LikeLike_Update@DrawAfterCapture`](src/bank4/like-like.asm#L126)
- Release also happens when the per-frame dungeon check finds no counting monster left in the room. [`Secret_UpdateRoomAllDead`](src/bank5/underworld-secrets.asm#L124)

**Spawn**
- Placed by the room's monster list with a spawn cloud; first facing from the walker rule. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113)

**Drops**
- Never drops an item (type `$17` is in the no-drop list). [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)
- Its death still advances the kill cycle and the kill counters. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

**Notes**
- The lost shield is not given back; the small shield remains. [`LikeLike_Update@IncCaptureTime`](src/bank4/like-like.asm#L111)
- Killing the Like-Like within 95 frames of the capture keeps the magic shield. [`LikeLike_Update@IncCaptureTime`](src/bank4/like-like.asm#L111)
- A Like-Like in object slot 1 carries the room item, as Gibdo and Stalfos do. [`Item_MoveAndDrawRoomItem`](src/bank7/room-item-draw.asm#L65)
- While holding Link it is drawn in the two sprites reserved for drawing over Link. [`Sprite_DrawObjMirroredOverLink`](src/bank4/push-block.asm#L218)

# Vire

Hopping wanderer that turns into two red Keese when a hit fails to kill it.

**Types**
- `$12`; attribute byte `$01`; init is the common walker init. [`Vire_Update`](src/bank4/vire.asm#L51) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- `$40` (nibble 4). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: 1 heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- After hurting Link it keeps its direction (special-cased by type). [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Weapons**
- Invincibility mask is never set (0): every weapon type reaches it. [`Vire_CheckCollisions`](src/bank4/vire.asm#L154)
- Killed outright by a hit of `$40` or more: magic sword or its beam, bomb, silver arrow. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- Split by any weaker hit that harms it: wooden or white sword and their beams, rod swing, Link's magic shot, wooden arrow, fire. [`Vire_CheckCollisions`](src/bank4/vire.asm#L154) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The split is triggered by the invincibility timer that the knockback code starts on a hit, tested on the same frame; so the first surviving hit always splits it and HP damage never accumulates. [`Vire_CheckCollisions`](src/bank4/vire.asm#L154)
- Boomerang: stun only (about 160 frames), no invincibility timer, so no split. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82) [`Vire_CheckCollisions`](src/bank4/vire.asm#L154)

**Behaviour**
- State 0: wanderer, turn rate `$80`, q-speed `$20` (0.5 pixel per frame). [`Vire_State0Wander`](src/bank4/vire.asm#L116)
- Hop: while facing left or right, `Vire_HopYOffsets[abs(grid offset)]` is added to Y every frame; the 16 entries sum to 0, so it rises and falls once per square walked. [`Vire_HopYOffsets`](src/bank4/vire.asm#L102) [`Vire_State0Wander`](src/bank4/vire.asm#L116)
- No hop while facing up or down, with the clock, or while stunned. [`Vire_State0Wander`](src/bank4/vire.asm#L116)
- State 1 (after a surviving hit): the hit has turned it to the weapon's direction; it flies that way at q-speed `$FF` (about 4 pixels per frame) until a tile or the room bound blocks it, then state 2. [`Damage_HandleWeaponHit@CheckZolVire`](src/bank1/weapon-damage-death.asm#L143) [`Zol_State1Slide`](src/bank4/zol-gel.asm#L98)
- In state 1 no collisions are checked: it cannot be hit and cannot hurt Link. [`Vire_CheckCollisions`](src/bank4/vire.asm#L154)
- State 2: on the same frame the Vire is destroyed and up to two red Keese (`$1C`) are made at its position with its direction. [`Vire_Update@SplitUp`](src/bank4/vire.asm#L61) [`Shooter_Shoot`](src/bank4/shoot-helpers-candle.asm#L94)
- Food: when the room's first monster type is `$12`, food on the ground becomes the chase target, so Vires walk to it. [`Boomerang_UpdateOrFood@AttractMonsters`](src/bank7/boomerang-arrow-food.asm#L107)

**Spawn**
- Placed by the room's monster list with a spawn cloud; first facing from the walker rule. [`Walker_Init`](src/bank4/walker-wanderer-common.asm#L113)
- Each Keese of a split needs a free slot in `$01`-`$0B`; the Vire's own slot was just freed, so the first always fits, the second only if another slot is free. [`Vire_Update@LoopMakeKeese`](src/bank4/vire.asm#L70) [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
- `RoomObjCount` is raised by 1 at the split whether one or two Keese appear. [`Vire_Update@SplitUp`](src/bank4/vire.asm#L61)

**Drops**
- Killed outright: drop group 2 (rate `$68`, 41%), and the death counts as a normal kill. [`Drop_Group2Types`](src/bank4/drop-items.asm#L111)
- Split: the Vire's slot is simply freed; no kill counter moves and nothing drops. [`Vire_Update@SplitUp`](src/bank4/vire.asm#L61) [`Obj_DestroyMonster`](src/bank7/object-stubs-destroy-monster.asm#L153)
- The red Keese never drop (no-drop list), and a red Keese's death advances neither the kill cycle nor the room kill count. [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Killing a red Keese still adds 1 to `WorldKillCount` and `HelpDropCount` (every weapon kill does). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

**Notes**
- Because red Keese add nothing to `RoomKillCount` while the split raises `RoomObjCount`, a room cleared by splitting its Vires is not saved as fully killed by the count comparison. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Keese have HP 0, so any hit (even the boomerang) kills them. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

# Wizzrobe (blue, red)

Two different monsters sharing collision and drawing code: the blue one walks, teleports and shoots along rows and columns; the red one appears near Link, shoots once and vanishes.

**Types**
- `$23` blue, `$24` red; attribute byte `$81`; both initialized without a spawn cloud. [`Wizzrobe_Blue_Update`](src/bank4/wizzrobe.asm#L95) [`Wizzrobe_Red_Update`](src/bank4/wizzrobe.asm#L550) [`Obj_ResetMetastateAndTimer`](src/bank7/flute-secret-pond.asm#L69)
- Their shots are monster magic shots: type `$58` (blue), `$59` (red). [`Wizzrobe_ShootMagicShot`](src/bank4/wizzrobe.asm#L534)

**HP**
- Blue `$A0` (wooden sword 10 hits, white 5, magic 3, bomb 3); red `$40` (4, 2, 1, bomb 1). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)

**Damage to Link**
- Touch: blue 2 hearts, red 1 heart, before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Magic shot `$58` 2 hearts, `$59` 4 hearts, before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Both shots are blocked only with the magic shield, with Link facing against the shot and not swinging or using an item (state `$00`-`$0F`). [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)

**Weapons**
- Wizzrobes do not use the shared collision pass: their own routine tests only weapon slots `$0E` (sword beam or magic shot), `$10` and `$11` (bomb or fire) and `$0D` (sword). [`Wizzrobe_DrawAndCheckCollisions`](src/bank4/wizzrobe.asm#L683)
- Slots `$0F` (boomerang, food) and `$12` (arrow, rod swing) are never tested: boomerang, arrows and the rod itself pass through, no stun, no damage. [`Wizzrobe_DrawAndCheckCollisions`](src/bank4/wizzrobe.asm#L683)
- Invincibility mask `$F6`: of the tested weapons only the sword, the sword beam and the bomb blast hurt. [`Wizzrobe_DrawAndCheckCollisions`](src/bank4/wizzrobe.asm#L683)
- Fire and Link's magic shot are tested but masked; the magic shot still ends on the overlap. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- A hit starts the invincibility timer (about 32 frames): weapon tests are skipped during it, but the touch on Link is still tested. [`Wizzrobe_DrawAndCheckCollisions@CheckLink`](src/bank4/wizzrobe.asm#L697)
- No knockback: neither update runs the knockback mover. [`Wizzrobe_Blue_Update`](src/bank4/wizzrobe.asm#L95)
- Their magic shots cannot be hit by Link's weapons (attribute bit `$20`). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**Behaviour**
- Blue walking phase: `ObjTimer` = `Random,X` OR `$70` (`$70`-`$7F` or `$F0`-`$FF` frames); while the timer is >= `$10` it steps 1 pixel on every even frame (0.5 pixel per frame). [`Wizzrobe_Blue_AlignWithNearestSquareAndRandomizeTimer`](src/bank4/wizzrobe.asm#L375) [`Wizzrobe_Blue_WalkEveryOtherFrame`](src/bank4/wizzrobe.asm#L154)
- Blue: timer `$0F`-`$02` it stands still (14 frames); at 1 it picks a random diagonal spot `$20` pixels away on both axes and teleports there if that tile is walkable, else starts another walking phase. [`Wizzrobe_Blue_WalkOrTeleport`](src/bank4/wizzrobe.asm#L123) [`Wizzrobe_Blue_ChooseTeleportTarget`](src/bank4/wizzrobe.asm#L320)
- Blue teleport: `$20` frames at 1 pixel per frame per axis, ignoring tiles on the way; on frames with odd distance left it is neither drawn nor collision-tested (cannot be hit, cannot hurt). [`Wizzrobe_Blue_BeginTeleporting`](src/bank4/wizzrobe.asm#L363) [`Wizzrobe_DrawAndCheckCollisionsIntermittently`](src/bank4/wizzrobe.asm#L103)
- Blue walking into a block tile (`$B0`-`$B3`) or a tile >= `$F4` starts a `$20`-pixel teleport straight through it; any other unwalkable tile reverses it. [`Wizzrobe_Blue_MoveAndCheckTile`](src/bank4/wizzrobe.asm#L174)
- Blue turns toward Link every `$40` steps (128 frames), alternating the horizontal and vertical axis; each teleport flips which axis is next. [`Wizzrobe_Blue_TurnTowardLink`](src/bank4/wizzrobe.asm#L241) [`Wizzrobe_Blue_TurnTowardLinkIfNeeded`](src/bank4/wizzrobe.asm#L230)
- Blue shoots only when not teleporting and `FrameCounter` AND `$1F` = 0 (once per 32 frames), and only if it already faces Link: along a shared 16-pixel row (Y high nibbles equal). [`Wizzrobe_Blue_TryShooting`](src/bank4/wizzrobe.asm#L469)
- Blue column shot: Link's X must equal the Wizzrobe's X rounded down to a multiple of `$10` exactly, and it must face him vertically. [`Wizzrobe_Blue_TryShooting_CheckColumn`](src/bank4/wizzrobe.asm#L503)
- Red cycle, 208 frames, `ObjState` counting down from `$FF`: 1 frame choosing a place, 63 fading in (`$FE`-`$C0`), 64 solid (`$BF`-`$80`), 16 fading out (`$7F` jumps to `$4F`, then to `$40`), 64 hidden (`$3F`-`$00`). [`Wizzrobe_Red_Update`](src/bank4/wizzrobe.asm#L550) [`Wizzrobe_Red_State1FadeOut`](src/bank4/wizzrobe.asm#L597)
- Red appears 2, 3, 4 or 5 squares from Link in a cardinal direction (one random byte), facing Link, square-aligned. [`Wizzrobe_Red_AppearOffsetsX`](src/bank4/wizzrobe.asm#L577) [`Wizzrobe_Red_AppearDirs`](src/bank4/wizzrobe.asm#L588)
- Red: the spot is accepted only with Y in `$5D`-`$C3` and a walkable tile; a rejected spot is retried every frame until one fits. [`Wizzrobe_Red_State3Appear`](src/bank4/wizzrobe.asm#L615)
- Red shoots one magic shot `$59` per cycle at state `$B0`, the 16th solid frame, in the direction it faces. [`Wizzrobe_Red_State2Solid`](src/bank4/wizzrobe.asm#L667)
- Red while fading in or out is drawn and collision-tested only on every other frame; while hidden it is not tested at all. [`Wizzrobe_DrawAndCheckCollisionsIntermittently`](src/bank4/wizzrobe.asm#L103) [`Wizzrobe_Red_State0Hidden`](src/bank4/wizzrobe.asm#L659)
- Shots: each needs a free monster slot and fewer than 4 counted monster shots in flight; the sound plays even when the shot is refused. [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70) [`Wizzrobe_ShootMagicShot`](src/bank4/wizzrobe.asm#L534)
- A magic shot flies at q-speed `$C0` (3 pixels per frame), passes over tiles, and ends at the room bound or when it harms Link. [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82) [`MonsterShot_Update`](src/bank4/monster-shots-fireballs.asm#L102)

**Spawn**
- Placed by the room's monster list, no spawn cloud. Blue starts a walking phase on its first update. [`Wizzrobe_Blue_WalkOrTeleport@WalkAgain`](src/bank4/wizzrobe.asm#L133)
- Red starts with state 0, so its first update already chooses a place near Link: its list position is not used. [`Wizzrobe_Red_Update`](src/bank4/wizzrobe.asm#L550)

**Drops**
- Blue: drop group 0 (rate `$50`, 31%). Red: drop group 2 (rate `$68`, 41%). [`Drop_Group0Types`](src/bank4/drop-items.asm#L100) [`Drop_Group2Types`](src/bank4/drop-items.asm#L111)

**Notes**
- Clock: both types skip all movement, state counting and shooting, and are drawn and weapon-tested every frame; a Red Wizzrobe frozen while hidden or fading is fully visible and hittable where it last stood. [`Wizzrobe_Red_Update`](src/bank4/wizzrobe.asm#L550) [`Wizzrobe_Blue_Update`](src/bank4/wizzrobe.asm#L95)
- With the clock neither the Wizzrobe nor a shot can touch Link. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- A Red Wizzrobe is vulnerable on 64 solid frames plus every other frame of the 79 fading ones per 208-frame cycle. [`Wizzrobe_Red_Update_JumpTable`](src/bank4/wizzrobe.asm#L569)
- Ganon reuses the Blue Wizzrobe's step routine. [`Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile`](src/bank4/wizzrobe.asm#L164)

# Lanmola (red, blue)

Two five-segment worms. Each segment is its own object; kills shorten the worm from the tail.

**Types**
- `$3A` red (speed 1), `$3B` blue (speed 2); attribute byte `$41` (self-updating, half width for collisions). [`Lanmola_Init`](src/bank4/zelda-lanmola-inits.asm#L75) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)
- One Lanmola entry in a room makes ten segments in slots `$01`-`$0A`: worm 1 in `$01`-`$05`, worm 2 in `$06`-`$0A`; heads in `$05` and `$0A`, tails in `$01` and `$06`. [`Lanmola_Init`](src/bank4/zelda-lanmola-inits.asm#L75) [`Lanmola_Update`](src/bank4/lanmola.asm#L43)

**HP**
- `$20` per segment for both types (wooden sword 2 hits, white sword 1). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Lanmola_Init`](src/bank4/zelda-lanmola-inits.asm#L75)
- A killed segment is revived with HP `$20`; each worm therefore takes 5 segment kills. [`Lanmola_Update@FindTail`](src/bank4/lanmola.asm#L116)

**Damage to Link**
- Touch of any segment: 2 hearts before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Weapons**
- Invincibility mask is never set (0): sword, beam, rod, magic shot, arrow, bomb and fire all hurt any segment. [`Lanmola_Update`](src/bank4/lanmola.asm#L43) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- Boomerang: sets the segment's stun timer; the segment keeps moving (only the clock stops movement), but a stunned segment's touch does not hurt Link. [`Lanmola_Update`](src/bank4/lanmola.asm#L43) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- No knockback: any shove a hit sets is never played, and a kill clears it; a damaging hit still gives the segment about 32 untouchable frames. [`Lanmola_Update`](src/bank4/lanmola.asm#L43) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- The segment's direction is saved around the collision test, so a hit never turns the chain. [`Lanmola_Update`](src/bank4/lanmola.asm#L43)

**Behaviour**
- Speed: (object type - `$39`) pixels per frame, 1 for `$3A`, 2 for `$3B`; no position fraction. [`Lanmola_Speed`](src/ObjVars.inc#L192) [`Lanmola_Move`](src/bank4/lanmola.asm#L340)
- Only the heads steer. Every 8 pixels the head's routine passes each segment's direction to the one behind it, so segments trail 8 pixels apart. [`Lanmola_UpdateHead@PropagateDirs`](src/bank4/lanmola.asm#L190)
- Every 16 pixels the head chooses: `Random,X` < `$80` turn toward Link; else `Random+1,X` >= `$80` straight on; else a perpendicular turn. [`Lanmola_UpdateHead`](src/bank4/lanmola.asm#L167)
- Toward Link: the horizontal direction is used only if it is not the reverse of the head's and equals the direction Link faces; otherwise the vertical direction toward Link, even when that reverses the head. [`Lanmola_UpdateHead@TurnTowardLink`](src/bank4/lanmola.asm#L297)
- A chosen direction blocked by the room bounds or an unwalkable tile is replaced by the next allowed one (rotating 8, 4, 2, 1), never the reverse. [`Lanmola_UpdateHead@FindNonOpposite`](src/bank4/lanmola.asm#L266)
- Body segments test no tiles and no bounds; they follow the directions handed down. [`Lanmola_Move`](src/bank4/lanmola.asm#L340)
- With the clock the segments stop but are still drawn and can be hit. [`Lanmola_Update`](src/bank4/lanmola.asm#L43)
- It never shoots. [`Lanmola_Update`](src/bank4/lanmola.asm#L43)

**Spawn**
- All ten segments start at (`$40`, `$8D`) with no spawn cloud; only the heads start with a direction (up), and a segment with direction 0 is neither moved, drawn nor tested. [`Lanmola_Init`](src/bank4/zelda-lanmola-inits.asm#L75) [`Lanmola_Update`](src/bank4/lanmola.asm#L43)
- The segments emerge one by one as directions are passed down the chain. [`Lanmola_UpdateHead@PropagateDirs`](src/bank4/lanmola.asm#L190)
- `RoomObjCount` is set to 8, not 10. It makes no difference: the count is read again only when the room is left, against a `RoomKillCount` of at most 2. [`Lanmola_Init`](src/bank4/zelda-lanmola-inits.asm#L75) [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)

**Drops**
- Segment kill while the worm has more than its head: the hit segment lives on; the worm's lowest live slot (the tail) is moved to the hit segment's position and becomes a dead dummy (`$5D`) that plays the death sparkle. [`Lanmola_Update@FindTail`](src/bank4/lanmola.asm#L116)
- A dead dummy drops nothing and advances neither the kill cycle nor the room kill count. [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- When the head is the worm's last segment, the kill is left to run as an ordinary death of type `$3A`/`$3B`: kill cycle +1, `RoomKillCount` +1, drop group 3 (rate `$68`, 41%). [`Lanmola_Update@FindTail`](src/bank4/lanmola.asm#L116) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- So a room of Lanmolas gives at most 2 drops and 2 room kills for 10 segment kills. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Every one of the 10 segment kills adds 1 to `WorldKillCount` and `HelpDropCount` (counted when the killing hit lands). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

**Notes**
- Hitting any segment has the same effect: the worm always loses its tail end, and the head is always the last to go. [`Lanmola_Update@FindTail`](src/bank4/lanmola.asm#L116)
- Starting from `HelpDropCount` 0 and taking no damage, the tenth segment kill (the second head) brings the count to `$0A`, so that head's death gives a forced help drop (or a fairy if `WorldKillCount` reads `$10`); the first head's death gets only the ordinary 41% roll. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)
- With `RoomKillCount` at most 2 against `RoomObjCount` 8, the count comparison on leaving never marks the room fully killed. [`Lanmola_Init`](src/bank4/zelda-lanmola-inits.asm#L75) [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)
- The direction search in the head routine has no exit if all three allowed directions are blocked, but no Lanmola room lets that happen: in their layouts (`$25`, `$26`, `$1A`, `$0A`, `$24`, `$1D`) the only squares with a single open neighbour lie inside the ring of blocks of layout `$1A`, which opens only by its push block, and that moves only when every Lanmola is dead. [`Lanmola_UpdateHead@FindNonOpposite`](src/bank4/lanmola.asm#L266) [`PushBlock_State0Idle`](src/bank4/push-block.asm#L74)

# Aquamentus

The dragon that stands at the right of its room and breathes three fireballs at a time. Q1 rooms: level 1
`$35` and level 7 `$2A` (both hold a heart container as room item); Q2 rooms are in `data/levelblock-uw-*-q2.md`.

**Types**
- Object type `$3D`, one routine, no variants. [`Aquamentus_Update`](src/bank4/aquamentus.asm#L54) [`Obj_Update_JumpTable`](src/bank7/object-update-table.asm#L172)
- Attribute byte `$89`: its own update draws it and checks its collisions; it does not turn around after hitting Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- `$60` (HP nibble 6). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)
- Hits to kill: wooden sword 6, white sword 3, magic sword 2 (the sword beam counts as the sword held); rod swing or magic shot 3; wooden arrow 3, silver arrow 2; bomb blast 2. [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182) [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)

**Damage to Link**
- Touch: `$0100` = 1 heart (table byte `$01`); blue ring halves it, red ring quarters it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Fireball (type `$55`): `$0080` = half a heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- The magic shield parries a type `$55` fireball when Link's state is `$00`-`$0F` (not swinging or using an item) and he faces against one of the fireball's direction bits; the small shield never does. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- Only the 16x16 box at the top left of the drawn 24x32 body (its head) touches Link: the test is Link's middle within 9 pixels of (X+8, Y+8) on both axes. [`Obj_GetMiddle`](src/bank1/monster-collisions.asm#L173) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73) [`Aquamentus_Draw`](src/bank4/aquamentus.asm#L258)

**Weapons**
- Invincibility mask `$E2`: the boomerang (damage type `$02`) and fire (`$20`) do nothing; the boomerang does not stun it and turns back with the parry tune. [`Aquamentus_Init`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Hurt by sword, sword beam and rod swing (type `$01`), arrow (`$04`), bomb blast (`$08`) and magic shot (`$10`). [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- Every weapon is tested against the same point (X+8, Y+8), the head, not the whole drawn body. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Obj_GetMiddle`](src/bank1/monster-collisions.asm#L173)
- A hit that hurts sets `ObjInvincibilityTimer` to `$10` (minus 1 every second frame, about 32 frames): meanwhile no weapon is tested, its touch does not hurt Link and the body flashes. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119)
- Never knocked back: the shove bytes are cleared every frame. [`Obj_PlayBossDeathCryAndResetShove`](src/bank4/aquamentus.asm#L70)
- The boss cry (sample `$02`) plays on the frame of a hit; the death cry when it dies. [`Sample_PlayBossHitCryIfHit`](src/bank4/walker-wanderer-common.asm#L57) [`Sample_PlayBossDeathCryIfDying`](src/bank4/walker-wanderer-common.asm#L70)

**Behaviour**
- Each frame: move, shoot, draw, collisions; with the clock (`InvClock`) it neither moves nor shoots. [`Aquamentus_Update`](src/bank4/aquamentus.asm#L54)
- Moves only horizontally, 1 pixel on every 8th frame (`FrameCounter AND 7` = 0): 0.125 pixels per frame. Y stays `$80`. [`Aquamentus_Move`](src/bank4/aquamentus.asm#L86) [`Aquamentus_StepsX`](src/bank4/aquamentus.asm#L75)
- It paces in legs of 7 or `$F` pixels; one `Random,X` byte per leg gives the length (bit 3) and the direction (bit 0: 0 right, 1 left). [`Aquamentus_Move`](src/bank4/aquamentus.asm#L86)
- Range: found at X < `$88` it is put at `$88` facing right, at X >= `$C8` at `$C7` facing left, each with a new 7-pixel leg; X stays within `$87`-`$C8`. [`Aquamentus_Move@CheckRightLimit`](src/bank4/aquamentus.asm#L125)
- Volley: when `ObjTimer` reads 0, three type `$55` fireballs are created at (X+4, Y) and the timer is set to `Random,X OR $70`: `$70`-`$7F` or `$F0`-`$FF` frames to the next volley. [`Aquamentus_Shoot`](src/bank4/aquamentus.asm#L166) [`Fireball_ShootType55`](src/bank4/aquamentus.asm#L232)
- The spawn cloud leaves `ObjTimer` at 6, so the first volley comes 6 frames after Aquamentus starts acting. [`Obj_AnimateMetastate`](src/bank7/object-animation.asm#L54) [`Aquamentus_Shoot`](src/bank4/aquamentus.asm#L166)
- Each fireball takes the highest free object slot of `$0B`-`$01`; with no free slot it is not made. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58) [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
- Each fireball aims once at Link's position on its first update, stays in place for `$0F` more frames (it can already hit Link), then flies straight; it is destroyed at the room boundary or on touching Link. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- Fireball speed: one of 9 (X, Y) q-speed pairs, from `$70`/`$00` (1.75 pixels per frame along one axis) to `$50`/`$50` (1.25 on both). [`Fireball_QSpeedsX`](src/bank4/monster-shots-fireballs.asm#L268) [`Fireball_QSpeedsY`](src/bank4/monster-shots-fireballs.asm#L272)
- Spread: on even frames while Aquamentus's timer runs, the second fireball of a volley moves 1 pixel down and the third 1 pixel up, from creation on; the first is not shifted. [`Aquamentus_Shoot@SpreadOutFireballs`](src/bank4/aquamentus.asm#L195) [`Aquamentus_ObjFireballOffset`](src/ObjVars.inc#L64)
- The mouth is drawn open while `ObjTimer` < `$20`: the `$20` frames before a volley. [`Aquamentus_Draw`](src/bank4/aquamentus.asm#L258)

**Spawn**
- Made in object slot 1 from monster list id `$3D` (ids `$32`-`$61` always make one object). [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- The init moves it to (`$B0`, `$80`) whatever spawn spot the room gave it, sets the mask and requests roar sample `$10`; it then appears from a spawn cloud. [`Aquamentus_Init`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L40)
- Non-recurring: one kill saves the room as all dead, and such a room makes nothing again. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)

**Drops**
- Drop group 3 (type `$3D` is in no group list): an item only when `Random,X` < `$68` (104 in 256). [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- Group 3 items by `WorldKillCycle` (advanced for this kill first), columns 0-9: heart, heart, fairy, rupee, heart, fairy, heart, heart, heart, rupee. [`Drop_ItemTable`](src/bank4/drop-items.asm#L132) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Forced instead: a fairy when `WorldKillCount` = `$10`, else a bomb or 5 rupees when `HelpDropCount` >= `$0A`. [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216)
- The heart container is the room's item, not a drop: secret trigger 7 shows it when no monster is left. [`Secret_CheckUnderworld`](src/bank5/underworld-secrets.asm#L61) [`Secret_RunTrigger_JumpTable`](src/bank5/underworld-secrets.asm#L105)

**Notes**
- The spread loop shifts every type `$55` object in the room by its own offset byte, not only this volley's. [`Aquamentus_Shoot@LoopObject`](src/bank4/aquamentus.asm#L201)
- Fireballs are not counted in the monster shot limit; only free object slots limit them. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58)
- While Link's invincibility timer, the clock or Link's stun timer is nonzero, fireballs pass through him without being destroyed. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)

# Dodongo

Ordinary damage never hurts a Dodongo. It dies from the second bomb it swallows, or from one sword hit while
a bomb's blast or cloud has it stunned. Only the bomb in object slot `$10` counts for either.

**Types**
- Object types `$31` and `$32` share init, update, HP, attribute and damage values; no Dodongo code tests the type. [`Obj_Update_JumpTable`](src/bank7/object-update-table.asm#L172) [`Dodongo_Init`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L118) [`Dodongo_Update`](src/bank4/dodongo.asm#L83)
- Single Dodongo = monster list id `$32`: ids `$32`-`$61` always make one object, in slot 1. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- Triple Dodongo = list id `$31`: made as often as the room's count field says (slots 1, 2, 3 for a count of 3). [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`LevelInfo_FoeCounts`](src/Variables.inc#L716)
- Q1 rooms: single in level 2 `$0E` (the boss room); triple in level 5 `$56` and level 7 `$0C`, `$58`. [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70) [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)
- Facing left or right the body is 32 pixels wide with `ObjX` its left end; facing up or down it is one 16-pixel box at `ObjX`. [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296) [`Dodongo_Draw`](src/bank4/dodongo.asm#L611)

**HP**
- `ObjHP` is `$F0` (nibble `$F`) and decides nothing: the bomb death never reads it, and the sword death comes with HP left. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296)
- Swallowed bombs are counted per object slot in `Dodongo_ObjBombHits`; 2 kill. The count is never reset by Dodongo code (room entry zeroes it). [`Dodongo_ObjBombHits`](src/ObjVars.inc#L135) [`Dodongo_State1Bloated_Wait`](src/bank4/dodongo.asm#L211)

**Damage to Link**
- Touch: `$0100` = 1 heart before rings, from either 16-pixel half of the body. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296)
- A bloated Dodongo still hurts on touch; a stunned one does not (no touch is tested while the monster's stun timer is nonzero). [`Dodongo_CheckCollisionsStandardSize`](src/bank4/dodongo.asm#L347) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- It shoots nothing. [`Dodongo_Update`](src/bank4/dodongo.asm#L83)

**Weapons**
- General pass: invincibility mask `$FF`. Sword, sword beam, rod, magic shot, arrow and boomerang do no damage, stun or knockback and play the parry tune; bomb blast and fire do nothing at all. [`Dodongo_CheckCollisionsStandardSize`](src/bank4/dodongo.asm#L347) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- Shots that overlap it still end: the beam spreads, the arrow sparks, the boomerang turns back. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64) [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Stunned (state 2) only: the mask becomes `$FE` for one extra test of slot `$0D`, and that test acts only in sword state 2 (the extended step of a swing). [`Dodongo_CheckCollisionsStandardSize`](src/bank4/dodongo.asm#L347) [`Sword_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L192)
- The rod (slot `$12`), the sword beam (slot `$0E`) and every other weapon are never given that test: only the sword itself kills a stunned Dodongo. [`Dodongo_CheckCollisionsStandardSize`](src/bank4/dodongo.asm#L347)
- One sword hit of any grade kills: it subtracts `$10`/`$20`/`$40` from HP `$F0` and starts knockback, whose `ObjInvincibilityTimer` = `$10` the Dodongo code takes as the kill. [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)
- Only slot `$10` is examined for a bomb; a bomb in slot `$11` is neither eaten nor stuns. [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400)
- A bomb goes to slot `$11` when slot `$10` still holds a bomb, which is allowed once that bomb is in its blast or cloud (states `$13`, `$14`: `$18` + `$0C` frames). [`Bomb_Wield`](src/bank1/bomb-candle-wield.asm#L63) [`Bomb_StateTimes`](src/bank7/bomb.asm#L55)

**Behaviour**
- Each frame: state handler, collisions, bomb test, draw. `ObjState`: 0 walking, 1 bloated, 2 stunned. [`Dodongo_Update`](src/bank4/dodongo.asm#L83) [`Dodongo_UpdateState_JumpTable`](src/bank4/dodongo.asm#L108)
- State 0: walks at q-speed `$20` (0.5 pixel per frame, the room-entry default) as a wanderer with turn rate `$20`. [`Dodongo_State0Move`](src/bank4/dodongo.asm#L125) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- Turning, at grid points only: when `Random+1,X` <= `$20` (33 in 256) and the chase target is within 8 pixels on one axis, it faces the target along the other axis; otherwise, once its turn timer (a random 0-`$FF` frames, set at each turn) is 0, it turns perpendicular, toward the chase target. [`Wanderer_MoveAndTurn`](src/bank4/walker-wanderer-common.asm#L224)
- Swallowing needs all of: Dodongo in state 0; the bomb in slot `$10` with its fuse burning (bomb state `$12`, `$30` frames); the bomb's centre in the mouth box. [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400) [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469) [`Bomb_StateTimes`](src/bank7/bomb.asm#L55)
- Mouth box, d = Dodongo centre minus bomb centre (bomb centre = bomb X+8, Y+8; Dodongo centre = (`ObjX`+`$10`, `ObjY`+8) facing left or right, (`ObjX`+8, `ObjY`+9) facing up or down): right dx -16..-1, dy -4..+3; left dx 0..+15, dy -4..+3; down dy -16..-1, dx -8..+7; up dy 0..+15, dx -8..+7. [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469) [`Dodongo_MouthNegativeLimitsX`](src/bank4/dodongo.asm#L366) [`Dodongo_MouthPositiveLimitsX`](src/bank4/dodongo.asm#L371) [`Dodongo_MouthNegativeLimitsY`](src/bank4/dodongo.asm#L376) [`Dodongo_MouthPositiveLimitsY`](src/bank4/dodongo.asm#L381)
- In words: the bomb must lie in the front half of the body, on the Dodongo's own row (or column): a bomb dropped where the head will walk over it while the fuse burns is eaten. [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469)
- A swallowed bomb leaves slot `$10` at once and never explodes; the Dodongo goes to state 1, substate 0. [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469)
- Stun needs all of: Dodongo in state 0; the bomb in slot `$10` in its blast (state `$13`, `$18` frames) or cloud (state `$14`, `$0C` frames); the two centres within -12..+11 pixels on both axes. [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400) [`Dodongo_IsBombInRange`](src/bank4/dodongo.asm#L538) [`Dodongo_BombNegativeLimits`](src/bank4/dodongo.asm#L527)
- So a bomb stuns instead of being eaten when it is outside the mouth box for the whole fuse but within 12 pixels of the centre during the following `$24` frames: placed too far ahead, off the row, or beside the body. [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400)
- Bloated, first bomb: substate 0 `$20` frames (the bomb is counted on its first frame), substate 1 `$40` frames (drawn bloated), substate 4 one frame, then state 0: 97 frames standing still. [`Dodongo_State1Bloated_Wait`](src/bank4/dodongo.asm#L211) [`Dodongo_BloatedWaitTimes`](src/bank4/dodongo.asm#L96) [`Dodongo_State1Bloated_End`](src/bank4/dodongo.asm#L276)
- Bloated, second bomb (count >= 2): substates 0 and 1 as above, substate 2 another `$40` frames (drawn flickering), then substate 3: the death starts on the 161st frame after the bomb was swallowed. [`Dodongo_State1Bloated_Wait`](src/bank4/dodongo.asm#L211) [`Dodongo_State1Bloated_Die`](src/bank4/dodongo.asm#L271)
- Stunned: `ObjStunTimer` is set to `$20` and drops 1 every 10 frames; the state ends when it reads 1: 301-310 frames. The Dodongo stands still, eats no bomb, and can be killed by the sword from the first frame. [`Dodongo_State2Stunned`](src/bank4/dodongo.asm#L171) [`ObjStunTimer`](src/Variables.inc#L52)
- After the stun it stays in place until the timer reaches 0 (up to 10 more frames). [`Walker_Move`](src/bank7/walker-move.asm#L64)
- With the clock (`InvClock`) it does not walk, but the bomb test still runs. [`Walker_Move`](src/bank7/walker-move.asm#L64) [`Dodongo_Update`](src/bank4/dodongo.asm#L83)

**Spawn**
- Appears from a spawn cloud at the spot the room gave its slot; the init requests roar sample `$20` and faces it right when `Random,X` < `$80`, else left. [`Dodongo_Init`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L118) [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)
- Type `$32` is non-recurring: once killed, its room makes nothing again. Type `$31` is below that range: a triple room counts its kills like an ordinary room and refills once it has left the six-room history. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)

**Drops**
- Both deaths start the death sparkle directly, not through the shared kill routine: `WorldKillCount` and `HelpDropCount` do not go up. When the sparkle ends, `WorldKillCycle` and `RoomKillCount` step and the drop is chosen. [`Dodongo_State1Bloated_Die`](src/bank4/dodongo.asm#L271) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Sword death: the kill sets `HelpDropCount` = `IsHelpDropBomb` = `$0A`, which forces the next drop set up (normally this Dodongo's) to be a bomb (item `$00`) with no random roll, and then zeroes both bytes. [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296) [`Drop_SetUpItem@SetHelpItem`](src/bank4/drop-items.asm#L244)
- Exceptions to the forced bomb: `WorldKillCount` = `$10` gives a fairy instead; Link harmed before the sparkle ends zeroes both bytes and the drop becomes the ordinary one. [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216) [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210)
- Bomb death: the ordinary drop of group 3 (types `$31`/`$32` are in no group list): only when `Random,X` < `$68`, item by kill cycle column 0-9: heart, heart, fairy, rupee, heart, fairy, heart, heart, heart, rupee. [`Drop_SetUpItem@RandomlyCancel`](src/bank4/drop-items.asm#L253) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122) [`Drop_ItemTable`](src/bank4/drop-items.asm#L132)
- Bomb death with the help counters already at their cap from earlier kills gives the forced bomb or 5 rupees like any monster. [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216)
- The room item (heart container in level 2 `$0E`, bombs or 5 rupees in the triple rooms) is separate: secret trigger 7 shows it when no monster is left. [`Secret_CheckUnderworld`](src/bank5/underworld-secrets.asm#L61) [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70)

**Notes**
- (trace) Level 2 room `$0E`, two bombs placed 16 pixels in front of the walking Dodongo's nose: each was swallowed on the frame the bomb's X was 24 pixels right of `ObjX` (dx = -16); it stood 97 frames after the first, the death began 161 frames after the second, and 19 frames later the slot held a fairy (group 3, kill cycle 4 -> 5, roll passed). Two bombs used; `WorldKillCount` and `HelpDropCount` unchanged. [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469) [`Dodongo_State1Bloated_Wait`](src/bank4/dodongo.asm#L211) [`Drop_ItemTable`](src/bank4/drop-items.asm#L132)
- (trace) Same room, one bomb placed 52 pixels ahead of the Dodongo's centre: not eaten; the Dodongo was stunned in the cloud (bomb state `$14`) on the frame the centres were 12 pixels apart. One wooden sword stab: HP `$F0` -> `$E0`, death at once, both help bytes `$0A`; 19 frames later the slot held a bomb (item `$00`) and both bytes read 0. One bomb used. [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400) [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296) [`Drop_SetUpItem@SetHelpItem`](src/bank4/drop-items.asm#L244)
- (trace) Left alone, that stun lasted 311 frames and Link standing on the stunned Dodongo lost no health. [`Dodongo_State2Stunned`](src/bank4/dodongo.asm#L171) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Bomb economy: eating costs 2 bombs per Dodongo and gives a random drop; smoke and sword costs 1 and returns a bomb drop. [`Dodongo_State1Bloated_Wait`](src/bank4/dodongo.asm#L211) [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296)
- One ticking bomb feeds one Dodongo (the first to test it removes it), but one blast or cloud can stun every Dodongo within range, each testing slot `$10` itself. [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469) [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400)
- The first frame of the bloat empties slot `$10` a second time: a bomb or candle fire put there on that frame is lost. [`Dodongo_State1Bloated_Wait`](src/bank4/dodongo.asm#L211)
- The bomb test still runs on the death frame (state is back to 0): a ticking bomb in the mouth box is removed with no effect. [`Dodongo_Update`](src/bank4/dodongo.asm#L83)
- The stun timer is not cleared by the sword death, and a bloated Dodongo cannot be stunned or fed (both need state 0). [`Dodongo_CheckCollisions`](src/bank4/dodongo.asm#L296) [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400)

# Manhandla

Five objects: four hands around a base that cannot die. Killing the last hand kills the whole thing; every
dead hand makes the rest faster.

**Types**
- Object type `$3C` for all five segments: hands in object slots 1-4 (up, down, left, right of the base, 16 pixels away), base in slot 5. [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155) [`Manhandla_SegmentOffsetsX`](src/bank4/gleeok-manhandla-gohma-inits.asm#L139) [`Manhandla_SegmentOffsetsY`](src/bank4/gleeok-manhandla-gohma-inits.asm#L143)
- Attribute byte `$89` (copied from slot 1 to all five). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155)
- Q1 rooms: level 3 `$4D` (boss room), level 4 `$10`, level 8 `$2E`, `$6E`, `$7D`. [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70) [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)

**HP**
- `$40` per segment (nibble 4, copied from slot 1). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155)
- Hits to kill one hand: wooden sword 4, white sword 2, magic sword 1; rod swing or magic shot 2; wooden arrow 2, silver arrow 1; bomb blast 1. [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182) [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)

**Damage to Link**
- Touch of any segment: `$0100` = 1 heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Fireball (type `$56`): `$0100` = 1 heart before rings; no shield parries type `$56`. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- A hand that was just hit is not tested against Link while its invincibility timer runs (`$10`, about 32 frames); the base always is. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206)

**Weapons**
- Invincibility mask `$E2` on all five: boomerang (no stun) and fire do nothing. [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155)
- Hurt by sword, sword beam, rod swing, arrow, bomb blast, magic shot. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- A bomb blast (state `$13`) hits every segment whose middle is under `$18` pixels from the bomb's middle on both axes, and its damage `$40` equals a hand's HP: with the hands at their start spacing, a bomb less than 8 pixels from the base's position on both axes kills all four hands in one frame. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Collision_TestPointsWithThresholds`](src/bank1/object-overlap-test.asm#L98) [`Manhandla_SegmentOffsetsY`](src/bank4/gleeok-manhandla-gohma-inits.asm#L143)
- No knockback: the Manhandla code never runs the knockback move, and resets the shove bytes when a segment dies. [`Manhandla_Move`](src/bank4/manhandla.asm#L316) [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206)
- The base takes hits but cannot die: its dying metastate is reset in the same frame and its invincibility timer is cleared every frame, so a weapon overlapping it hits on every frame. [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206)
- Once the base's HP is used up, each further hit on it runs the kill counting again (`WorldKillCount` + 1, `HelpDropCount` + 1 up to `$0A`). [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- The boss cry plays when a hand is hit, the death cry when the last hand dies. [`Sample_PlayBossHitCryIfHit`](src/bank4/walker-wanderer-common.asm#L57) [`Sample_PlayBossDeathCry`](src/bank4/drop-items.asm#L88)

**Behaviour**
- The update runs once per segment per frame, the base (slot 5) before the hands (slots 4 to 1); only the base steers and changes speed, for all five. [`Manhandla_Update`](src/bank4/manhandla.asm#L60)
- Speed is 16-bit (pixels.fraction per frame) and starts at `$0080` = 0.5 pixel per frame. It applies to each axis of the 8-way direction, so a diagonal moves the full amount on both. [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155) [`Manhandla_Move`](src/bank4/manhandla.asm#L316)
- Speed-up: in each frame in which the base finds that a hand died, every segment gains `$0080`: 1.0, 1.5, 2.0 pixels per frame with 3, 2, 1 hands left. [`Manhandla_Update@LoopSegment`](src/bank4/manhandla.asm#L72) [`Manhandla_SegmentJustDied`](src/ObjVars.inc#L168)
- Hands that die in the same frame give only one step, so killing several at once leaves the rest slower. [`Manhandla_Update`](src/bank4/manhandla.asm#L60) [`Manhandla_CheckCollisions@FlagHandDied`](src/bank4/manhandla.asm#L283)
- Direction: one random 8-way direction at init. Every `$10` frames the base turns: `Random,X` >= `$80` at most one 45-degree step toward the chase target, else at random (`Random+1,X` >= `$A0` none, >= `$50` clockwise, else counterclockwise); all segments copy it. [`Manhandla_Update@TurnIfNeeded`](src/bank4/manhandla.asm#L95) [`Dir_TurnObjTowardTarget8`](src/bank4/fairy-flyer-common.asm#L407) [`Dir_TurnObjRandomly8`](src/bank4/fairy-flyer-common.asm#L527)
- No tile collision. A segment that crosses a room bound is reversed, and the base hands that direction to all segments on its next update. [`Manhandla_Move`](src/bank4/manhandla.asm#L316) [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309) [`Manhandla_BounceDir`](src/ObjVars.inc#L170)
- Fireballs: a hand may shoot only on the frame its animation frame changes to 0, and then only if `Random+1,X` >= `$E0` (1 in 8) and object slot 7 is empty. [`Manhandla_Update`](src/bank4/manhandla.asm#L60)
- The animation accumulator grows by the pixels moved plus a random 0-3 each frame, so faster hands reach that frame, and shoot, more often. [`Manhandla_Move@IncFrameAccum`](src/bank4/manhandla.asm#L385)
- Fireball limit: a fireball takes the highest free slot, so at most slots `$0B`-`$07` (5) hold fireballs. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58) [`Manhandla_Update`](src/bank4/manhandla.asm#L60)
- Each fireball (type `$56`) aims once at Link, waits `$0F` frames, then flies straight at 1.25-1.75 pixels per frame. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286) [`Fireball_QSpeedsX`](src/bank4/monster-shots-fireballs.asm#L268)
- The base does not shoot. [`Manhandla_Update`](src/bank4/manhandla.asm#L60)

**Spawn**
- The room makes one type `$3C` object in slot 1; its init turns slots 1-5 into the five segments, with no spawn cloud, and requests roar sample `$40`. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155)
- The base stays on the spawn spot the room gave slot 5 (every slot `$01`-`$09` gets one); the hands are placed around it. [`Manhandla_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L155) [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)
- Non-recurring: once killed the room makes nothing again. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)

**Drops**
- A hand that dies while another hand remains becomes a dead dummy (type `$5D`): no `RoomKillCount` or `WorldKillCycle` step and no drop. [`Manhandla_CheckCollisions@MakeDeadDummy`](src/bank4/manhandla.asm#L290) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308) [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96)
- The last hand keeps type `$3C` through its death sparkle: it is Manhandla's one counted kill and its one drop, group 3 (item only when `Random,X` < `$68`). The base becomes a dying dead dummy. [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206) [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- Every hand's death adds 1 to `WorldKillCount` and to `HelpDropCount` when the killing hit lands. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- The room item (heart container in level 3 `$4D`) is shown by the room's secret trigger when no monster is left. [`Secret_CheckUnderworld`](src/bank5/underworld-secrets.asm#L61)

**Notes**
- What counts as the kill: the death of the last of slots 1-4 still holding type `$3C`. Nothing done to the base ends the fight. [`Manhandla_CheckCollisions@CountHands`](src/bank4/manhandla.asm#L242)
- Hits on the base inflate `WorldKillCount` and `HelpDropCount`: a bomb blast overlapping it scores on every one of its `$18` frames (the first uses up the base's HP `$40`), and a bomb hit that brings `HelpDropCount` to `$0A` makes the next forced drop a bomb. [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- The collision check restores each segment's direction and timer afterwards, so hitting Link never turns Manhandla. [`Manhandla_CheckCollisions`](src/bank4/manhandla.asm#L206)

# Gleeok

A fixed body with one to four necks. Each neck's head must be cut off with the sword; cut-off heads fly around
and cannot be destroyed. The boss dies when every neck has lost its head.

**Types**
- Object types `$42`-`$45`: number of necks = type - `$41` (1 to 4). The Gleeok object is slot 1. [`Gleeok_Update`](src/bank4/gleeok.asm#L175) [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128)
- Type `$46`: a cut-off flying head. [`Gleeok_UpdateFlyingHead`](src/bank4/gleeok.asm#L78)
- Q1 rooms: level 4 `$13` type `$43` (2 heads), level 6 `$18` type `$44` (3), level 8 `$3C` type `$45` (4). No room of either quest uses type `$42`. [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70) [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)
- The body is not an object: 6 sprites at fixed X `$74`-`$84`, Y `$57` and `$67`; it cannot be hit and does not hurt. [`Gleeok_WriteBodySprites`](src/bank4/gleeok-head-body.asm#L412) [`Gleeok_DrawBody`](src/bank4/gleeok-head-body.asm#L391)

**HP**
- Slots 1-6 get HP `$A0` at init. Only slot 5 (the head of the neck being processed) and slot 1 (its base) are ever hit-tested. [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L59) [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128)
- All necks are worked on in the same slots and `ObjHP` is not part of the per-neck data: every attached head shares the one HP byte of slot 5, whichever head is hit. [`Gleeok_Update@LoadNeckBytes`](src/bank4/gleeok.asm#L195) [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L59)
- The first head falls after `$A0` damage in total; cutting a head sets the byte to `$60` for the next. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128)
- Sword hits: first head 10 wooden, 5 white, 3 magic; each later head 6, 3, 2. Rod swing (`$20`): 5, then 3. [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182) [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- Flying head: HP `$F0` from the table, but it cannot die. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Gleeok_UpdateFlyingHead`](src/bank4/gleeok.asm#L78)

**Damage to Link**
- Neck base (slot 1, types `$42`-`$45`) and flying head (`$46`): `$0100` = 1 heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Attached head: slot 5 has object type 0, so the damage byte is entry 0 of the table, `$60`: `$0060` = 3/8 of a heart. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128) [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164) [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Fireball (type `$56`): `$0100` = 1 heart; no shield parries it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- The middle neck segments (slots 2-4) are drawn only: they neither hurt Link nor take hits. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128)

**Weapons**
- Invincibility mask `$FE` on slots 1-6: only damage type `$01` hurts (sword, sword beam, rod swing). Boomerang, arrow, bomb, fire and magic shot do nothing. [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L59) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- A hit that hurts sets the invincibility timer of slot 5 (or 1) to `$10`, about 32 frames; the timer is shared like the HP, so meanwhile no head (or no base) can be hit or hurts Link. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Gleeok_Update@LoadNeckBytes`](src/bank4/gleeok.asm#L195)
- No knockback: the shove bytes are cleared after every collision check. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128)
- The neck base can be hit (the body writhes: its animation runs at 7 frames per step instead of `$11` for 6 steps) but its death is cancelled every time. [`Gleeok_CheckSegmentCollisions@ResetMetastate`](src/bank4/gleeok-head-body.asm#L240) [`Gleeok_DrawBody`](src/bank4/gleeok-head-body.asm#L391)
- The flying head is hit-tested (its slot has no mask), but its metastate, shove bytes and invincibility timer are zeroed every frame: it is never killed or shoved. [`Gleeok_UpdateFlyingHead`](src/bank4/gleeok.asm#L78)

**Behaviour**
- Each frame the body is drawn, then every living neck is loaded into slots 1-6, drawn and hit-tested; a neck moves, and its head may shoot, only when `FrameCounter AND 3` equals its index: once every 4 frames. [`Gleeok_Update`](src/bank4/gleeok.asm#L175)
- Attached head: 1 pixel in X and 1 in Y per move; the X step reverses every `$30` moves and the Y step every `$18` moves (first reversal after half that). No random numbers. [`Gleeok_MoveHead`](src/bank4/gleeok-head-body.asm#L285)
- Necks start at X `$7C`, Y `$6F` (base) to `$83` (head); heads of necks 0-3 start moving after 0, `$0C`, `$18`, `$24` of their moves (0, 48, 96, 144 frames). [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L59) [`Gleeok_SegmentStartYs`](src/bank4/gleeok-manhandla-gohma-inits.asm#L48) [`Gleeok_DecHeadDelay`](src/bank4/gleeok-head-body.asm#L340)
- Reach: a segment (head included) at or beyond its limit from the one below is pulled 2 pixels back; the limit per axis is at most `$0B` pixels. The base never moves. [`Gleeok_MoveNeck`](src/bank4/gleeok.asm#L290) [`Gleeok_CalcSegmentLimits`](src/bank4/gleeok.asm#L456)
- Attached head fireball: on its move frame, type `$56` when `Random+5` < `$20` (1 in 8) and object slot `$0B` is empty. [`Gleeok_Update`](src/bank4/gleeok.asm#L175)
- Flying head fireball: on a frame where its distance counter is even, `Random+1,X` < `$20` and slot `$0B` is empty. [`Gleeok_UpdateFlyingHead`](src/bank4/gleeok.asm#L78)
- Fireball limit: the new fireball takes the highest free slot, which is `$0B`, so one Gleeok fireball exists at a time, attached and flying heads together. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58) [`Gleeok_Update`](src/bank4/gleeok.asm#L175)
- Cutting a head (slot 5 killed): shared HP = `$60`, a flying head (type `$46`) appears at the head's position in slot 7 + neck index (`$07`-`$0A`), the neck's bit is set in `GleeokDeadNeckMask` and the neck is skipped from then on. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128) [`GleeokDeadNeckMask`](src/Variables.inc#L405)
- Flying head: a blue-Keese-style flyer with start speed `$BF` and top speed `$E0`; flight states 0 speed up, 1 decide, 2 chase Link, 3 wander; it never slows or rests. [`Gleeok_InitFlyingHead`](src/bank4/gleeok-manhandla-gohma-inits.asm#L240) [`Gleeok_FlyingHeadFlight_JumpTable`](src/bank4/gleeok.asm#L118)
- Its decide state picks chase when `Random+1,X` < `$D0` (208 in 256), else wander; 5 turns `$10` frames apart, then it decides again. [`Flyer_GleeokHeadDecideState`](src/bank4/gleeok.asm#L130) [`Flyer_SetStateAndTurns`](src/bank4/bubble-keese.asm#L216)
- Death: when the set bits of `GleeokDeadNeckMask` equal the neck count, slot 1 starts its death sparkle (metastate `$11`), the death cry plays and slots `$02`-`$0A` are emptied, which removes the flying heads. Slot `$0B` (a fireball) is left. [`Gleeok_CheckSegmentCollisions@BossDied`](src/bank4/gleeok-head-body.asm#L250)

**Spawn**
- The room makes one object in slot 1; the init sets up all four necks whatever the type, without a spawn cloud, and requests roar sample `$10`. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Gleeok_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L59)
- Non-recurring: once killed the room makes nothing again. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)

**Drops**
- The dying slot 1 is one counted kill (`WorldKillCycle`, `RoomKillCount`) with one drop of group 3: item only when `Random,X` < `$68`. [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308) [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- Every cut-off head, and every cancelled "death" of a neck base, adds 1 to `WorldKillCount` and (up to `$0A`) to `HelpDropCount`. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- The heart container of the level 4 and level 8 rooms is the room item, shown by the secret trigger when no monster is left. [`Secret_CheckUnderworld`](src/bank5/underworld-secrets.asm#L61) [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70)

**Notes**
- Total sword hits on heads for a kill: 2 heads `$A0` + `$60`; 3 heads `$A0` + 2 x `$60`; 4 heads `$A0` + 3 x `$60`. Hits on any head count toward the head that falls next. [`Gleeok_CheckSegmentCollisions`](src/bank4/gleeok-head-body.asm#L128)
- Heads and neck bases are drawn in the first sprite records, in front of Link, and do not flash when hit; the body flashes instead while slot 5's timer runs. [`Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions`](src/bank4/gleeok-head-body.asm#L65) [`Gleeok_WriteBodySprites`](src/bank4/gleeok-head-body.asm#L412)
- The flying head's hits still subtract from its HP `$F0`; once that is used up, each further hit runs the kill counting (`WorldKillCount`, `HelpDropCount`) although the death is cancelled. [`Gleeok_UpdateFlyingHead`](src/bank4/gleeok.asm#L78) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- The Gleeok sprite palette row is chosen for types `$43`-`$45` only; type `$42` would get the level's own row. [`Palette_BossObjTypes`](src/bank7/init-mode5-play.asm#L49)
- The horizontal distance in the neck stretch code is a value minus itself (always 0), so its horizontal contraction handlers never run. [`Gleeok_StretchNeck`](src/bank4/gleeok.asm#L493) [`Gleeok_ContractSegmentX`](src/bank4/gleeok.asm#L612)

# Digdogger

A large body that no weapon can touch. Playing the recorder in its room makes it shrink and split into small
Digdoggers, which are ordinary monsters.

**Types**
- `$38`: big Digdogger that splits into 3; `$39`: big one that splits into 1 (its init rewrites the type to `$38`); `$18`: small (child) Digdogger. One update routine for all. [`Digdogger_InitWith3Children`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L65) [`Digdogger_InitWith1Child`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L99) [`Digdogger_Update`](src/bank4/digdogger.asm#L202)
- The number of children is the room-wide byte `ChildDigdoggerCount`, set by the init. [`ChildDigdoggerCount`](src/Variables.inc#L420) [`Digdogger_InitWith1Child`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L99)
- Q1 rooms: level 5 `$24` type `$39` (boss room); level 7 `$1C` and `$39` type `$38`, `$6C` type `$39`. [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70) [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)
- Attribute bytes: big `$A9` (bit `$20`: Link's weapons are not tested against it), small `$81`. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)
- The big body is 32x32 pixels: bounds and touch are tested at four 16x16 corners, at (0,0), (`$10`,`$10`), (`$10`,0), (0,`$10`) from its position. [`Digdogger_CheckBigCollisionsAndDraw`](src/bank4/digdogger.asm#L253) [`Digdogger_CornerOffsetsX`](src/bank4/digdogger.asm#L72) [`Digdogger_CornerOffsetsY`](src/bank4/digdogger.asm#L77)

**HP**
- Big: `$F0` (nibble `$F`), never reduced. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- Small: `$80` (nibble 8), loaded when the child's slot is initialized. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Obj_Init`](src/bank7/object-init-table.asm#L45)
- Hits to kill a small one: wooden sword 8, white sword 4, magic sword 2; rod swing or magic shot 4; wooden arrow 4, silver arrow 2; bomb blast 2; candle fire 8. [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182) [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)

**Damage to Link**
- Touch of either size: `$0200` = 2 hearts before rings (table bytes `$02` for `$18`, `$38`, `$39`). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- No projectiles. [`Digdogger_Update`](src/bank4/digdogger.asm#L202)
- No touch is tested while it is stunned, or on most frames of the shrink. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`Digdogger_HandleFlute@Shrinking`](src/bank4/digdogger.asm#L108)

**Weapons**
- Big: attribute bit `$20` makes the collision pass skip every weapon test; nothing hits, stuns or is parried, weapons pass through. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)
- Small: no Digdogger code sets an invincibility mask (the byte is 0 from the room-entry clear), so every weapon hurts it and the boomerang stuns it (`ObjStunTimer` `$10`, in 10-frame units). [`Digdogger_HandleFlute@LoopMakeChild`](src/bank4/digdogger.asm#L139) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82) [`Util_ClearRam0300UpTo`](src/bank7/ppu-tablejump-input.asm#L162)
- A stunned Digdogger does not move. [`Digdogger_Update`](src/bank4/digdogger.asm#L202)
- A hit small one gets the usual invincibility timer `$10` (about 32 frames), but no Digdogger routine moves it by knockback. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Digdogger_Update`](src/bank4/digdogger.asm#L202)

**Behaviour**
- The recorder in a dungeon only sets `FluteUsedState` from 0 to 1 (plus the tune and the `$98`-frame freeze in which nothing is updated); once nonzero, further uses change nothing. [`Flute_Wield@FlagFluteUsed`](src/bank7/flute.asm#L127) [`Flute_Wield`](src/bank7/flute.asm#L49) [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59)
- `FluteUsedState` is one byte for the room, zeroed on room entry: the recorder must be played in the Digdogger's room, any time after entering. [`FluteUsedState`](src/Variables.inc#L439) [`Digdogger_Update`](src/bank4/digdogger.asm#L202)
- On its first update after the freeze a big Digdogger sets `ObjTimer` to `$40` and `FluteUsedState` to 2 and stops moving. [`Digdogger_HandleFlute`](src/bank4/digdogger.asm#L91)
- Shrink: `$40` frames without moving, drawn small except on the frames the timer reads `$30`, `$20`, `$10`, when it is drawn big and tested for touch. [`Digdogger_HandleFlute@Shrinking`](src/bank4/digdogger.asm#L108)
- Split, when the timer reads 0 (about `$98` + `$40` frames after the recorder if nothing delays it): `FluteUsedState` back to 1, `RoomObjCount` = number of children, death cry, the parent's type cleared. [`Digdogger_HandleFlute@MakeChildren`](src/bank4/digdogger.asm#L128)
- Children are written to the slots right after the parent's (not tested for being free), all at the position of object slot 1, each with a random 8-way direction. [`Digdogger_HandleFlute@LoopMakeChild`](src/bank4/digdogger.asm#L139)
- A child slot unused since room entry is still uninitialized with a spawn cloud pending: the child first loads attribute `$81` and HP `$80`, plays the cloud, then moves. [`Digdogger_HandleFlute@LoopMakeChild`](src/bank4/digdogger.asm#L139) [`Obj_Init`](src/bank7/object-init-table.asm#L45) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- With the clock (`InvClock`) or while stunned, the flute handling is skipped: a big Digdogger neither starts nor continues its shrink. [`Digdogger_Update`](src/bank4/digdogger.asm#L202)
- Speed is 16-bit and changes by 1 per frame toward a target, flipping at the target: big `$003F` at start, then between `$0040` and `$0080`; small from `$003F` up to `$0180`, then between `$0140` and `$0180`. [`Digdogger_SpeedUp`](src/bank4/digdogger.asm#L364) [`Digdogger_SlowDown`](src/bank4/digdogger.asm#L389) [`Digdogger_SetTargetSpeed`](src/bank4/digdogger.asm#L414)
- Pixels per frame = speed high byte + carry of (low byte AND `$E0`) added to a fraction: big 0.25-0.5 (0.125 at the very start), small 1.25-1.5 once up to speed, which takes `$141` frames from the split. [`Digdogger_Move`](src/bank4/digdogger.asm#L437) [`Digdogger_SpeedUp`](src/bank4/digdogger.asm#L364)
- The speed applies to each axis of the 8-way direction; no tile collision, only the room bounds, which reverse the direction. [`Digdogger_Move`](src/bank4/digdogger.asm#L437) [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309)
- Every `$10` frames it turns at most one 45-degree step: `Random,X` >= `$80` toward the chase target, else at random (`Random+1,X` >= `$A0` none, >= `$50` clockwise, else counterclockwise). [`Digdogger_ChangeSpeedAndTurn`](src/bank4/digdogger.asm#L215) [`Dir_TurnObjTowardTarget8`](src/bank4/fairy-flyer-common.asm#L407) [`Dir_TurnObjRandomly8`](src/bank4/fairy-flyer-common.asm#L527)

**Spawn**
- One object in slot 1 from list id `$38` or `$39`, from a spawn cloud at the room's spawn spot, random 8-way direction, roar sample `$40`. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [`Digdogger_InitWith3Children`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L65)
- Non-recurring: the first kill of a small one saves the room as all dead, so it makes nothing again once left, even with other small ones alive. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110) [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)

**Drops**
- The split removes the parent without the death code: no kill counted, no drop. [`Digdogger_HandleFlute@MakeChildren`](src/bank4/digdogger.asm#L128)
- Each small one is an ordinary kill of drop group 3 (type `$18` is in no group list): item only when `Random,X` < `$68`. [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- The room item (heart container in level 5 `$24`) is shown by the room's secret trigger when no monster is left. [`Secret_CheckUnderworld`](src/bank5/underworld-secrets.asm#L61) [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70)

**Notes**
- Before the recorder the big Digdogger cannot be affected in any way; without the recorder the room cannot be cleared. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Digdogger_HandleFlute`](src/bank4/digdogger.asm#L91)
- The shrink path does not test the size: while one big Digdogger shrinks (`FluteUsedState` = 2), every other Digdogger in the room takes it too and splits when its own `ObjTimer` reads 0. [`Digdogger_HandleFlute@Shrinking`](src/bank4/digdogger.asm#L108)
- After any split `ChildDigdoggerCount` reads 3, because each child runs the 3-children init. [`Digdogger_HandleFlute@LoopMakeChild`](src/bank4/digdogger.asm#L139) [`Digdogger_InitWith3Children`](src/bank4/aquamentus-digdogger-dodongo-inits.asm#L65)
- A corner of the big body that crosses a room bound reverses the direction; two such corners in one frame cancel out. [`Digdogger_CheckBigCollisionsAndDraw`](src/bank4/digdogger.asm#L253)

# Gohma (blue, red)

A stationary-looking boss that paces out and back from one spot. Only an arrow flying up into the fully open eye deals damage.

**Types**
- `$33` (sprite palette row 1) and `$34` (row 2); both run the same update. [`Gohma_Update`](src/bank4/gohma.asm#L63) [`Gohma_SetSpriteAttributes`](src/bank4/gohma.asm#L301)
- Attribute byte `$C9`: draws and collides itself, half-width collision middle (X + 4), does not turn round after hitting Link. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**HP**
- Type `$33`: `$60` (3 wooden or 2 silver arrows). Type `$34`: `$20` (1 arrow of either kind). [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367)

**Damage to Link**
- Touch: damage byte `$02` = `$0200`, 2 hearts before rings, on any of the five collision parts. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Gohma_CheckCollisions`](src/bank4/gohma.asm#L322)
- Fireball type `$56`: `$0100`, 1 heart before rings; no shield blocks it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Each ring grade halves both (blue 1/2, red 1/4). [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)

**Weapons**
- Invincibility mask `$FB`: every damage type except the arrow (4) is refused. Sword, sword shot, rod, magic shot and boomerang only play the parry sound; bomb and fire do nothing, silently. [`Gohma_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L221) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- The boomerang does not stun it (damage type 2 is masked). [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- An arrow harms only when all three hold: it hit collision part 3 or 4 (the eye), the eye state is 3 (fully open), and the arrow's direction is up (`8`). Otherwise only the parry sound. [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367)
- Arrow damage is `$20` wooden, `$40` silver. [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249)
- Collision runs in 5 parts, 8 pixels apart, from `ObjX - $10` (part 5) to `ObjX + $10` (part 1), tested left to right; part 4 is at `ObjX - 8`, part 3 at `ObjX`. [`Gohma_CheckCollisions`](src/bank4/gohma.asm#L322)
- X window of the eye: an arrow turns into a spark on the first part it overlaps, so the eye is reached only when the arrow's hit point X is `ObjX - 1` to `ObjX + $0E`; with Link facing up the hit point is the arrow's X + 6, so the arrow's X must be Gohma's `ObjX - 7` to `ObjX + 8`. [`Gohma_CheckCollisions`](src/bank4/gohma.asm#L322) [`Collision_CheckThinWeaponVsMonster_Thresholds`](src/bank1/object-overlap-test.asm#L47)
- An arrow that touches any part is used up (spark), hit or not. [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367) [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- An arrow on part 3 or 4 with the eye at state 3 starts Gohma's invincibility timer (`$10`, about 32 frames with no weapon or Link test on any part), whatever the arrow's direction: a sideways arrow into the open eye does no damage but still buys that pause. [`Shove_Begin@Gohma`](src/bank1/shove-knockback.asm#L199)
- Knockback never moves Gohma. [`Gohma_Update`](src/bank4/gohma.asm#L63)

**Behaviour**
- Eye states: 0 and 1 closed (alternating every 8 frames), 2 half open, 3 fully open. [`Gohma_Update@AnimateClosedEye`](src/bank4/gohma.asm#L215) [`Gohma_ObjEyeState`](src/CommonVars.inc#L17)
- An opening lasts `$80` frames: `$11` frames at state 2, `$60` (96) frames at state 3, `$0F` frames at state 2. [`Gohma_Update@AnimateEye`](src/bank4/gohma.asm#L152) [`Gohma_ObjEyeOpenTimer`](src/ObjVars.inc#L183)
- Openings start when a counter reaches 0; it is reloaded with `$C0 OR Random,X` and drops by 1 on odd `FrameCounter` frames only, so openings start 384 to 510 frames apart. The counter is 0 on room entry: the eye starts opening on the first update. [`Gohma_Update@DecNextOpenEyeCounter`](src/bank4/gohma.asm#L167) [`Gohma_ObjNextOpenEyeCounter`](src/ObjVars.inc#L181)
- Movement: 1 pixel every 2 frames. Direction by `Random,X`: `>= $B0` right, `>= $60` left, else down (never up first). It goes `$20` pixels (64 frames), reverses, comes back `$20` pixels, then picks again after one still frame. [`Gohma_Update`](src/bank4/gohma.asm#L63) [`Gohma_ObjDistTraveled`](src/ObjVars.inc#L179)
- It therefore always returns to its start point and stays within X `$60`-`$A0`, Y `$70`-`$90`; no tile or room-bound test is made. [`Gohma_Update@Move`](src/bank4/gohma.asm#L90)
- Fireball: type `$56` every `$41` (65) frames, independent of the eye; the first on the first update. Only a full object table (slots `$01`-`$0B`) stops it; fireballs are not counted against the 4-shot limit. [`Gohma_Update@ShootAnimateCollide`](src/bank4/gohma.asm#L197) [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58)
- The update has no clock test: with the clock Gohma still moves, cycles its eye and shoots, but neither it nor its fireballs can touch Link. [`Gohma_Update`](src/bank4/gohma.asm#L63) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)

**Spawn**
- Placed at (`$80`, `$70`) with no spawn cloud; roar sample request `$20`; shoot timer set to 1. [`Gohma_Init`](src/bank4/gleeok-manhandla-gohma-inits.asm#L221)

**Drops**
- Ordinary death; the types are in no drop list, so drop group 3 (rate `$68`/256, about 41%, unless a forced drop applies). [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157) [`Drop_GroupRates`](src/bank4/drop-items.asm#L122)
- One kill saves the dungeon room as all dead when Gohma is the room's first monster type (types `$32`-`$48` except `$3A`/`$3B`). [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)

**Notes**
- A counted hit ends in the parry sound request too (fall-through), replacing the "harmed" sound: the sound is not a reliable sign of a miss. [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367)
- Because parts are tested left to right, an arrow slightly left of the eye (hit point X `<= ObjX - 2`) sparks on the leg part and is wasted. [`Gohma_CheckCollisions`](src/bank4/gohma.asm#L322)

# Moldorm

Two five-segment worms made from one room object. Killing any segment shortens the worm from the tail.

**Types**
- `$41`; attribute byte `$C1` (draws and collides itself, half-width collision middle, does not turn round after hitting Link). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Moldorm_Update`](src/bank4/moldorm.asm#L59)
- Ten segments in object slots `$01`-`$0A`: worm A slots `$01`-`$05` (head `$05`), worm B slots `$06`-`$0A` (head `$0A`). [`Moldorm_Init`](src/bank4/moldorm-init.asm#L38)

**HP**
- `$20` per segment, copied from slot `$01` to all ten. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Moldorm_Init@LoopSegment`](src/bank4/moldorm-init.asm#L43)
- A segment that is "killed" gets HP `$20` back at once (see Notes); damage never carries from one segment to another. [`Moldorm_Update`](src/bank4/moldorm.asm#L59)

**Damage to Link**
- Touch: damage byte `$80` = `$0080`, half a heart before rings, per segment. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Weapons**
- No invincibility mask is set (the byte is 0 after room entry), so every weapon's damage counts: sword `$10`/`$20`/`$40`, sword shot the same, rod `$20`, magic shot `$20`, arrow `$20`/`$40`, bomb `$40`, fire `$10`. [`Moldorm_Init`](src/bank4/moldorm-init.asm#L38) [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- So one kill takes 2 wooden sword hits or 2 fire hits, and 1 hit of anything stronger. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)
- The boomerang sets the segment's stun timer but does no damage; nothing in the update tests the stun timer, so the segment keeps moving, yet its touch cannot hurt Link while stunned. [`Moldorm_Update`](src/bank4/moldorm.asm#L59) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- A hit starts the usual `$10` invincibility timer (about 32 frames without any test on that segment), but never moves or turns it: direction and timer are restored after the collision pass and the update never runs the knockback. [`Moldorm_Update`](src/bank4/moldorm.asm#L59) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)

**Behaviour**
- All segments move 1 pixel every 2 frames (flyer speed `$80`), in 8 directions, with no tile collision. [`Flyer_Move`](src/bank4/fairy-flyer-common.asm#L250) [`Moldorm_Init`](src/bank4/moldorm-init.asm#L38)
- Only the heads steer. When a head's timer runs out it picks chase (state 2) if `Random,X >= $40`, else wander (state 3), with 8 turns `$10` frames apart: a state lasts `$80` frames. [`Moldorm_DecideFlightState`](src/bank4/moldorm.asm#L281) [`Moldorm_ChangeFlyingState`](src/bank4/moldorm.asm#L189)
- Chase turns toward the chase target (not always Link); wander turns by `Random+1,X`: `>= $A0` straight, `$50`-`$9F` one step right, `< $50` one step left. [`Moldorm_Chase`](src/bank4/moldorm.asm#L172) [`Moldorm_Wander`](src/bank4/moldorm.asm#L210)
- Every `$10` frames (8 pixels) each segment takes the old direction of the one ahead, so the body follows the head's path. [`Moldorm_PropagateDirs`](src/bank4/moldorm.asm#L229)
- A segment whose direction is 0 is skipped entirely: not moved, not drawn, not hittable, harmless. At the start only the heads have a direction; one more segment leaves the start point every `$10` frames. [`Moldorm_Update`](src/bank4/moldorm.asm#L59) [`Moldorm_PropagateDirs`](src/bank4/moldorm.asm#L229)
- A head at a room bound does not reverse at once: the reversed direction is stored and applied at the next propagation. [`Moldorm_DeferBounce`](src/bank4/fairy-flyer-common.asm#L358) [`Moldorm_ObjBounceDir`](src/ObjVars.inc#L114)
- The clock stops all movement and steering; segments are still drawn and hittable. [`Moldorm_Update`](src/bank4/moldorm.asm#L59)

**Spawn**
- One type `$41` object becomes ten segments, all at (`$80`, `$70`), no spawn cloud; head directions come from `Random+5` and `Random+10` (AND 7, 8-way). [`Moldorm_Init`](src/bank4/moldorm-init.asm#L38)
- `RoomObjCount` is set to 8, not 10. It makes no difference: the count is read again only when the room is left, and there one kill already marks a room whose first type is `$41` as cleared. [`Moldorm_Init`](src/bank4/moldorm-init.asm#L38) [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)

**Drops**
- Only the two head deaths (type `$41`) advance `WorldKillCycle`, add to `RoomKillCount` and can drop an item (drop group 3, rate `$68`/256). [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308) [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)
- The 8 tail deaths are dead dummies (type `$5D`): no drop, no kill cycle step, no room kill. [`Drop_NoDropTypes`](src/bank4/drop-items.asm#L96) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Every one of the 10 kills still adds 1 to `WorldKillCount` and (up to `$0A`) to `HelpDropCount` when the hit lands. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

**Notes**
- Killing any segment removes the worm's tail instead: the lowest slot of that worm still holding type `$41` becomes a dead dummy at the hit segment's position, and the hit segment lives on with HP `$20`. [`Moldorm_Update`](src/bank4/moldorm.asm#L59)
- A worm therefore dies after 5 kills anywhere on it; the fifth kills the head itself. It does not matter which segment is hit. [`Moldorm_Update`](src/bank4/moldorm.asm#L59)
- `RoomKillCount` ends at 2, below the `RoomObjCount` of 8, but type `$41` is non-recurring: one head kill saves the dungeon room as all dead when Moldorm is the room's first monster type. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)
- Each kill plays the boss death cry. [`Moldorm_Update`](src/bank4/moldorm.asm#L59)

# Patra

A slow flyer circled by eight children. The body cannot be hit by any weapon until every child is dead.

**Types**
- Patra `$47` with eight children of type `$25`; Patra `$48` with eight children of type `$26`. Patra is object slot `$01`, the children slots `$02`-`$09`. [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L42)
- The two types differ only in the children's orbit patterns and turn rate. [`Patra_UpdateChild_State1Orbit`](src/bank4/patra.asm#L261)

**HP**
- Patra `$B0` (both types): 11 wooden, 6 white or 3 magic sword hits. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)
- Child `$60` (both types): 6 wooden, 3 white or 2 magic sword hits. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)

**Damage to Link**
- Touch of Patra or of a child: damage byte `$02` = `$0200`, 2 hearts before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Patra shoots nothing. [`Patra_Update`](src/bank4/patra.asm#L62)

**Weapons**
- Invincibility mask `$FE` on Patra and on every child: only damage type 1 hurts (sword, sword shot, rod swing). Arrow, boomerang and magic shot are parried with the sound; bomb and fire silently. [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L42) [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- No boomerang stun (type 2 masked). [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- While any slot `$02`-`$09` holds a child (`$25` or `$26`), Patra is tested against Link only; weapons are not tested against the body at all. [`Patra_Update@LoopChildren`](src/bank4/patra.asm#L77)
- Children test nothing (weapons or Link) until the last child (slot `$09`) has appeared and set Patra's state to 1; a child still waiting to appear is not drawn either. [`Patra_UpdateChild`](src/bank4/patra.asm#L191) [`Patra_UpdateChild_State1Orbit`](src/bank4/patra.asm#L261)
- After a hit the target has the usual `$10` invincibility timer (about 32 frames) before it can be hit again. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`Shove_Begin`](src/bank1/shove-knockback.asm#L65)

**Behaviour**
- Patra flies as a flyer: start speed `$1F`, speeds up 1 unit per frame until `(speed AND $E0) >= $40`, and never slows again: 1 pixel every 4 frames from then on (1 pixel per 8 frames while still at `$20`). [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L42) [`Flyer_Move`](src/bank4/fairy-flyer-common.asm#L250)
- Then it alternates: chase the chase target if `Random,X >= $40` (3 in 4), else wander; 7 turns `$10` frames apart, then choose again. [`Patra_DecideFlightState`](src/bank4/patra.asm#L148) [`Patra_ControlFlight_JumpTable`](src/bank4/patra.asm#L134)
- Children appear one at a time straight above Patra: slot `$02` at once, each later slot when slot `$02`'s angle reaches its entry, so they end up 1/8 circle apart. Start radius `$2C` pixels (type `$25`), `$18` (type `$26`). [`Patra_ChildStartAngles`](src/bank4/patra.asm#L164) [`Patra_UpdateChild@Ready`](src/bank4/patra.asm#L215)
- Orbit: 32 angle steps per circle, decreasing `$70`/256 step per frame for type `$25` (about 73 frames per orbit) and `$60`/256 for type `$26` (about 85 frames). [`Patra_UpdateChild_State1Orbit`](src/bank4/patra.asm#L261) [`Obj_DecreaseOrbitAngle`](src/bank4/orbit-rotation-math.asm#L181)
- Two patterns, chosen by a maneuver index (0 or 1) held by Patra. Type `$25`: 0 = circle, 1 = ring half as tall (ellipse). Type `$26`: 0 = small circle, 1 = circle twice as wide. [`Patra_Child1RotationCosineBits`](src/bank4/patra.asm#L169) [`Patra_Child2RotationBits`](src/bank4/patra.asm#L174) [`Patra_Child1RotationSineBits`](src/bank4/patra.asm#L181)
- The index flips when the maneuver timer (the `ObjTimer` of slot `$02`) is 0 and the child in slot `$03` is at angle 0; the timer is then reloaded with `$FF` frames. So the pattern changes at most once per 255 frames. [`Patra_Update@TryChangeManeuver`](src/bank4/patra.asm#L96) [`Patra_ManeuverTimes`](src/bank4/patra.asm#L52)
- Children follow Patra by adding its last per-frame displacement to their own position before rotating. [`Patra_UpdateChild_State1Orbit`](src/bank4/patra.asm#L261) [`Patra_Update`](src/bank4/patra.asm#L62)
- No clock test in Patra's or the children's update: they keep flying and orbiting with the clock, but cannot touch Link. [`Patra_Update`](src/bank4/patra.asm#L62) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)

**Spawn**
- Placed at (`$80`, `$70`) facing up; room flyer top speed `$40`; roar sample request `$40`. The init writes the child type and mask into slots `$02`-`$09`. [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L42)
- The children have no spawn cloud. [`Obj_Init_JumpTable`](src/bank7/object-init-table.asm#L177) [`Obj_ResetMetastateAndTimer`](src/bank7/flute-secret-pond.asm#L69)

**Drops**
- A killed child becomes a dead dummy (type `$5D`): no drop, no `WorldKillCycle` step, no `RoomKillCount`. [`DeadDummy_SetObjType`](src/bank4/zelda-guard-fire.asm#L130) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308)
- Each child kill still adds to `WorldKillCount` and `HelpDropCount` (up to `$0A`). [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- Patra itself dies as an ordinary monster in drop group 3 (rate `$68`/256). [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)

**Notes**
- The first pattern flip does not wait `$FF` frames: the `$FF` that Patra's init puts in the timer is zeroed on the next frame by the init of the child in slot `$02`, so the index goes 0 -> 1 the first time the child in slot `$03` reaches angle 0, 63 frames after Patra's init for type `$47` and 74 for type `$48`; later flips follow every 292-293 frames for type `$47` (trace). [`Patra_Init`](src/bank4/patra-ganon-inits.asm#L42) [`Patra_Update@TryChangeManeuver`](src/bank4/patra.asm#L96)
- The sword shot counts as sword damage against children and body alike, at the damage of the sword held. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)

# Ganon

Type `$3E`. The fight has an entry scene, an invisible sword phase, and a brown phase in which only the silver arrow kills.

**Types**
- `$3E`, one variant; 32x32 pixels, drawn as four 16x16 quarters. [`Ganon_DrawBody`](src/bank4/ganon-draw-collisions.asm#L297)

**HP**
- `$F0`, used only as a hit counter for the sword phase: 15 wooden (`$10`), 8 white (`$20`) or 4 magic (`$40`) sword hits. [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105) [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)
- At 0 HP he does not die: HP returns to `$F0` and he turns brown. [`Ganon_CheckCollisions@CheckHarmed`](src/bank4/ganon-draw-collisions.asm#L408)

**Damage to Link**
- Touch: damage byte `$04` = `$0400`, 4 hearts before rings. His middle is (X + `$10`, Y + `$10`); Link is hit when both axis distances to Link's middle are under 9 pixels. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360)
- The touch is tested on every fight frame, also while he is invisible. [`Ganon_ScenePhase2_Fight`](src/bank4/ganon.asm#L156)
- Fireball type `$56`: `$0100`, 1 heart before rings; no shield blocks it. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Weapons**
- Invincibility mask `$FA` (only sword-class 1 and arrow 4 unmasked), but his own collision routine tests only two weapons: the sword in slot `$0D` and an arrow in slot `$12`. Sword shot, rod, bombs, fire, boomerang and magic shot are never tested. [`Ganon_Init`](src/bank4/patra-ganon-inits.asm#L96) [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360)
- Blue (state 0): the sword hits only while his timer is 0, i.e. while he is invisible, and only in sword state 2 (fully extended). [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360) [`Sword_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L192)
- Each sword hit makes him visible and motionless for `$40` frames, during which he cannot be hurt; no knockback, no flashing. [`Ganon_CheckCollisions@CheckHarmed`](src/bank4/ganon-draw-collisions.asm#L408) [`Ganon_CheckCollisions@UndoDeath`](src/bank4/ganon-draw-collisions.asm#L417)
- Brown (state nonzero): the sword is not tested. Only an arrow in flight (slot `$12`, state `$10`) is tested, and only when `InvArrow` = 2 (silver arrow); a wooden arrow passes through. [`Ganon_CheckCollisions@CheckArrowCollision`](src/bank4/ganon-draw-collisions.asm#L427) [`InvArrow`](src/Variables.inc#L559)
- The silver arrow hit starts the death; nothing else can kill him. [`Ganon_CheckCollisions@CheckArrowCollision`](src/bank4/ganon-draw-collisions.asm#L427) [`Ganon_ObjDyingCount`](src/ObjVars.inc#L202)

**Behaviour**
- Scene phase 0: Link is halted holding a Triforce piece in the dark for `$40` frames, then the room fades to light. Phase 1: `$C0` more frames, then control returns and the fight (phase 2) begins. [`Ganon_ScenePhase0_LiftInDark`](src/bank4/ganon.asm#L71) [`Ganon_ScenePhase1_LiftInLight`](src/bank4/ganon.asm#L127)
- Blue, timer 0: not drawn; moves 1 pixel per frame with the Blue Wizzrobe's teleport movement (passes over blocks and water, turns toward Link at intervals). [`Ganon_MoveAndShoot`](src/bank4/ganon.asm#L195) [`Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile`](src/bank4/wizzrobe.asm#L164)
- While moving he shoots a type `$56` fireball whenever `FrameCounter AND $3F` = 0 (every 64 frames) if a monster slot is free; it aims once at Link's position. [`Ganon_MoveAndShoot`](src/bank4/ganon.asm#L195) [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- When the visible timer reaches 1 he jumps to Y `$A0`, X `$30` (even `FrameCounter`) or `$B0` (odd) and goes invisible again. [`Ganon_RandomizeLocation`](src/bank4/ganon.asm#L180) [`Ganon_StartXs`](src/bank4/ganon.asm#L223)
- Brown: state starts at `$FF` and drops by 1 on odd frames (about 510 frames). He stands still, does not shoot, is drawn every frame until the state is below `$30`, then flickers. At 0 he turns blue at a start spot with full HP: the sword phase starts over. [`Ganon_UpdateBrownState`](src/bank4/ganon.asm#L234)
- Dying: a count rises 1 per frame. Below `$50` the body is drawn; at `$50` he becomes ashes (8 rays burst out, boss death cry); at `$A0` the room item appears at the ashes and `RoomKillCount` is incremented. [`Ganon_UpdateDying`](src/bank4/ganon.asm#L271) [`Ganon_SetUpBurstRays`](src/bank4/ganon-draw-collisions.asm#L86)

**Spawn**
- No spawn cloud. The init halts Link (state `$40`) with Link's timer at `$40` and requests the roar. [`Ganon_Init`](src/bank4/patra-ganon-inits.asm#L96)

**Drops**
- None: his object is never destroyed and never becomes a dropped item. [`Ganon_UpdateDying`](src/bank4/ganon.asm#L271)
- The Triforce of Power is the room item (object slot `$13`), activated 160 frames after the arrow hit, unless the room flags say it was already taken. [`Ganon_ActivateRoomItem`](src/bank4/ganon-draw-collisions.asm#L541)
- Taking it halts Link for `$C0` frames of fanfare; `InvPowerTriforce` is set only when the fanfare ends. [`Item_BeginPowerTriforceFanfare`](src/bank1/tile-block-power-triforce.asm#L152) [`Item_UpdatePowerTriforceFanfare`](src/bank1/tile-block-power-triforce.asm#L99)
- Secret trigger 3 opens the room's shutters once `InvPowerTriforce` is nonzero. [`Secret_Trigger3PowerTriforce`](src/bank5/underworld-secrets.asm#L255)

**Notes**
- Clock: Ganon's movement and shooting have no clock test, and his touch test skips the clock and stun tests other monsters use; it tests only Link's invincibility timer. The clock adds `$10` to that timer every frame and the timer loses 1 on even frames, so the byte wraps and reads 0 on exactly 1 frame in 33 (trace of the timer with the clock set): his touch can hurt Link on that frame only. [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360) [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- His fireballs do use the ordinary test: with the clock they fly through Link. [`MonsterShot_CheckLinkCollision`](src/bank4/monster-shots-fireballs.asm#L249) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Shield: nothing of Ganon's can be blocked; type `$56` fireballs are never parried and his body is not a shot. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- The cancelled sword "death" still ran the kill bookkeeping: `WorldKillCount` + 1 and `HelpDropCount` + 1 (up to `$0A`) each time he turns brown. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216) [`Ganon_CheckCollisions@UndoDeath`](src/bank4/ganon-draw-collisions.asm#L417)
- A missed brown phase costs the whole sword count again: his HP is already back at `$F0` when he turns brown. [`Ganon_UpdateBrownState`](src/bank4/ganon.asm#L234)
- The death burst uses object slots `$02`-`$09` for its rays without testing that they are free. [`Ganon_SetUpBurstRays`](src/bank4/ganon-draw-collisions.asm#L86)

# Trap

The dungeon blade trap. It cannot be hurt; it charges when Link lines up with it and creeps back.

**Types**
- `$49` and `$4A` are generators: `$49` makes six traps, `$4A` four. Every trap made is type `$49`. [`Trap_Init_Full`](src/bank1/trap-rupee-stash.asm#L134)
- Attribute byte `$A1`: bit `$20` means Link's weapons are never tested against it, so it cannot be damaged, stunned or killed. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)

**Damage to Link**
- Touch: damage byte `$01` = `$0100`, 1 heart before rings (blue ring half, red ring a quarter). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)

**Shield**
- None: a trap is not a shot (type below `$53`), so it harms on overlap whatever Link faces. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Behaviour**
- Start spots (X, Y) by object slot: 1 (`$20`,`$5D`), 2 (`$20`,`$BD`), 3 (`$D0`,`$5D`), 4 (`$D0`,`$BD`); slots 5 and 6 (type `$49` only) (`$40`,`$8D`), (`$B0`,`$8D`). [`Trap_StartXs`](src/bank1/trap-rupee-stash.asm#L112) [`Trap_StartYs`](src/bank1/trap-rupee-stash.asm#L117)
- Trigger lines (state 0), comparing `ObjX`/`ObjY` of Link and trap: if `|Link Y - trap Y| < $0E` and the X's differ, it wants to charge horizontally toward Link; otherwise, if `|Link X - trap X| < $0E` and the Y's differ, vertically toward Link. [`Trap_Update_Full`](src/bank1/trap-rupee-stash.asm#L189)
- The charge starts only in a direction the slot allows: slot 1 right or down, 2 right or up, 3 left or down, 4 left or up, 5 right only, 6 left only (always toward the room's middle). A refused horizontal wish is not followed by the vertical test that frame. [`Trap_AllowedDirs`](src/bank1/trap-rupee-stash.asm#L167) [`Trap_Update_Full@TryCharge`](src/bank1/trap-rupee-stash.asm#L216)
- Out (state 1): q-speed `$70`, at most 1.75 pixels per frame, until less than 5 pixels from X `$78` (horizontal) or Y `$90` (vertical). [`Trap_Update_Full@CheckState`](src/bank1/trap-rupee-stash.asm#L288)
- Back (state 2): q-speed `$20`, 0.5 pixels per frame, until its coordinate equals the start coordinate exactly; then state 0. It does not sense Link while charging or returning. [`Trap_Update_Full@FinishState2`](src/bank1/trap-rupee-stash.asm#L314)
- Charge lengths: corner traps about 84 pixels horizontally, 47 (top) or 41 (bottom) vertically; side traps about 52 pixels. [`Trap_StartXs`](src/bank1/trap-rupee-stash.asm#L112) [`Trap_StartYs`](src/bank1/trap-rupee-stash.asm#L117)
- With the clock, or while Link is invincible, the trap still senses and moves; it only cannot hurt him. [`Trap_Update_Full`](src/bank1/trap-rupee-stash.asm#L189) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)

**Created by**
- A room whose monster list id is `$49` or `$4A` (the generator is the room's only object), or a monster list whose first entry is `$4A` followed by three empty entries (lists `$6D`, `$6E`, `$76`, `$77`: four traps plus Zols, Keese, Like-Likes or Wizzrobes from slot 5 up; Q1 level 3 rooms `$3B`, `$4C`, `$5A`). The generator sits in slot 1 and overwrites itself with the first trap; traps fill slots 1-6 or 1-4. Each shows the spawn cloud first. [`Trap_Init_Full`](src/bank1/trap-rupee-stash.asm#L134) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143) [objlists.md](data/objlists.md)
- Init and update live in bank 1 behind fixed-bank trampolines. [`Trap_Init`](src/bank7/object-stubs-destroy-monster.asm#L64) [`Trap_Update`](src/bank7/object-stubs-destroy-monster.asm#L70)

**Notes**
- The return trip is 3.5 times slower than the charge: baiting a trap and passing behind it while it creeps back is safe until it reaches its start coordinate. [`Trap_Update_Full`](src/bank1/trap-rupee-stash.asm#L189)
- Traps never die, so they never count toward a room's kills. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

# Statue fireballs

Dungeon statues are not objects: a per-frame routine creates type `$55` fireballs at fixed points. Nothing can be done to a statue.

**Types**
- The fireballs are type `$55` (see `# Fireball`). [`Statue_Update`](src/bank4/statues.asm#L86)
- Pattern 0: room layout id `$24`, 4 sources at (`$24`,`$C0`), (`$C8`,`$BC`), (`$24`,`$64`), (`$C8`,`$5C`). Pattern 1: layout id `$23`, 2 sources at (`$64`,`$94`), (`$88`,`$8C`). The fireball appears 4 pixels right of the source. [`Statue_RoomLayoutIds`](src/bank4/statues.asm#L45) [`Statue_FireballXs`](src/bank4/statues.asm#L64) [`Statue_FireballYs`](src/bank4/statues.asm#L69)
- Pattern 2: 2 sources at (`$48`,`$82`) and (`$A8`,`$86`), used in any dungeon room once the room's person has been attacked; it takes priority over the layout test. [`Statue_Update`](src/bank4/statues.asm#L86) [`Person_CheckCollisions`](src/bank1/underworld-person-update.asm#L89)
- Q1 rooms with layout `$24` (the `lay` column of `data/levelblock-uw-*.md`): level 2 `$4F`, level 5 `$24`, level 6 `$58`, level 7 `$0A` `$1C` `$39` `$6C`, level 8 `$1E` `$5C` `$5E`. Layout `$23`: level 7 `$0C` `$69` `$78`, level 8 `$3E` `$4B` `$7D`, level 9 `$37` `$53`. [`Statue_RoomLayoutIds`](src/bank4/statues.asm#L45)

**Damage to Link**
- `$0080`, half a heart before rings (type `$55`). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Shield**
- Magic shield only, with Link idle and facing against the fireball on one axis. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)

**Behaviour**
- Runs once per frame whenever `CurLevel` is not 0 (no game mode test). [`Statue_Update`](src/bank4/statues.asm#L86)
- Each source has its own timer, 0 on room entry, so every statue may fire on the first frame. [`Statue_FireballTimers`](src/ObjVars.inc#L99) [`Statue_Update@NextStatue`](src/bank4/statues.asm#L137)
- When a timer runs out: with `Random,X >= $F0` (X = statue number 0-3) nothing is shot and the statue waits `$100` frames; otherwise the timer is reloaded with `$50`, `$80`, `$F0` or `$60` frames (`Random,X AND 3`) and it shoots. The next chance comes 81, 129, 241 or 97 frames later. [`Statue_FireballStartTimes`](src/bank4/statues.asm#L54) [`Statue_Update@NextStatue`](src/bank4/statues.asm#L137)
- No shot when Link is within `$18` pixels of the source on both axes (difference -`$18` to +`$17`); the timer is reloaded anyway. [`Statue_Update@LoopAxis`](src/bank4/statues.asm#L194)
- Nothing happens at all, and timers do not count, on a frame where the highest empty object slot of `$01`-`$0B` is below `$06` or none is empty. [`Statue_Update@Shoot`](src/bank4/statues.asm#L106)
- Statue fireballs are not counted in `MonsterShotCount`: the only limit is a free slot. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58)

**Created by**
- The play update, in dungeons, after the object loop. [`Statue_Update`](src/bank4/statues.asm#L86)
- Pattern 2 is switched on by a weapon "kill" of the dungeon person in slot 1: his death is cancelled and `IsPersonFireballsEnabled` is set until the room is left. [`Person_CheckCollisions`](src/bank1/underworld-person-update.asm#L89) [`IsPersonFireballsEnabled`](src/Variables.inc#L383)

**Notes**
- Keeping object slots `$06`-`$0B` full (six or more live objects in the high slots) silences the statues completely. [`Statue_Update@Shoot`](src/bank4/statues.asm#L106)
- When several statues shoot in one frame, the frame's first fireball ends up at the start point of the last statue that shot. [`Statue_Update@Shoot`](src/bank4/statues.asm#L106)

# Fireball

Aimed projectile of statues and several monsters. It aims once, waits, then flies straight.

**Types**
- `$55`: blockable by the magic shield. Shot by statues, the person's fires, Zora and Aquamentus. [`Statue_Update`](src/bank4/statues.asm#L86) [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301) [`Fireball_ShootType55`](src/bank4/aquamentus.asm#L232)
- `$56`: never blockable. Shot by Gohma, Ganon, Manhandla and Gleeok (necks and flying heads). [`Gohma_Update`](src/bank4/gohma.asm#L63) [`Ganon_MoveAndShoot`](src/bank4/ganon.asm#L195) [`Manhandla_Update`](src/bank4/manhandla.asm#L60) [`Gleeok_Update`](src/bank4/gleeok.asm#L175) [`Gleeok_UpdateFlyingHead`](src/bank4/gleeok.asm#L78)

**Damage to Link**
- `$55`: damage byte `$80` = `$0080`, half a heart. `$56`: `$01` = `$0100`, 1 heart. Both before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Shield**
- `$55`: magic shield only; Link's state must be `$00`-`$0F` (not swinging or using an item) and his direction must oppose the fireball's on one axis. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138) [`InvMagicShield`](src/Variables.inc#L604)
- A fireball's direction always holds one horizontal and one vertical bit, so the shield works if Link faces against either component. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- `$56`: never parried. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Behaviour**
- Appears 4 pixels right of its shooter. On its first update it takes the directions to Link's own position (not the chase target) and picks one of 9 speed pairs. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58) [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- It then stays in place for `$0F` frames, drawn and already able to hit Link, and from the `$10`th frame flies in the fixed direction; no homing. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- Speed pairs (X/Y q-speed, pixels per frame = value / 64): from `$70`/`$00` (1.75 along one axis) through `$50`/`$50` (1.25 on both) to `$00`/`$70`. [`Fireball_QSpeedsX`](src/bank4/monster-shots-fireballs.asm#L268) [`Fireball_QSpeedsY`](src/bank4/monster-shots-fireballs.asm#L272)
- Destroyed at the room boundary and on any touch of Link, harmful or blocked; it never bounces and never tests tiles. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- While Link's invincibility timer, the clock or Link's stun timer is nonzero, no touch is reported: the fireball flies through Link and lives on. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Link's weapons cannot touch it (attribute bit `$20`). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**Created by**
- `Fireball_Shoot` with the type in A: needs only a free object slot in `$01`-`$0B` (the highest is used); no 4-shot limit and no `MonsterShotCount`. [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58) [`MonsterShot_Destroy`](src/bank4/monster-shots-fireballs.asm#L189)

**Notes**
- Standing next to the shooter does not help: the fireball can hit during its `$0F` waiting frames. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)
- When fireball and Link share a coordinate, the direction on that axis is left or up. [`Fireball_Update`](src/bank4/monster-shots-fireballs.asm#L286)

# Monster shots (rock, sword shot, magic shot)

Straight-flying shots, types `$53`, `$54`, `$57`-`$5A`, sharing one update. Shield rules for every projectile type `$53`-`$5C` are listed here.

**Types**
- `$53` flying rock: Octoroks. [`Octorok_Update`](src/bank4/octorok.asm#L48)
- `$57` sword shot: Lynels, and Stalfos in Q2 only. [`Lynel_Update`](src/bank4/moblin-lynel-shooting.asm#L69) [`Stalfos_Update`](src/bank4/stalfos.asm#L43)
- `$58` magic shot: Blue Wizzrobe. `$59` magic shot: Red Wizzrobe. [`Wizzrobe_ShootMagicShot58`](src/bank4/wizzrobe.asm#L523) [`Wizzrobe_Red_State2Solid`](src/bank4/wizzrobe.asm#L667)
- `$54` (rock-like, faster) and `$5A` have live table entries but are never created: every shot type the code makes is an immediate (`$53`, `$55`-`$59`, `$5B`, `$5C`), no room's monster list id is `$53`-`$61`, and no monster list contains them. [`MonsterShot_InitType54`](src/bank4/walker-wanderer-common.asm#L91) [`Shooter_Shoot`](src/bank4/shoot-helpers-candle.asm#L94) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)

**Damage to Link**
- Before rings: `$53` and `$54` `$0080` (half a heart); `$57` and `$58` `$0200` (2 hearts); `$59` and `$5A` `$0400` (4 hearts). [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Shield**
- A parry needs Link's state `$00`-`$0F` (not swinging the sword or using an item) and Link's direction opposing the shot's on one axis (up against down, or left against right). [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Any shield: `$53` flying rock, `$54`, `$5B` monster arrow, `$5C` monster boomerang. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- Magic shield only: `$55` fireball, `$57` sword shot, `$58` and `$59` magic shots. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138) [`InvMagicShield`](src/Variables.inc#L604)
- Never blocked: `$56` fireball and `$5A`. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- A blocked shot goes to state `$30`: it drifts 2 pixels per frame the way Link faced (plus 1 sideways) for `$10` frames, harmless, then is destroyed. [`MonsterShot_CheckLinkCollision`](src/bank4/monster-shots-fireballs.asm#L249) [`MonsterShot_Bounce`](src/bank4/monster-shots-fireballs.asm#L209)

**Behaviour**
- Q-speed `$C0` = 3 pixels per frame (`$54`: `$E0` = 3.5), in the shooter's facing direction, from the shooter's position. [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82) [`Shooter_Shoot`](src/bank4/shoot-helpers-candle.asm#L94)
- A shot can touch Link only while its state is `$10`-`$1F` (flying). [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Destroyed at the room boundary or when it harms Link. Only `$53` and `$54` are also destroyed by an unwalkable tile ahead; sword and magic shots pass over tiles. [`MonsterShot_Update`](src/bank4/monster-shots-fireballs.asm#L102)
- While Link's invincibility timer, the clock or Link's stun timer is nonzero the shot flies through Link. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Link's weapons cannot touch a shot (attribute bit `$20`). [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70)

**Created by**
- `Shooter_ShootLimited`: needs a free slot in `$01`-`$0B` and fewer than 4 counted shots in flight (`MonsterShotCount`; fireballs are not counted, arrows and boomerangs are). [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70) [`MonsterShotCount`](src/Variables.inc#L295)
- Walking shooters (Octorok, Moblin, Lynel) stand still for `$30` frames and release the shot 32 frames in; being hit cancels it; with the clock or while stunned no shot is made. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- The count is given back when the shot is destroyed. [`MonsterShot_DestroyCounted`](src/bank4/monster-shots-fireballs.asm#L198)

**Notes**
- Swinging the sword or using an item drops the shield: any shot then harms Link even head-on. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- No spawn cloud: a shot's init clears the metastate, at the cost of one frame before it first moves. [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82)

# Monster arrow

**Types**
- `$5B`, shot only by Moblins (`$03`, `$04`). [`Moblin_Update`](src/bank4/moblin-lynel-shooting.asm#L55)

**Damage to Link**
- Damage byte `$80` = `$0080`, half a heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Shield**
- Any shield, with Link idle (state `$00`-`$0F`) and facing against the arrow. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- A blocked arrow bounces for `$10` frames (state `$30`), harmless, then is destroyed. [`MonsterArrow_Update`](src/bank4/moblin-lynel-shooting.asm#L236) [`MonsterShot_Bounce`](src/bank4/monster-shots-fireballs.asm#L209)

**Behaviour**
- Flies at q-speed `$80` = 2 pixels per frame in the Moblin's facing direction; no range limit. [`MonsterArrow_Update`](src/bank4/moblin-lynel-shooting.asm#L236)
- Tiles do not stop it. At the room bounds it becomes a spark (state `$20`) for 3 frames, then is destroyed. [`MonsterArrow_Update`](src/bank4/moblin-lynel-shooting.asm#L236) [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192)
- Destroyed on the frame it harms Link. [`MonsterArrow_Update`](src/bank4/moblin-lynel-shooting.asm#L236)
- Passes through Link while his invincibility timer, the clock or his stun timer is nonzero. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)

**Created by**
- The Moblin's shoot timer: the Moblin stops for `$30` frames and the arrow leaves 32 frames in. Type `$03` tests for a shot every frame, type `$04` only on frames with `Random,X >= $F8`. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Counted: needs a free slot and fewer than 4 counted shots in flight. [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70)

**Notes**
- Slower than the other straight shots (2 against 3 pixels per frame). [`MonsterArrow_Update`](src/bank4/moblin-lynel-shooting.asm#L236) [`MonsterShot_Init`](src/bank4/walker-wanderer-common.asm#L82)

# Monster boomerang

**Types**
- `$5C`, thrown only by Goriyas (`$05`, `$06`). [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156)

**Damage to Link**
- Damage byte `$01` = `$0100`, 1 heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Shield**
- Any shield, with Link idle (state `$00`-`$0F`) and facing against the boomerang's direction. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- Blocked or not, a boomerang that touches Link becomes a spark for 3 frames and then returns fast. [`Boomerang_DrawAndCheckCollision`](src/bank7/boomerang-return-draw.asm#L217)

**Behaviour**
- Flies out at q-speed `$A0` = 2.5 pixels per frame for `$51` pixels in the Goriya's facing direction; tiles do not stop it, the room bounds turn it back. [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156) [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192)
- Then slows (1 pixel per frame for `$10` frames), returns slowly for `$10` frames, then fast (2 pixels per frame along an axis) until within 8 pixels of the thrower on both axes. [`Boomerang_UpdateSlowAndReturn`](src/bank7/boomerang-return-draw.asm#L52)
- It can hurt Link only in state `$1x` (flying out), and tests him only every other frame. [`Boomerang_AnimateAndCheckCollision`](src/bank7/boomerang-return-draw.asm#L199) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- The Goriya stands still (state `$80`) until it catches the boomerang; the catch sets its timer to `$30`, `$50` or `$70` frames before the next throw. [`Boomerang_UpdateSlowAndReturn@MonsterCatches`](src/bank7/boomerang-return-draw.asm#L141)

**Created by**
- Throw conditions: blue Goriya `$05` any frame, red `$06` only when `Random,X` is `$23` or `$77`; its timer 0; no clock; not stunned; "wants to shoot" set; a free slot; fewer than 4 counted shots. [`Walker_SetInputDirAndTryThrowBoomerang`](src/bank4/goriya.asm#L156) [`Shooter_ShootLimited`](src/bank4/shoot-helpers-candle.asm#L70)

**Notes**
- Killing the Goriya while the boomerang is out removes the boomerang at once. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- A returning boomerang is harmless: only the outward flight can hit. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

# Standing fire

The two flames beside every cave person and dungeon person.

**Types**
- `$40`. [`StandingFire_Update`](src/bank4/walker-wanderer-common.asm#L160)

**Damage to Link**
- Touch: damage byte `$80` = `$0080`, half a heart before rings, with the normal knockback. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200) [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164)

**Shield**
- None: it is not a shot (type below `$53`). [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Behaviour**
- Does not move. Each frame it only tests its touch on Link and draws itself; Link's weapons are never tested against it, so it cannot be put out. [`StandingFire_Update`](src/bank4/walker-wanderer-common.asm#L160)
- Harmless while Link is halted (state `$40`, as during the person's text), invincible, or holding the clock. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Created by**
- Every cave or dungeon person setup: object slots 2 and 3 become type `$40` at X `$48` and `$A8`, at the person's Y (`$80`). [`Person_SetUpWithFires`](src/bank1/cave-init-person-text.asm#L304)
- The fires stay when a once-only cave person has already given its item and is removed. [`Cave_Init`](src/bank1/cave-init-person-text.asm#L101)

**Notes**
- These are not the sources of the fireballs shot after a person is attacked; those come from fixed points (`$48`,`$82`) and (`$A8`,`$86`) of the statue routine. [`Statue_Update`](src/bank4/statues.asm#L86)
- Link's own candle fire is a different object (slots `$10`/`$11`) that only borrows this type's animation. [`Fire_Update`](src/bank7/fire.asm#L44)

# Guard fire

The four killable flames in front of Zelda.

**Types**
- `$3F`; four of them in object slots 2-5 at (`$60`,`$B5`), (`$70`,`$9D`), (`$80`,`$9D`), (`$90`,`$B5`). [`Zelda_Init`](src/bank4/zelda-lanmola-inits.asm#L45) [`Zelda_RoomObjXs`](src/bank4/zelda-lanmola-inits.asm#L31) [`Zelda_RoomObjYs`](src/bank4/zelda-lanmola-inits.asm#L36)

**Damage to Link**
- Touch: damage byte `$80` = `$0080`, half a heart before rings. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)

**Shield**
- None: it is not a shot (type below `$53`). [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Behaviour**
- Does not move and is never knocked back. Unlike the standing fire it runs the full collision pass: HP `$10`, no invincibility mask, so one hit of any damaging weapon kills it (the boomerang only stuns it). [`GuardFire_Update`](src/bank4/zelda-guard-fire.asm#L120) [`Obj_TypeHpPairs`](src/bank7/object-update-table.asm#L105)
- A killed fire turns into a dead dummy (type `$5D`): no drop, no `WorldKillCycle` step, no `RoomKillCount`; `WorldKillCount` and `HelpDropCount` still rise. [`DeadDummy_SetObjType`](src/bank4/zelda-guard-fire.asm#L130) [`Obj_EndMetastate`](src/bank7/object-update-table.asm#L308) [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)

**Created by**
- Zelda's init (type `$37`, slot 1); each fire is then initialized as an ordinary type `$3F` object on its own first update. [`Zelda_Init`](src/bank4/zelda-lanmola-inits.asm#L45)

**Notes**
- Zelda's trigger does not test the fires: it fires when Link's X is `$70`-`$80` and his Y is exactly `$95`, whether or not any fire is alive. [`Zelda_Update`](src/bank4/zelda-guard-fire.asm#L42)

# Zelda

The goal object of level 9. She is scenery with one trigger; the four fires around her are separate objects.

**Types**
- Object type `$37`, always object slot `$01`, at (`$78`, `$88`); her init also places four guard fires (type `$3F`) in slots `$02`-`$05`. [`Zelda_Init`](src/bank4/zelda-lanmola-inits.asm#L45)
- Guard fire positions: (`$60`,`$B5`), (`$70`,`$9D`), (`$80`,`$9D`), (`$90`,`$B5`). [`Zelda_RoomObjXs`](src/bank4/zelda-lanmola-inits.asm#L31), [`Zelda_RoomObjYs`](src/bank4/zelda-lanmola-inits.asm#L36)
- Her room is level 9 room `$32` (see `data/levelblock-uw-7to9-q1.md`). [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)

**Conditions**
- Trigger (state 0): Link's X in `$70`-`$80` and Link's Y exactly `$95`. Nothing else is tested. [`Zelda_Update`](src/bank4/zelda-guard-fire.asm#L42)
- The guard fires need not be dead: the trigger does not look at them. [`Zelda_Update`](src/bank4/zelda-guard-fire.asm#L42)

**Effect**
- On the trigger Link is halted (`ObjState` = `$40`), moved to (`$88`, `$88`) facing left, song request `$06` is made and Zelda's timer is set to `$80` frames. [`Zelda_Update`](src/bank4/zelda-guard-fire.asm#L42)
- When the `$80` frames end: `GameMode` = `$13` (win game), `GameSubmode` = 0, Link's state 0, and the tile map is filled with blank tile `$24`. [`Zelda_State1WaitForEnding`](src/bank4/zelda-guard-fire.asm#L90)

**Notes**
- Zelda's update draws her and runs no collision check: weapons and contact do nothing. [`Zelda_Update`](src/bank4/zelda-guard-fire.asm#L42)
- A guard fire is an ordinary collision target; when it dies its type becomes `$5D` (dead dummy), so the death steps neither the kill cycle nor the room kill count. [`GuardFire_Update`](src/bank4/zelda-guard-fire.asm#L120), [`DeadDummy_SetObjType`](src/bank4/zelda-guard-fire.asm#L130)

# Old men and shopkeepers (cave and dungeon persons)

Every person stands in object slot `$01` at (`$78`, `$80`) between two standing fires, halts Link, and types a text.
Cave persons are types `$6A`-`$7D`; dungeon persons are `$4B`-`$52`. Rooms, wares and prices of each cave are listed
under `# Caves` in `data/levelblock-ow.md`; all texts are in `data/person-text.md`.

## Shared behaviour

- Setup puts the person at (`$78`, `$80`) with HP 0, makes slots `$02` and `$03` standing fires (type `$40`) at X `$48` and `$A8`, and halts Link (`ObjState` = `$40`). [`Person_SetUpWithFires`](src/bank1/cave-init-person-text.asm#L304)
- Text is typed one character per 6 frames; tile `$25` (fast blank) takes a cell but no time. [`Person_UpdateState_Textbox`](src/bank1/person-textbox.asm#L51)
- Link is released (`ObjState` = 0) in the frame the last character is typed. [`Person_UpdateState_Textbox`](src/bank1/person-textbox.asm#L51), [`Link_Unhalt`](src/bank1/cave-init-person-text.asm#L130)
- A person that shows prices waits `$0A` frames before the first character. [`Cave_WritePriceList`](src/bank1/cave-person-prices.asm#L229), [`Person_ComplexState0Begin`](src/bank1/underworld-person-update.asm#L147)
- Link cannot move up while his Y < `$8E` (highest Y is `$8D`): always in a cave, and in a dungeon room while slot `$01` holds type `$36` or `$4B`-`$52`. [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289), [`Walker_Move@CheckPersonBlocking`](src/bank7/walker-move.asm#L149)
- Rupees never change at once: prices and gifts are posted to `InvRupeesToSubtract` / `InvRupeesToAdd` and counted off later. [`Inv_PostRupeeDebit`](src/bank1/cave-hint-money-game.asm#L176), [`Inv_PostRupeeCredit`](src/bank1/cave-hint-money-game.asm#L165)

## Cave persons (`$6A`-`$7D`)

- The person's type is `$6A` + (cave index - `$10`), the cave index being bits 7-2 of the room's attribute B. [`Spawn_AssignPositions`](src/bank5/spawn-positions.asm#L64)
- Text selector and the "pick up" / "pay" flags come from one byte per type; wares and prices from `LevelBlockAttrsE` (3 ware bytes per cave, prices 60 bytes further). [`Cave_PersonTextSelectors`](src/bank1/cave-init-person-text.asm#L71), [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)
- Wares sit at X `$58`, `$78`, `$98`, Y `$98`. Link touches one when his X equals the ware's X exactly and |Y - `$98`| < 6; nothing is tested while `InvRupeesToSubtract` is nonzero. [`Cave_WareXs`](src/bank1/cave-person-prices.asm#L155), [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- Once-only persons: `$6A`-`$6D`, `$71`, `$72`, `$7B`-`$7D`. If bit `$10` of the room's flags is set on entry, the person is removed and Link is released at once (the fires stay). [`Cave_Init`](src/bank1/cave-init-person-text.asm#L101), [`RoomFlags_GetItemTaken`](src/bank1/take-item.asm#L151)
- The flag belongs to the overworld room, not to the person type: two rooms leading to the same kind of cave are tracked separately. [`Cave_Init@CheckTaken`](src/bank1/cave-init-person-text.asm#L119)
- Persons `$6E`-`$70` and `$73`-`$7A` (roads, hints, money game, shops) appear on every visit. [`Cave_Init`](src/bank1/cave-init-person-text.asm#L101)
- `$6A` wooden sword: free, no requirement. [`Cave_PersonTextSelectors`](src/bank1/cave-init-person-text.asm#L71)
- `$6C` white sword needs `HeartValues` >= `$40` (5 heart containers); `$6D` magic sword needs >= `$B0` (12 containers). Current health does not matter. [`Cave_State2TalkOrShopOrDoorCharge@TryToTake`](src/bank1/cave-shop.asm#L161)
- `$6B` "take any one": red potion (left) or heart container (right), free, once per room. [`Cave_State2TalkOrShopOrDoorCharge@Take`](src/bank1/cave-shop.asm#L185)
- `$72` gives the letter (free, once). [`Cave_PersonTextSelectors`](src/bank1/cave-init-person-text.asm#L71)
- `$74` medicine shop: blue potion 40, red potion 68. Until `InvLetter` = 2 the person is drawn but shows no text or wares, and Link is released on the first frame. [`Cave_UpdatePerson`](src/bank1/cave-person-prices.asm#L66)
- Showing the letter: press B with the letter selected (`SelectedItemSlot` = `$0F`) in that cave; `InvLetter` goes 1 -> 2 and the selected slot becomes the potion (7). [`Cave_UpdatePerson@UseLetter`](src/bank1/cave-person-prices.asm#L105)
- Shops (left, middle, right): `$77` magic shield 130, bomb 20, wooden arrow 80; `$78` magic shield 160, key 100, blue candle 60; `$79` magic shield 90, food 100, heart 10; `$7A` key 80, blue ring 250, food 60. [`Cave_Init_SetUpTextAndWares@CopyCaveBytes`](src/bank1/cave-init-person-text.asm#L176)
- Buying needs `InvRupees` >= price; with fewer nothing happens. [`Cave_State2TalkOrShopOrDoorCharge@PickedUpWare`](src/bank1/cave-shop.asm#L146)
- Taking or buying a ware sets room flag `$10`, gives the item, and starts a `$40`-frame timer after which the person and the other wares vanish: one ware per visit, also in shops. [`Cave_State2TalkOrShopOrDoorCharge@Take`](src/bank1/cave-shop.asm#L185), [`Cave_State4DelayThenHide`](src/bank1/cave-shop.asm#L221)
- An item taken in a cave is lifted: `ItemLiftTimer` = `$80` and Link stays halted while it runs (a bomb is not held up, but the timer and song still start). [`Item_Take`](src/bank1/take-item.asm#L259), [`Link_CheckLiftItem`](src/bank7/status-bar-lift-item.asm#L162)
- `$6E` "take any road", `$6F` and `$73` free hints: talk only, then stay. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- `$71` door repair: when his text ends 20 rupees are posted to subtract, with no test of Link's rupees, and room flag `$10` is set so he is gone afterwards. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- `$75` paid hints: 5 and 10 buy "THIS AIN'T ENOUGH TO TALK.", 20 buys "GO UP,UP, THE MOUNTAIN AHEAD.". [`Cave_HintTextSelectorsType75`](src/bank1/cave-hint-money-game.asm#L42)
- `$76` paid hints: 10 buys "AIN'T ENOUGH", 30 the Lost Woods route ("GO NORTH,WEST,SOUTH,WEST..."), 50 "BOY, YOU'RE RICH!". [`Cave_HintTextSelectorsOther`](src/bank1/cave-hint-money-game.asm#L48)
- A hint is charged when the ware is touched (needs `InvRupees` >= price), whatever text follows; the person stays. [`Cave_State2TalkOrShopOrDoorCharge@PickedUp`](src/bank1/cave-shop.asm#L117), [`Cave_State5HintOrMoneyGame`](src/bank1/cave-hint-money-game.asm#L61)
- `$7B`, `$7C`, `$7D` secret rupees: touching the middle rupee credits 30, 100 and 10 rupees (the middle price byte), once per room. [`Cave_State5HintOrMoneyGame@CheckGiftOrMoneyGame`](src/bank1/cave-hint-money-game.asm#L94)
- `$70` money game: the three amounts are fixed when the cave is entered: a loss of 10, a loss of 10 or 40 (`Random+2` bit 0), a win of 20 or 50 (`Random+2` bit 1). [`Cave_Init_SetUpTextAndWares@FoundPermutation`](src/bank1/cave-init-person-text.asm#L241), [`Cave_MoneyGameLossAmounts`](src/bank1/cave-init-person-text.asm#L77)
- Their order is one of 6 permutations chosen by `Random+1` (41/256 for the first, 43/256 for each other), so the win is behind the left or middle rupee 86/256 of the time each and behind the right one 84/256. [`Cave_MoneyGamePermutations`](src/bank1/cave-init-person-text.asm#L83), [`Cave_Init_SetUpTextAndWares@ChoosePermutation`](src/bank1/cave-init-person-text.asm#L233)
- Playing needs `InvRupees` >= 10; the chosen amount is credited when it is 20 or 50, else debited; one play per visit. [`Cave_State5HintOrMoneyGame@HandleMoneyGame`](src/bank1/cave-hint-money-game.asm#L119)
- A cave person's update only draws: it runs no collision check, so it cannot be hit. [`Cave_UpdatePerson`](src/bank1/cave-person-prices.asm#L66)

## Dungeon persons (`$4B`-`$52`)

- The text is chosen by level group and type: levels 1, 2, 5, 7; levels 3, 4, 6, 8; level 9 (types `$4B`-`$4E`). [`Person_InitUnderworld_JumpTable`](src/bank1/underworld-person-init.asm#L53), [`Person_TextSelectorsLevels1257`](src/bank1/underworld-person-init.asm#L67), [`Person_TextSelectorsLevels3468`](src/bank1/underworld-person-init.asm#L134), [`Person_TextSelectorsLevel9`](src/bank1/underworld-person-init.asm#L162)
- In levels 3, 4, 6, 8 and 9 every person only talks and stays. [`Person_UpdateUnderworld_Full`](src/bank1/underworld-person-update.asm#L44), [`Person_UpdateUnderworldTalker_JumpTable`](src/bank1/underworld-person-update.asm#L64)
- More-bombs offer (type `$4F` in levels 1, 2, 5, 7; price text "-100"): Link at X = `$78` exactly, |Y - `$98`| < 6 and `InvRupees` >= 100, tested every frame after the text. [`Person_ComplexState2SellBombCapacity`](src/bank1/underworld-person-update.asm#L171)
- The purchase posts 100 rupees, adds 4 to `InvMaxBombs` (8 in a new file), fills `InvBombs` to the new capacity and sets room flag `$10`; the man flickers `$40` frames and is gone for good. [`Person_ComplexState2SellBombCapacity@CheckCanPay`](src/bank1/underworld-person-update.asm#L195), [`InvMaxBombs`](src/Variables.inc#L607), [`Person_DestroyIfTaken`](src/bank1/underworld-person-init.asm#L121)
- Life-or-money man (type `$51`): a heart container at X `$58` and a rupee at X `$98` (Y `$98`), costs "-1" and "-50"; Link chooses by standing on one (exact X, |Y - `$98`| < 6). [`Person_LifeOrMoneyItemXs`](src/bank1/life-or-money.asm#L63), [`Person_LifeOrMoneyState2Choose`](src/bank1/life-or-money.asm#L121)
- Rupee choice needs `InvRupees` >= 50, else nothing happens. [`Person_LifeOrMoneyState2Choose@Pay`](src/bank1/life-or-money.asm#L147)
- Heart choice with 4 or more containers: one container and one full heart are removed. With 3 or fewer: containers kept, full hearts and `HeartPartial` set to 0. [`Person_LifeOrMoneyState2Choose@PayWithHeartContainer`](src/bank1/life-or-money.asm#L166)
- Either payment sets `ShutterTrigger` = 1 (the shutters open) and room flag `$10`; he vanishes after `$40` frames and is not made again. [`Person_LifeOrMoneyState2Choose@Paid`](src/bank1/life-or-money.asm#L207), [`Person_InitLifeOrMoney_Full`](src/bank1/underworld-person-init.asm#L105)
- Level 9, type `$4B`: with all eight Triforce pieces (`InvTriforce` = `$FF`) he sets `ShutterTrigger`, releases Link and disappears during init; otherwise he stays and blocks the way up. [`Person_InitUnderworldLevel9`](src/bank1/underworld-person-init.asm#L172)
- Attacking a dungeon person (or the Grumble Goriya) does not kill him: the hit sets `IsPersonFireballsEnabled`, and from then on fireballs are shot at Link from (`$48`,`$82`) and (`$A8`,`$86`), beside the two fires. [`Person_CheckCollisions`](src/bank1/underworld-person-update.asm#L89), [`Statue_Update`](src/bank4/statues.asm#L86)
- The fireballs stop only on leaving: the flag lies in the RAM that room entry zeroes. [`Room_EnterCommon`](src/bank5/enter-room.asm#L143), [`IsPersonFireballsEnabled`](src/Variables.inc#L383)

# Grumble Goriya

The hungry Goriya that blocks a passage until it is given the food. It is a dungeon person in all but its update.

**Types**
- Object type `$36`, slot `$01`, at (`$78`, `$80`) with the two fires; text "GRUMBLE,GRUMBLE...". [`Grumble_Init_Full`](src/bank1/underworld-person-init.asm#L216)
- First-quest room: level 7 room `$28` (see `data/levelblock-uw-7to9-q1.md`). [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)

**Behaviour**
- States: 0 type the text (Link halted until it ends), 1 wait for the food, 2 blank the text (1 frame), 3 flicker then vanish. [`Grumble_Update_JumpTable`](src/bank1/grumble-goriya.asm#L61)
- While it stands there Link cannot move up past Y `$8D`. [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289), [`Walker_Move@CheckPersonBlocking`](src/bank7/walker-move.asm#L149)

**Conditions**
- The food test is bit 7 of the state of object slot `$0F`: the food is out (placed in state `$80`). Where the food or Link is does not matter. [`Grumble_State1WaitForFood`](src/bank1/grumble-goriya.asm#L73), [`Food_Wield`](src/bank5/wield-items.asm#L247)
- The test starts only after the text has been typed (state 1 follows state 0). [`Grumble_Update_JumpTable`](src/bank1/grumble-goriya.asm#L61)

**Effect**
- When the food is seen: Link is halted (`ObjState` = `$40`), tune-1 request `$04` plays, room flag `$10` is set at once, and a `$40`-frame timer starts. [`Grumble_State1WaitForFood`](src/bank1/grumble-goriya.asm#L73), [`Person_FlagTakenAndBlankText`](src/bank1/grumble-goriya.asm#L92)
- When the timer ends: the food object is removed, `InvFood` = 0 (the food is gone for good), Link is released and the Goriya's slot is freed. [`Grumble_State3TakeFood`](src/bank1/grumble-goriya.asm#L109)
- What changes in the room: slot `$01` is empty, so the block on moving up no longer applies. The routine opens no shutter and gives no item. [`Grumble_State3TakeFood`](src/bank1/grumble-goriya.asm#L109)
- Saved: room flag `$10`. On later visits the Goriya is not made and Link is not halted. [`Grumble_Init_Full`](src/bank1/underworld-person-init.asm#L216)

**Notes**
- A weapon hit does not kill it; it turns on the person fireballs, as for an old man. [`Grumble_Update_Full`](src/bank1/grumble-goriya.asm#L41), [`Person_CheckCollisions`](src/bank1/underworld-person-update.asm#L89)

# Rupee stash

A room full of rupees lying on the floor (the "secret money" rooms of the levels).

**Types**
- Object type `$35`. Its init turns object slots `$0A`-`$01` into ten type `$35` objects, one rupee each; the stash object itself is slot `$01`. [`RupeeStash_Init_Full`](src/bank1/trap-rupee-stash.asm#L75)
- Layout, a diamond in rows of 1, 2, 4, 2, 1: (`$78`,`$70`); (`$70`,`$80`) (`$80`,`$80`); (`$60`,`$90`) (`$70`,`$90`) (`$80`,`$90`) (`$90`,`$90`); (`$70`,`$A0`) (`$80`,`$A0`); (`$78`,`$B0`). [`RupeeStash_RupeeXs`](src/bank1/trap-rupee-stash.asm#L55), [`RupeeStash_RupeeYs`](src/bank1/trap-rupee-stash.asm#L61)

**Conditions**
- A rupee is taken when |Link X - rupee X| < 9 and |Link Y - rupee Y| < 9. No button, weapon or other state is tested. [`RupeeStash_Update_Full`](src/bank1/trap-rupee-stash.asm#L338)

**Effect**
- Each rupee is worth 1 (`InvRupeesToAdd` + 1); its slot is freed with no drop and no kill counted. The whole stash is 10 rupees. [`RupeeStash_Update_Full`](src/bank1/trap-rupee-stash.asm#L338), [`Item_TakeOneRupee`](src/bank1/take-item.asm#L557)
- Taking any rupee sets `RoomObjCount` = 0, which makes the room count as cleared when Link leaves; a cleared room does not make the stash again. [`RupeeStash_Update_Full`](src/bank1/trap-rupee-stash.asm#L338), [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110), [`RoomHistory_ModifyObjCountUW`](src/bank5/room-history-uw.asm#L50)

**Notes**
- Rupees left behind after the first one is taken are lost once Link leaves; a stash left untouched is made again in full. [`RoomHistory_SaveKillCountUW`](src/bank5/room-history-uw.asm#L110)
- The rupees are not dropped items: they never expire and weapons cannot collect them. [`RupeeStash_Update_Full`](src/bank1/trap-rupee-stash.asm#L338)

# Pond fairy

The fairy over a pond that refills all of Link's hearts.

**Types**
- Object type `$2F`, placed at (`$78`, `$7D`); its code reads state and timer from object slot `$01`. [`PondFairy_Init`](src/bank4/zora-ow-inits.asm#L270), [`PondFairy_Update`](src/bank4/pond-fairy.asm#L43)
- First-quest rooms: overworld `$39` and `$43` (see `data/levelblock-ow.md`). [`Level_BlockOW`](src/bank6/level-blocks-common-data.asm#L66)

**Conditions**
- Trigger (state 0): Link's Y exactly `$AD` and X in `$70`-`$80`. Current health is not tested. [`PondFairy_Update`](src/bank4/pond-fairy.asm#L43)

**Effect**
- On the trigger Link is halted (`ObjState` = `$40`) and `World_IsFillingHearts` is set. [`PondFairy_Update`](src/bank4/pond-fairy.asm#L43)
- Healing rate: `HeartPartial` + 6 per frame; at `$F8` or more the next frame completes the heart. An empty heart takes 43 frames. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- The refill ends when full hearts equal the heart containers (`HeartPartial` = `$FF`). [`World_FillHearts@CompleteHeart`](src/bank5/cave-cellar-enter.asm#L356)
- Link is released `$50` frames after the refill ends (state 2 timer), so even at full health (where the refill ends on its first frame) a trigger costs about `$50` frames. [`PondFairy_UpdateStates1To3`](src/bank4/pond-fairy.asm#L75)
- The fairy then stays in state 3 and never rearms: one refill per room visit. [`PondFairy_UpdateStates1To3@CheckState2`](src/bank4/pond-fairy.asm#L91)

**Notes**
- While Link's state is exactly `$40` the monster-touch test returns at once, so nothing can hurt him by contact during the refill. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Eight hearts orbit the pond during states 1 and 2, using object slots `$02`-`$09` for their positions; they appear one after another, 4 angle steps apart. [`PondFairy_MoveHearts`](src/bank4/pond-fairy.asm#L146), [`PondFairy_HeartStartAngles`](src/bank4/pond-fairy.asm#L135)
- The fairy's update draws it and runs no collision check. [`PondFairy_Update`](src/bank4/pond-fairy.asm#L43)

# Item fairy

The fairy a dead monster can leave. It is a dropped item (next section) with item id `$23` that flies instead of lying still.

**Types**
- Item id `$23` in a type `$60` object; set up as a flyer when the drop is decided. [`Drop_SetUpItem@Commit`](src/bank4/drop-items.asm#L260), [`Fairy_Init`](src/bank4/fairy-flyer-common.asm#L82)
- Sources: the drop table (subject to the group's random test) and the forced fairy when `WorldKillCount` = `$10`; see `# Drop tables`. [`Drop_SetUpItem@LookUpItem`](src/bank4/drop-items.asm#L216)

**Behaviour**
- Starts facing up with speed `$7F`, and sets the room's flyer top speed `FlyerMaxSpeedFrac` to `$A0`. [`Fairy_Init`](src/bank4/fairy-flyer-common.asm#L82)
- Speeds up by 1 per frame: 0.5 px/frame for the first `$20` frames, then 0.625 px/frame (`$A0`) for the rest of its life; it never slows or stops. [`Flyer_State0SpeedUp`](src/bank4/fairy-flyer-common.asm#L210), [`Flyer_Move`](src/bank4/fairy-flyer-common.asm#L250)
- At top speed it only wanders: every `$10` frames it turns 45 degrees either way or keeps its direction, at random. It never moves toward Link. [`Fairy_DecideFlightState`](src/bank4/fairy-flyer-common.asm#L165), [`Flyer_State3Wander`](src/bank4/fairy-flyer-common.asm#L507)
- No tile collision: only the room bounds turn it around. [`Flyer_BoundByRoom`](src/bank4/fairy-flyer-common.asm#L309)
- Lifetime and pickup are those of any dropped item: about 510 frames, not takeable for about the first 32, takeable by Link or an active sword, boomerang or arrow slot. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)
- It does not flash during the untakeable period: it is drawn by its own routine, two frames alternating every 4 frames. [`Drop_UpdateItem@AnimateFairy`](src/bank4/drop-items.asm#L315), [`Fairy_Draw`](src/bank4/fairy-flyer-common.asm#L130)

**Effect**
- Taken: up to 3 full hearts are added, one at a time while full hearts < containers - 1; once they are equal the partial heart is set full (`$FF`) and the rest is discarded. [`Item_TakeHeartsNoSound`](src/bank1/take-item.asm#L475), [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- The healing is immediate: no halt and no gradual refill, unlike the pond fairy. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)

**Notes**
- `FlyerMaxSpeedFrac` is one byte for the whole room: after a fairy drop, a Keese that speeds up again tops out at `$A0` instead of its own `$C0`. [`Fairy_Init`](src/bank4/fairy-flyer-common.asm#L82), [`Flyer_CompareMaxSpeed`](src/bank4/fairy-flyer-common.asm#L234)

# Dropped items

What a dead monster leaves behind. Which item drops, and how often, is in `# Drop tables`; what each item does to the
inventory in full is in `items.md`.

**Types**
- Object type `$60`, created in the dead monster's own object slot (`$01`-`$0B`) when its death sparkle ends; no free slot is needed. [`Obj_EndMetastate@DropItem`](src/bank7/object-update-table.asm#L345), [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)
- `Item_ObjItemId` holds the item id, `Item_ObjItemLifetime` the life left; possible ids are heart `$22`, rupee `$18`, 5 rupees `$0F`, bomb `$00`, clock `$21`, fairy `$23`. [`Drop_ItemTable`](src/bank4/drop-items.asm#L132), [`Item_ObjItemId`](src/CommonVars.inc#L22)

**Behaviour**
- Lifetime starts at `$FF` and loses 1 on every frame with an odd `FrameCounter`; at 0 the slot is freed: about 510 frames (8.5 s). [`Drop_SetUpItem@Commit`](src/bank4/drop-items.asm#L260), [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)
- While the lifetime is >= `$F0` (the first 16 steps, about 32 frames) the item cannot be taken. [`Item_TryTake`](src/bank1/take-item.asm#L206)
- During that period it flashes: drawn only when the lifetime value is odd. [`Item_DrawObject`](src/bank7/room-item-draw.asm#L101)
- The item keeps aging while Link is halted (`ObjState` & `$C0` = `$40`) but nothing can take it then. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)
- It lies where the monster died; only a fairy moves. [`Drop_UpdateItem@AnimateFairy`](src/bank4/drop-items.asm#L315)

**Conditions**
- Takers, tested in this order: slot `$12` (arrow or rod), slot `$0D` (sword), slot `$0F` (boomerang), then Link. [`Drop_ItemTakerObjSlots`](src/bank4/drop-items.asm#L280), [`Drop_UpdateItem@LoopItemTaker`](src/bank4/drop-items.asm#L331)
- A weapon slot counts while its state is nonzero with bit 7 clear; the object in the slot is not identified, and the food (state `$80`) never counts. [`Drop_UpdateItem@LoopItemTaker`](src/bank4/drop-items.asm#L331), [`Food_Wield`](src/bank5/wield-items.asm#L247)
- Slots `$0E` (sword shot, magic shot) and `$10`/`$11` (bombs, fire) are not takers. [`Drop_ItemTakerObjSlots`](src/bank4/drop-items.asm#L280)
- Range, with the taker's position in place of Link's: |taker X - item X| < 9 and |taker Y + 3 - item Y| < 9 (taker Y - item Y from -11 to +5). [`Item_TryTake`](src/bank1/take-item.asm#L206)
- The item goes to Link at once wherever the weapon is: a thrown boomerang or an arrow collects at a distance. [`Drop_UpdateItem@TryToTake`](src/bank4/drop-items.asm#L353), [`Item_Take`](src/bank1/take-item.asm#L259)

**Effect**
- Heart `$22`: one full heart, up to the containers. [`Item_TakeHearts`](src/bank1/take-item.asm#L463)
- Rupee `$18`: `InvRupeesToAdd` + 1. [`Item_TakeOneRupee`](src/bank1/take-item.asm#L557)
- 5 rupees `$0F`: `InvRupeesToAdd` + 5. [`Item_Take5Rupees`](src/bank1/take-item.asm#L328)
- Bomb `$00`: `InvBombs` + 4, capped at `InvMaxBombs`. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337), [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- Clock `$21`: `InvClock` = 1 (monsters frozen, Link untouchable); cleared when Link leaves the room. [`Item_Take`](src/bank1/take-item.asm#L259), [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354)
- Fairy `$23`: up to 3 full hearts (see `# Item fairy`). [`Item_TakeHeartsNoSound`](src/bank1/take-item.asm#L475)

**Notes**
- A taken item is marked by item id `$FF` and its slot is freed in the same frame. [`Drop_UpdateItem@SkipTaking`](src/bank4/drop-items.asm#L360)
- In mode 5 a pickup does not halt Link (the lift happens only when `GameMode` <> 5). [`Item_Take@SkipLift`](src/bank1/take-item.asm#L281)
- The room item (object slot `$13`) is a different object: it does not expire and only Link takes it. [`Item_TryTakeRoomItem`](src/bank1/take-item.asm#L173)
