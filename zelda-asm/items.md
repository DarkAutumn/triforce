# Items and weapons

Everything the code says about each item and weapon: two system sections first (object slots and what may be
out at once; state values and frame counts), then one section per item family with every variant in the
heading, so a search for `# Magic Rod` or `# Candle` lands on it. Every bullet ends with the label that proves
it. `(trace)` marks a result observed in the running game; `(unverified)` a statement not confirmed in the
code. Monsters are in [enemies.md](enemies.md), shared rules in [mechanics.md](mechanics.md).

Contents (generated; `L` = line in this file):

<!-- toc -->
- L36 Weapon object slots and limits
- L83 Animation and state values
- L134 Sword (Wooden, White, Magic) and sword beam
- L186 Bombs
- L237 Arrows and Bow (wooden, silver)
- L280 Boomerang (wooden, magic)
- L327 Candle (blue, red)
- L373 Recorder (flute, whistle)
- L433 Food (bait, meat)
- L473 Potion (blue, red) and Letter
- L523 Magic Rod and Book
- L574 Raft
- L610 Ladder
- L651 Keys and Magic Key
- L687 Power Bracelet
- L717 Ring (blue, red)
- L742 Shield (small, magic)
- L785 Hearts, Heart Containers and Fairies
- L830 Rupees
- L870 Clock
- L909 Map and Compass
- L942 Triforce pieces and the Triforce of Power
<!-- /toc -->

# Weapon object slots and limits

Link's weapons are not typed objects. Each kind lives in a fixed object slot `$0D`-`$12` and is told apart only by the slot and its state byte (`ObjState` + slot; 0 = slot free). "Busy" always means a test of that state byte.

**Slots**
- `$0D` sword: states 1-5; started by A, run by the swing routine. [`Sword_Wield`](src/bank5/wield-items.asm#L59) [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- `$0E` sword shot (`$10` flying, `$11` spreading) or magic shot (`$80`, bit 7 set): made by the sword or rod swing on entering its third state. [`SwordShot_UpdateOrMagicShot`](src/bank7/sword-shot-magic-shot.asm#L56) [`SwordShot_SetUpWithState`](src/bank7/sword-and-rod.asm#L272)
- `$0F` boomerang (`$1x`-`$5x`, bit 7 clear) or food (`$80`-`$82`). [`Boomerang_UpdateOrFood`](src/bank7/boomerang-arrow-food.asm#L70)
- `$10` and `$11` bomb (`$11`-`$14`) or fire (`$21`, `$22`): the high nibble of the state picks the routine. [`Bomb_UpdateBombOrFire`](src/bank7/bomb.asm#L71)
- `$12` arrow (`$10` flying, `$20`/`$28` spark) or rod swing (`$31`-`$35`): states `$30` and up go to the sword/rod routine, the rest to the arrow routine. [`Rod_UpdateRodOrArrow`](src/bank7/sword-and-rod.asm#L57)
- Weapon slots have a state, position, direction, timer, animation counter, q-speed and grid offset, but no object type, HP or metastate: those arrays cover slots `$00`-`$0C` only. [`ObjType`](src/Variables.inc#L303) [`ObjState`](src/Variables.inc#L200)
- Slot `$13` is the room item; slot `$0C` is not updated by the play loop. [`RoomItemId`](src/Variables.inc#L195) [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)

**Order within a frame**
- Update: Link, then slots `$0D`, `$0E`, `$0F`, `$10`, `$11`, `$12`, then object slots `$0B` down to `$01`; a monster's collision test therefore sees the weapon states left by this frame's weapon updates. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- If Link's update changed the game mode, no weapon slot is updated that frame. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- Each monster tests the weapon slots in the order `$0F`, `$0E`, `$10`, `$11`, `$0D`, `$12`, then its own touch on Link. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- The monster's invincibility timer is tested once, before the first weapon: two weapons overlapping it in one frame both deal their damage (trace: a magic shot's `$20` and a bomb blast's `$40` on one frame). [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)

**Slot busy tests (what may coexist)**
- Sword: refused when `InvSword` = 0 or slot `$0D`'s state is not 0. [`Sword_Wield`](src/bank5/wield-items.asm#L59)
- Sword shot: refused when slot `$0E`'s state is not 0 (a flying or spreading sword shot, or a magic shot). [`SwordShot_Make`](src/bank7/sword-and-rod.asm#L327)
- Magic shot: refused only when slot `$0E` holds a magic shot (bit 7 set); it overwrites a sword shot that is flying or spreading. [`Sword_UpdateSwordOrRod@MakeMagicShot`](src/bank7/sword-and-rod.asm#L258)
- Boomerang: refused only when slot `$0F`'s state is not 0 with bit 7 clear (a boomerang is out); a throw replaces food lying on the ground. [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)
- Food: refused when slot `$0F`'s state is not 0 (boomerang out or food already down). [`Food_Wield`](src/bank5/wield-items.asm#L247)
- Bomb, slot choice: `$10` if it is empty or holds a fire, else `$11` if it is empty or holds a fire, else refused (two bombs). [`Bomb_Wield`](src/bank1/bomb-candle-wield.asm#L63)
- Bomb, second rule: the other slot must be empty or in a state of `$13` or more (a bomb already exploding, or any fire), so only one bomb can be in its fuse. [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87)
- A bomb placed in a slot that holds a fire replaces the fire, even when the other slot is empty. [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87)
- With a bomb placed on frame 0 the second bomb is accepted from frame 49, the first frame after the blast state `$13` begins (trace). [`Bomb_Wield`](src/bank1/bomb-candle-wield.asm#L63) [`Bomb_StateTimes`](src/bank7/bomb.asm#L55)
- Fire (candle): needs an empty slot, `$10` first, else `$11`; a fire never replaces a bomb or a fire. [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220)
- The book's fire is claimed through the same routine and the same two slots. [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)
- Arrow: refused when slot `$12`'s state is not 0 with bit 7 clear; no state with bit 7 is ever stored there, so a rod swing or an arrow (flying or sparking) blocks it. [`Arrow_Wield`](src/bank5/wield-items.asm#L187)
- Rod: refused when slot `$12`'s state is not 0 (rod swing or arrow). [`Rod_Wield`](src/bank5/wield-items.asm#L294)
- Possible together: sword, one shot, boomerang or food, two of bomb/fire, arrow or rod. Never together: sword shot and magic shot; boomerang and food; arrow and rod; two bombs in fuse. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)

**Link's state gate**
- A and B are read only while Link's state is 0, and the state is tested once before both: A and B newly pressed on one frame start the sword and the B item together (trace: sword with rod, sword with bomb). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- Every wield routine except the potion and the flute sets Link's state to `$10`, which is what keeps a second weapon from being started until the state is 0 again. [`Link_UseItem_SetState10`](src/bank1/bomb-candle-wield.asm#L143) [`Potion_Wield`](src/bank5/wield-items.asm#L269) [`Flute_Wield`](src/bank7/flute.asm#L49)
- Near a screen edge Link faces, a new B press is discarded (down Y >= `$BE`, up Y < `$54`, right X >= `$D1`, left X < `$1F`); farther out A is discarded too (overworld Y >= `$D6`, Y < `$45`, X >= `$E9`, X < `$07`; dungeon `$C6`, `$55`, `$D9`, `$17`), also with that edge behind him. [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54) [`Link_BorderBounds`](src/bank5/border-input-mask.asm#L38)

**Shared rules**
- A new weapon is put `$10` pixels from Link's position in the direction he faces and takes his direction. [`Link_PlaceWeaponAhead`](src/bank1/bomb-candle-wield.asm#L154)
- Shots, arrows and boomerangs are stopped by the room bounds, not by tiles; for weapon slots the bounds are wider than for Link: left X < `$06` overworld / `$16` dungeon, right X >= `$EC` / `$DC`, up Y < `$3F` / `$4F`, down Y >= `$DF` / `$CF`. [`Room_BoundDirHorizontally`](src/bank1/room-bounds.asm#L116) [`Room_BoundDirVertically`](src/bank1/room-bounds.asm#L185) [`Obj_MoveShot`](src/bank1/shot-aim-boomerang-speed.asm#L49)
- Dropped items can be taken by the position of slot `$12`, `$0D` or `$0F` (state not 0, bit 7 clear, so not by food), then by Link. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297) [`Drop_ItemTakerObjSlots`](src/bank4/drop-items.asm#L280)
- Room change: the states of slots `$0D`-`$12` are zeroed (and the ladder released) when Link leaves a room, enters a room, cave or cellar, and at the end of mode `$A`; nothing that is out survives. [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- A room change also sets Link's state to 0, ending a swing or item use. [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281) [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354)

# Animation and state values

Link's state is `ObjState` of slot 0 (`$AC`); a weapon's state is `ObjState` + slot (`$B9` sword, `$BA` shot, `$BB` boomerang or food, `$BC`/`$BD` bomb or fire, `$BE` arrow or rod). Frame 0 is the frame on which the button press is read; values are those at the end of the frame, which is what the monsters' collision tests see.

**Link's state byte**
- 0 = idle, the only state in which A and B are read. [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- `$10` -> `$11` -> `$31` -> 0 = swinging the sword or using an item; `$20` -> `$21` -> `$31` -> 0 = catching the boomerang. [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223)
- A step happens only when Link's animation counter reaches 0; the counter is then reloaded with 6, so a step lasts 6 frames unless other code writes the counter. [`Anim_AnimateObjWalking`](src/bank7/object-animation.asm#L121) [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223)
- While (state AND `$F0`) is `$10` or `$20` Link's moving direction is zeroed; `$31` does not stop him. A doorway or the laid ladder can still move him. [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104)
- Directions are still processed during a use, so Link can turn; the sword and rod take his facing every frame (trace: turning from down to up in mid-swing turned the sword with him). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- While (state AND `$30`) is `$10` or `$20` Link is drawn in the item-use pose (frame + 4); `$31` uses the walk frames again. [`Link_EndMoveAndAnimate@ChangeLookForState`](src/bank7/link-animate-ladder.asm#L312)
- (state AND `$C0`) = `$40` = halted: Link's update does only the invincibility timer step; no input, movement or animation. [`Link_Update`](src/bank7/mode5-play-update.asm#L330)
- `$40` is stored by cave and dungeon persons, by the item lift (`$7F` frames after a pickup outside mode 5), the whirlwind, the raft, the pond fairy, the Wallmaster, the Triforce of Power fanfare, Zelda and Ganon; released by storing 0. [`Person_SetUpWithFires`](src/bank1/cave-init-person-text.asm#L304) [`Link_CheckLiftItem`](src/bank7/status-bar-lift-item.asm#L162) [`Link_Unhalt`](src/bank1/cave-init-person-text.asm#L130)
- Ending an item use keeps bits `$C0`, so a halt set meanwhile survives. [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223)
- A state of `$10` or more disables the shield; a state of exactly `$40` also makes monsters and shots unable to touch Link. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Sword swing frame table** (trace, agrees with the timers in [`Sword_Wield`](src/bank5/wield-items.asm#L59) and [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)). The rod uses the same table with states `$31`-`$35` in slot `$12` (`$BE`).

| Frame | Link `$AC` | Sword `$B9` | Event |
|---|---|---|---|
| 0 | `$11` | 1 | A read; sword placed, not drawn; Link stops |
| 1-3 | `$11` | 1 | |
| 4-11 | `$11` | 2 | sword drawn, fully extended; the only frames it can hit |
| 12 | `$31` | 3 | sword shot (or the rod's magic shot) is made |
| 13 | 0 | 4 | Link moves again |
| 14 | 0 | 5 | sword not drawn; A/B are read again (the sword slot is still busy) |
| 15 | 0 | 0 | slot free |
| 16 | | | earliest frame a new swing can start |

- The wield routine loads timer 5 and the swing routine counts it once in the same frame, so state 1 shows for 4 frames; state 2 gets timer 8; states 3, 4 and 5 get 1 each. [`Sword_Wield`](src/bank5/wield-items.asm#L59) [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- Each state change of the sword or rod writes the same count to Link's animation counter; that is why Link's `$11` lasts through frame 11 and `$31` one frame, instead of 6 and 6. [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- Link stands still on frames 0-12 and first moves on frame 13: the movement test sees his state before that frame's animation step (trace). [`Walker_Move@SetMovingDir`](src/bank7/walker-move.asm#L104)
- A press on frame 13 is ignored (Link's state is still `$31` when it is read); B is accepted from frame 14; A starts a new swing from frame 16: at best one swing per 16 frames (trace). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Sword_Wield`](src/bank5/wield-items.asm#L59)

**Other items: Link's state**
- Bomb, arrow, boomerang, food (Link's counter set to 1): `$11` on frames 0-5, `$31` on frames 6-11, 0 from frame 12; Link stands still on frames 0-6; A/B are read again from frame 13 (trace). [`Link_UseItem_SetAnimCounter1`](src/bank1/bomb-candle-wield.asm#L131) [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)
- Candle: Link's counter is not set, so `$10` lasts (counter - 1) frames, 0 to 5, before the same 6 + 6 (trace: counter 5 gave `$10` on frames 0-3, `$11` on 4-9, `$31` on 10-15). [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220) [`Link_UseItem_SetState10`](src/bank1/bomb-candle-wield.asm#L143)
- Boomerang catch: state OR `$20` with counter 1: `$20` one frame, `$21` 6 frames, `$31` 6 frames, during which A and B are ignored (trace: caught on frame 79; `$20` on 79, `$21` on 80-85, `$31` from 86). [`Boomerang_UpdateSlowAndReturn`](src/bank7/boomerang-return-draw.asm#L52)
- Potion and flute leave Link's state at 0: the potion pauses the game for the refill, the flute freezes all of play for `$98` frames. [`Potion_Wield`](src/bank5/wield-items.asm#L269) [`Flute_Wield`](src/bank7/flute.asm#L49)

**Weapon states and frames**
- Sword shot: `$10` flying at q-speed `$C0` (3 pixels a frame) from frame 12 until a room bound or a monster; `$11` spreading for 22 frames; then 0 (trace, fired down from Y `$4D`: `$10` on frames 12-55, `$11` on 56-77). [`SwordShot_UpdateOrMagicShot`](src/bank7/sword-shot-magic-shot.asm#L56) [`SwordShot_UpdateSpread`](src/bank7/sword-shot-magic-shot.asm#L255)
- Magic shot: `$80` flying at q-speed `$A0` (2.5 pixels a frame); when stopped the slot is freed in the same frame (trace, same shot: `$80` on frames 12-64). [`SwordShot_SetUpWithState`](src/bank7/sword-and-rod.asm#L272) [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)
- Boomerang: `$1x` out at 3 pixels a frame (low 3 bits = spin step, +1 every 2 frames); `$3x` slowing, 1 pixel a frame for `$10` frames; `$4x` slow return for `$10` frames; `$5x` fast return until within 8 pixels of Link on both axes. [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192) [`Boomerang_UpdateSlowAndReturn`](src/bank7/boomerang-return-draw.asm#L52)
- Wooden boomerang, Link standing still (trace): `$1x` frames 0-15, `$3x` 16-31, `$4x` 32-47, `$5x` 48-78, caught on frame 79. [`Boomerang_RangeLimits`](src/bank5/wield-items.asm#L85)
- A boomerang blocked at a room bound sparks (`$20`, then `$28`) for 3 frames and returns in `$50`; one that overlaps a monster goes straight to `$50` (trace, magic boomerang thrown down from Y `$4D`: `$1x` frames 0-43, spark 44-46, `$5x` from 47). [`Boomerang_HandleBlockedOrArrow`](src/bank7/boomerang-arrow-food.asm#L404) [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Food: `$80`, `$81`, `$82`, `$FF` frames each (765 frames), then 0. [`Boomerang_UpdateOrFood`](src/bank7/boomerang-arrow-food.asm#L70) [`Food_Wield`](src/bank5/wield-items.asm#L247)
- Bomb: `$11` when placed (gone within the same frame), `$12` fuse on frames 0-47, `$13` blast on frames 48-71 (the only damaging state), `$14` clouds on frames 72-83, slot free on frame 84 (trace). [`Bomb_Update`](src/bank7/bomb.asm#L89) [`Bomb_StateTimes`](src/bank7/bomb.asm#L55)
- Fire: `$21` moving 0.5 pixel a frame for `$10` pixels (frames 0-30), `$22` standing for `$3F` frames (31-93), slot free on frame 94 (trace). The book's fire starts in `$22` with `$4F` frames. [`Fire_Update`](src/bank7/fire.asm#L44) [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220)
- Arrow: `$10` flying at 3 pixels a frame with no range limit; at a room bound or on a monster `$20` for one frame, `$28` for two, slot free on the next (trace: `$20` on frame 44, `$28` on 45-46, 0 on 47). [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192) [`Boomerang_UpdateOrArrow_State2Spark`](src/bank7/boomerang-arrow-food.asm#L352)

# Sword (Wooden, White, Magic) and sword beam

The code calls the beam the "sword shot".

**Ids**
- Item ids `$01` wooden, `$02` white, `$03` magic sword; all three use item slot 0. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)
- The sword is always on A; it has no selected-slot index (selected slot 0 is the boomerang). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`SelectedItemSlot`](src/Variables.inc#L553)

**RAM**
- `InvSword` (`$657`): 0 none, 1 wooden, 2 white, 3 magic. [`InvSword`](src/Variables.inc#L556)
- Sword object: slot `$0D` (state at `$B9`); sword shot: slot `$0E` (state at `$BA`). [`Sword_Wield`](src/bank5/wield-items.asm#L59) [`SwordShot_Make`](src/bank7/sword-and-rod.asm#L327)
- `SwordBlockedLongTimer` (`$4C`, 10-frame units) and `IsSwordBlocked` (`$52E`): the sword cannot be started while either is not 0. [`SwordBlockedLongTimer`](src/Variables.inc#L62) [`IsSwordBlocked`](src/Variables.inc#L474)

**Use**
- A newly pressed while Link's state is 0: slot `$0D` gets state 1 and timer 5, Link's state becomes `$10`, the sword noise plays. No cost. [`Sword_Wield`](src/bank5/wield-items.asm#L59) [`Link_WieldWeapon`](src/bank5/wield-items.asm#L219)
- Timeline (table in "Animation and state values"): hitting state 2 on frames 4-11, shot on frame 12, Link free on frame 13, slot free on frame 15, next swing from frame 16. [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- The sword is repositioned from Link every frame; in state 2 its top-left corner is at Link's X, Y plus (-1, -10) facing up, (+1, +13) down, (-11, +3) left, (+11, +3) right. [`Sword_LinkToWeaponOffsetsX`](src/bank7/sword-and-rod.asm#L68) [`Sword_LinkToWeaponOffsetsY`](src/bank7/sword-and-rod.asm#L74)
- Drawn in states 2-4 only (frames 4-13), palette row 3 + `InvSword`. [`Sword_UpdateSwordOrRod@CalcSwordAttrs`](src/bank7/sword-and-rod.asm#L208)

**Limits**
- Refused when `InvSword` = 0 or a swing is in progress (slot `$0D` not 0). [`Sword_Wield`](src/bank5/wield-items.asm#L59)
- Flashing Bubble (`$2B`) touch: `SwordBlockedLongTimer` = `$10`, no sword for 151 to 160 frames. [`Bubble_Update`](src/bank4/bubble-keese.asm#L93)
- Bubble `$2D` touch: `IsSwordBlocked` = 1 until a Bubble `$2C` touch or the end of any heart refill clears it. B items are not affected by either block. [`Bubble_Update@BlockOrUnblock`](src/bank4/bubble-keese.asm#L137) [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343) [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- No A press is read while Link's state is not 0 (item use, boomerang catch, halted) or in the outer border band at a screen edge. [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54)

**Effect**
- Hits only while the sword's state is 2 (8 frames per swing). Damage type 1; damage `$10` wooden, `$20` white, `$40` magic, in `ObjHP` units (HP x `$10`). [`Sword_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L192) [`Sword_DamagePoints`](src/bank1/sword-bomb-arrow-collision.asm#L182)
- Hit test: the sword's point (X+6, Y+8 when Link faces up or down; X+8, Y+6 when left or right) against the monster's middle, less than `$10` pixels along Link's facing axis and less than `$0C` across it. [`Collision_CheckStabVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L215) [`Collision_CheckThinWeaponVsMonster_Thresholds`](src/bank1/object-overlap-test.asm#L47)
- A monster whose invincibility mask has bit 1 takes nothing (parry tune); a Darknut facing exactly against the sword takes nothing. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- A monster that is hit is shoved `$40` pixels the way the sword points and gets invincibility timer `$10` (about 32 frames), during which no weapon is tested against it. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101)
- The sword's position picks up dropped items in any state of the swing. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)

**Sword beam**
- Made on frame 12 of a swing (sword entering state 3) when slot `$0E` is empty and health is full. [`SwordShot_Make`](src/bank7/sword-and-rod.asm#L327)
- Full health test: high nibble of `HeartValues` (containers - 1) = low nibble (full hearts), and `HeartPartial` >= `$80`; up to just under half a heart may be missing (trace: `$22`/`$80` fires; `$22`/`$7F` and `$21`/`$FF` do not). [`SwordShot_Make`](src/bank7/sword-and-rod.asm#L327) [`HeartValues`](src/Variables.inc#L593) [`HeartPartial`](src/Variables.inc#L595)
- `IsSwordShotForced` would skip the health test, but nothing ever sets it. [`IsSwordShotForced`](src/Variables.inc#L467)
- Starts `$10` pixels ahead of Link in state `$10` at q-speed `$C0` (3 pixels a frame); facing left or right with the shot's X < `$14` or >= `$EC` it is dropped at once. [`SwordShot_SetUpWithState`](src/bank7/sword-and-rod.asm#L272)
- Flies until a room bound (tiles do not stop it) or until it overlaps a monster, immune or not. [`SwordShot_UpdateOrMagicShot`](src/bank7/sword-shot-magic-shot.asm#L56) [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)
- Damage type 1 and the damage of the sword held (`$10`/`$20`/`$40`); hit test less than `$0C` pixels on both axes. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)
- Spread: when stopped it becomes state `$11`: four corner sprites move diagonally 1 pixel a frame, from 2 to 23 pixels out, for 22 frames, doing no damage. [`SwordShot_StartSpread`](src/bank7/sword-shot-magic-shot.asm#L239) [`SwordShot_UpdateSpread`](src/bank7/sword-shot-magic-shot.asm#L255)
- The slot stays busy while the beam flies and spreads, so a swing made meanwhile gets no beam. [`SwordShot_Make`](src/bank7/sword-and-rod.asm#L327)

**Pickup**
- Class 2 (grade 1, 2, 3): the grade is stored only if it is not lower than the one held; a better sword is never downgraded. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_TakeClass2Grade`](src/bank1/take-item.asm#L536)
- Given by cave persons `$6A` (wooden, no condition), `$6C` (white: `HeartValues` >= `$40`, 5 containers) and `$6D` (magic: >= `$B0`, 12 containers); only the container count matters, not current health. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- Caves: wooden in overworld room `$77`, white in `$0A`, magic under the gravestone of `$21` (Q1) or the rock of `$09` (Q2); see `data/levelblock-ow.md`. [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)
- Taken in a cave (not mode 5), the sword is held up and Link is halted for `$7F` frames. [`Item_Take`](src/bank1/take-item.asm#L259) [`Link_CheckLiftItem`](src/bank7/status-bar-lift-item.asm#L162)

**Notes**
- The sword in the status bar is drawn from `InvItems`+0. [`StatusBar_DrawSword`](src/bank7/status-bar-lift-item.asm#L130)
- Sword and rod are not exclusive: see "Magic Rod and Book", Limits. [`Rod_Wield`](src/bank5/wield-items.asm#L294)

# Bombs

A bomb is an object in slot `$10` or `$11` (shared with fires) that sits where it was put, burns a fuse, blasts and fades. Monster HP and weapon damage are in the same units: `$10` is one wooden sword hit.

**Ids**
- Item id `$00`, item slot `$01`; selected-slot index (B button) 1. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)

**RAM**
- `InvBombs` (`$658`): bombs held, 0 to `InvMaxBombs`. [`InvBombs`](src/Variables.inc#L558)
- `InvMaxBombs` (`$67C`): capacity; 8 in a newly registered file and after the switch to Q2. [`InvMaxBombs`](src/Variables.inc#L607) [`ModeERegister_Update`](src/bank2/register-eliminate-update.asm#L42) [`Profile_SwitchToSecondQuest`](src/bank2/second-quest-bank-stub.asm#L45)
- A bomb's `ObjState` is `$11`-`$14`; a fire in the same slots is `$21`/`$22`; 0 = empty slot. [`Bomb_UpdateBombOrFire`](src/bank7/bomb.asm#L71)

**Use**
- B newly pressed while Link's state is 0 calls the handler; it needs `InvBombs` > 0. [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68) [`Bomb_Wield`](src/bank1/bomb-candle-wield.asm#L63)
- Placing costs 1 `InvBombs`, plays tune `$20`, and starts the bomb in state `$11` with timer 0. [`Bomb_Wield@Activate`](src/bank1/bomb-candle-wield.asm#L99)
- The bomb is put `$10` pixels from Link's position in the direction he faces; Link's state becomes `$10` (item use) with his animation counter at 1. [`Link_PlaceWeaponAhead`](src/bank1/bomb-candle-wield.asm#L154) [`Link_UseItem_SetAnimCounter1`](src/bank1/bomb-candle-wield.asm#L131)
- Weapon slots update after Link in the same frame, so state `$11` ends on the frame of placement. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129) [`Bomb_Update`](src/bank7/bomb.asm#L89)
- States: `$12` fuse `$30` (48) frames, `$13` blast `$18` (24) frames, `$14` fading clouds `$0C` (12) frames, then the slot is freed: `$54` (84) frames in all. [`Bomb_StateTimes`](src/bank7/bomb.asm#L55) [`Bomb_Update`](src/bank7/bomb.asm#L89)
- The explosion sound is requested once, on entering state `$13`; the screen flashes grayscale at timer `$16` and `$11` (off at `$12` and `$0D`) in state `$13` only. [`Bomb_Update`](src/bank7/bomb.asm#L89) [`Bomb_UpdateFlashEffect`](src/bank1/bomb-candle-wield.asm#L328)

**Limits**
- Slot choice: `$10` if it is empty or holds a fire, else `$11` if it is empty or holds a fire, else refused (both slots hold bombs). [`Bomb_Wield`](src/bank1/bomb-candle-wield.asm#L63)
- Then the other slot must be empty or in a state >= `$13` (a bomb already blasting or fading, or any fire); a bomb in state `$11`/`$12` there refuses the new bomb. [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87)
- Result: at most two bombs exist, and at most one of them unexploded; the second can be placed `$30` frames after the first, when its blast begins. [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87) [`Bomb_StateTimes`](src/bank7/bomb.asm#L55)
- A bomb placed in a slot that holds a fire overwrites the fire; with a fire in `$10` and nothing in `$11` the bomb still takes `$10`. [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87)
- With an unexploded bomb in one slot and a fire in the other, no bomb can be placed until the first blasts. [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87)
- A refused bomb costs nothing. [`Bomb_Wield`](src/bank1/bomb-candle-wield.asm#L63)

**Effect**
- Only state `$13` (the 24 blast frames) hurts monsters: damage type 8, damage `$40` (four wooden sword hits). [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126)
- Blast area: the bomb's middle (X+8, Y+8) less than `$18` pixels from the monster's middle on both axes. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126)
- A hit monster is knocked back unless its invincibility mask holds type 8; an immune hit plays no parry tune. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Damage_PlayParryTuneUnlessBombOrFire`](src/bank1/weapon-damage-death.asm#L199)
- A Darknut facing exactly against the bomb's stored direction (the direction Link faced when placing it) takes no damage. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- The bomb routines hold no test against Link: nothing there harms him. [`Bomb_Update`](src/bank7/bomb.asm#L89)
- Overworld rock wall (tile object `$63`): opens while a bomb in slot `$10` or `$11` is in state `$13` with midpoints less than `$10` pixels apart on both axes, and only in the quest the secret belongs to. [`RockWall_Update`](src/bank4/rock-wall-tree-secrets.asm#L214) [`Secret_IsForOtherQuest`](src/bank4/rock-wall-tree-secrets.asm#L331)
- Dungeon bomb wall: tested once per bomb, on the frame the state changes `$13` -> `$14` (the end of the blast), never during the fuse or blast. [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136)
- The wall test needs a dungeon (not a cellar, mode 9), the bomb's `ObjX` and `ObjY` each less than `$18` pixels from a hotspot, that door not already open, no door command running, and door type 4 (bombable) on that side. [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136)
- Hotspots (bomb X, Y): up wall `$78,$5D`; down `$78,$BD`; left `$20,$8D`; right `$D0,$8D`; searched right, left, down, up and only the first in range counts. [`Bomb_WallHotspotsX`](src/bank7/bomb.asm#L60) [`Bomb_WallHotspotsY`](src/bank7/bomb.asm#L64) [`Bomb_TryOpenWall@LoopHotspot`](src/bank7/bomb.asm#L154)
- A passing test starts door command 6 (open) for that side. [`Bomb_TryOpenWall`](src/bank7/bomb.asm#L136) [`TriggeredDoorCmd`](src/Variables.inc#L85)

**Pickup**
- Class 1 (amount), value 4: every bomb item taken, dropped or bought, adds 4, capped at `InvMaxBombs`. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)
- A bomb taken outside mode 5 (a cave) is not held up over Link's head, though the lift timer and item song still start. [`Item_Take`](src/bank1/take-item.asm#L259)
- More-bombs offer: the dungeon person of type `$4F` in levels 1, 2, 5, 7 sells capacity; Link must stand at X = `$78` exactly, within 5 pixels of Y = `$98`, with `InvRupees` >= 100. [`Person_ComplexState2SellBombCapacity`](src/bank1/underworld-person-update.asm#L171)
- The purchase posts 100 rupees to subtract, adds 4 to `InvMaxBombs`, refills `InvBombs` to the new maximum and sets the room's item-taken flag, so the man is gone on later visits. [`Person_ComplexState2SellBombCapacity`](src/bank1/underworld-person-update.asm#L171) [`Person_InitUnderworldLevels1257`](src/bank1/underworld-person-init.asm#L77)
- Q1 has two such rooms: level 5 room `$17` and level 7 room `$48` (capacity 8 -> 12 -> 16); the code itself sets no cap. [`Level_BlockUW1to6Q1`](src/bank6/level-blocks-common-data.asm#L70) [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)

**Notes**
- A kill counts toward the forced bomb drop only when the kill that brings `HelpDropCount` to exactly `$0A` is made with a bomb. [`Damage_KillMonster`](src/bank1/weapon-damage-death.asm#L216)
- Dodongo reads only the bomb in slot `$10`, for eating (fuse state `$12`) and for the stun (states `$13`, `$14`); a bomb in slot `$11` does nothing to it. [`Dodongo_CheckBombHit`](src/bank4/dodongo.asm#L400) [`Dodongo_TryEatBomb`](src/bank4/dodongo.asm#L469)

# Arrows and Bow (wooden, silver)

The arrow is an object in slot `$12`, which it shares with the rod. The bow enables the shot; the arrow item only sets the damage.

**Ids**
- Item ids: `$08` wooden arrow, `$09` silver arrow (item slot `$02`), `$0A` bow (item slot `$03`). [`Item_IdToSlot`](src/bank1/take-item.asm#L81)
- Selected-slot index 2 shoots; index 3 (bow) does nothing. [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103) [`Link_WieldNothing`](src/bank5/wield-items.asm#L237)

**RAM**
- `InvArrow` (`$659`): 0 none, 1 wooden, 2 silver. [`InvArrow`](src/Variables.inc#L559)
- `InvBow` (`$65A`): 0 or 1. [`InvBow`](src/Variables.inc#L560)
- Arrow `ObjState` in slot `$12`: `$10` flying, `$20`/`$28` spark; `$31`-`$35` in that slot is a rod swing. [`Rod_UpdateRodOrArrow`](src/bank7/sword-and-rod.asm#L57)

**Use**
- B with slot 2 selected: slot `$12` gets state `$10`, q-speed `$C0` (3 pixels per frame), Link's direction, `$10` pixels ahead of him and 3 pixels right when he faces up or down. [`Arrow_Wield`](src/bank5/wield-items.asm#L187) [`Link_WieldWeapon`](src/bank5/wield-items.asm#L219)
- Link's state becomes `$10` with animation counter 1, as for other item uses. [`Link_UseItem_SetAnimCounter1`](src/bank1/bomb-candle-wield.asm#L131)
- The arrow flies straight with no range limit until blocked at a room bound or stopped by the collision code. [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192)
- Stopped arrows show a spark for 3 frames; on the next frame slot `$12` is freed. [`Boomerang_UpdateOrArrow_State2Spark`](src/bank7/boomerang-arrow-food.asm#L352) [`Boomerang_HandleBlockedOrArrow`](src/bank7/boomerang-arrow-food.asm#L404)

**Limits**
- Refused without the bow (`InvBow` = 0). [`Arrow_Wield`](src/bank5/wield-items.asm#L187)
- Refused with `InvRupees` = 0; each shot posts 1 rupee to `InvRupeesToSubtract`. [`Arrow_Wield`](src/bank5/wield-items.asm#L187) [`InvRupeesToSubtract`](src/Variables.inc#L610)
- Refused while slot `$12` is busy: one arrow at a time (flight or spark), and none during a rod swing. [`Arrow_Wield`](src/bank5/wield-items.asm#L187) [`Rod_UpdateRodOrArrow`](src/bank7/sword-and-rod.asm#L57)
- `InvArrow` is not tested at the shot; the submenu is what keeps the arrow slot from being selected without the bow. [`Arrow_Wield`](src/bank5/wield-items.asm#L187) [`Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow`](src/bank5/submenu-items.asm#L344)

**Effect**
- Damage type 4; damage `$20` when `InvArrow` = 1 (wooden), `$40` for any other value (silver). [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249)
- Hit box: less than `$0B` pixels on both axes; only a flying arrow (state below `$20`) hits, a spark does not. [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249)
- What stops it: any monster it overlaps, immune or not, turns it into a spark; so it never passes through a monster, except a Pols Voice. [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- What stops it: the room bounds, widened for weapon slots (left X < bound - `$0B`, right X >= bound + `$0C`, up Y < bound - `$0F`, down Y >= bound + `$12`); tiles are not tested. [`Obj_MoveShot`](src/bank1/shot-aim-boomerang-speed.asm#L49) [`Room_BoundDirHorizontally`](src/bank1/room-bounds.asm#L116) [`Room_BoundDirVertically`](src/bank1/room-bounds.asm#L185)
- Pols Voice (`$16`): an overlapping arrow sets its HP to 0 and kills it past the invincibility mask, and the arrow keeps flying. [`Collision_CheckShotVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L287)
- A hit monster is knocked back in the arrow's direction; a Darknut facing exactly against it parries. [`Damage_ParryOrShove`](src/bank1/sword-bomb-arrow-collision.asm#L318)
- Gohma routes every weapon hit to its own handler (the arrow and the eye). [`Gohma_HandleWeaponCollision`](src/bank4/gohma.asm#L367)
- Ganon's arrow test runs only when `InvArrow` = 2 and an arrow is in state `$10`. [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360)

**Pickup**
- Arrows are class 2 (grade): wooden stores 1, silver 2, and a lower grade never replaces a higher one. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_TakeClass2Grade`](src/bank1/take-item.asm#L536)
- The bow is class 0: `InvBow` = 1. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- An active arrow (flying or spark) picks up a dropped item it touches, with Link's pickup range applied at the arrow's position. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297) [`Drop_ItemTakerObjSlots`](src/bank4/drop-items.asm#L280)
- The arrow's palette row is 4 for wooden, 5 for silver (`InvArrow` - 1 added to the attribute). [`Arrow_SetAttrAndDraw`](src/bank7/boomerang-arrow-food.asm#L308)

# Boomerang (wooden, magic)

The boomerang is an object in slot `$0F`, which it shares with the food. It does no damage; it stuns, and it fetches dropped items.

**Ids**
- Item ids `$1D` wooden, `$1E` magic (item slots `$1D`, `$1E`); selected-slot index 0 for either. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)

**RAM**
- `InvBoomerang` (`$674`) and `InvMagicBoomerang` (`$675`): 0 or 1 each; the magic one decides the range whenever it is held. [`InvBoomerang`](src/Variables.inc#L602) [`InvMagicBoomerang`](src/Variables.inc#L603)
- `ObjState` in slot `$0F`, high nibble: `$1x` flying out, `$2x` spark, `$3x` slowing, `$4x` slow return, `$5x` fast return; low 3 bits = spin step; bit 7 set = food. [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192) [`Boomerang_UpdateOrFood`](src/bank7/boomerang-arrow-food.asm#L70)
- `ObjMovingLimit` for slot `$0F`: range in state `$1x`, then a countdown in states `$3x` and `$4x`. [`ObjMovingLimit`](src/Variables.inc#L308)

**Use**
- B with slot 0 selected: state `$10`, q-speed `$C0` (3 pixels per frame), `$10` pixels ahead of Link; Link's state `$10`, his animation counter 1. [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)
- Direction = the held direction bits if any (two bits for a diagonal throw), else Link's facing. [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)
- A diagonal throw moves 3 pixels per frame on both axes; only the horizontal part counts toward the range. [`Boomerang_UpdateOrArrow`](src/bank7/boomerang-arrow-food.asm#L192) [`Obj_MoveShot`](src/bank1/shot-aim-boomerang-speed.asm#L49)
- Wooden: range `$31` pixels, reached after 17 frames (51 pixels); then state `$3x`: `$10` frames at q-speed `$40` (1 pixel per frame) in the same direction. [`Boomerang_RangeLimits`](src/bank5/wield-items.asm#L85) [`Boomerang_UpdateSlowAndReturn`](src/bank7/boomerang-return-draw.asm#L52)
- Then state `$4x`: `$10` frames toward Link at half the return speed; then `$5x` at full speed (2 pixels per frame along an axis, 1.5 + 1.5 on the diagonal) until caught. [`Boomerang_UpdateSlowAndReturn@MoveTowardThrower`](src/bank7/boomerang-return-draw.asm#L166) [`Boomerang_SetReturnSpeed`](src/bank1/shot-aim-boomerang-speed.asm#L186) [`Boomerang_ReturnQSpeedsX`](src/bank7/boomerang-arrow-food.asm#L147)
- Magic: limit `$FF` is never reached (the compared distance is at most `$80`), so it flies at 3 pixels per frame until a room bound or a monster stops it, and never passes through states `$3x`/`$4x`. [`Boomerang_RangeLimits`](src/bank5/wield-items.asm#L85) [`Boomerang_UpdateOrArrow@CompareLimit`](src/bank7/boomerang-arrow-food.asm#L263)
- Blocked at a room bound (the widened weapon bounds; tiles are not tested): spark for 3 frames, then state `$50`. [`Boomerang_HandleBlockedOrArrow`](src/bank7/boomerang-arrow-food.asm#L404) [`Boomerang_UpdateOrArrow_State2Spark`](src/bank7/boomerang-arrow-food.asm#L352)
- Caught when both X and Y are within 8 pixels of Link's: slot `$0F` is freed, Link's state is ORed with `$20` (catch pose) and his animation counter set to 1. [`Boomerang_UpdateSlowAndReturn@ReturnToThrower`](src/bank7/boomerang-return-draw.asm#L96)
- The catch pose runs `$20` -> `$21` -> `$31` -> idle; A and B are ignored until Link's state is 0 again. [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223) [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)

**Limits**
- Refused when neither boomerang is held, or while a boomerang is out (slot state nonzero with bit 7 clear): one at a time. [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)
- Food lying in slot `$0F` (state bit 7 set) does not refuse the throw: the boomerang overwrites it and the food on the ground is gone (`InvFood` is untouched). [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)
- No cost. [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)

**Effect**
- Hit test: the point (X+4, Y+8) of the boomerang less than `$0A` pixels from the monster's middle on both axes; damage type 2. [`Boomerang_CheckMonsterCollision`](src/bank1/weapon-damage-death.asm#L48)
- Any overlap, immune monster or not, sends the boomerang straight to state `$50` (fast return, no spark), from any state. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- Not immune: damage 0 and `ObjStunTimer` = `$10`, counted in 10-frame units (151 to 160 frames); wooden and magic stun alike. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82) [`ObjStunTimer`](src/Variables.inc#L52)
- A monster whose HP is 0 dies from the hit, since a result of 0 HP is a death. [`Damage_DealToMonster`](src/bank1/weapon-damage-death.asm#L181)
- Immune (mask bit 2): parry tune, no stun. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)
- A Zol or Vire hit by the boomerang does not take its direction (other weapons turn them). [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- No knockback: the boomerang path does not reach the shove code. [`Collision_CheckWeaponPointVsMonster`](src/bank1/weapon-damage-death.asm#L82)

**Pickup**
- Both are class 0: the slot byte becomes 1; taking the wooden one after the magic one changes nothing in use. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)

**Notes**
- Fetching: while the boomerang is active (any state, out or returning) a dropped item within pickup range of the boomerang's position (both axis distances < 9) is taken at once. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297) [`Item_TryTake`](src/bank1/take-item.asm#L206)
- Only dropped items (type `$60`) are fetched; the room item is tested against Link alone, and nothing is taken while Link is halted or during the drop's first `$10` lifetime steps (about 32 frames). [`Item_TryTakeRoomItem`](src/bank1/take-item.asm#L173) [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)
- Caught during the throw pose (Link's state `$10`/`$11`), the OR makes `$30`/`$31`, the last step of an item use. [`Boomerang_UpdateSlowAndReturn@ReturnToThrower`](src/bank7/boomerang-return-draw.asm#L96)
- A boomerang moving left in state `$3x` with X < 2 goes to state `$40` at once. [`Boomerang_UpdateSlowAndReturn`](src/bank7/boomerang-return-draw.asm#L52)
- Sound: effect 2 when the spin advances (every 2 frames), at most once per `$0A` frames. [`Boomerang_AnimateAndCheckCollision`](src/bank7/boomerang-return-draw.asm#L199) [`Boomerang_PlaySfx`](src/bank1/modeb-cave-save-sprites.asm#L223)

# Candle (blue, red)

The candle makes a fire in object slot `$10` or `$11` (shared with bombs). The same fire object is what the magic book leaves where a magic shot ends.

**Ids**
- Item ids `$06` blue, `$07` red (item slot `$04`); selected-slot index 4. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)

**RAM**
- `InvCandle` (`$65B`): 0 none, 1 blue, 2 red. [`InvCandle`](src/Variables.inc#L562)
- `HasUsedCandle` (`$513`): set by every candle fire; zeroed with the rest of `$0300`-`$051F` on every room entry (also caves and cellars). [`HasUsedCandle`](src/Variables.inc#L428) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)
- Fire `ObjState`: `$21` moving, `$22` standing. [`Fire_Update`](src/bank7/fire.asm#L44)
- `CandleState` (`$51F`): dark room lighting, 0 not lit, 1 brightening, 2 lit; `IsRoomBrightening` (`$51E`) is 1 during the fade. [`CandleState`](src/Variables.inc#L445) [`IsRoomBrightening`](src/Variables.inc#L444)

**Use**
- B with slot 4 selected: a fire in state `$21` appears `$10` pixels ahead of Link with his direction and q-speed `$20` (0.5 pixel per frame); Link's state becomes `$10`; flame sound. [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220) [`Link_UseItem_SetState10`](src/bank1/bomb-candle-wield.asm#L143)
- The fire moves until it has gone `$10` pixels (31-32 frames), then stands (state `$22`) for `$3F` (63) frames and frees its slot: about 95 frames in all. [`Fire_Update`](src/bank7/fire.asm#L44)
- The moving fire is not stopped by tiles or room bounds: its grid offset is held at 0 for the move and nothing else is tested. [`Fire_Update`](src/bank7/fire.asm#L44)

**Limits**
- Needs an empty slot: `$10`, else `$11`; a fire never replaces a bomb or another fire. [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220)
- Blue (`InvCandle` = 1): refused while `HasUsedCandle` is set, so one fire per room entry. [`Candle_Wield@NotInUse`](src/bank1/bomb-candle-wield.asm#L233)
- Red (any other value): the flag is not tested; the only limit is the two slots, so two fires at once, or one beside a bomb. [`Candle_Wield@NotInUse`](src/bank1/bomb-candle-wield.asm#L233)
- No cost. A bomb placed later may overwrite a fire. [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220) [`Bomb_Wield@FoundSlot`](src/bank1/bomb-candle-wield.asm#L87)
- The book's fire goes through the same routine but saves and restores `HasUsedCandle` and Link's state: it neither needs nor spends the blue candle's use. [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)

**Effect**
- Monsters: damage type `$20`, damage `$10` (one wooden sword hit), in both fire states; the fire's middle less than `$0E` pixels from the monster's middle on both axes. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126)
- A hit monster is knocked back unless its mask holds type `$20`; an immune hit plays no parry tune. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Damage_PlayParryTuneUnlessBombOrFire`](src/bank1/weapon-damage-death.asm#L199)
- Link: his own fire, moving or standing, hurts him when the midpoints are less than `$0E` pixels apart on both axes and his invincibility timer is 0. [`Fire_Update@DrawAndCheckCollisions`](src/bank7/fire.asm#L92)
- Damage to Link `$0080` (half a heart with no ring; halved per ring grade) plus knockback. [`Fire_Update@DrawAndCheckCollisions`](src/bank7/fire.asm#L92) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Burning a tree (tile object `$64`): only a standing fire with `ObjTimer` < 2 (its last frames), midpoints less than `$10` pixels apart on both axes; the tree becomes stairs and room flag `$80` is set. [`Tree_Update`](src/bank4/rock-wall-tree-secrets.asm#L249) [`Secret_RevealAndFlagStairsObj`](src/bank4/rock-wall-tree-secrets.asm#L283)
- So a tree's stairs appear about 94 frames after the candle is used, and the fire must have come to rest on the tree. [`Tree_Update`](src/bank4/rock-wall-tree-secrets.asm#L249) [`Fire_Update`](src/bank7/fire.asm#L44)
- A tree does nothing in the quest its secret does not belong to. [`Secret_IsForOtherQuest`](src/bank4/rock-wall-tree-secrets.asm#L331)
- Lighting: only a standing fire, only in a dungeon (`CurLevel` <> 0), calls the room light routine each frame. [`Fire_Update@StandingFire`](src/bank7/fire.asm#L75)
- In a dark room (bit 7 of the room's attribute byte E) the first call starts the fade to light; the fade has 4 palette steps 10 frames apart. [`Candle_State0Begin`](src/bank4/shoot-helpers-candle.asm#L147) [`Candle_State1Brightening`](src/bank4/shoot-helpers-candle.asm#L173) [`Room_IsDark_Bank4`](src/bank4/shoot-helpers-candle.asm#L184)
- While the fade runs, the play update does nothing else: Link, monsters and weapons are frozen. [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59)
- A lit room stays lit until the room changes; `CandleState` is reset then, so the room is dark again on the next visit. [`Candle_UpdateRoomLight`](src/bank4/shoot-helpers-candle.asm#L132) [`CandleState`](src/Variables.inc#L445)

**Pickup**
- Class 2 (grade): blue stores 1, red 2; taking the blue candle while holding the red one changes nothing. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_TakeClass2Grade`](src/bank1/take-item.asm#L536)

**Notes**
- The handler does not test that a candle is held; the submenu selection is what guards it. [`Candle_Wield`](src/bank1/bomb-candle-wield.asm#L220)
- Candle fire and book fire are not told apart by trees or by the room light. [`Tree_Update`](src/bank4/rock-wall-tree-secrets.asm#L249) [`Fire_Update`](src/bank7/fire.asm#L44)
- The candle entry sets Link's state to `$10` without setting his animation counter, unlike the bomb, arrow and food entries. [`Link_UseItem_SetState10`](src/bank1/bomb-candle-wield.asm#L143)

# Recorder (flute, whistle)

The source calls it the flute. Using it always plays the tune and freezes play; what follows depends on dungeon or overworld, room and quest.

**Ids**
- Item id `$05`, item slot `$05`; selected-slot index 5. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)

**RAM**
- `InvFlute` (`$65C`): 0 or 1. [`InvFlute`](src/Variables.inc#L563)
- `FluteTimer` (`$3C`): frames left of the freeze. [`FluteTimer`](src/Variables.inc#L49)
- `FluteUsedState` (`$51B`, dungeons): 0, 1 once played in this room, 2 while a Digdogger shrinks. [`FluteUsedState`](src/Variables.inc#L439)
- `IsWhirlwindSummoned` (`$508`), `WhirlwindTeleportingState` (`$522`: 0 none, 1 carrying Link away, 2 dropping him off), `TeleportingLevelIndex` (`$523`: value AND 7 = level - 1). [`IsWhirlwindSummoned`](src/Variables.inc#L422) [`WhirlwindTeleportingState`](src/Variables.inc#L456) [`TeleportingLevelIndex`](src/Variables.inc#L457)
- `SecretColorCycle` (`$51A`): flute secret progress, 0 none, 1-`$0B` running, `$0C` done. [`SecretColorCycle`](src/Variables.inc#L435)

**Use**
- Every use requests the tune (`Tune1Request` = `$10`) and sets `FluteTimer` = `$98` (152 frames); no object slot is used and Link's state stays 0. [`Flute_Wield`](src/bank7/flute.asm#L49)
- While `FluteTimer` is nonzero the whole play update is skipped: no input, no pause, no movement of Link, monsters or weapons. [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59)
- Objects the flute creates (whirlwind, flute secret) are made on the frame of use but first update when the freeze ends. [`Flute_Wield`](src/bank7/flute.asm#L49) [`FluteSecret_Init`](src/bank7/flute-secret-pond.asm#L63)

**Limits**
- No cost and no slot; it can be played again as soon as the freeze ends. [`Flute_Wield`](src/bank7/flute.asm#L49)
- Overworld outside mode 5 (a cave, mode `$B`; the shortcut stairs room, mode `$C`): tune and freeze only. [`Flute_Wield`](src/bank7/flute.asm#L49)
- One whirlwind per room entry (`IsWhirlwindSummoned`, cleared on room entry) and none while one carries Link. [`Whirlwind_Summon@MakeWhirlwind`](src/bank1/whirlwind.asm#L291)
- One flute secret per room entry: refused when `SecretColorCycle` <> 0. [`Flute_Wield@RevealSecret`](src/bank7/flute.asm#L95)

**Effect**

In a dungeon:
- `FluteUsedState` goes 0 -> 1; a use while it is nonzero changes nothing more. [`Flute_Wield@FlagFluteUsed`](src/bank7/flute.asm#L127)
- Digdogger: after the freeze a big Digdogger starts a `$40`-frame shrink (`FluteUsedState` 2), then splits into `ChildDigdoggerCount` small ones (3 or 1) and the big one is removed. [`Digdogger_HandleFlute`](src/bank4/digdogger.asm#L91) [`ChildDigdoggerCount`](src/Variables.inc#L420)
- The split does not start or progress while Link holds the clock or the Digdogger is stunned. [`Digdogger_Update`](src/bank4/digdogger.asm#L202)
- The byte is zeroed on room entry, so the flute must be played in the Digdogger's room. [`FluteUsedState`](src/Variables.inc#L439) [`Room_EnterCommon`](src/bank5/enter-room.asm#L143)

On the overworld, secrets:
- Level 7 entrance: in room `$42` in Q1 the flute makes the flute secret object instead of a whirlwind. [`Flute_SecretRoomIdsOW`](src/bank7/flute.asm#L36)
- Q2 secret rooms: `$06`, `$29`, `$2B`, `$30`, `$3A`, `$3C`, `$58`, `$60`, `$6E`, `$72`; in Q1 these rooms summon the whirlwind, and in Q2 room `$42` does. [`Flute_SecretRoomIdsOW`](src/bank7/flute.asm#L36) [`Flute_Wield@Found`](src/bank7/flute.asm#L82)
- The secret object (type `$5E`) needs an empty object slot in `$01`-`$09` (the highest is used); with none, only the tune and freeze happen. [`Flute_Wield@FindEmptySlot`](src/bank7/flute.asm#L103)
- The object steps a colour cycle once every 8 frames; step `$0A` raises the walkable-tile limit to `$99`, and the 11th step puts stairs at X = `$60`, Y = `$90` (the same spot in every such room) and sets room flag `$80`: 81 to 88 frames after the freeze ends. [`FluteSecret_Update`](src/bank7/flute-secret-pond.asm#L101) [`FluteSecret_CuePondPaletteRow`](src/bank7/flute-secret-pond.asm#L127) [`FluteSecret_RevealPondStairs`](src/bank7/flute-secret-pond.asm#L157)
- Leaving such a room by scrolling waits while the cycle runs backwards to 0. [`FluteSecret_ReversePondCycle`](src/bank7/flute-secret-pond.asm#L172) [`Mode7Scroll_Init_Sub0`](src/bank5/mode7-scroll-init.asm#L157)

On the overworld, the whirlwind:
- Everywhere else on the overworld in mode 5 the flute calls the whirlwind summon. [`Flute_Wield@SummonWhirlwind`](src/bank7/flute.asm#L120)
- Summon conditions: at least one Triforce piece (`InvTriforce` <> 0), none already summoned in this room, none carrying Link, and a free object slot in `$01`-`$0B`; the room id is not tested. [`Whirlwind_Summon`](src/bank1/whirlwind.asm#L243) [`Whirlwind_Summon@MakeWhirlwind`](src/bank1/whirlwind.asm#L291)
- Destination: each use steps `TeleportingLevelIndex` by +1 when Link faces right or up, by -1 when he faces left or down. [`Whirlwind_StepLevelIndex`](src/bank1/whirlwind.asm#L314)
- It then keeps stepping the same way, wrapping within levels 1-8, past levels whose piece Link lacks (bit level - 1 of `InvTriforce`). [`Whirlwind_Summon@TestLevelPiece`](src/bank1/whirlwind.asm#L258) [`InvTriforce`](src/Variables.inc#L597)
- The index is kept between rooms and steps on every overworld use that reaches the summon, even when no whirlwind is made. [`Whirlwind_Summon`](src/bank1/whirlwind.asm#L243)
- The whirlwind (type `$2E`) is created at X = 0 on Link's Y, plays a spawn cloud if its slot was unused, then moves right 2 pixels per frame and is destroyed at X >= `$F0` (its 120th update). [`Whirlwind_SetUp`](src/bank1/whirlwind.asm#L66) [`Whirlwind_Update_Full`](src/bank1/whirlwind.asm#L99)
- Pickup is the ordinary monster touch test, which reports nothing while Link's invincibility timer or stun timer is nonzero, he holds the clock, he is halted (state `$40`) or a Like-Like holds him; a missed whirlwind leaves at the right edge and the room allows no second one. [`Whirlwind_Update_Full@CheckCollision`](src/bank1/whirlwind.asm#L145) [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- On pickup Link faces right, is halted (state `$40`) and hidden, and rides to the right edge; then the game leaves the room (mode 6) and scrolls right. [`Whirlwind_Update_Full@CheckCollision`](src/bank1/whirlwind.asm#L145) [`Whirlwind_Update_Full@CheckRightEdge`](src/bank1/whirlwind.asm#L188)
- Landing rooms for levels 1-8: `$37`, `$3C`, `$74`, `$45`, `$0B`, `$22`, `$42`, `$6D` (the table holds the room west of each), the same in both quests. [`Whirlwind_PrevRoomIds`](src/bank1/whirlwind.asm#L81)
- In the landing room the whirlwind takes object slot `$09` at X = 0 and Link is released at X = `$80`, Y = `$8D`, `$AD`, `$8D`, `$8D`, `$AD`, `$8D`, `$AD`, `$5D` for levels 1-8. [`Whirlwind_InitDropOffAndBeginUpdate`](src/bank1/whirlwind.asm#L360) [`Whirlwind_DropOffYs`](src/bank1/whirlwind.asm#L348)
- Drop-off timing: 1 frame of init, 9 + 6 + 6 frames of spawn cloud, then 64 updates to X = `$80`: 86 frames after play starts in the room. [`Whirlwind_InitDropOffAndBeginUpdate`](src/bank1/whirlwind.asm#L360) [`Whirlwind_Update_Full`](src/bank1/whirlwind.asm#L99)

**Pickup**
- Class 0: `InvFlute` = 1. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- The whirlwind's pickup deals 0 damage but runs the harm path: Link's invincibility timer becomes `$18`, and `WorldKillCount`, `HelpDropCount` and `IsHelpDropBomb` are zeroed. [`Whirlwind_Update_Full@CheckCollision`](src/bank1/whirlwind.asm#L145) [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- The big Digdogger cannot be hit by any weapon; only the small ones made by the flute can. [`Obj_TypeAttributes`](src/bank7/object-update-table.asm#L70) [`Digdogger_HandleFlute`](src/bank4/digdogger.asm#L91)

# Food (bait, meat)

The food is an object lying in slot `$0F` (shared with the boomerang). It redirects the monsters' chase target and is what the Grumble Goriya wants.

**Ids**
- Item id `$04`, item slot `$06`; selected-slot index 6. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)

**RAM**
- `InvFood` (`$65D`): 0 or 1. [`InvFood`](src/Variables.inc#L564)
- Food `ObjState` in slot `$0F`: `$80`, `$81`, `$82` (bit 7 set marks food). [`Boomerang_UpdateOrFood`](src/bank7/boomerang-arrow-food.asm#L70)
- `ChaseTargetX`/`ChaseTargetY` (`$61`/`$62`): the point chasing monsters steer toward. [`ChaseTargetX`](src/Variables.inc#L144) [`ChaseTargetY`](src/Variables.inc#L149)

**Use**
- B with slot 6 selected: slot `$0F` gets state `$80` and timer `$FF`, `$10` pixels ahead of Link; Link's state `$10`, his animation counter 1. [`Food_Wield`](src/bank5/wield-items.asm#L247) [`Link_UseItem_SetWeaponState`](src/bank1/bomb-candle-wield.asm#L119)
- Lifetime: three timer periods of `$FF` frames (states `$80`, `$81`, `$82`), 765 frames, then the slot is freed. [`Boomerang_UpdateOrFood`](src/bank7/boomerang-arrow-food.asm#L70)

**Limits**
- Refused while slot `$0F` is in use: a boomerang is out, or food is already down. [`Food_Wield`](src/bank5/wield-items.asm#L247)
- Placing it does not test or change `InvFood`: the food is not consumed and can be put down again once the slot is free. [`Food_Wield`](src/bank5/wield-items.asm#L247)
- Throwing the boomerang removes food lying on the ground. [`Boomerang_Wield`](src/bank5/wield-items.asm#L124)

**Effect**
- Attraction is decided by one byte, the type of the first monster the room was populated with (`RoomObjTemplateType`), not by each monster. [`Boomerang_UpdateOrFood@CheckMonsterType`](src/bank7/boomerang-arrow-food.asm#L93) [`RoomObjTemplateType`](src/Variables.inc#L305)
- Attracting types: `$03`-`$0A` (Moblins, Goriyas, Octoroks), `$12` Vire, `$1B` blue Keese, `$1C` red Keese. [`Boomerang_UpdateOrFood@CheckMonsterType`](src/bank7/boomerang-arrow-food.asm#L93)
- How: every frame the food writes its own position to `ChaseTargetX`/`ChaseTargetY`, after the play update has copied Link's position there, so every monster in the room that steers by the chase target heads for the food. [`Boomerang_UpdateOrFood@AttractMonsters`](src/bank7/boomerang-arrow-food.asm#L107) [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- With any other first type (Lynel `$01`/`$02`, black Keese `$1D`, Darknut, ...) the food changes nothing. [`Boomerang_UpdateOrFood@CheckMonsterType`](src/bank7/boomerang-arrow-food.asm#L93)
- Food never collides with monsters: no damage, no stun. [`Boomerang_CheckMonsterCollision`](src/bank1/weapon-damage-death.asm#L48)
- Food does not pick up dropped items (slot states with bit 7 set are skipped). [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)

**Pickup**
- Class 0: `InvFood` = 1. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- Grumble Goriya (type `$36`): once its text is typed it tests only that food is active in slot `$0F`; where the food or Link is does not matter. [`Grumble_State1WaitForFood`](src/bank1/grumble-goriya.asm#L73)
- On seeing the food: Link is halted, the secret tune plays, the room's item-taken flag is set at once and a `$40`-frame timer starts. [`Grumble_State1WaitForFood`](src/bank1/grumble-goriya.asm#L73) [`Person_FlagTakenAndBlankText`](src/bank1/grumble-goriya.asm#L92)
- When the timer ends: the food object is removed, `InvFood` = 0 (the only place the food is used up), Link is released and the Goriya's slot is freed. [`Grumble_State3TakeFood`](src/bank1/grumble-goriya.asm#L109)
- While the Goriya stands in slot 1, Link cannot move up when his Y < `$8E`. [`Person_BlockLinkMovingUp`](src/bank1/modeb-cave-save-sprites.asm#L289)
- With the flag set the Goriya is not created on later visits. [`Grumble_Init_Full`](src/bank1/underworld-person-init.asm#L216)
- Q1 has one Grumble room: level 7 room `$28`. [`Level_BlockUW7to9Q1`](src/bank6/level-blocks-common-data.asm#L74)

# Potion (blue, red) and Letter

The potion is a dose counter, not two items; the letter only unlocks the medicine shops. Shop wares and
prices are listed per cave in `data/levelblock-ow.md` ("Caves").

**Ids**
- Blue potion: item id `$1F`, class 1, value 1 (adds one dose). [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- Red potion: item id `$20`, class 2, grade 2 (sets two doses). [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- Both use item slot 7, which is also the B-button selection index 7. [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)
- Letter: item id `$15`, class 0, item slot `$0F`; selection index `$0F`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`SelectedItemSlot`](src/Variables.inc#L553)

**RAM**
- `$065E` holds the doses: 0 none, 1 (drawn blue), 2 (drawn red). [`InvPotion`](src/Variables.inc#L565)
- `$0666` holds the letter: 0 none, 1 held, 2 shown to a medicine shop. [`InvLetter`](src/Variables.inc#L573)

**Use**
- B with slot 7 selected drinks one dose: `InvPotion` - 1, `World_IsFillingHearts` = 1, `PauseState` = 2. [`Potion_Wield`](src/bank5/wield-items.asm#L269)
- No object slot is used and Link's state stays 0. [`Potion_Wield`](src/bank5/wield-items.asm#L269)
- While `PauseState` is 2 the play update runs only the heart refill and the rupee count; Select cannot end this pause. [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59)
- Refill rate: the partial heart gains 6 per frame; at `$F8` or more the next frame completes the heart. An empty heart takes 43 frames. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- The refill ends when full hearts equal containers - 1 and the partial heart is set to `$FF`; it then clears `PauseState`, `World_IsFillingHearts` and `IsSwordBlocked`. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- Every refill frame requests square-0 tune `$10` (the tick). [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- B with the letter selected does nothing outside a medicine shop: slot `$0F` returns before the dispatch. [`Link_WieldItem`](src/bank5/wield-items.asm#L95)
- In a medicine shop (person type `$74`), B pressed with `SelectedItemSlot` = `$0F` sets `InvLetter` to 2, selects slot 7 and starts the shop in the same frame. [`Cave_UpdatePerson`](src/bank1/cave-person-prices.asm#L66)

**Limits**
- The only refusal is `InvPotion` = 0. Health is not tested: at full health a dose is still spent and the refill ends after one frame. [`Potion_Wield`](src/bank5/wield-items.asm#L269), [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- Doses are capped at 2. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)
- Until `InvLetter` = 2 a medicine shop person stands there with no text and no wares, and Link is not halted. [`Cave_UpdatePerson`](src/bank1/cave-person-prices.asm#L66)
- The shop is paid like any other: `InvRupees` must be at least the price, and nothing can be taken while a debit is still counting down. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- One ware per visit: taking one starts a `$40`-frame timer after which the person and the other wares are removed. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55), [`Cave_State4DelayThenHide`](src/bank1/cave-shop.asm#L221)

**Effect**
- Hearts refill completely, whatever the dose colour; a red potion is simply two uses. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343), [`Potion_Wield`](src/bank5/wield-items.asm#L269)
- A finished refill also lifts a Bubble's sword block (`IsSwordBlocked` = 0). [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)

**Pickup**
- Blue potion: doses + 1, capped at 2. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)
- Red potion: doses = 2 unless the byte already holds more (it cannot). [`Item_TakeClass2Grade`](src/bank1/take-item.asm#L536)
- Medicine shop (cave `$1A`, person `$74`): blue potion 40 rupees, red potion 68 rupees. [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)
- The "take any one" cave (cave `$11`, person `$6B`) offers a free red potion or a heart container. [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)
- The letter is the free ware of cave `$18` (person `$72`), a once-only person: gone when the room's flag bit `$10` is set. [`Cave_Init`](src/bank1/cave-init-person-text.asm#L101)
- The letter is a plain class 0 item: `InvLetter` = 1. [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- When the potion slot is selected and empties, the selection moves to the letter slot `$0F` if Link has the letter. [`StatusBar_CheckMissingItem`](src/bank7/status-bar-lift-item.asm#L146)
- When the letter is selected and `InvPotion` becomes nonzero, the selection moves to slot 7. [`StatusBar_DrawItemsAndEnsureSelection`](src/bank7/status-bar-lift-item.asm#L64)
- The submenu draws potion and letter at the same spot and skips the letter while a potion is held. [`Submenu_DrawItems`](src/bank5/submenu-items.asm#L68)
- A potion bought in a shop is held up like every cave item: Link is halted for `$7F` frames. [`Item_Take`](src/bank1/take-item.asm#L259), [`Link_CheckLiftItem`](src/bank7/status-bar-lift-item.asm#L162)

# Magic Rod and Book

**Ids**
- Rod: item id `$10`, item slot 8, selected-slot index 8. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)
- Book: item id `$11`, item slot `$0A`; it cannot be selected and is never "used". [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`Inv_StepItemSlotInSearchDir`](src/bank5/submenu-items.asm#L404)

**RAM**
- `InvRod` (`$65F`) 0 or 1; `InvBook` (`$661`) 0 or 1; `SelectedItemSlot` (`$656`) = 8 with the rod on B. [`InvRod`](src/Variables.inc#L566) [`InvBook`](src/Variables.inc#L568) [`SelectedItemSlot`](src/Variables.inc#L553)
- Rod swing: slot `$12`, state at `$BE` = `$31`-`$35`. Magic shot: slot `$0E`, state at `$BA` = `$80`. Book fire: slot `$10` or `$11`, state `$22`. [`Rod_Wield`](src/bank5/wield-items.asm#L294) [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)

**Use**
- B newly pressed with slot 8 selected and Link's state 0: slot `$12` gets state `$31` and timer 5, Link's state becomes `$10`. No cost. [`Rod_Wield`](src/bank5/wield-items.asm#L294) [`Link_WieldWeapon`](src/bank5/wield-items.asm#L219)
- The wield routine does not test `InvRod`; the selection code only selects slot 8 when that byte is not 0. [`Rod_Wield`](src/bank5/wield-items.asm#L294) [`Inv_FindAndSelectOccupiedSlot`](src/bank5/submenu-items.asm#L321)
- The swing is the sword's state machine with the same timing: `$31` on frames 0-3, `$32` on 4-11, `$33` on 12, `$34` on 13, `$35` on 14, slot free on 15; Link's state as for the sword (trace). [`Rod_UpdateRodOrArrow`](src/bank7/sword-and-rod.asm#L57) [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)
- On frame 12 (entering `$33`) the magic shot is made in slot `$0E`: state `$80`, `$10` pixels ahead of Link, q-speed `$A0` (2.5 pixels a frame), with its own tune. No health test. [`Sword_UpdateSwordOrRod@MakeMagicShot`](src/bank7/sword-and-rod.asm#L258) [`SwordShot_SetUpWithState`](src/bank7/sword-and-rod.asm#L272)
- Facing left or right with the shot's X < `$14` or >= `$EC`, the shot is dropped at once. [`SwordShot_SetUpWithState`](src/bank7/sword-and-rod.asm#L272)
- The rod is drawn in states `$32`-`$34` with palette row 5. [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100)

**Limits**
- Refused while slot `$12` is busy: a rod swing, or an arrow flying or sparking. [`Rod_Wield`](src/bank5/wield-items.asm#L294)
- One magic shot at a time: a swing whose frame 12 comes while a magic shot is still flying makes none. [`Sword_UpdateSwordOrRod@MakeMagicShot`](src/bank7/sword-and-rod.asm#L258)
- A new B press is discarded in the border bands near a screen edge. [`Link_FilterInput`](src/bank5/border-input-mask.asm#L54)
- Sword and rod, what the code enforces: they use different slots (`$0D`, `$12`) and neither wield routine tests the other slot; the only gate is Link's state, which both set to `$10`. [`Sword_Wield`](src/bank5/wield-items.asm#L59) [`Rod_Wield`](src/bank5/wield-items.asm#L294) [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- A and B pressed on the same frame start both swings; they run in step and both can hit on frames 4-11 (trace). [`Link_HandleInput`](src/bank5/link-handle-input.asm#L68)
- In that case the sword's beam is made first on frame 12 (slot `$0D` is updated before `$12`) and the rod's magic shot overwrites it in the same frame: slot `$0E` ends frame 12 as `$80` even at full health (trace). [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129) [`Sword_UpdateSwordOrRod@MakeMagicShot`](src/bank7/sword-and-rod.asm#L258)
- A first, then B: B is accepted from frame 14, while the sword is still in states 4-5; a beam made on frame 12 that is still flying or spreading is replaced by the magic shot 12 frames after the B press (trace: B on frame 14, slot `$0E` `$10` -> `$80` on frame 26). [`Sword_UpdateSwordOrRod@MakeMagicShot`](src/bank7/sword-and-rod.asm#L258)
- In that sequence the sword's last two state changes set Link's animation counter to 1, so his state is 0 again on frame 16 with the rod still in `$31`; A on frame 17 starts a new swing, and sword state 2 then overlaps rod states `$32`-`$33`, both hitting (trace). [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100) [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223)
- The magic shot and the sword beam can never coexist: one slot. [`SwordShot_UpdateOrMagicShot`](src/bank7/sword-shot-magic-shot.asm#L56)

**Effect**
- Rod swing (melee): damage type 1, the sword's type, damage `$20` (as the white sword), with the sword's hit test. [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249) [`Collision_CheckStabVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L215)
- Unlike the sword it is tested in every state `$31`-`$35`, frames 0-14 of the swing. [`Arrow_CheckArrowOrRodVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L249)
- In `$31` (frames 0-3) the rod is not drawn but already hits where it is placed: over Link's head, at Link's X, Y plus (-1, -9) facing up, (+1, -14) down, (0, -11) left, (-8, -11) right. [`Sword_LinkToWeaponOffsetsX`](src/bank7/sword-and-rod.asm#L68) [`Sword_LinkToWeaponOffsetsY`](src/bank7/sword-and-rod.asm#L74)
- Monsters immune to sword damage (mask bit 1) are immune to the swing; the Darknut's frontal block applies. [`Damage_HandleWeaponHit`](src/bank1/weapon-damage-death.asm#L127)
- Magic shot: damage type `$10`, damage `$20`, hit test less than `$0C` pixels on both axes. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)
- The shot ends on the first monster it overlaps, immune or not, or at a room bound; tiles do not stop it. [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64) [`SwordShot_UpdateOrMagicShot`](src/bank7/sword-shot-magic-shot.asm#L56)
- Book: when a magic shot ends and `InvBook` is not 0, a standing fire (state `$22`, timer `$4F` = 79 frames; a candle's standing fire gets `$3F`) is made at the shot's position in fire slot `$10`, else `$11`, if one is empty (trace: `$22` on frames 65-143 for a shot stopped on frame 65). [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)
- The book's fire does not use the candle's once-per-room flag: `HasUsedCandle` is saved, cleared for the call and restored, so the fire is neither refused after the blue candle was used nor counted as its use; no candle is needed (trace: flag still 0 afterwards). [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172) [`HasUsedCandle`](src/Variables.inc#L428)
- Link's state is saved and restored around that call: the book's fire does not put him in the item-use state. [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)
- The fire is an ordinary standing fire: damage type `$20`, damage `$10`, less than `$0E` pixels; it lights a dark dungeon room; it burns Link for half a heart. [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126) [`Fire_Update`](src/bank7/fire.asm#L44)
- Shot plus fire on one monster (trace, monster held in place): `$20` from the shot on the hit frame, then `$10` from the fire each time the monster's invincibility timer runs out (31 and 63 frames later) until the fire ends. [`Shove_Begin`](src/bank1/shove-knockback.asm#L65) [`Bomb_CheckBombOrFireVsMonster`](src/bank1/sword-bomb-arrow-collision.asm#L126)

**Pickup**
- Both are class 0 items: the inventory byte is set to 1. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_Take`](src/bank1/take-item.asm#L259)
- Both are room items of dungeon treasure cellars (rod id `$10`, book id `$11` in the level block listings under `data/`), so they are held up on pickup. [`Item_TryTakeRoomItem`](src/bank1/take-item.asm#L173) [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- The fire does not hit the monster on the frame the shot does: placing the fire overwrites [02], the monster's middle X used by the rest of that monster's weapon tests, with `$F0` (trace: a bomb blast overlapping the same monster on that frame added its `$40` without the book and nothing with it). [`Link_PlaceWeaponAhead`](src/bank1/bomb-candle-wield.asm#L154) [`SwordShot_CheckMonsterCollision`](src/bank1/sword-bomb-arrow-collision.asm#L64)
- If both fire slots are busy and slot `$11` holds a moving candle fire (`$21`), that fire is turned into the book's standing fire at the shot's position. [`SwordShot_HandleBlocked`](src/bank7/sword-shot-magic-shot.asm#L172)
- The rod's position picks up dropped items during the swing, as the sword's does. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297)

# Raft

The raft works only at the two docks. There is no water-tile test: a ride starts from Link's exact
position.

**Ids**
- Item id `$0C`, class 0, item slot 9; it is not a B-button item. [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`Link_WieldItem_JumpTable`](src/bank5/wield-items.asm#L103)

**RAM**
- `$0660`: 0 or 1. [`InvRaft`](src/Variables.inc#L567)

**Use**
- Overworld rooms `$3F` and `$55` get a dock object (type `$61`) in object slot `$0B` on entry. [`Room_SetUpTileObjOW`](src/bank5/enter-room.asm#L429)
- The dock's X is `$80` in room `$55` and `$60` otherwise (room `$3F`). [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- Ride up starts when Link's X equals the dock X exactly and his Y is exactly `$7D` (dock state 2). [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- Ride down starts when Link's X equals the dock X and his Y is exactly `$3D`, the top screen edge (dock state 1). [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- No button is needed; the "secret found" tune plays when the ride starts. [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- During the ride Link's state is `$40` (halted: no input, no movement of his own) and he faces up (ride up) or down (ride down). [`Dock_Update`](src/bank4/dock-raft.asm#L39), [`Dock_RaftLinkDirs`](src/bank4/dock-raft.asm#L28)
- Speed: Link's Y and the raft's Y change by 1 pixel per frame, written directly (no q-speed). [`Dock_Update@HandleOtherStates`](src/bank4/dock-raft.asm#L94)
- Ride up: 64 frames from Y `$7D` to `$3D`, then the room is left through the top with Link facing up. [`Dock_Update@HandleOtherStates`](src/bank4/dock-raft.asm#L94), [`Mode5Play_GoToNextMode`](src/bank7/walker-collision-screen-edge.asm#L281)
- Ride down: 66 frames from Y `$3D` to `$7F`; Link's grid offset is set to 2 and his state to 0. [`Dock_Update@State1`](src/bank4/dock-raft.asm#L117)

**Limits**
- Without the raft the dock update returns at once. [`Dock_Update`](src/bank4/dock-raft.asm#L39)
- The ride down ends at `$7F`, not `$7D`, so it does not restart a ride up. [`Dock_Update@State1`](src/bank4/dock-raft.asm#L117)

**Effect**
- The raft sprite (item slot 9's image) is drawn 6 pixels below Link, only during a ride. [`Dock_Update@Draw`](src/bank4/dock-raft.asm#L139)
- A halted Link (state exactly `$40`) cannot be touched by monsters or shots, so the whole ride is safe. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Pickup**
- Plain class 0 item: `InvRaft` = 1. [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- The start test runs every frame the dock is idle, so walking onto the exact spot is enough; one pixel off in X never starts a ride. [`Dock_Update`](src/bank4/dock-raft.asm#L39)

# Ladder

Link lays the ladder automatically when he walks into water; it is an object in a monster slot.

**Ids**
- Item id `$0D`, class 0, item slot `$0C`; not a B-button item. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)
- The laid ladder is object type `$5F`, which has no update routine of its own. [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246), [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)

**RAM**
- `$0663`: 0 or 1. [`InvLadder`](src/Variables.inc#L570)
- `LadderSlot` = object slot of the laid ladder, 0 = none. [`LadderSlot`](src/Variables.inc#L104)

**Use**
- Rooms: every dungeon room; in the overworld only rooms `$17`, `$18`, `$19`, `$27`, `$4F`, `$5F`. [`Ladder_RoomIdsOW`](src/bank7/link-animate-ladder.asm#L76), [`Link_EndMoveAndAnimate@CheckLadderRoom`](src/bank7/link-animate-ladder.asm#L190)
- Tiles: the tile ahead of Link in his facing direction must be `$F4` in a dungeon, or `$8D`-`$98` in the overworld. [`Link_EndMoveAndAnimate@CheckWaterOW`](src/bank7/link-animate-ladder.asm#L240)
- Overworld tiles `$8D` and `$91` never match: the tile lookup returns `$26` for walkable tiles, and both are in the walkable list. [`Tile_WalkableTilesOW`](src/bank7/tile-collision.asm#L43), [`Link_EndMoveAndAnimate@CheckWaterOW`](src/bank7/link-animate-ladder.asm#L240)
- Other conditions, all required and tested after Link's move each frame: game mode 5; grid offset 0; not in a doorway; Link not halted; no ladder out. [`Link_EndMoveAndAnimate@CheckLadder`](src/bank7/link-animate-ladder.asm#L209)
- A direction must be held and equal Link's facing direction (`ObjInputDir` = `ObjDir`). [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- The ladder takes the highest free monster slot of `$0B`..`$01`, state 1, facing as Link. [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246), [`Obj_FindEmptyMonsterSlot`](src/bank7/object-stubs-destroy-monster.asm#L169)
- Position relative to Link: up (0, -5); down (0, +`$13`); left (-`$10`, +3); right (+`$10`, +3). [`Ladder_LinkOffsetsX`](src/bank7/link-animate-ladder.asm#L58), [`Ladder_LinkOffsetsY`](src/bank7/link-animate-ladder.asm#L62)
- Distance to the ladder = |Link Y + 3 - ladder Y| for a vertical ladder, |Link X - ladder X| for a horizontal one. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- State 2 is set on every frame Link is less than `$10` pixels from it (he is on it). [`Ladder_CheckAndDraw@SetState2`](src/bank5/ladder.asm#L120)
- On the ladder with a direction held Link may move forward in the ladder's direction or back the way he came; any other direction is stopped. [`Ladder_CheckAndDraw@HandleInput`](src/bank5/ladder.asm#L125)
- Stashed (slot freed, `LadderSlot` = 0) when: Link leaves its line (X differs for a vertical ladder, Y + 3 for a horizontal one); the distance exceeds `$10`; or the distance is exactly `$10` and Link faces another way or the state is already 2 (he crossed). [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)

**Limits**
- It spans one square: 16 pixels of water. After crossing, the ladder is stashed and can be laid again at the next water tile. [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110)
- No free monster slot: no ladder. [`Link_EndMoveAndAnimate@SetUpLadder`](src/bank7/link-animate-ladder.asm#L246)
- While the ladder is out, the screen-edge room change is not tested. [`Walker_AcceptMovingDir`](src/bank7/walker-collision-screen-edge.asm#L200)
- The ladder has no timer; it stays as long as Link keeps to its line within `$10` pixels. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- Every room change clears `LadderSlot` together with the weapon slots. [`Obj_ResetWeaponSlotsAndLadder`](src/bank5/mode3-mode6-init.asm#L263)

**Pickup**
- Plain class 0 item: `InvLadder` = 1. [`Item_Take`](src/bank1/take-item.asm#L259)

**Notes**
- On the frame the ladder is stashed Link does not move: the move that follows acts on the freed ladder slot. [`Ladder_CheckAndDraw@StashLadder`](src/bank5/ladder.asm#L110), [`Walker_Move`](src/bank7/walker-move.asm#L64)
- The ladder can move Link during a sword swing or item use, because it sets the moving direction after the code that zeroed it. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44), [`Walker_Move`](src/bank7/walker-move.asm#L64)
- A ladder laid upward with exactly Up held repeats the tile test 8 pixels higher and lets Link go on only onto a walkable tile. [`Ladder_CheckAndDraw`](src/bank5/ladder.asm#L44)
- Overworld room `$5F`, a ladder room, holds the overworld's only room-item heart container at X `$C0`, Y `$90`. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51), [`Ladder_RoomIdsOW`](src/bank7/link-animate-ladder.asm#L76)

# Keys and Magic Key

**Ids**
- Key: item id `$19`, class 1, value 1, item slot `$17`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- Magic key: item id `$0B`, class 0, item slot `$0D`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)

**RAM**
- `$066E`: key count 0-`$FF`, one counter for all levels. [`InvKeys`](src/Variables.inc#L584)
- `$0664`: magic key, 0 or 1. [`InvMagicKey`](src/Variables.inc#L571)

**Use**
- Keys act when Link pushes into a dungeon doorway whose door type is 5 or 6 (locked door). [`Door_Touch_JumpTable`](src/bank5/touch-door.asm#L62), [`Door_TouchKey`](src/bank5/touch-door.asm#L160)
- Order of tests: door already open: pass; a door command running: blocked while Link's timer is nonzero; magic key: open, nothing spent; no keys: wall; else `InvKeys` - 1 and open. [`Door_TouchKey`](src/bank5/touch-door.asm#L160)
- Opening sets the door's open command and Link's `ObjTimer` to `$20`; Link stays blocked until the door's bit is in `CurOpenedDoors` or the timer runs out. [`Door_TouchKey`](src/bank5/touch-door.asm#L160), [`Door_TriggerOpen`](src/bank5/shutters-door-flags.asm#L117)
- The open command saves the door's opened flag in this room and in the next one, so a door is paid for once. [`Door_Update`](src/bank5/update-doors.asm#L60)

**Limits**
- Only door types 5 and 6 take a key; walls (1), false walls (2, 3), bombable walls (4) and shutters (7) never do. [`Door_Touch_JumpTable`](src/bank5/touch-door.asm#L62)
- Only one door command exists (`TriggeredDoorCmd`), so a key cannot unlock a door while another door is opening or closing. [`Door_TouchKey`](src/bank5/touch-door.asm#L160)
- The magic key never decrements `InvKeys`: the magic key test comes before the key count is read. [`Door_TouchKey`](src/bank5/touch-door.asm#L160)

**Effect**
- With the magic key the status bar shows "XA" in place of the key count and `InvKeys` is not displayed. [`StatusBar_FormatText`](src/bank1/status-bar-rupees.asm#L148)
- Without it the count is drawn as "X" and digits (three digits without the "X" from 100 up). [`Text_FormatDecimalCountByte`](src/bank1/status-bar-rupees.asm#L230)

**Pickup**
- A key adds 1; the sum is capped at `$FF`. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337)
- A key cancels the item tune and plays tune 0 number 8 instead. [`Sfx_PlayKeyTakenTune`](src/bank1/take-item.asm#L569)
- Magic key: plain class 0 item, `InvMagicKey` = 1. [`Item_Take`](src/bank1/take-item.asm#L259)
- Buying: a key costs 100 rupees in shop cave `$1E` and 80 in shop cave `$20` (see `data/levelblock-ow.md`). [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145), [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- A shop sells one ware per visit, so one key per cave entry. [`Cave_State4DelayThenHide`](src/bank1/cave-shop.asm#L221)

**Notes**
- Keyless pass: while any door command runs and Link's `ObjTimer` is 0, a locked door lets him through unopened and unpaid. It needs a command that did not set his timer: a shutter opening or a bombed wall opening. [`Door_TouchKey`](src/bank5/touch-door.asm#L160), [`Door_Touch_BlockWhileTimer`](src/bank5/touch-door.asm#L193)
- A close command sets Link's timer to `$30` and a key unlock to `$20`, which is what blocks the pass in those cases. [`Door_TouchKey`](src/bank5/touch-door.asm#L160), [`Door_Update`](src/bank5/update-doors.asm#L60)

# Power Bracelet

**Ids**
- Item id `$14`, class 0, item slot `$0E`; not a B-button item. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)

**RAM**
- `$0665`: 0 or 1. [`InvBracelet`](src/Variables.inc#L572)

**Use**
- A rock tile object (type `$62`) waits for a push only when `InvBracelet` is nonzero; a gravestone (`$65`) needs no item. [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62)
- Push condition: Link's X equals the rock's X, |Link Y + 3 - rock Y| < `$11`, and the vertical bits of his input are exactly up (Link below) or down (Link above). [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62), [`Rock_PushDirs`](src/bank4/rock-wall-tree-secrets.asm#L51)
- Rocks are never pushed sideways. [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62)
- The rock slides `$10` pixels in 32 frames (q-speed `$20`), then stairs (tile `$70`) appear at the room's shortcut position, not where the rock stood. [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62)
- While it slides, Link cannot move if he is within `$10` pixels of it on both axes. [`Link_CheckSlidingTileObjBlocking`](src/bank1/tile-block-power-triforce.asm#L45)

**Limits**
- A room's secret can belong to one quest only (bits 7-6 of room attribute F); in the other quest its rock does nothing. [`Secret_IsForOtherQuest`](src/bank4/rock-wall-tree-secrets.asm#L331), [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62)
- Rock rooms in the overworld data: `$1D`, `$23`, `$49`, `$79` (both quests, all lead to the shortcut cave) and `$09`, `$11`, `$1B` (flagged second quest only); see `data/levelblock-ow.md`. [`Secret_IsForOtherQuest`](src/bank4/rock-wall-tree-secrets.asm#L331)

**Effect**
- A finished push sets room flag `$20` for the room where it began; that flag makes the stairs reappear every time the room is laid out. [`Rock_Update`](src/bank4/rock-wall-tree-secrets.asm#L62), [`Secret_RestoreStairsOW`](src/bank5/ow-shortcut-map-square.asm#L54)
- Shortcut stairs: the stairs in rooms `$1D`, `$23`, `$49`, `$79` lead to the shortcut cave (cave `$14`, mode `$C`, "take any road"). [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43)
- In that cave the three stairs are at Y `$9D`, X `$50`, `$80`, `$B0`; stairs n (1-3) lead to the room n entries after the current room in the 4-entry shortcut room array, wrapping. [`Warp_CheckSubroomExit`](src/bank5/cave-cellar-exits.asm#L43), [`LevelInfo_CellarRoomIdArray`](src/Variables.inc#L726)
- Arriving by shortcut stairs marks the destination room visited (flag `$20`), so its stairs are shown there too. [`ModeALeaveCellar_GoFromCellar`](src/bank5/cave-cellar-exits.asm#L173), [`Secret_RestoreStairsOW`](src/bank5/ow-shortcut-map-square.asm#L54)

**Pickup**
- The bracelet is the secret under one Armos in overworld room `$24` (the statue at X `$E0`, Y `$80`): when that Armos finishes waking (its fade-in timer, `$3F` from the touch, reaches 0) the room item becomes item `$14` at its position. [`Armos_InitOrFlyingGhiniFadeIn`](src/bank4/ghini-armos.asm#L154), [`Armos_SecretRoomIds`](src/bank4/ghini-armos.asm#L138), [`Armos_SecretXs`](src/bank4/ghini-armos.asm#L143)
- Once the room's item-taken flag `$10` is set the bracelet is neither drawn nor takeable. [`Item_TryTakeRoomItem`](src/bank1/take-item.asm#L173)
- Plain class 0 item: `InvBracelet` = 1. [`Item_Take`](src/bank1/take-item.asm#L259)

# Ring (blue, red)

**Ids**
- Blue ring: item id `$12`, class 2, grade 1. Red ring: item id `$13`, class 2, grade 2. Item slot `$0B`. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111), [`Item_IdToSlot`](src/bank1/take-item.asm#L81)

**RAM**
- `$0662`: 0 none, 1 blue, 2 red. [`InvRing`](src/Variables.inc#L569)

**Effect**
- Damage to Link is a 16-bit value: [0D] whole hearts, [0E] 256ths of a heart. It is shifted right once per ring grade (`LSR` [0D], `ROR` [0E]). [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Blue ring: damage / 2. Red ring: damage / 4. The shift carries from the heart byte into the fraction and drops bits below 1/256. [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193)
- Examples: a 1-heart touch (`$01:$00`) becomes `$00:$80` (blue) or `$00:$40` (red); a half-heart touch (`$00:$80`) becomes `$00:$40` or `$00:$20`. [`Link_BeHarmed`](src/bank1/link-collision-harm.asm#L193), [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- Every harm path that goes through `Link_BeHarmed` is divided: monster and shot touches and Link's own candle fire (`$00:$80`). [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164), [`Fire_Update`](src/bank7/fire.asm#L44)
- The life-or-money man's heart container charge is not damage and is not reduced. [`Person_LifeOrMoneyState2Choose`](src/bank1/life-or-money.asm#L121)
- Tunic colour (NES palette value): `$29` no ring, `$32` blue ring, `$16` red ring. [`Link_TunicColorsByRing`](src/bank1/take-item.asm#L164)

**Pickup**
- The grade is stored only if it is not lower than the one held: a blue ring never replaces a red one. [`Item_TakeClass2Grade`](src/bank1/take-item.asm#L536)
- When the grade is stored, the tunic colour is patched at once into the save slot's menu palette row and the level palette, and the palette transfer is cued. [`Item_TakeClass2Grade`](src/bank1/take-item.asm#L536), [`Palette_PatchLinkColorAndCueLevelPalettes`](src/bank7/init-mode-dispatch.asm#L319)
- The blue ring is sold for 250 rupees in shop cave `$20` (see `data/levelblock-ow.md`). [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)

**Notes**
- The ring does not change knockback or the invincibility time after a hit; only the amount subtracted. [`Link_HarmByObjType`](src/bank1/link-collision-harm.asm#L164)
- A hit whose damage the ring reduces to 0 still zeroes the kill counters behind the fairy and help drops. [`Link_BeHarmed@ResetHelp`](src/bank1/link-collision-harm.asm#L210)

# Shield (small, magic)

The shield is passive: it is one test inside the routine that decides whether a monster's shot touching Link harms him.

**Ids**
- Magic shield: item id `$1C`, item slot `$1F`. The small shield has no item id and no inventory byte: it is what Link has while `InvMagicShield` is 0. [`Item_IdToSlot`](src/bank1/take-item.asm#L81) [`InvMagicShield`](src/Variables.inc#L604)

**RAM**
- `InvMagicShield` (`$676`): 0 or 1. [`InvMagicShield`](src/Variables.inc#L604)
- `IsShotCollidingWithLink` (`$34B`): set to 1 when a shot overlaps Link, blocked or not. [`IsShotCollidingWithLink`](src/Variables.inc#L293)

**Use**
- Only shots are ever blocked: object types `$53` and up, in a state `$10`-`$1F`, whose middle is less than 9 pixels from Link's middle on both axes. A monster (type below `$53`) that touches Link harms him at once. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Condition 1, Link's state: `$00`-`$0F`. Any swing, item use or boomerang catch (`$10`, `$11`, `$20`, `$21`, `$31`) disables the shield. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Condition 2, facing: Link's direction and the shot's direction together must contain both up and down, or both left and right; a diagonal shot qualifies if either of its components opposes Link's facing. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Condition 3, type: `$53` (flying rock), `$54`, `$5B` (monster arrow) and `$5C` (monster boomerang) are blocked by either shield. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- `$55` (fireball), `$57` (monster sword shot), `$58` and `$59` (monster magic shots) are blocked only with the magic shield. [`Link_CheckMonsterTouch_NoReset@FacingOpposite`](src/bank1/link-collision-harm.asm#L138)
- `$56` (the fireball of Gohma, Manhandla, Gleeok and Ganon) and `$5A` are never blocked. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73) [`Fireball_Shoot`](src/bank4/monster-shots-fireballs.asm#L58)
- Neither movement nor input is tested: Link may walk or stand, as long as he faces the shot. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Limits**
- During a sword or rod swing the shield is off on frames 0-12 (Link's state `$11`, then `$31`); during a bomb, arrow, boomerang or food use on frames 0-11; during a boomerang catch for 13 frames. [`Sword_UpdateSwordOrRod`](src/bank7/sword-and-rod.asm#L100) [`Link_AdvanceItemUseState`](src/bank7/object-animation.asm#L223)
- A shot from the side or from behind always harms. [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Nothing is tested at all (no harm, no block) while Link's invincibility timer, the clock, Link's stun timer or the shot's stun timer is not 0, while Link's state is exactly `$40`, or while a Like-Like holds him. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48) [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)

**Effect**
- A block plays the parry tune and cancels the collision: no damage, no knockback, no invincibility timer. [`Link_CheckMonsterTouch_NoReset@Parry`](src/bank1/link-collision-harm.asm#L151)
- Damage avoided, before rings: `$53`, `$54`, `$55`, `$5B` half a heart; `$5C` 1 heart; `$57`, `$58` 2 hearts; `$59` 4 hearts. Never blockable: `$56` 1 heart, `$5A` 4 hearts. [`Damage_ObjTypeDamageToLink`](src/bank1/monster-collisions.asm#L200)
- A blocked straight shot goes to state `$30` and drifts 2 pixels a frame the way Link faces for `$10` frames, harmless, then is destroyed. [`MonsterShot_CheckLinkCollision`](src/bank4/monster-shots-fireballs.asm#L249) [`MonsterShot_Bounce`](src/bank4/monster-shots-fireballs.asm#L209)

**Pickup**
- Class 0 item: `InvMagicShield` = 1. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111) [`Item_Take`](src/bank1/take-item.asm#L259)
- Sold in overworld caves for 160, 130 or 90 rupees depending on the shop (persons `$78`, `$77`, `$79` in `data/levelblock-ow.md`); buying needs `InvRupees` >= the price. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)

**Like-Like**
- A Like-Like (`$17`) whose touch harms Link captures him; its capture timer is 1 on that frame and counts 1 per frame. [`Collision_CheckMonsterVsWeaponsAndLink`](src/bank1/monster-collisions.asm#L101) [`LikeLike_Update`](src/bank4/like-like.asm#L46)
- When the timer reaches `$60`, 95 frames after the capture, `InvMagicShield` is set to 0, and again on every later frame of the capture; the small shield remains. [`LikeLike_Update@IncCaptureTime`](src/bank4/like-like.asm#L111)
- Killing the Like-Like before that keeps the shield: while held, Link's direction input is discarded but A and B still work. [`Link_Update`](src/bank7/mode5-play-update.asm#L330) [`LikeLike_Update`](src/bank4/like-like.asm#L46)
- The Like-Like's code never restores the shield, also not when it dies or lets Link go. [`LikeLike_Update@IncCaptureTime`](src/bank4/like-like.asm#L111)

**Notes**
- The drawn shield is cosmetic and separate from the test: without the magic shield a shield tile shows only on the facing-down walk frames; with it, on the sideways and facing-down walk frames; never facing up or in the item-use pose. Facing up still blocks. [`Link_EndMoveAndAnimate@HandleMagicShield`](src/bank7/link-animate-ladder.asm#L352) [`Link_ShieldHalfTiles`](src/bank7/link-animate-ladder.asm#L68) [`Link_MagicShieldHalfTiles`](src/bank7/link-animate-ladder.asm#L72)

# Hearts, Heart Containers and Fairies

Link's health is two bytes. One heart is `$0100` in damage terms: one step of the full-heart nibble, or
256 steps of the partial byte.

**Ids**
- Heart: item id `$22`, class 1, value 0 (adds 1 heart), item slot `$19`. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- Fairy: item id `$23`, class 1, value 2 (adds 3 hearts), item slot `$14`. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- Heart container: item id `$1A`, class 1, item slot `$18`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)

**RAM**
- `$066F`: high nibble = heart containers - 1, low nibble = full hearts. [`HeartValues`](src/Variables.inc#L593)
- `$0670`: fill of the heart after the full ones: 0 empty, `$01`-`$7F` drawn as half, `$80`-`$FF` drawn as full. [`HeartPartial`](src/Variables.inc#L595), [`Text_FormatHeartRows`](src/bank1/format-decimal-hearts.asm#L81)
- Full health is low nibble = high nibble with `HeartPartial` = `$FF`: the last heart is always the partial one. [`Item_TakeHeartsNoSound`](src/bank1/take-item.asm#L475), [`Inv_CompareHeartsToContainers`](src/bank1/take-item.asm#L507)
- A new file starts with `HeartValues` = `$22`, `HeartPartial` = `$FF`: 3 containers, all full. [`ModeERegister_Update`](src/bank2/register-eliminate-update.asm#L42)
- `$066B` (the fairy's slot) is never written: a fairy only heals. [`InvFairy`](src/Variables.inc#L578)

**Effect**
- Heart: one full heart is added while full hearts < containers - 1; `HeartPartial` is left as it is, so health rises by exactly one heart. [`Item_TakeHeartsNoSound@AddWholeHeart`](src/bank1/take-item.asm#L490)
- When full hearts already equal containers - 1, `HeartPartial` is set to `$FF` and any remaining hearts are discarded. [`Item_TakeHeartsNoSound@FillHeartPartial`](src/bank1/take-item.asm#L497)
- Fairy: the same loop three times (up to 3 hearts), without the heart tune. [`Item_Take_CheckClass1`](src/bank1/take-item.asm#L337), [`Item_TakeHeartsNoSound`](src/bank1/take-item.asm#L475)
- Heart container: `HeartValues` + `$11`, one more container and one more full heart; the other hearts are not refilled. [`Item_Take_CheckClass1@TakeHeartContainer`](src/bank1/take-item.asm#L392)
- Damage subtracts the fraction from `HeartPartial` and whole hearts from the low nibble; a borrow turns one full heart into a partial of `$FF`, costing 1/256 heart extra. [`Link_BeHarmed@BorrowHeart`](src/bank1/link-collision-harm.asm#L238)
- Link dies when the whole-heart damage exceeds his full hearts, or a borrow is needed with no full heart left: `GameMode` = `$11`. [`Link_BeHarmed@HandleDied`](src/bank1/link-collision-harm.asm#L254)
- Full refills (potion, pond fairy, end of a level) all run the same tick: +6 to `HeartPartial` per frame, 43 frames per empty heart. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)
- Pond fairy (object type `$2F`): Link at Y `$AD` with X in `$70`-`$80` is halted and refilled; he is released `$50` frames after the hearts are full, and the fairy does not rearm in that visit. [`PondFairy_Update`](src/bank4/pond-fairy.asm#L43)

**Limits**
- Container cap: 16. With `HeartValues` >= `$F0` a heart container does nothing. [`Item_Take_CheckClass1@TakeHeartContainer`](src/bank1/take-item.asm#L392)
- After the continue menu (continue, save or retry) health is 3 full hearts whatever the container count. [`Mode8Continue_Update_Full`](src/bank5/mode8-continue-mode10-stairs.asm#L68)
- With no full heart (low nibble 0) the low-health beep (tune 0 bit `$40`) is requested on every frame the object loop runs. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)

**Pickup**
- Dropped hearts and fairies come from the drop table; a fairy is forced when `WorldKillCount` is `$10` as a death ends. [`Drop_ItemTable`](src/bank4/drop-items.asm#L132), [`Drop_SetUpItem`](src/bank4/drop-items.asm#L157)
- A dropped fairy flies: 0.5 px/frame for `$20` frames, then 0.625, wandering with random turns. [`Fairy_Init`](src/bank4/fairy-flyer-common.asm#L82)
- Dropped items can be taken by Link or by the position of his arrow or rod (slot `$12`), sword (`$0D`) or boomerang (`$0F`), from about 32 frames after they appear until about 510 frames. [`Drop_UpdateItem`](src/bank4/drop-items.asm#L297), [`Drop_ItemTakerObjSlots`](src/bank4/drop-items.asm#L280)
- Pickup range: |Link X - item X| < 9 and |Link Y + 3 - item Y| < 9. [`Item_TryTake`](src/bank1/take-item.asm#L206)
- A heart costs 10 rupees in shop cave `$1F`. [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)
- Heart containers: dungeon room items; the free choice in "take any one" caves (cave `$11`, against a red potion); and the room item of overworld room `$5F` at X `$C0`, Y `$90`. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51), [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145)

**Notes**
- Sword caves test containers only: `HeartValues` >= `$40` (5 containers) for person `$6C`, >= `$B0` (12) for the other; current health cannot change the result. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- Life-or-money man, "life" choice with 4 or more containers: one container and one full heart removed. With 3 or fewer: containers kept, full hearts and `HeartPartial` set to 0, and no death is started by that routine. [`Person_LifeOrMoneyState2Choose`](src/bank1/life-or-money.asm#L121)
- Any refill that runs to the end clears `IsSwordBlocked`. [`World_FillHearts`](src/bank5/cave-cellar-enter.asm#L343)

# Rupees

**Ids**
- Rupee: item id `$18`, class 1, value 1, item slot `$16`. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)
- 5 rupees: item id `$0F`, class 1, value 5, item slot `$1C`. [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)

**RAM**
- `$066D`: rupees, 0-`$FF`. [`InvRupees`](src/Variables.inc#L583)
- `$067D`: rupees still to be credited. `$067E`: rupees still to be taken. [`InvRupeesToAdd`](src/Variables.inc#L608), [`InvRupeesToSubtract`](src/Variables.inc#L610)
- `$0673`, the 5-rupee item's slot, is never written. [`InvFiveRupees`](src/Variables.inc#L600)

**Effect**
- `InvRupees` never jumps. On each even frame it gains 1 while `InvRupeesToAdd` is nonzero and loses 1 while `InvRupeesToSubtract` is nonzero: 30 rupees a second each way. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95)
- Both counters can step in the same frame. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95)
- Cap: when `InvRupees` = `$FF` the pending credit is zeroed, so rupees stop at 255 and the rest of a gain is lost. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95)
- Floor: when `InvRupees` = 0 the pending debit is zeroed; a debt is never carried. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95)
- The count runs in the play update of modes 5, 9, `$A`, `$B`, `$C` (also while paused or in the submenu) and in mode `$12` submode 2. [`World_UpdateHeartsAndRupees`](src/bank7/mode5-play-update.asm#L311), [`Mode12EndLevel_Update_Sub2FillHearts`](src/bank5/mode12-lay-out-room.asm#L129)
- It does not run while the flute tune plays or a dark room is brightening, and it skips a frame when a PPU transfer is already queued. [`Mode5Play_Update`](src/bank7/mode5-play-update.asm#L59), [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95)
- Each counted rupee requests tune 0 bit `$10` and the status bar text is rewritten. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95), [`StatusBar_FormatText`](src/bank1/status-bar-rupees.asm#L148)

**Pickup**
- A rupee adds 1 to `InvRupeesToAdd` and plays tune 1 number 1. [`Item_TakeOneRupee`](src/bank1/take-item.asm#L557)
- A 5-rupee item is five single rupees: `InvRupeesToAdd` + 5. [`Item_Take5Rupees`](src/bank1/take-item.asm#L328)
- Rupee stash (object type `$35`): ten rupee objects in slots `$01`-`$0A` in a diamond; each is taken when Link is less than 9 pixels from it on both axes, +1 each. [`RupeeStash_Init_Full`](src/bank1/trap-rupee-stash.asm#L75), [`RupeeStash_Update_Full`](src/bank1/trap-rupee-stash.asm#L338)
- Taking one stash rupee zeroes `RoomObjCount`: the room is saved as cleared on leaving and rupees left behind are lost. [`RupeeStash_Update_Full`](src/bank1/trap-rupee-stash.asm#L338)
- Secret rupee caves (persons `$7B`-`$7D`) give 30, 100 or 10 rupees, once. [`Cave_State5HintOrMoneyGame`](src/bank1/cave-hint-money-game.asm#L61), [`Cave_Init`](src/bank1/cave-init-person-text.asm#L101)
- Money game: the three amounts are fixed when the cave person is set up: one win of 20 or 50, one loss of 10, one loss of 10 or 40. [`Cave_Init_SetUpTextAndWares`](src/bank1/cave-init-person-text.asm#L145), [`Cave_MoneyGamePermutations`](src/bank1/cave-init-person-text.asm#L83)
- A forced help drop is 5 rupees unless it is a bomb. [`Drop_SetUpItem@SetHelpItem`](src/bank4/drop-items.asm#L244)

**Limits**
- Shops, paid hints: `InvRupees` must be at least the price, and nothing can be touched while `InvRupeesToSubtract` is nonzero. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- An arrow costs 1 rupee and cannot be shot with `InvRupees` = 0. [`Arrow_Wield`](src/bank5/wield-items.asm#L187)
- Door repair (person `$71`) posts a debit of 20 without testing what Link has. [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- The bomb capacity offer costs 100 and the life-or-money man 50; both test only `InvRupees`, not the pending debit. [`Person_ComplexState2SellBombCapacity`](src/bank1/underworld-person-update.asm#L171), [`Person_LifeOrMoneyState2Choose`](src/bank1/life-or-money.asm#L121)

**Notes**
- Because a debit is dropped when `InvRupees` reaches 0, charges posted without a balance test (door repair) or on top of a pending debit cost at most what Link has. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95)
- A price of 100 takes 200 frames to count off, during which the shop ignores Link. [`Inv_ChangeRupees`](src/bank1/status-bar-rupees.asm#L95), [`Cave_State2TalkOrShopOrDoorCharge`](src/bank1/cave-shop.asm#L55)
- Rupee items flash between sprite palettes 1 and 2 every 8 frames. [`Item_DrawBySlot@Flash`](src/bank7/room-item-draw.asm#L166)

# Clock

**Ids**
- Item id `$21`, class 0, item slot `$15`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`Item_IdToDescriptor`](src/bank1/take-item.asm#L111)

**RAM**
- `$066C`: 1 while the clock is in effect. There is no timer on it. [`InvClock`](src/Variables.inc#L580)

**Effect**
- Monsters that move through the shared walker move do not move while `InvClock` is nonzero. [`Walker_Move`](src/bank7/walker-move.asm#L64)
- Monster touches and shots do not register on Link: the touch test returns at once while `InvClock` is nonzero. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- `$10` is added to Link's invincibility timer every frame, before his update; the timer loses only 1 per two frames. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129), [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119)
- Shooters keep running their shoot timer but the shot is skipped. [`Shooter_RunShootTimer`](src/bank4/moblin-lynel-shooting.asm#L96)
- Frozen with their own test: wanderers, jumpers (Tektite, Boulder: in mid-air too), Keese, Peahat, flying Ghini, Pols Voice, Rope, Wallmaster's trip, Vire's hop, Moldorm, Lanmola, Digdogger, Aquamentus (no move, no shot). [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185), [`Jumper_Update`](src/bank4/tektite-boulder.asm#L166), [`Keese_Update`](src/bank4/bubble-keese.asm#L163), [`Aquamentus_Update`](src/bank4/aquamentus.asm#L54)
- Wizzrobes neither move, fade nor shoot, and are drawn and hittable every frame. [`Wizzrobe_Blue_Update`](src/bank4/wizzrobe.asm#L95)
- A Zora's update does nothing at all: it is not drawn and cannot be hit. [`Zora_Update`](src/bank4/zora-ow-inits.asm#L301)
- Frozen monsters still take weapon hits; only their movement and their touch on Link stop. [`Wanderer_Update`](src/bank4/walker-wanderer-common.asm#L185), [`Lanmola_Update`](src/bank4/lanmola.asm#L43)

**Limits**
- Not frozen (no `InvClock` test in their update, and they do not use the walker move): Gohma, Manhandla, Gleeok, Patra, Ganon, statue fireballs, shots and fireballs already in flight. [`Gohma_Update`](src/bank4/gohma.asm#L63), [`Manhandla_Update`](src/bank4/manhandla.asm#L60), [`Gleeok_Update`](src/bank4/gleeok.asm#L175), [`Patra_Update`](src/bank4/patra.asm#L62), [`Statue_Update`](src/bank4/statues.asm#L86)
- Those still cannot hurt Link through the ordinary touch test, which the clock disables. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)
- Ganon's touch does not test `InvClock`; it tests only Link's invincibility timer. [`Ganon_CheckCollisions`](src/bank4/ganon-draw-collisions.asm#L360), [`Link_CheckMonsterTouch_NoReset`](src/bank1/link-collision-harm.asm#L73)
- Link's own candle fire also tests only the invincibility timer. [`Fire_Update`](src/bank7/fire.asm#L44)
- That timer is not constantly nonzero: +`$10` per frame and -1 per two frames makes the byte wrap to 0 on one frame in every 33 (arithmetic of the two routines), and on that frame Ganon and fire can hurt Link. [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129), [`Obj_DecInvincibilityTimer`](src/bank7/object-stubs-destroy-monster.asm#L119)
- A Like-Like that already holds Link keeps eating: its hold tests neither the clock nor its stun timer. [`LikeLike_Update`](src/bank4/like-like.asm#L46)
- Blade traps still sense Link and charge; they only cannot hurt him. [`Trap_Update_Full`](src/bank1/trap-rupee-stash.asm#L189)
- Leevers: a Blue Leever stops moving but still sinks and resurfaces; a Red Leever is held only in its surfaced state 3. [`Leever_UpdateBlue`](src/bank4/leever.asm#L79), [`Leever_UpdateRed`](src/bank4/leever.asm#L252)
- The boulder spawner keeps making Boulders. [`Boulder_UpdateSpawner`](src/bank4/tektite-boulder.asm#L62)
- A Rope stops moving but its rush test still sets direction and speed. [`Rope_Update`](src/bank4/rope.asm#L51)
- Cleared (`InvClock` = 0) on every room entry (modes 4, 9, `$B`, `$C`: caves and cellars too), on leaving a room (mode 6), by the continue menu and when a file is loaded. [`Link_ResetStateAndClock`](src/bank7/init-mode-dispatch.asm#L354), [`Room_EnterCommon`](src/bank5/enter-room.asm#L143), [`Mode6Leave_Init`](src/bank5/mode3-mode6-init.asm#L214), [`Mode1Menu_Update_Sub1Load`](src/bank2/mode1-menu-update.asm#L104)

**Pickup**
- Plain class 0 item: `InvClock` = 1. [`Item_Take`](src/bank1/take-item.asm#L259)
- Source: the drop table only. Group 1 at kill cycle 6 and group 2 at kill cycle 3. [`Drop_ItemTable`](src/bank4/drop-items.asm#L132)

**Notes**
- A whirlwind cannot pick Link up while he holds the clock (same touch test). [`Whirlwind_Update_Full`](src/bank1/whirlwind.asm#L99)
- A Wallmaster or Like-Like cannot capture Link under the clock: capture needs a touch that harms him. [`Link_CheckMonsterTouch`](src/bank1/link-collision-harm.asm#L48)

# Map and Compass

Both are kept as one bit per level; neither does anything in the overworld.

**Ids**
- Compass: item id `$16`, class 0, item slot `$10` (level 9: slot `$12`). [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421)
- Map: item id `$17`, class 0, item slot `$11` (level 9: slot `$13`). [`Item_IdToSlot`](src/bank1/take-item.asm#L81), [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421)

**RAM**
- `$0667`: compasses of levels 1-8, bit (level - 1). `$0669`: compass of level 9, bit 0. [`InvCompass`](src/Variables.inc#L574), [`InvCompass9`](src/Variables.inc#L576)
- `$0668`: maps of levels 1-8, bit (level - 1). `$066A`: map of level 9, bit 0. [`InvMap`](src/Variables.inc#L575), [`InvMap9`](src/Variables.inc#L577)
- The ownership tests return 0 in the overworld. [`Inv_HasCompass`](src/bank5/submenu-map.asm#L143), [`Inv_HasMap`](src/bank5/submenu-map.asm#L153)

**Effect**
- Compass: in a dungeon whose compass Link has, a marker sprite is drawn on the status bar map at the room `LevelInfo_TriforceRoomId`. [`StatusBar_UpdateTriforceMarker`](src/bank7/next-room-room-flags.asm#L51), [`LevelInfo_TriforceRoomId`](src/Variables.inc#L723)
- The marker blinks (attribute 2 for the first 16 of every 32 frames, else 3) while the level's Triforce piece is missing, and always in level 9; with the piece it stays at attribute 3. [`StatusBar_UpdatePositionMarker`](src/bank1/map-marker-curtain.asm#L72)
- Marker position: Y = `$17` + 4 * room row; X = column * 8 + `$12` + the level's map X offset (dungeon). [`StatusBar_UpdatePositionMarker`](src/bank1/map-marker-curtain.asm#L72)
- Map: the status bar map of the level is drawn. On entering a level it is cued only if Link has the level's map; the overworld map is always drawn. [`Mode3Unfurl_Init_Sub6_CueStatusBarMap`](src/bank5/mode3-mode6-init.asm#L102)
- Taking the map sets `StatusBarMapTrigger`, and the play update cues the status bar map transfer on the next frame the transfer buffer is free. [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421), [`Mode5Play_UpdateWorld`](src/bank7/mode5-play-update.asm#L129)
- The submenu shows the compass and map icons only for the current level. [`Submenu_DrawItems`](src/bank5/submenu-items.asm#L68)
- The submenu's room-by-room map sheet does not depend on the map: it shows rooms whose visited flag (`$20`) is set. [`Submenu_WriteScanningMapRoomMark`](src/bank5/submenu-map.asm#L183)

**Limits**
- Link's own position marker on the status bar map needs neither item. [`StatusBar_UpdateLinkPositionMarker`](src/bank1/map-marker-curtain.asm#L53)
- Neither item changes monsters, doors or room contents: the only readers of the bits are the two ownership tests and their three callers above. [`Inv_HasCompass`](src/bank5/submenu-map.asm#L143), [`Inv_HasMap`](src/bank5/submenu-map.asm#L153)

**Pickup**
- The pickup sets bit (level - 1) of the slot; in level 9 the slot index is raised by 2 and bit 0 is set. [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421), [`Util_BitMasks`](src/bank7/next-room-room-flags.asm#L110)
- Taken in the overworld (`CurLevel` = 0) nothing is stored. [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421)

**Notes**
- A level's compass and map bits are independent of the level's Triforce bit; only the marker's blinking reads `InvTriforce`. [`StatusBar_UpdatePositionMarker`](src/bank1/map-marker-curtain.asm#L72)

# Triforce pieces and the Triforce of Power

**Ids**
- Triforce piece: item id `$1B`, class 0, item slot `$1A`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)
- Triforce of Power: item id `$0E`, class 0, item slot `$1B`. [`Item_IdToSlot`](src/bank1/take-item.asm#L81)

**RAM**
- `$0671`: pieces, bit (level - 1) for levels 1-8; `$FF` = all eight. [`InvTriforce`](src/Variables.inc#L597)
- `$0672`: 1 once the Triforce of Power fanfare has ended. [`InvPowerTriforce`](src/Variables.inc#L598)

**Pickup** (Triforce piece)
- A piece is a dungeon room item; it is placed 8 pixels left of the room's item position. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51)
- Taking it sets the room's item-taken flag `$10` and bit (level - 1) of `InvTriforce`, ends the mode and sets `GameMode` = `$12`. [`Item_TryTake`](src/bank1/take-item.asm#L206), [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421)
- Mode `$12` init: song request 4, play area tile map blanked, object sprites hidden, Link drawn holding up item `$1B`. [`Mode12EndLevel_Init`](src/bank5/mode12-lay-out-room.asm#L56)
- Submode 0: wait `$30` frames. Submode 1: `$30` frames of palette flash, 4 frames white and 4 frames the level palette. [`Mode12EndLevel_Update_Sub0Wait`](src/bank5/mode12-lay-out-room.asm#L93), [`Mode12EndLevel_Update_Sub1Flash`](src/bank5/mode12-lay-out-room.asm#L101)
- Submode 2: hearts refill to full (`World_IsFillingHearts` = 2; 43 frames per empty heart); the sequence waits for it. [`Mode12EndLevel_StartFillingHearts`](src/bank5/mode12-lay-out-room.asm#L118), [`Mode12EndLevel_Update_Sub2FillHearts`](src/bank5/mode12-lay-out-room.asm#L129)
- Submode 3: `$80` frames wait, then the curtain closes from both screen edges, 16 steps of 5 frames. [`Mode12EndLevel_Update_Sub3CloseCurtain`](src/bank5/mode12-lay-out-room.asm#L138)
- Submode 4: `$80` frames wait, then `CurLevel` = 0, `GameMode` = 2 (load level) and `UndergroundExitType` = 2: Link comes out of the level's entrance on the overworld. [`Mode12EndLevel_Update_Sub4LeaveLevel`](src/bank5/mode12-lay-out-room.asm#L156), [`Mode_LeaveLevelToOW`](src/bank5/next-room-calc.asm#L107)
- Pending rupees are counted only during submode 2 of this mode. [`Mode12EndLevel_Update_Sub2FillHearts`](src/bank5/mode12-lay-out-room.asm#L129)

**Effect** (Triforce pieces)
- Level 9's first old man (person type `$4B`): with `InvTriforce` = `$FF` he sets `ShutterTrigger`, releases Link and is removed when the room is set up; with any piece missing he stays and blocks the way up. [`Person_InitUnderworldLevel9`](src/bank1/underworld-person-init.asm#L172)
- The recorder's whirlwind goes only to levels whose piece Link has; with `InvTriforce` = 0 no whirlwind is made. [`Whirlwind_Summon`](src/bank1/whirlwind.asm#L243)
- The compass marker stops blinking once the level's piece is held. [`StatusBar_UpdatePositionMarker`](src/bank1/map-marker-curtain.asm#L72)
- The overworld submenu draws the Triforce from `InvTriforce`: bit n = the piece of level n + 1. [`Submenu_BuildTriforceOW`](src/bank5/submenu-states.asm#L440)

**Pickup** (Triforce of Power)
- In Ganon's room the room item starts absent and is made to appear when Ganon dies. [`Room_SetUpItemAndPushBlock`](src/bank5/create-room-objects.asm#L51), [`Ganon_ActivateRoomItem`](src/bank4/ganon-draw-collisions.asm#L541)
- Taking it (in a dungeon only) plays tune 1 number 2 instead of 8 and starts the fanfare: Link halted (state `$40`), his `ObjTimer` = `$C0`. [`Item_Take`](src/bank1/take-item.asm#L259), [`Item_BeginPowerTriforceFanfare`](src/bank1/tile-block-power-triforce.asm#L152)
- For those `$C0` frames the background palettes alternate every 4 frames between the level's palettes and white. [`Item_UpdatePowerTriforceFanfare`](src/bank1/tile-block-power-triforce.asm#L99)
- Only when the timer ends: `InvPowerTriforce` = 1, Link's state = 0, the level 9 song is requested again. [`Item_UpdatePowerTriforceFanfare@EndFanfare`](src/bank1/tile-block-power-triforce.asm#L132)
- Rooms with secret trigger 3 open their shutters while `InvPowerTriforce` is nonzero (the way on to Zelda). [`Secret_Trigger3PowerTriforce`](src/bank5/underworld-secrets.asm#L255)
- No hearts are refilled by the Triforce of Power: the refill branch in the fanfare routine is unreachable. [`Item_UpdatePowerTriforceFanfare@FillHearts`](src/bank1/tile-block-power-triforce.asm#L124)

**Limits**
- Both items do nothing when taken in the overworld (`CurLevel` = 0). [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421)
- A Triforce piece taken in level 9 would set bit 0 of item slot `$1C` instead of `InvTriforce` and would not start mode `$12`, because the slot index is raised by 2 before the Triforce test. [`Item_TakeClass0PerLevel`](src/bank1/take-item.asm#L421)

**Notes**
- Taking a piece is a full heal and an exit to the overworld, about `$30` + `$30` + refill + `$80` + 80 + `$80` frames long. [`Mode12EndLevel_Update_Full_JumpTable`](src/bank5/mode12-lay-out-room.asm#L85)
- Both Triforce items flash between sprite palettes 1 and 2 every 8 frames. [`Item_DrawBySlot@Flash`](src/bank7/room-item-draw.asm#L166)
