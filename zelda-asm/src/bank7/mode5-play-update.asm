; ============================================================================
; FILE:     bank7/mode5-play-update.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1811-2135
; SUBJECTS: Link, Mode5Play, Obj, World
; TOPIC:    The per-frame update of play (mode 5, also modes 9, $A, $B, $C): flute freeze, pause, submenu,
;           Link, the monsters' chase target, weapon slots $0D-$12, object slots $0B-$01, per-room checks,
;           hearts and rupees. Also Link's own update and the grid alignment helper.
; SEE ALSO: bank7/update-mode-dispatch.asm (Mode_Update_JumpTable: which modes run this);
;           bank5/link-handle-input.asm (Link_HandleInput) and bank7/walker-move.asm (Walker_Move): Link's
;           movement; bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate); bank7/object-update-table.asm
;           (Obj_Update); bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink); bank7/flute.asm
;           (FluteTimer); bank7/boomerang-arrow-food.asm (food replaces the chase target);
;           bank5/cave-cellar-enter.asm (World_FillHearts); bank1/status-bar-rupees.asm (Inv_ChangeRupees).
; KEY FACTS:
;   - While FluteTimer is nonzero the whole update is skipped: no input, no pause, no object moves
;     (Mode5Play_Update).
;   - Frame order: Link, then weapon slots $0D, $0E, $0F, $10, $11, $12, then object slots $0B down to
;     $01, then room checks (statues, secrets, shutters, doors in the UW; Zora in the OW), then the room
;     item, then hearts and rupees (Mode5Play_UpdateWorld).
;   - If Link's update changed the game mode (IsUpdatingMode = 0), weapons and objects are not updated
;     that frame (Mode5Play_UpdateWorld).
;   - Select toggles PauseState 0/1 unless it is 2; while paused only hearts and rupees are updated
;     (Mode5Play_Update). Start opens the submenu (SubmenuState = 1) and nothing else runs that frame.
;   - The clock (InvClock) adds $10 to Link's invincibility timer every frame; the timer drops only 1
;     per 2 frames, and the byte wraps (Mode5Play_UpdateWorld, Obj_DecInvincibilityTimer).
;   - Chase target: while IsChasingOtherTarget = 0 it is Link's position, copied every frame. When
;     ChaseLongTimer reaches 0 it is reloaded with Random+1 AND 7 (units of 10 frames) and the flag
;     toggles; on the switch to 1 the target becomes (X EOR $FF, Y EOR $FF) = (255 - X, 255 - Y) of
;     Link's position, unless a weapon routine changed the target X in between (food)
;     (Mode5Play_UpdateWorld@CheckChaseTarget and below).
;   - An object whose ObjMetastate is 0 and whose ObjAttr bit $01 is clear gets its collision test, and
;     unless bit $04 is set its walking animation and drawing, from this loop. Bit $04 is tested only
;     when bit $01 is clear, so attribute $05 acts like $01 here (Mode5Play_UpdateWorld@LoopObject).
;   - Low nibble of HeartValues = 0 (at most the partial heart is left) requests the low-hearts beep
;     on every frame the object loop runs (Mode5Play_UpdateWorld, after the object loop).
;   - Link's update is skipped while (ObjState AND $C0) = $40 (halted); while IsLinkParalyzed is set his
;     direction input is discarded but A and B still reach Link_HandleInput (Link_Update).
;   - At grid offset 0 an object is snapped to X = multiple of 8 and Y = (multiple of 8) + 5
;     (Obj_SnapToGrid).
; CONTENTS: (generated; L = line in this file)
;   L0059 Mode5Play_Update              code  Runs one frame of play: flute freeze, room brightening, pause, submenu, else the world update.
;   L0129 Mode5Play_UpdateWorld         code  Updates Link, the chase target, the weapon slots, object slots $0B-$01 and the per-room checks.
;   L0311 World_UpdateHeartsAndRupees   code  Steps the heart refill and the rupee count, one tick each (World_FillHearts, Inv_ChangeRupees).
;   L0330 Link_Update                   code  Updates Link for one frame: invincibility timer, input, movement, animation, grid alignment.
;   L0357 Link_EndMoveAndAnimateInRoom  code  Finishes Link's move in a room: animate, draw behind UW door frames, snap to the grid.
;   L0375 Obj_SnapToGrid                code  Snaps an object that is on a grid point to X = multiple of 8, Y = (multiple of 8) + 5.
;   L0387 Obj_SnapToGrid_Return         code  Shared return of Link_Update, Link_EndMoveAndAnimateInRoom and Obj_SnapToGrid.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs one frame of play: flute freeze, room brightening, pause, submenu, else the world update.
; OUT:  bank 5 is switched in on most paths (bank 4 after the brightening path).
; NOTE: Mode_Update_JumpTable uses this routine for modes 5, 9 (cellar), $A, $B (cave) and $C.
; NOTE: While FluteTimer is nonzero ($98 frames after the flute is used) nothing at all is updated here;
;       the timer itself is counted down in the NMI handler, once per frame.
; NOTE: While paused or in the submenu only World_FillHearts and Inv_ChangeRupees run for the world;
;       the NMI handler also stops its timers then.
; ----------------------------------------------------------------------------
Mode5Play_Update:
    ; While the flute tune plays, do nothing.
    LDA FluteTimer
    BNE Mode_GoToNext_Return

    ; While a candle fire is fading a dark room to light, only run that fade.
    LDA IsRoomBrightening
    BEQ @CheckMenuAndPause
    LDA #$04
    JSR Mmc1_SwitchBank
    JMP Candle_UpdateRoomLight

@CheckMenuAndPause:
    LDA SubmenuState
    BNE @CheckMenu              ; The submenu is open or scrolling: go update it.
    LDA PauseState
    CMP #$02
    BEQ @CheckPaused            ; Paused by the game (state 2): Select is ignored.

    ; Not paused, or paused with Select: Select toggles PauseState between 0 and 1.
    LDA ButtonsPressed
    AND #$20                    ; Select
    BEQ @CheckPaused
    LDA PauseState
    EOR #$01
    STA PauseState
    BNE @CheckPaused            ; Just paused.
    LDA #$0F                    ; Just unpaused: enable the two squares, triangle and noise channels.
    STA ApuStatus

@CheckPaused:
    ; While paused: clear the grayscale bit of the PPU mask, update hearts and rupees, and return.
    LDA PauseState
    BEQ @CheckMenu
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Ppu_ClearGrayscaleInMask
    JMP World_UpdateHeartsAndRupees

@CheckMenu:
    ; Not paused.
    JSR Sprite_HideObjectSprites
    LDA ButtonsDown             ; Link's input direction = the direction bits held on controller 1.
    AND #$0F
    STA ObjInputDir
    LDA SubmenuState
    BEQ @NotInMenu

    ; The submenu is open or scrolling: update it (it ends with World_UpdateHeartsAndRupees) and return.
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Ppu_ClearGrayscaleInMask
    JMP Submenu_UpdateAndDrawMeters

@NotInMenu:
    LDA ButtonsPressed
    AND #$10                    ; Start
    BEQ Mode5Play_UpdateWorld        ; Start not pressed: go update the world.
    INC SubmenuState            ; Start pressed: submenu state 1 (begin scrolling in); nothing else runs.
    RTS

; ----------------------------------------------------------------------------
; Updates Link, the chase target, the weapon slots, object slots $0B-$01 and the per-room checks.
; OUT:  CurObjIndex = $0B; bank 5 switched in (falls into World_UpdateHeartsAndRupees).
; NOTE: Order within a frame: Link (slot 0); slots $0D sword or rod, $0E sword shot or magic shot, $0F
;       boomerang or food, $10 and $11 bomb or fire, $12 rod or arrow; then slots $0B down to $01.
;       Slot $0C is not updated by this loop.
; NOTE: If Link's update left the mode (IsUpdatingMode = 0), everything up to @FinishUpdatePlay is skipped.
; NOTE: Random use: Random+1 AND 7 reloads ChaseLongTimer (0-7 units of 10 frames) each time it expires.
; ----------------------------------------------------------------------------
Mode5Play_UpdateWorld:
    ; With the clock, add $10 to Link's invincibility timer every frame. The timer counts down only 1
    ; per 2 frames, so it keeps rising and wraps around as a byte.
    LDA InvClock
    BEQ @NotInvincible
    LDA ObjInvincibilityTimer
    CLC
    ADC #$10
    STA ObjInvincibilityTimer

@NotInvincible:
    JSR Link_Update
    LDA IsUpdatingMode
    BNE @CheckChaseTarget       ; Still updating this mode: go on.
    JMP @FinishUpdatePlay       ; Link's update ended the mode (room change, stairs, ...): skip the rest.

@CheckChaseTarget:
    ; While monsters chase Link (flag = 0), the chase target is Link's current position.
    LDA IsChasingOtherTarget
    BNE @UpdateWeapons
    LDA ObjX
    STA ChaseTargetX
    LDA ObjY
    STA ChaseTargetY

@UpdateWeapons:
    ; Update Link's weapons, one fixed object slot each.
    LDX #$0D                    ; Sword or rod
    JSR Sword_UpdateSwordOrRod
    LDX #$0E                    ; Sword shot or magic shot
    JSR SwordShot_UpdateOrMagicShot
    LDX #$0F                    ; Boomerang or food
    JSR Boomerang_UpdateOrFood
    LDX #$10                    ; Bomb or fire, first slot
    JSR Bomb_UpdateBombOrFire
    LDX #$11                    ; Bomb or fire, second slot
    JSR Bomb_UpdateBombOrFire
    LDX #$12                    ; Rod or arrow
    JSR Rod_UpdateRodOrArrow

    ; ChaseLongTimer counts down once per 10 frames (NMI handler). While it runs, keep the target.
    LDA ChaseLongTimer
    BNE @LoopObject

    ; Reload it with a random 0-7 (0 means it expires again on the next frame).
    LDA Random+1
    AND #$07
    STA ChaseLongTimer

    ; Toggle the flag. If monsters now chase Link again (0), there is nothing more to do.
    LDA IsChasingOtherTarget
    EOR #$01
    STA IsChasingOtherTarget
    BEQ @LoopObject

    ; The flag went 0 -> 1. The target was copied from Link at @CheckChaseTarget in this same frame, so
    ; its X still equals Link's X unless a weapon update above overwrote it (food on the ground sets the
    ; target to the food's position). If it is unchanged, replace the target with (X EOR $FF, Y EOR $FF):
    ; Link's position mirrored through the centre of the screen, where chasing monsters now head.
    ; If it was changed, leave it alone. The target then stays fixed until the flag returns to 0.
    LDA ChaseTargetX
    CMP ObjX
    BNE @LoopObject
    EOR #$FF
    STA ChaseTargetX
    LDA ChaseTargetY
    EOR #$FF
    STA ChaseTargetY

@LoopObject:
    ; Update object slots from CurObjIndex ($0B) down to 1.
    LDX CurObjIndex
    JSR Obj_DecInvincibilityTimer
    LDA ObjType, X
    BEQ @NextObject             ; Empty slot.
    LDA ObjType, X
    JSR Obj_Update

    ; Objects in the spawn cloud or death sparkle (metastate <> 0), and objects with attribute $01
    ; (they check collisions and draw themselves), are finished. Bit $04 is not looked at for them, so
    ; an attribute of $05 behaves like $01 here.
    LDX CurObjIndex
    LDA ObjMetastate, X
    BNE @NextObject
    LDA ObjAttr, X
    AND #$01                    ; Checks collisions and draws itself
    BNE @NextObject

    ; Bit $01 clear: unless bit $04 is set (draws itself), animate and draw it as a walking object.
    LDA ObjAttr, X
    AND #$04
    BNE :+
    JSR Anim_AnimateAndDrawObjectWalking
:
    ; Test it against Link and his weapons.
    LDX CurObjIndex
    JSR Collision_CheckMonsterVsWeaponsAndLink

@NextObject:
    DEC CurObjIndex
    BNE @LoopObject

    ; Leave CurObjIndex at $0B, the last slot for monsters and the room's tile object, ready for the
    ; next frame. Room entry sets the same value before the first frame of play.
    LDA #$0B
    STA CurObjIndex

    ; No full heart left (low nibble of HeartValues = 0): request the low-hearts beep.
    LDA HeartValues
    AND #$0F
    BNE @CheckUW
    LDA Tune0Request
    ORA #$40
    STA Tune0Request

@CheckUW:
    ; In the UW: statues, the Triforce map marker, room secrets, shutters, doors, and the Triforce of
    ; Power fanfare.
    LDA CurLevel
    BEQ @CheckOW
    LDA #$04
    JSR Mmc1_SwitchBank
    JSR Statue_Update
    JSR StatusBar_UpdateTriforceMarker
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Secret_CheckUnderworld
    JSR Shutter_OpenTriggered
    JSR Door_Update
    LDA #$01
    JSR Mmc1_SwitchBank
    JSR Item_UpdatePowerTriforceFanfare
    JMP @TransferStatusBarMap

@CheckOW:
    ; In the OW (CurLevel = 0; this includes caves, which run in mode $B).
    ; Only in mode 5: request the sea sound effect ($20) if bit 2 of the room's attribute byte A is set.
    LDA GameMode
    CMP #$05
    BNE @CheckZora
    LDY RoomId
    LDA LevelBlockAttrsA, Y
    AND #$04                    ; Sea sound flag
    ASL
    ASL
    ASL                         ; $04 -> effect $20; 0 requests nothing.
    JSR Sfx_Play

@CheckZora:
    ; Let a Zora appear if the room calls for one.
    LDA #$04
    JSR Mmc1_SwitchBank
    JSR Zora_TrySpawn

@TransferStatusBarMap:
    ; If the dynamic transfer buffer already holds something this frame, leave the trigger for later.
    LDA DynTileBufLen
    BNE @FinishUpdatePlay

    ; If the trigger is set (the map was picked up), cue the level's status bar map transfer.
    LDA StatusBarMapTrigger
    BEQ @FinishUpdatePlay
    LDA #$00
    STA StatusBarMapTrigger
    LDA #$44                    ; Transfer buffer selector of the level's status bar map.
    STA TileBufSelector

@FinishUpdatePlay:
    ; Done after Link's update whether or not the other objects were updated: the item lift, the room
    ; item (move, draw, pickup) and the status bar items.
    JSR Link_CheckLiftItem
    JSR Item_MoveAndDrawRoomItem
    JSR Item_TryTakeRoomItem
    JSR StatusBar_DrawItemsAndEnsureSelection

; ----------------------------------------------------------------------------
; Steps the heart refill and the rupee count, one tick each (World_FillHearts, Inv_ChangeRupees).
; OUT:  bank 5 switched in.
; NOTE: Runs on every frame the play update gets past the flute freeze and the room brightening, except
;       the frame Start opens the submenu: also while paused and in the submenu. The end-level mode
;       calls it too.
; ----------------------------------------------------------------------------
World_UpdateHeartsAndRupees:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR World_FillHearts
    JMP Inv_ChangeRupees

; Filler: 12 bytes of $FF between routines. They have no label and nothing refers to them.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF

; ----------------------------------------------------------------------------
; Updates Link for one frame: invincibility timer, input, movement, animation, grid alignment.
; OUT:  bank 5 switched in unless Link is halted; X = 0 unless the mode is $A (then X is whatever
;       Walker_Move left).
; NOTE: Does nothing but the timer while Link is halted: (ObjState AND $C0) = $40.
; NOTE: While IsLinkParalyzed is set (a Like-Like holds Link) the direction input is cleared, but
;       Link_HandleInput still runs, so the buttons are still handled.
; NOTE: Also called while Link walks into a cellar (mode 9 init), with a forced input direction.
; ----------------------------------------------------------------------------
Link_Update:
    LDX #$00
    JSR Obj_DecInvincibilityTimer

    ; If Link is halted, return.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ Obj_SnapToGrid_Return

    ; If Link is paralyzed, discard the direction input. ObjInputDir holds only the low nibble at this
    ; point (see @CheckMenu above), so masking with $F0 leaves 0.
    LDA IsLinkParalyzed
    BEQ :+
    LDA ObjInputDir
    AND #$F0
    STA ObjInputDir
:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Link_HandleInput
    JSR Walker_Move

; Finishes Link's move in a room: animate, draw behind UW door frames, snap to the grid.
; Entered by fall-through from Link_Update, and by jump after Obj_MoveByQSpeed where Link walks by himself
; (bank5/cave-cellar-enter.asm, bank7/update-mode-dispatch.asm). In mode $A it returns at once and
; leaves X alone; otherwise X = 0 on exit.
Link_EndMoveAndAnimateInRoom:
    LDA GameMode
    CMP #$0A
    BEQ Obj_SnapToGrid_Return             ; In mode $A, return.
    JSR Link_EndMoveAndAnimate
    LDA CurLevel
    BEQ @Exit                   ; In the UW, give Link's sprites background priority in horizontal doorways.
    JSR Link_PutSpritesBehindSideDoorways

@Exit:
    LDX #$00

; ----------------------------------------------------------------------------
; Snaps an object that is on a grid point to X = multiple of 8, Y = (multiple of 8) + 5.
; IN:   X = object slot (0 when reached by fall-through).
; OUT:  ObjX,X and ObjY,X rounded down as above when ObjGridOffset,X = 0; otherwise unchanged.
; NOTE: Examples of aligned positions: ($30, $7D), ($58, $B5). Also called by Shove_MovePixels for any slot.
; ----------------------------------------------------------------------------
Obj_SnapToGrid:
    LDA ObjGridOffset, X
    BNE Obj_SnapToGrid_Return
    LDA ObjX, X
    AND #$F8
    STA ObjX, X
    LDA ObjY, X
    AND #$F8
    ORA #$05
    STA ObjY, X

; Shared return of Link_Update, Link_EndMoveAndAnimateInRoom and Obj_SnapToGrid.
Obj_SnapToGrid_Return:
    RTS
