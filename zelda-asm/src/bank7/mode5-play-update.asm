; ============================================================================
; FILE:     bank7/mode5-play-update.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1811-2135
; SUBJECTS: Link
; TOPIC:    UpdateMode5Play, object loop, chase target, UpdatePlayer
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 UpdateMode5Play               code
;   L0083 BeginUpdateWorld              code
;   L0276 UpdateHeartsAndRupees         code
;   L0286 UpdatePlayer                  code
;   L0312 Link_EndMoveAndAnimateInRoom  code
;   L0330 EnsureObjectAligned           code  Params:
;   L0341 L1EDEA_Exit                   code
; ============================================================================

UpdateMode5Play:
    ; While the flute timer has not expired, return.
    LDA FluteTimer
    BNE L1EBF8_Exit

    ; If brightening the room, then animate it.
    LDA IsRoomBrightening
    BEQ @CheckMenuAndPause
    LDA #$04
    JSR SwitchBank
    JMP UpdateCandle

@CheckMenuAndPause:
    ; Check menu and pause.
    LDA SubmenuState
    BNE @CheckMenu              ; If the submenu is active, go update it.
    LDA PauseState
    CMP #$02
    BEQ @CheckPaused            ; If involuntarily paused, skip checking Select button here. But go do pause actions.

    ; Not paused or paused voluntarily.
    ; Check Select button.
    LDA ButtonsPressed
    AND #$20
    BEQ @CheckPaused            ; If pressed Select,
    LDA PauseState                  ; then toggle pause.
    EOR #$01
    STA PauseState
    BNE @CheckPaused            ; If not paused,
    LDA #$0F                    ; then enable sound.
    STA ApuStatus

@CheckPaused:
    ; If paused, then make sure to use NT 0,
    ; update hearts and rupees, and return.
    LDA PauseState
    BEQ @CheckMenu              ; If not paused, go update the submenu or world.
    LDA #$05
    JSR SwitchBank
    JSR MaskCurPpuMaskGrayscale
    JMP UpdateHeartsAndRupees

@CheckMenu:
    ; Not paused.
    JSR HideObjectSprites
    LDA ButtonsDown             ; Save current direction from input buttons.
    AND #$0F
    STA ObjInputDir
    LDA SubmenuState
    BEQ @NotInMenu              ; If not in the submenu, go check Start or update the world.

    ; In submenu.
    LDA #$05
    JSR SwitchBank
    JSR MaskCurPpuMaskGrayscale
    JMP UpdateMenuAndMeters

@NotInMenu:
    ; Not in submenu.
    LDA ButtonsPressed
    AND #$10
    BEQ BeginUpdateWorld        ; If Start not pressed, go update the world.
    INC SubmenuState               ; Else set menu state 1 to start scrolling.
    RTS

BeginUpdateWorld:
    ; If we have the clock, then
    ; force the player's invincibility.
    LDA InvClock
    BEQ @NotInvincible
    LDA ObjInvincibilityTimer
    CLC
    ADC #$10
    STA ObjInvincibilityTimer

@NotInvincible:
    ; Update the player.
    JSR UpdatePlayer
    LDA IsUpdatingMode
    BNE @CheckChaseTarget       ; If not updating anymore,
    JMP @FinishUpdatePlay       ; then we changed the mode, so don't update anything else.

@CheckChaseTarget:
    ; If Link can be chased, then set his coordinates as the target.
    LDA IsChasingOtherTarget
    BNE @UpdateWeapons
    LDA ObjX
    STA ChaseTargetX
    LDA ObjY
    STA ChaseTargetY

@UpdateWeapons:
    ; Update weapons and items.
    LDX #$0D
    JSR UpdateSwordOrRod
    LDX #$0E
    JSR UpdateSwordShotOrMagicShot
    LDX #$0F
    JSR UpdateBoomerangOrFood
    LDX #$10
    JSR UpdateBombOrFire
    LDX #$11
    JSR UpdateBombOrFire
    LDX #$12
    JSR UpdateRodOrArrow

    ; If it's not time to change the chase target, then
    ; skip this and go update objects.
    LDA ChaseLongTimer
    BNE @LoopObject

    ; Set the chase long-timer to a random value between 0 and 7.
    LDA Random+1
    AND #$07
    STA ChaseLongTimer

    ; Switch the flag indicating whether Link is the target.
    ; If the value is 0 (Link *is* the target), then
    ; skip this and go update objects.
    LDA IsChasingOtherTarget
    EOR #$01
    STA IsChasingOtherTarget
    BEQ @LoopObject

    ; If Link's X has not changed since the last frame
    ; (when Link was the target), then XOR the chase target
    ; coordinates (which were Link's coordinates last frame).
    ;
    ; This ends up reflecting Link's coordinates across the room;
    ; so that monsters chase away from him.
    ;
    ; But if Link's X has changed since the last frame, then
    ; leave the chase target coordinates with his old coordinates.
    LDA ChaseTargetX
    CMP ObjX
    BNE @LoopObject
    EOR #$FF
    STA ChaseTargetX
    LDA ChaseTargetY
    EOR #$FF
    STA ChaseTargetY

@LoopObject:
    ; Update objects from $B to 1.
    LDX CurObjIndex
    JSR DecrementInvincibilityTimer
    LDA ObjType, X
    BEQ @NextObject
    LDA ObjType, X
    JSR UpdateObject

    ; If the object is autonomous and not checking collisions itself,
    ; then see about checking collisions and drawing on its behalf.
    ;
    ; Else go loop again.
    LDX CurObjIndex
    LDA ObjMetastate, X
    BNE @NextObject
    LDA ObjAttr, X
    AND #$01                    ; Self checks collisions and draws
    BNE @NextObject

    ; If no attribute 1, then it indicates not to skip drawing. So, animate and draw.
    LDA ObjAttr, X
    AND #$04                    ; Self-draw attribute
    BNE :+
    JSR AnimateAndDrawObjectWalking
:
    ; Either way, we'll check collisions.
    LDX CurObjIndex
    JSR CheckMonsterCollisions

@NextObject:
    ; Bottom of the object update loop.
    DEC CurObjIndex
    BNE @LoopObject

    ; Set the current object slot to what it should be when we
    ; begin updating objects again next frame: $B
    ; the last object slot for monsters and tile objects
    ;
    ; Before the first frame of mode 5, the current object slot variable
    ; was set to $B in 05:87B9 (InitMode_EnterRoom).
    ;
    ; This can also be called the last slot for dynamically allocated objects.
    LDA #$0B
    STA CurObjIndex

    ; If full hearts = 0, then only a partial heart is left.
    ; So, turn on the "heart warning".
    LDA HeartValues
    AND #$0F
    BNE @CheckUW
    LDA Tune0Request
    ORA #$40
    STA Tune0Request

@CheckUW:
    ; If in UW, check statues, doors, secrets, and the power triforce.
    LDA CurLevel
    BEQ @CheckOW
    LDA #$04
    JSR SwitchBank
    JSR UpdateStatues           ; Update statues.
    JSR UpdateTriforcePositionMarker
    LDA #$05
    JSR SwitchBank
    JSR CheckUnderworldSecrets
    JSR CheckShutters
    JSR UpdateDoors
    LDA #$01
    JSR SwitchBank
    JSR CheckPowerTriforceFanfare    ; Check triforce fanfare.
    JMP @TransferStatusBarMap

@CheckOW:
    ; In OW.
    ; If game mode = 5, turn on sea sound effect if the room's
    ; attributes call for it.
    LDA GameMode
    CMP #$05
    BNE @CheckZora
    LDY RoomId
    LDA LevelBlockAttrsA, Y
    AND #$04                    ; Sea sound effect
    ASL
    ASL
    ASL
    JSR PlayEffect

@CheckZora:
    ; In OW, make a zora, if applicable.
    LDA #$04
    JSR SwitchBank
    JSR CheckZora               ; Check zora.

@TransferStatusBarMap:
    ; If there are tiles or palettes in the dynamic transfer buf, then
    ; do not handle the status bar map trigger.
    LDA DynTileBufLen
    BNE @FinishUpdatePlay

    ; Transfer the level's status bar map, if we got the signal.
    LDA StatusBarMapTrigger
    BEQ @FinishUpdatePlay
    LDA #$00
    STA StatusBarMapTrigger
    LDA #$44                    ; Cue the transfer of the level's status bar map.
    STA TileBufSelector

@FinishUpdatePlay:
    ; These actions are done after Link is updated, regardless
    ; of whether other objects were updated.
    JSR CheckLiftItem
    JSR MoveAndDrawRoomItem
    JSR TryTakeRoomItem
    JSR DrawStatusBarItemsAndEnsureItemSelected

UpdateHeartsAndRupees:
    LDA #$05
    JSR SwitchBank
    JSR World_FillHearts
    JMP World_ChangeRupees

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF

UpdatePlayer:
    LDX #$00
    JSR DecrementInvincibilityTimer

    ; If Link is halted, then return.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ L1EDEA_Exit

    ; If Link is paralyzed (by a like-like), then
    ; reset input direction.
    ;
    ; As far as I can tell, ObjInputDir could have been set to 0,
    ; instead of masking with $F0 for the same effect.
    LDA IsLinkParalyzed
    BEQ :+
    LDA ObjInputDir
    AND #$F0
    STA ObjInputDir
:
    LDA #$05
    JSR SwitchBank
    JSR Link_HandleInput
    JSR Walker_Move

Link_EndMoveAndAnimateInRoom:
    LDA GameMode
    CMP #$0A
    BEQ L1EDEA_Exit             ; If in mode $A, then return.
    JSR Link_EndMoveAndAnimate
    LDA CurLevel
    BEQ @Exit                   ; If in UW, then show link behind doors.
    JSR ShowLinkSpritesBehindHorizontalDoors

@Exit:
    LDX #$00

; Params:
; X: 0
;
; Ensure that, if grid offset = 0, then X is a multiple of 8,
; and Y is (multiple of 8) OR 5. For example, ($30, $7D) or ($58, $B5).
;
EnsureObjectAligned:
    LDA ObjGridOffset, X
    BNE L1EDEA_Exit
    LDA ObjX, X
    AND #$F8
    STA ObjX, X
    LDA ObjY, X
    AND #$F8
    ORA #$05
    STA ObjY, X

L1EDEA_Exit:
    RTS
