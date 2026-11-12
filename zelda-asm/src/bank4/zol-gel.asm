; ============================================================================
; FILE:     bank4/zol-gel.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1239-1569
; SUBJECTS: Gel, Zol
; TOPIC:    Zol (type $13): wander with pauses, and the split into two child Gels (type $14) when a hit
;           does not kill it. Gel (types $14 child, $15): start-up dash of a child, wander with pauses.
; SEE ALSO: bank1/weapon-damage-death.asm (Damage_HandleWeaponHit: a weapon turns the Zol to the
;           weapon's direction; Damage_DealToMonster); bank1/shove-knockback.asm (Shove_Begin: sets the
;           invincibility timer the split test reads); bank4/shoot-helpers-candle.asm (Shooter_ShootLimited:
;           creates the child Gel); bank4/walker-wanderer-common.asm (Wanderer_MoveAndTurn, Walker_Init);
;           bank7/object-update-table.asm (HP and attribute tables; child Gel kills are not counted);
;           bank4/vire.asm (Vire's state 1 is Zol_State1Slide).
; KEY FACTS:
;   - A Zol splits when a weapon hit leaves it alive with its invincibility timer set, i.e. the hit
;     did less than its HP ($20) and shoved it: wooden sword or its beam ($10) and candle fire ($10).
;     The boomerang only stuns it (no shove, so no split) (Zol_CheckCollisions).
;   - Split path: if Link faces left or right while the Zol is on a grid row ((Y AND $0F) = $0D), or
;     up or down while it is on a grid column ((X AND $0F) = 0), the Zol first slides at q-speed $FF
;     (about 4 px/frame) in the weapon's direction until a tile or the room boundary blocks it (state
;     1), then splits; otherwise it splits at once (Zol_CheckCollisions, Zol_State1Slide).
;   - While sliding (state 1) the Zol tests no collisions: it cannot be hit and cannot hurt Link
;     (Zol_CheckCollisions returns unless state = 0).
;   - The split destroys the Zol and creates two type $14 Gels at its position, facing left/right if
;     the Zol faced vertically, up/down if horizontally; RoomObjCount goes up by 1
;     (Zol_State2Split).
;   - Each child Gel first shows the spawn cloud (slot number + 12 frames, no collisions), then dashes
;     about 4 px/frame for 4 frames (less if blocked), snaps to the grid and wanders (Zol_CreateChildGel,
;     Gel_Move).
;   - If no object slot is free for the second Gel, Zol_CreateChildGel's writes go to slot 0: Link's
;     ObjState, ObjGridOffset and ObjDir are overwritten (Zol_CreateChildGel, Zol_State2Split).
;   - Walking speed: Zol q-speed $18 (0.375 px/frame), Gel $40 (1 px/frame). At each grid point they
;     pause: Zol $18/$28/$38/$48 frames, Gel $08/$18/$28/$38, chosen by Random,X; they start moving
;     when 4 frames of the pause are left (Zol_WalkAndPause, Zol_PauseTimes).
;   - A Gel placed by the room (type $15) starts in state 2, skipping the dash (Gel_Init).
; CONTENTS: (generated; L = line in this file)
;   L0057 Zol_Update                 code  Updates a Zol (type $13): runs its state, tests collisions (state 0 only) and draws it.
;   L0076 Zol_UpdateState            code  Runs the Zol's state handler: 0 wander, 1 slide after a hit, 2 split.
;   L0081 Zol_UpdateState_JumpTable  data  TABLE: Zol state handlers; index = ObjState.
;   L0087 Zol_State0Wander           code  Zol state 0: walks with pauses at q-speed $18 (0.375 px/frame).
;   L0098 Zol_State1Slide            code  Zol state 1: slides straight at q-speed $FF until a tile or the room boundary blocks it.
;   L0113 Zol_State2Split            code  Zol state 2: destroys the Zol and creates two child Gels facing opposite ways.
;   L0147 Zol_CreateChildGel         code  Creates one child Gel (type $14) at the Zol's position, in state 0.
;   L0170 Zol_CheckCollisions        code  Tests a wandering Zol against Link and his weapons and starts the split when it was hit.
;   L0235 Gel_Init                   code  Initializes a Gel placed by the room (type $15): state 2 (wander), then the walker init.
;   L0244 Gel_Update                 code  Updates a Gel (types $14, $15): moves, tests collisions in every state, and draws it.
;   L0281 Gel_Move                   code  Moves a Gel by state: 0 set up the dash, 1 dash away from the split, 2 wander with pauses.
;   L0350 Zol_WalkAndPause           code  Walks a Zol or Gel one grid square at a time as a wanderer, with a random pause at each square.
;   L0393 Zol_PauseTimes             data  TABLE: pause at a grid point; index = (Random,X AND 3), +4 for a Gel; value = frames in ObjTimer
;   L0403 Gel_MoveSplitting          code  Moves a struck Zol or a new child Gel straight ahead at q-speed $FF (about 4 px/frame).
;   L0446 Gel_CheckCollisions        code  Runs the common monster collision test (Link's weapons, then Link) for a Zol or Gel.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Zol (type $13): runs its state, tests collisions (state 0 only) and draws it.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Zol_Update:
    JSR Zol_UpdateState
    JSR Zol_CheckCollisions
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Every 8 screen frames, switch between frame image 0 and 1.
    LDY #$00
    LDA FrameCounter
    AND #$08
    BNE :+
    INY
:
    TYA
    JMP Anim_DrawObjectMirrored

; ----------------------------------------------------------------------------
; Runs the Zol's state handler: 0 wander, 1 slide after a hit, 2 split.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Zol_UpdateState:
    LDA ObjState, X
    JSR Util_TableJump

; TABLE: Zol state handlers; index = ObjState.
Zol_UpdateState_JumpTable:
    .ADDR Zol_State0Wander    ; 0: walk with pauses
    .ADDR Zol_State1Slide     ; 1: slide until blocked
    .ADDR Zol_State2Split     ; 2: turn into two child Gels

; Zol state 0: walks with pauses at q-speed $18 (0.375 px/frame).
Zol_State0Wander:
    LDA #$18                    ; Q-speed $18 (0.375 pixels a frame)
    JMP Zol_WalkAndPause

; ----------------------------------------------------------------------------
; Zol state 1: slides straight at q-speed $FF until a tile or the room boundary blocks it.
; IN:   X = object slot.
; NOTE: The direction is ObjDir,X, which the weapon hit set to the weapon's direction. No collision
;       test runs in this state. When blocked, the state becomes 2 (split on the next frame).
;       Also state 1 of the Vire (its state table points here).
; ----------------------------------------------------------------------------
Zol_State1Slide:
    JSR Gel_MoveSplitting
    BCC :+                      ; C = 0: moved, not blocked yet.
    INC ObjState, X
:
    RTS

; ----------------------------------------------------------------------------
; Zol state 2: destroys the Zol and creates two child Gels facing opposite ways.
; IN:   X = object slot.
; NOTE: Zol facing up or down: the Gels face left (2) and right (1). Zol facing left or right: up
;       (8) and down (4). The Zol's direction here is that of the weapon that hit it. The first Gel
;       always finds a slot (the Zol's own was just freed). If the second finds none, Y is 0 and
;       the direction below is written to Link's ObjDir; RoomObjCount was still raised by 1.
; ----------------------------------------------------------------------------
Zol_State2Split:
    ; One Zol becomes two Gels, so the room's object count goes up by 1.
    INC RoomObjCount
    JSR Obj_DestroyMonster          ; Keeps ObjX, ObjY, ObjDir and ObjGridOffset of the slot.
    JSR Zol_CreateChildGel

    ; If the Zol's direction is vertical (>= 4), then set the first child Gel's
    ; direction to left (2), else up (8).
    LDA ObjDir, X
    CMP #$04
    LDA #$02
    BCS :+
    LDA #$08
:
    STA a:ObjDir, Y
    PHA                         ; Save the direction of the first Gel.
    JSR Zol_CreateChildGel
    PLA                         ; Restore the direction of the first Gel.

    ; Set the second child Gel's direction to the opposite of the first one's
    ; (2 -> 1, 8 -> 4).
    LSR
    STA a:ObjDir, Y
    RTS

; ----------------------------------------------------------------------------
; Creates one child Gel (type $14) at the Zol's position, in state 0.
; IN:   X = the Zol's object slot (already destroyed).
; OUT:  Y = the Gel's object slot, or 0 if no slot in $01-$0B was free.
; NOTE: The result of Shooter_ShootLimited is not tested. On failure Y = 0 and the two writes below change
;       Link's ObjState (to 0) and ObjGridOffset. The new slot is left uninitialized with the spawn
;       cloud metastate, so the Gel is initialized (Walker_Init) and shows the cloud before it moves.
;       Type $14 is below $53, so it is not counted as a shot and has no limit of 4.
; ----------------------------------------------------------------------------
Zol_CreateChildGel:
    LDA #$14                    ; Child Gel object type

    ; Use the shooting operation to create the Gel object.
    STA $00
    JSR Shooter_ShootLimited

    ; Shots start in state $10. But a child Gel needs to start in state 0.
    LDA #$00
    STA a:ObjState, Y

    ; Give the child the same grid offset as the parent.
    LDA ObjGridOffset, X
    STA ObjGridOffset, Y
    RTS

; ----------------------------------------------------------------------------
; Tests a wandering Zol against Link and his weapons and starts the split when it was hit.
; IN:   X = object slot.
; NOTE: Runs only in state 0. After a hit that neither killed the Zol nor left its invincibility
;       timer at 0, the state becomes 1 (slide, then split) or 2 (split now). The alignment test
;       uses Link's facing direction (ObjDir of slot 0), not the Zol's.
; ----------------------------------------------------------------------------
Zol_CheckCollisions:
    ; If not in state 0, then return.
    LDA ObjState, X
    BNE @Exit

    ; Check collisions.
    ;
    ; If the Zol is dying (metastate <> 0), or was not shoved by a hit
    ; (invincibility timer = 0: no hit, or a boomerang hit), then return.
    JSR Gel_CheckCollisions
    LDA ObjMetastate, X
    BNE @Exit
    LDA ObjInvincibilityTimer, X
    BEQ @Exit

    ; If vertically aligned with the square grid, then
    ; use a horizontal direction mask (3).
    ;
    ; Objects are aligned vertically when the low nibble of their Y is $D.
    LDY #$00
    LDA ObjY, X
    AND #$0F
    CMP #$0D
    BNE :+
    LDY #$03
:
    STY $00

    ; If horizontally aligned with the square grid, then
    ; use a vertical direction mask ($C).
    LDY #$00
    LDA ObjX, X
    AND #$0F
    BNE :+
    LDY #$0C
:
    TYA

    ; Combine the two masks: the directions in which the Zol can slide along the grid.
    ORA $00

    ; Test the mask against Link's facing direction (slot 0; this operand has no X index).
    ;
    ; If Link faces a direction in the mask, then go to state 1 to slide.
    ; Else go to state 2, to split in two at once.
    ;
    ; Example: a Zol walking left or right between grid columns has mask 3. Struck by Link
    ; facing left or right it slides; struck by Link facing up or down it splits where it is.
    ;
    ; The weapon hit has already set the Zol's own direction to the weapon's direction
    ; (Damage_HandleWeaponHit); the slide in state 1 uses that.
    AND ObjDir
    BNE :+
    INC ObjState, X
:
    INC ObjState, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Initializes a Gel placed by the room (type $15): state 2 (wander), then the walker init.
; IN:   X = object slot.
; NOTE: Child Gels (type $14) use Walker_Init alone and so begin in state 0.
; ----------------------------------------------------------------------------
Gel_Init:
    LDA #$02
    STA ObjState, X
    JMP Walker_Init

; ----------------------------------------------------------------------------
; Updates a Gel (types $14, $15): moves, tests collisions in every state, and draws it.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Gel_Update:
    JSR Gel_Move
    JSR Gel_CheckCollisions

    ; Gels are narrow. So, for drawing, temporarily shift the X coordinate
    ; 4 pixels to the right.
    LDA ObjX, X
    PHA                         ; Save the original X coordinate.
    CLC
    ADC #$04
    STA ObjX, X
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Every two screen frames, switch between frame image 0 and 1.
    LDY #$00
    LDA FrameCounter
    AND #$02
    BNE :+
    INY
:
    ; Set sprite attributes 3: normal sprite, sprite palette 3.
    LDA #$03
    JSR Anim_SetSpriteDescriptorAttributes
    TYA                         ; Pass the frame image in A register.
    JSR Anim_DrawObjectNotMirrored
    PLA                         ; Restore the original X coordinate.
    STA ObjX, X
    RTS

; ----------------------------------------------------------------------------
; Moves a Gel by state: 0 set up the dash, 1 dash away from the split, 2 wander with pauses.
; IN:   X = object slot.
; NOTE: State 0 lasts one frame and sets ObjTimer to 5. State 1 moves at q-speed $FF
;       (Gel_MoveSplitting) on the 4 frames the timer reads 4 to 1, unless blocked; when it reads 0
;       or the Gel is blocked, the Gel is snapped to the nearest grid square and goes to state 2.
;       State 2 falls into Zol_WalkAndPause with q-speed $40 (1 px/frame).
; ----------------------------------------------------------------------------
Gel_Move:
    ; Go handle state 0 or 2 as appropriate.
    LDY ObjState, X
    BEQ @State0
    DEY
    BNE @State2

    ; State 1.
    ;
    ; If the timer has not expired, then move.
    LDA ObjTimer, X
    BEQ @SnapToGrid
    JSR Gel_MoveSplitting

    ; If not blocked while moving, then return.
    BCC @Exit

@SnapToGrid:
    ; Blocked or timer expired: snap to the grid.
    ;
    ; Set X coordinate to ((X + 8) AND $F0).
    LDA ObjX, X
    CLC
    ADC #$08
    AND #$F0
    STA ObjX, X

    ; Set Y coordinate to (((Y + 8) AND $F0) OR $D).
    ; The carry of the X addition above is not cleared first.
    LDA ObjY, X
    ADC #$08
    AND #$F0
    ORA #$0D
    STA ObjY, X

    ; Reset the grid offset, and go to state 2.
    LDA #$00
    STA ObjGridOffset, X
    INC ObjState, X

@Exit:
    RTS

@State0:
    ; State 0.
    ;
    ; Go to state 1 with a 5 frame timer. The q-speed $20 stored here is
    ; replaced with $FF by Gel_MoveSplitting before any movement.
    LDA #$20
    STA ObjQSpeedFrac, X
    LDA #$05
    STA ObjTimer, X
    INC ObjState, X
    RTS

@State2:
    ; State 2.
    ;
    ; Set q-speed $40 (1 pixel a frame), and fall through.
    LDA #$40

; ----------------------------------------------------------------------------
; Walks a Zol or Gel one grid square at a time as a wanderer, with a random pause at each square.
; IN:   A = q-speed ($18 Zol, $40 Gel); X = object slot.
; NOTE: While ObjTimer >= 5 the monster stands still. Below 5 it moves as a wanderer with turn rate
;       $20. Each time it is on a grid point with the timer at 0, a new pause is loaded from
;       Zol_PauseTimes by Random,X AND 3; the last 4 frames of the pause are already movement. Does
;       not handle knockback; the clock and stun stop the movement inside Walker_Move.
; ----------------------------------------------------------------------------
Zol_WalkAndPause:
    STA ObjQSpeedFrac, X

    ; If object timer >= 5, then return: still pausing.
    LDA ObjTimer, X
    CMP #$05
    BCS @Exit
    LDA #$20                    ; Turn rate $20
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_MoveAndTurn

    ; If the monster is between squares, or timer <> 0; then return.
    LDA ObjGridOffset, X
    ORA ObjTimer, X
    BNE @Exit

    ; Choose a random index (0 to 3) to look up an amount of time.
    LDA Random, X
    AND #$03
    TAY

    ; If the monster is Zol ($13), then go use this index as is.
    LDA ObjType, X
    CMP #$13
    BEQ @SetDelay

    ; Else it's a Gel. Add 4 to the index to look in the set of times
    ; for Gel.
    INY
    INY
    INY
    INY

@SetDelay:
    ; Set the pause before the next square.
    LDA Zol_PauseTimes, Y
    STA ObjTimer, X

@Exit:
    RTS

; TABLE: pause at a grid point; index = (Random,X AND 3), +4 for a Gel; value = frames in ObjTimer
; (movement resumes when 4 are left).
Zol_PauseTimes:
    .BYTE $18, $28, $38, $48, $08, $18, $28, $38    ; Zol: $18-$48; Gel: $08-$38

; ----------------------------------------------------------------------------
; Moves a struck Zol or a new child Gel straight ahead at q-speed $FF (about 4 px/frame).
; IN:   X = object slot; ObjDir,X = direction.
; OUT:  C = 1 if blocked (nothing moved), 0 if moved.
; NOTE: The tile ahead is tested only when the grid offset is 0 (unwalkable = at or above
;       RoomFirstUnwalkableTile); the room boundary is tested every frame.
; ----------------------------------------------------------------------------
Gel_MoveSplitting:
    LDA #$FF
    STA ObjQSpeedFrac, X

    ; Set moving direction [0F] to facing direction.
    LDA ObjDir, X
    STA $0F

    ; If at a grid point, then check for a tile collision.
    LDA ObjGridOffset, X
    BNE @CheckRoomBoundary
    JSR Tile_GetAheadOfObj

    ; If blocked by a tile, then return C=1 (set by the compare).
    CMP RoomFirstUnwalkableTile
    BCS @Exit

@CheckRoomBoundary:
    ; If blocked by the room boundary, then return C=1.
    JSR Room_BoundDir
    SEC
    BEQ @Exit

    ; Move.
    JSR Obj_MoveByQSpeed

    ; If the grid offset has reached a multiple of $10 (a square length), reset it to 0.
    ; Return C=0.
    LDA ObjGridOffset, X
    AND #$0F
    CLC
    BNE @Exit
    STA ObjGridOffset, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Runs the common monster collision test (Link's weapons, then Link) for a Zol or Gel.
; IN:   X = object slot.
; NOTE: The [02]/[03] values stored here are overwritten at once: Collision_CheckMonsterVsWeaponsAndLink begins
;       with Obj_GetMiddle, which sets both.
; ----------------------------------------------------------------------------
Gel_CheckCollisions:
    LDA ObjX, X
    STA $02
    LDA ObjY, X
    STA $03
    JMP Collision_CheckMonsterVsWeaponsAndLink
