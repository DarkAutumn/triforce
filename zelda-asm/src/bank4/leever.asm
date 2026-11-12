; ============================================================================
; FILE:     bank4/leever.asm             BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 2592-2966
; SUBJECTS: Burrower
; TOPIC:    Blue Leever (type $0F) and Red Leever (type $10) updates, and the six-state burrower cycle
;           (under ground, emerging, out, sinking) that the Blue Leever shares with the Zora (type $11).
; SEE ALSO: bank4/zora-ow-inits.asm (InitLeever; CheckZora and UpdateZora, which calls UpdateBurrower);
;           bank4/walker-wanderer-common.asm (Wanderer_TargetPlayer: the Blue Leever's steering);
;           bank7/walker-move.asm (Walker_Move, MoveObject); bank1/monster-collisions.asm
;           (CheckMonsterCollisions); bank7/object-update-table.asm (UpdateObject_JumpTable $0F-$11).
; KEY FACTS:
;   - Burrower states 0-5: 0 under ground (not drawn), 1 and 5 mound, 2 and 4 half out, 3 fully out.
;     Each lasts the frames in BlueLeeverStateTimes ($80, $20, $0F, $FF, $10, $60: 542 in all), then the
;     next begins (UpdateBurrower@CycleState).
;   - A Leever can hurt Link and be hit only in state 3; a Zora in states 2, 3 and 4
;     (Burrower_AnimateDrawAndCheckCollisions).
;   - Zora cycle: mound $20 frames (state 1), body up for states 2-4 ($0F frames, about $22, $10; in
;     state 3 it shoots a fireball 2 frames in and its timer is cut to $20), mound $60 (state 5); then
;     the object is destroyed and a new Zora is made on a random water square, with full HP
;     (UpdateBurrower, and UpdateZora, CheckZora in bank4/zora-ow-inits.asm).
;   - The Blue Leever moves in every state, under ground too, steering toward the chase target with turn
;     rate $A0; q-speed per state $08, $0A, $10, $20, $10, $0A = 0.125 to 0.5 px/frame
;     (UpdateBlueLeever, BlueLeeverStateQSpeeds).
;   - The clock or a stun stops a Blue Leever's movement but not its state timers: it still sinks and
;     comes back (UpdateBlueLeever does not test them; Walker_Move does).
;   - A Red Leever waits in state 0 until RedLeeverLongTimer is 0 and fewer than 2 Red Leevers are out,
;     then surfaces $28 pixels from Link: ahead of him in the direction he faces (3 in 4), else behind
;     him (UpdateRedLeever).
;   - It refuses to surface above Y = $5D, on an unwalkable tile, or when the X would wrap around the
;     screen; it tries again every frame (UpdateRedLeever@CheckSafeToSpawn).
;   - A Red Leever emerges for $18 frames, then runs straight at 0.5 px/frame (q-speed $20) toward where
;     Link was, for as long as nothing blocks it: it sinks ($18 frames) only at an unwalkable tile or the
;     room bound (UpdateRedLeever@CheckStunned, RedLeeverStateTimes).
;   - Each surfacing sets RedLeeverLongTimer to 2, so the next Red Leever surfaces 11-20 frames later at
;     the earliest (the timer loses 1 every 10th frame). The clock is not tested in states 0, 1, 2, 4
;     and 5: Red Leevers still surface with it, then stay out in state 3
;     (UpdateRedLeever@CheckSafeToSpawn, UpdateRedLeever@CheckStunned).
;   - ActiveRedLeeverCount goes down when a Red Leever finishes its cycle or dies
;     (RedLeever_CycleStateDrawAndCheckCollisions, Burrower_AnimateDrawAndCheckCollisions).
; CONTENTS: (generated; L = line in this file)
;   L0057 BlueLeeverStateQSpeeds                      data  TABLE: q-speed of a Blue Leever or Zora in each burrower state; index = ObjState (0-5); value =
;   L0063 BlueLeeverStateTimes                        data  TABLE: length of each burrower state of a Blue Leever or Zora; index = ObjState (0-5); value = frames
;   L0069 BlueLeeverStateAnimTimes                    data  TABLE: frames per animation frame of a Blue Leever or Zora; index = ObjState (0-5); value = frames
;   L0079 UpdateBlueLeever                            code  Updates a Blue Leever (type $0F): walks toward the chase target, then runs the burrower cycle.
;   L0094 UpdateBurrower                              code  Runs the six-state burrower cycle of a Blue Leever or Zora, then draws it and checks collisions.
;   L0152 Burrower_AnimateDrawAndCheckCollisions      code  Animates and draws a burrower (Leever or Zora) and checks its collisions in the states where it is out.
;   L0222 RedLeeverStateQSpeeds                       data  TABLE: q-speed of a Red Leever in each state; index = ObjState (0-5); value = q-speed (pixels per
;   L0227 RedLeeverStateTimes                         data  TABLE: length of each Red Leever state; index = ObjState (0-5); value = frames (ObjTimer value set on
;   L0235 RedLeeverStateAnimTimes                     data  TABLE: frames per animation frame of a Red Leever; index = ObjState (0-5); value = frames (reload value
;   L0252 UpdateRedLeever                             code  Updates a Red Leever (type $10): surfaces near Link, runs in a straight line, sinks when blocked.
;   L0447 RedLeever_CycleStateDrawAndCheckCollisions  code  Advances a Red Leever to its next state (5 wraps to 0 and frees its place), then draws it.
;   L0470 RedLeever_AnimateAndCheckCollisions         code  Animates and draws a Red Leever with its state's animation time; collisions only in state 3.
; ============================================================================

; TABLE: q-speed of a Blue Leever or Zora in each burrower state; index = ObjState (0-5); value =
; q-speed (pixels per frame = value / 64). The Zora gets the value too but is never moved.
BlueLeeverStateQSpeeds:
    ; 0 under ground 0.125, 1 mound 0.156, 2 half out 0.25, 3 out 0.5, 4 half out 0.25, 5 mound 0.156
    .BYTE $08, $0A, $10, $20, $10, $0A

; TABLE: length of each burrower state of a Blue Leever or Zora; index = ObjState (0-5); value = frames
; (the ObjTimer value set on entering the state).
BlueLeeverStateTimes:
    ; 0 under ground 128, 1 mound 32, 2 half out 15, 3 out 255, 4 half out 16, 5 mound 96
    .BYTE $80, $20, $0F, $FF, $10, $60

; TABLE: frames per animation frame of a Blue Leever or Zora; index = ObjState (0-5); value = frames
; (reload value of ObjAnimCounter).
BlueLeeverStateAnimTimes:
    .BYTE $10, $0B, $01, $05, $01, $0B

; ----------------------------------------------------------------------------
; Updates a Blue Leever (type $0F): walks toward the chase target, then runs the burrower cycle.
; IN:   X = object slot.
; NOTE: Turn rate $A0 (Wanderer_TargetPlayer). It moves in all six states, at the state's q-speed.
;       The clock and stun are not tested here; Walker_Move stops the movement for them, but the state
;       timers go on, so a frozen Blue Leever still sinks and surfaces. Falls into UpdateBurrower.
; ----------------------------------------------------------------------------
UpdateBlueLeever:
    LDA #$A0                    ; Turn rate $A0
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_TargetPlayer

; ----------------------------------------------------------------------------
; Runs the six-state burrower cycle of a Blue Leever or Zora, then draws it and checks collisions.
; IN:   X = object slot.
; NOTE: When ObjTimer is 0 the state advances (5 wraps to 0) and the state's q-speed and timer are
;       loaded from BlueLeeverStateQSpeeds and BlueLeeverStateTimes. States: 0 under ground, 1 mound,
;       2 half out, 3 out, 4 half out, 5 mound. A new Zora has state 0 and timer 0, so it goes to state 1
;       on its first update; a new Blue Leever stays in state 0 for (slot + 1) * $10 frames first
;       (InitLeever). A Zora leaving state 1 gets its body frame image stored in ObjDir:
;       3 (back) when its Y >= Link's Y, else 2 (front).
; ----------------------------------------------------------------------------
UpdateBurrower:
    ; If object timer <> 0, then go animate, draw, and check for collisions.
    LDA ObjTimer, X
    BNE @Animate

    ; If not zora or state <> 1, go cycle the state.
    LDA ObjType, X
    CMP #$11
    BNE @CycleState
    LDY ObjState, X
    DEY
    BNE @CycleState

    ; Else is zora and state = 1.
    ;
    ; If zora's Y >= Link's Y, then set zora's frame image to
    ; 3 (back), else 2 (front).
    ;
    ; The frame image number is kept in ObjDir, which a Zora does not use as a direction.
    LDY #$03
    LDA ObjY, X
    CMP ObjY
    BCS :+
    DEY
:
    STY ObjDir, X

@CycleState:
    ; Cycle the state between 0 and 5.
    LDA ObjState, X
    CLC
    ADC #$01
    CMP #$06
    BCC :+
    LDA #$00
:
    STA ObjState, X

    ; Look up and set the speed and timer for the new state.
    TAY
    LDA BlueLeeverStateQSpeeds, Y
    STA ObjQSpeedFrac, X
    LDA BlueLeeverStateTimes, Y
    STA ObjTimer, X

@Animate:
    ; Advance the animation counter, reloading it with the state's value when it runs out.
    LDY ObjState, X
    LDA BlueLeeverStateAnimTimes, Y

; ----------------------------------------------------------------------------
; Animates and draws a burrower (Leever or Zora) and checks its collisions in the states where it is out.
; IN:   X = object slot; A = reload value for ObjAnimCounter when it runs out (frames).
; NOTE: State 0 draws nothing. Leevers: frame image (state - 1) * 2 + ObjAnimFrame; collisions with
;       Link and his weapons only in state 3. Zora: mound images in states 1 and 5, the body image kept
;       in ObjDir in states 2-4; collisions in states 2, 3 and 4. A Red Leever (type $10) that starts
;       dying here decrements ActiveRedLeeverCount.
; ----------------------------------------------------------------------------
Burrower_AnimateDrawAndCheckCollisions:
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; If state = 0 (under ground), return.
    LDA ObjState, X
    BEQ @Exit

    ; If is zora, and state >= 2 and < 5, then use frame image 2 or 3
    ; (front or back), which was stored in ObjDir by UpdateBurrower.
    LDY ObjType, X
    CPY #$11
    BNE @CalcFrameImage
    LDY ObjState, X
    CPY #$02
    BCC @CalcFrameImage
    CPY #$05
    BCS @CalcFrameImage
    LDA ObjDir, X
    BNE @Draw

@CalcFrameImage:
    ; A = state here. Zora in state 1 or 5, and Leevers in states 1 to 5:
    ;
    ; frame image := ((state - 1) * 2) + animation frame
    ;
    ; For the Zora this yields mound images 0/1 or 8/9.
    SEC
    SBC #$01
    ASL
    CLC
    ADC ObjAnimFrame, X

@Draw:
    JSR DrawObjectMirrored

    ; If not zora and state <> 3,
    ; or zora and state <> 2, 3, 4;
    ; then return without checking collisions.
    LDA ObjType, X
    CMP #$11
    BNE @NoState3Collisions
    LDA ObjState, X
    CMP #$02
    BEQ @CheckCollisions
    CMP #$04
    BEQ @CheckCollisions

@NoState3Collisions:
    LDA ObjState, X
    CMP #$03
    BNE @Exit

@CheckCollisions:
    JSR CheckMonsterCollisions

    ; If not dying (metastate 0), then return.
    LDA ObjMetastate, X
    BEQ @Exit

    ; If the monster is a red leever, then decrement the red leever count.
    LDA ObjType, X
    CMP #$10
    BNE @Exit
    DEC ActiveRedLeeverCount

@Exit:
    RTS

; TABLE: q-speed of a Red Leever in each state; index = ObjState (0-5); value = q-speed (pixels per
; frame = value / 64). It moves only in state 3: $20 = 0.5 px/frame.
RedLeeverStateQSpeeds:
    .BYTE $00, $00, $00, $20, $00, $00

; TABLE: length of each Red Leever state; index = ObjState (0-5); value = frames (ObjTimer value set on
; entering the state).
RedLeeverStateTimes:
    ; 0 under ground: no timer (waits for RedLeeverLongTimer). 1 mound 16, 2 half out 8.
    ; 3 out: $FF, rewritten every frame it moves, so the state ends only when it is blocked.
    ; 4 half out 8, 5 mound 16.
    .BYTE $00, $10, $08, $FF, $08, $10

; TABLE: frames per animation frame of a Red Leever; index = ObjState (0-5); value = frames (reload value
; of ObjAnimCounter).
RedLeeverStateAnimTimes:
    .BYTE $10, $08, $08, $05, $08, $08

; ----------------------------------------------------------------------------
; Updates a Red Leever (type $10): surfaces near Link, runs in a straight line, sinks when blocked.
; IN:   X = object slot.
; NOTE: State 0 (under ground, not drawn): waits until RedLeeverLongTimer is 0 and fewer than 2 Red
;       Leevers are out (ActiveRedLeeverCount). It then takes Link's facing direction, reversed when
;       Random >= $C0 (1 in 4), and places itself $28 pixels from Link along it; after surfacing it turns
;       around, so it runs toward where Link stood. No surfacing at Y < $5D, on an unwalkable tile, or
;       when the X wrapped around the screen; it retries each frame. A surfacing sets RedLeeverLongTimer
;       to 2 (10-frame units: 11-20 frames). The ObjTimer value the init leaves is not read in state 0.
;       States 1, 2: emerging, $10 + 8 frames, no movement, no collisions. State 3: moves at q-speed $20
;       (0.5 px/frame) until an unwalkable tile or the room bound is ahead; shoving, stun and the clock
;       hold it in state 3. States 4, 5: sinking, 8 + $10 frames, then state 0 and the count drops.
;       Only state 3 tests the clock and stun: the other states run on with them.
; ----------------------------------------------------------------------------
UpdateRedLeever:
    LDA ObjState, X
    BEQ :+
    JMP @CheckOtherStates
:
    ; State 0. Under ground: decide whether to surface.
    ;
    ; If the long timer has not expired, then return.
    LDA RedLeeverLongTimer
    BNE @Exit

    ; Only allow two red leevers out at a time.
    LDA ActiveRedLeeverCount
    CMP #$02
    BCS @Exit

    ; Face in the same direction as Link: this puts the leever ahead of him.
    LDA ObjDir
    STA ObjDir, X

    ; Randomly (Random >= $C0, 1 in 4) face the opposite direction: this puts it behind him.
    LDA Random+1, X
    CMP #$C0
    BCC :+
    JSR ReverseObjDir
:
    ; If facing horizontally, go place the leever to the left or right of Link.
    LDA ObjDir, X
    AND #$0C
    BEQ @PlaceHorizontally

    ; Begin with the leever's X set to Link's X.
    LDA ObjX
    STA ObjX, X

    ; If facing down, then store Y offset $28 in [00].
    ; Else (up) store Y offset -$28.
    LDY #$28
    LDA ObjDir, X
    AND #$08
    BEQ :+
    LDY #$D8
:
    STY $00

    ; Add Link's Y and the vertical offset.
    LDA ObjY
    CLC
    ADC $00

    ; Align the Y coordinate to a square row:
    ; (Y AND $F0) OR $D
    AND #$F0
    ORA #$0D
    STA ObjY, X

    ; If Y coordinate >= $5D (top of the third square row), then
    ; go see if the tile is valid to come out of.
    ;
    ; Else it's too close to the top edge of the screen. So, return.
    CMP #$5D
    BCS @CheckSafeToSpawn

@Exit:
    RTS

@PlaceHorizontally:
    ; Begin with the leever's Y set to Link's Y (not aligned).
    LDA ObjY
    STA ObjY, X

    ; If Y coordinate < $5D (top of the third square row), then
    ; it's too close to the top edge of the screen. So, return.
    CMP #$5D
    BCC @Exit

    ; If facing right, then store X offset $28 in [00].
    ; Else (left) store X offset -$28.
    LDY #$28
    LDA ObjDir, X
    AND #$02
    BEQ :+
    LDY #$D8
:
    STY $00

    ; [01] holds Link's X coordinate.
    LDA ObjX
    STA $01

    ; Add Link's X and the horizontal offset.
    CLC
    ADC $00

    ; AND the result with $F8, so it's aligned with a tile horizontally.
    AND #$F8

    ; Assign that result to the leever's X coordinate and [02].
    STA ObjX, X
    STA $02

    ; If the leever's X [02] >= Link's X [01], then swap [02] and [01].
    CMP $01
    BCC @Subtract
    PHA
    LDA $01
    STA $02
    PLA
    STA $01

@Subtract:
    ; [01] - [02] is now the absolute distance between the two.
    LDA $01
    SEC
    SBC $02

    ; If the absolute distance >= $30, then return.
    ; It means that the leever's X coordinate wrapped around
    ; the left or right edge of the screen.
    CMP #$30
    BCS @Exit

@CheckSafeToSpawn:
    ; The leever can't come out of an unwalkable tile. So, return.
    JSR GetCollidableTileStill
    CMP RoomFirstUnwalkableTile
    BCS @Exit
    INC ActiveRedLeeverCount

    ; Set animation counter to 1, so that it will roll over this frame
    ; and pick up the reload value for state 1.
    LDA #$01
    STA ObjAnimCounter, X

    ; Set the long timer to 2 (10-frame units): the wait before the next red leever may surface.
    ASL
    STA RedLeeverLongTimer
    ; State 0 -> 1. This call also draws the leever; the code below draws it a second time this frame.
    JSR RedLeever_CycleStateDrawAndCheckCollisions

    ; Turn around, so that the leever faces back toward where Link is.
    JSR ReverseObjDir

@CheckOtherStates:
    ; If not in state 3, go check the object timer in order to draw and such.
    LDA ObjState, X
    CMP #$03
    BNE @AnimateIfTime

    ; State 3. Out of the ground and moving.
    ;
    ; If being shoved, go shove, animate, and draw the leever.
    LDA ObjShoveDir, X
    BEQ @CheckStunned
    JSR Obj_Shove
    JMP RedLeever_AnimateAndCheckCollisions

@CheckStunned:
    ; If Link has the clock or the leever is stunned, then
    ; only animate, draw, and check for collisions. The state does not advance.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE RedLeever_AnimateAndCheckCollisions

    ; If the leever is blocked by an unwalkable tile or the room boundary,
    ; then go to the next state (4, sinking), in addition to animating, drawing,
    ; and checking for collisions.
    LDA ObjDir, X
    STA $0F
    JSR GetCollidingTileMoving
    CMP RoomFirstUnwalkableTile
    BCS RedLeever_CycleStateDrawAndCheckCollisions
    JSR BoundByRoom
    BEQ RedLeever_CycleStateDrawAndCheckCollisions

    ; Move in direction [0F] at the state's q-speed.
    JSR MoveObject

    ; If the low nibble of the grid offset is 0 (a whole square was crossed), reset the grid offset.
    LDA ObjGridOffset, X
    AND #$0F
    BNE :+
    STA ObjGridOffset, X
:
    ; Set timer to $FF again, so that state 3 does not time out while the leever moves.
    LDA #$FF
    STA ObjTimer, X

@AnimateIfTime:
    ; If object timer has not expired, then
    ; go animate, draw, and check for collisions.
    LDA ObjTimer, X
    BNE RedLeever_AnimateAndCheckCollisions

; Advances a Red Leever to its next state (5 wraps to 0 and frees its place), then draws it.
RedLeever_CycleStateDrawAndCheckCollisions:
    ; Cycle the state between 0 and 5.
    ;
    ; If a cycle was completed (wrapped around to 0),
    ; then decrease the active red leever count.
    LDA ObjState, X
    CLC
    ADC #$01
    CMP #$06
    BCC :+
    DEC ActiveRedLeeverCount
    LDA #$00
:
    STA ObjState, X

    ; Look up and set the speed and timer for the new state.
    TAY
    LDA RedLeeverStateQSpeeds, Y
    STA ObjQSpeedFrac, X
    LDA RedLeeverStateTimes, Y
    STA ObjTimer, X

; Animates and draws a Red Leever with its state's animation time; collisions only in state 3.
RedLeever_AnimateAndCheckCollisions:
    LDY ObjState, X
    LDA RedLeeverStateAnimTimes, Y
    JMP Burrower_AnimateDrawAndCheckCollisions
    ; Unreachable: follows an unconditional JMP and has no label.
    RTS
