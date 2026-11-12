; ============================================================================
; FILE:     bank4/fairy-flyer-common.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 11447-11868
; SUBJECTS: Dir, Fairy, Flyer, Moldorm
; TOPIC:    The fairy item (item id $23): init, flight and drawing. The flight code shared by all 8-way
;           flyers: speed-up / slow-down / chase / wander / delay states, the pixel mover, the room-boundary
;           bounce, and the 8-way turn helpers (toward the chase target, at random, reverse).
; SEE ALSO: bank4/drop-items.asm (Drop_SetUpItem and Drop_UpdateItem create and run the fairy item);
;           bank4/bubble-keese.asm, bank4/flying-ghini-peahat.asm, bank4/gleeok.asm, bank4/patra.asm,
;           bank4/moldorm.asm (each has its own flight table and "decide" state using the states here);
;           bank4/digdogger.asm, bank4/manhandla.asm, bank4/tektite-boulder.asm, bank4/gohma.asm (use the
;           turn and bounce helpers); bank4/pond-fairy.asm (uses Fairy_Draw);
;           bank1/room-bounds.asm (Room_BoundDirHorizontally, Room_BoundDirVertically).
; KEY FACTS:
;   - Flyer speed is Flyer_ObjSpeed AND $E0, in 1/256 pixel per frame; 1 pixel is moved on each carry of
;     Flyer_ObjPosFrac. A diagonal direction moves 1 pixel on both axes on the same carry. Flyer_Move
;   - Flight states used by every flight table: 0 speed up, 1 decide (per type), 2 chase, 3 wander,
;     4 slow down, 5 delay. Flyer_State0SpeedUp, Flyer_State2Chase, Flyer_State3Wander, Flyer_State4SlowDown, Flyer_State5Delay
;   - Speed changes by 1 per frame. Speed-up ends (state 1) when (speed AND $E0) >= FlyerMaxSpeedFrac;
;     slow-down ends when speed drops below $20: ObjTimer = $40-$7F from Random,X, then state 5, then
;     state 0 when the timer runs out. Flyer_State0SpeedUp, Flyer_State4SlowDown, Flyer_State5Delay
;   - Chase and wander make one turn of at most 45 degrees every $10 frames, Flyer_ObjTurns - 1 turns in
;     all, then return to state 1. Flyer_State2Chase, Flyer_State3Wander
;   - A chase turn steers toward ChaseTargetX/Y, which is not always Link; when no adjacent direction
;     passes the test in Dir_TurnObjTowardTarget8_TestCandidate the flyer turns one step clockwise, even away from the target.
;     Dir_TurnObjTowardTarget8
;   - A wander turn reads Random+1,X: < $50 one step counter-clockwise, $50-$9F one step clockwise,
;     >= $A0 no turn. Dir_TurnObjRandomly8
;   - After each pixel moved the flyer is tested against the room bounds and its direction is reversed
;     (180 degrees) when it crossed one. Boulders (type $20) skip the vertical test. Moldorm (type $41)
;     does not reverse at once: its heads (slots $05, $0A) store the direction in Moldorm_ObjBounceDir.
;     Flyer_BoundByRoom, Dir_ReverseObjDir8, Moldorm_DeferBounce
;   - The fairy item starts facing up at speed $7F with top speed $A0: it moves at 0.5 px/frame for $20
;     frames, then at 0.625, and from then on only wanders: 5 random turns, $10 frames apart, per cycle.
;     Its init overwrites FlyerMaxSpeedFrac, the one top speed of all flyers in the room.
;     Fairy_Init, Fairy_DecideFlightState
;   - Dir_EightWayDirs lists the 8 directions clockwise from up; index + 1 = 45 degrees clockwise.
; CONTENTS: (generated; L = line in this file)
;   L0082 Fairy_Init                               code  Turns a new item object into a flying fairy: jingle, facing up, speed $7F, top speed $A0.
;   L0107 Flyer_ResetState                         code  Zeroes a flyer's position fraction, turn count, distance, flight state and invincibility timer.
;   L0121 Fairy_Update                             code  Runs the fairy item for one frame: flight state, movement, then drawing.
;   L0130 Fairy_Draw                               code  Draws a fairy at the object's position, flapping between two frames every 4 frames.
;   L0148 Fairy_ControlFlight                      code  Dispatches the fairy item's flight state (Flyer_ObjFlyingState,X: 0-3).
;   L0153 Fairy_ControlFlight_JumpTable            data  TABLE: fairy item flight handlers; index = Flyer_ObjFlyingState; value = routine address.
;   L0165 Fairy_DecideFlightState                  code  Fairy flight state 1: go to state 3 (wander) with the turn counter at 6.
;   L0172 Fairy_DoNothingInFlightState2            code  Fairy flight state 2: returns at once. Also the RTS of Fairy_DecideFlightState.
;   L0177 Dir_EightWayDirs                         data  TABLE: the 8 directions in clockwise order; index = 8-way direction index 0-7 (+1 = 45 degrees
;   L0186 Flyer_State5Delay                        code  Flight state 5: wait for ObjTimer,X to run out, then go to state 0 (speed up).
;   L0200 Flyer_State4SlowDown                     code  Flight state 4: lose 1 speed unit per frame; below $20 start a random delay and go to state 5.
;   L0210 Flyer_State0SpeedUp                      code  Flight state 0: gain 1 speed unit per frame; at the room's top speed go to state 1 (decide).
;   L0228 Flyer_SetFlyingState                     code  Stores A as the flight state of slot X and returns. Tail of the speed states; also jumped to.
;   L0234 Flyer_CompareMaxSpeed                    code  Tail of Flyer_State0SpeedUp and Flyer_State4SlowDown: A = speed AND $E0 (nonzero); state 1 once A >= top sp...
;   L0250 Flyer_Move                               code  Moves a flyer 1 pixel along its 8-way direction whenever its position fraction overflows.
;   L0309 Flyer_BoundByRoom                        code  Tests an 8-way mover against the room bounds and reverses its direction if it crossed one.
;   L0330 Dir_ReverseObjDir8                       code  Turns an 8-way mover around (180 degrees); a Moldorm head defers the turn instead.
;   L0349 Flyer_BoundByRoom_Return                 code  Shared exit of Flyer_BoundByRoom and Dir_ReverseObjDir8.
;   L0358 Moldorm_DeferBounce                      code  Moldorm part of Dir_ReverseObjDir8: a head stores the reversed direction for its next turn.
;   L0368 Moldorm_DeferBounce_Return               code  Shared exit of Moldorm_DeferBounce and Flyer_State2Chase.
;   L0377 Flyer_State2Chase                        code  Flight state 2: every $10 frames turn one step toward the chase target; then state 1.
;   L0386 Flyer_GoToState1Decide                   code  Sets flight state 1 (decide) for slot X. Shared by Flyer_State2Chase and Flyer_State3Wander.
;   L0391 Flyer_State2Chase_DelayAndTurn           code  Part of Flyer_State2Chase: starts the $10-frame delay, then falls into the turn.
;   L0407 Dir_TurnObjTowardTarget8                 code  Turns an 8-way mover at most one 45-degree step, toward the chase target when a test allows it.
;   L0466 Dir_TurnObjTowardTarget8_ScanCandidates  code  Candidate loop of Dir_TurnObjTowardTarget8 (step 2).
;   L0475 Dir_TurnObjTowardTarget8_NextCandidate   code  Advances to the next candidate (one step clockwise); after the third, turns clockwise.
;   L0484 Dir_SetObjDir8FromIndex                  code  Sets ObjDir,X to Dir_EightWayDirs[Y]. Also the tail of Dir_TurnObjRandomly8.
;   L0489 Dir_TurnObjTowardTarget8_Return          code  Shared exit of Dir_TurnObjTowardTarget8.
;   L0496 Dir_TurnObjTowardTarget8_TestCandidate   code  Test of step 2 in Dir_TurnObjTowardTarget8: A = candidate direction, [00] = target direction.
;   L0507 Flyer_State3Wander                       code  Flight state 3: every $10 frames turn at random (Dir_TurnObjRandomly8); then state 1.
;   L0527 Dir_TurnObjRandomly8                     code  Turns an 8-way mover one 45-degree step clockwise, counter-clockwise, or not at all, at random.
;   L0549 Dir_GetObjDir8Index                      code  Finds the 8-way index (0-7, clockwise from up) of an object's direction in Dir_EightWayDirs.
;   L0561 Dir_GetObjDir8Index_Return               code  Shared exit of Dir_GetObjDir8Index and Flyer_State3Wander.
; ============================================================================

; ----------------------------------------------------------------------------
; Turns a new item object into a flying fairy: jingle, facing up, speed $7F, top speed $A0.
; IN:   X = object slot of the item (its item id is already $23).
; OUT:  writes ObjDir,X, the flyer fields of slot X, and FlyerMaxSpeedFrac (one byte for the room).
; NOTE: Called from Drop_SetUpItem when the item to drop is a fairy.
;       FlyerMaxSpeedFrac is shared by all flyers in the room: any flyer that runs Flyer_State0SpeedUp after
;       a fairy drop (Keese, whose init set $C0) is held to $A0 until an init writes the byte again.
; ----------------------------------------------------------------------------
Fairy_Init:
    ; Request square-1 tune $08, the same request Item_Take makes for a taken item.
    LDA #$08
    STA Tune1Request
    JSR Flyer_ResetState

    ; Face up ($08) at first.
    LDA #$08
    STA ObjDir, X

    ; Speed $7F. Flyer_State0SpeedUp adds 1 before the first move, so the fairy never moves at $7F: it
    ; starts at $80 (0.5 px/frame).
    LDA #$7F
    STA Flyer_ObjSpeed, X

    ; Top speed $A0 = 0.625 px/frame, reached on the $21st frame of Flyer_State0SpeedUp.
    LDA #$A0
    STA FlyerMaxSpeedFrac
    RTS

; ----------------------------------------------------------------------------
; Zeroes a flyer's position fraction, turn count, distance, flight state and invincibility timer.
; IN:   X = object slot.
; OUT:  A = 0. Flyer_ObjSpeed and ObjDir are left for the caller to set.
; ----------------------------------------------------------------------------
Flyer_ResetState:
    LDA #$00
    STA Flyer_ObjPosFrac, X
    STA Flyer_ObjTurns, X
    STA Flyer_ObjDistTraveled, X
    STA Flyer_ObjFlyingState, X
    STA ObjInvincibilityTimer, X
    RTS

; ----------------------------------------------------------------------------
; Runs the fairy item for one frame: flight state, movement, then drawing.
; IN:   X = object slot of the item.
; NOTE: Called from Drop_UpdateItem while the item's id is $23, so it runs every frame the item exists.
; ----------------------------------------------------------------------------
Fairy_Update:
    JSR Fairy_ControlFlight
    JSR Flyer_Move

; ----------------------------------------------------------------------------
; Draws a fairy at the object's position, flapping between two frames every 4 frames.
; IN:   X = object slot.
; NOTE: Also the drawing routine of the pond fairy (PondFairy_Update).
; ----------------------------------------------------------------------------
Fairy_Draw:
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorRedPaletteRow

    ; A = 2 here (the red palette row attribute). Frame [0C] = bit 2 of FrameCounter: 0 or 1,
    ; switching every 4 frames. Y = $14 is the fairy's item slot (Item_IdToSlot[$23]).
    ASL
    AND FrameCounter
    LSR
    LSR
    STA $0C
    LDY #$14
    JMP Anim_WriteItemSprites

; ----------------------------------------------------------------------------
; Dispatches the fairy item's flight state (Flyer_ObjFlyingState,X: 0-3).
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Fairy_ControlFlight:
    LDA Flyer_ObjFlyingState, X
    JSR Util_TableJump

; TABLE: fairy item flight handlers; index = Flyer_ObjFlyingState; value = routine address.
Fairy_ControlFlight_JumpTable:
    .ADDR Flyer_State0SpeedUp             ; 0: speed up to FlyerMaxSpeedFrac, then state 1
    .ADDR Fairy_DecideFlightState    ; 1: always choose state 3 with 6 in the turn counter
    .ADDR Fairy_DoNothingInFlightState2           ; 2: nothing (the fairy never enters it; other flyers chase here)
    .ADDR Flyer_State3Wander              ; 3: random turns every $10 frames, then state 1

; ----------------------------------------------------------------------------
; Fairy flight state 1: go to state 3 (wander) with the turn counter at 6.
; IN:   X = object slot.
; NOTE: No random choice and no slow-down: once at top speed the fairy wanders at 0.625 px/frame for
;       as long as it exists. A counter of 6 gives 5 turns (Flyer_State3Wander).
; ----------------------------------------------------------------------------
Fairy_DecideFlightState:
    LDA #$03
    STA Flyer_ObjFlyingState, X
    LDA #$06
    STA Flyer_ObjTurns, X

; Fairy flight state 2: returns at once. Also the RTS of Fairy_DecideFlightState.
Fairy_DoNothingInFlightState2:
    RTS

; TABLE: the 8 directions in clockwise order; index = 8-way direction index 0-7 (+1 = 45 degrees
;        clockwise, +4 = reverse); value = direction bits (1 right, 2 left, 4 down, 8 up).
Dir_EightWayDirs:
    ; up, up-right, right, down-right, down, down-left, left, up-left
    .BYTE $08, $09, $01, $05, $04, $06, $02, $0A

; ----------------------------------------------------------------------------
; Flight state 5: wait for ObjTimer,X to run out, then go to state 0 (speed up).
; IN:   X = object slot.
; NOTE: After a slow-down (speed AND $E0) is 0, so Flyer_Move moves nothing while this state waits.
; ----------------------------------------------------------------------------
Flyer_State5Delay:
    LDA ObjTimer, X
    BNE :+
    LDA #$00
    STA Flyer_ObjFlyingState, X
:
    RTS

; ----------------------------------------------------------------------------
; Flight state 4: lose 1 speed unit per frame; below $20 start a random delay and go to state 5.
; IN:   X = object slot.
; NOTE: Shares the tail of Flyer_State0SpeedUp. The delay is $40-$7F frames from Random,X. From top speed
;       $A0 the slow-down takes $81 frames (speed $A0 down to $1F).
; ----------------------------------------------------------------------------
Flyer_State4SlowDown:
    DEC Flyer_ObjSpeed, X
    JMP :+

; ----------------------------------------------------------------------------
; Flight state 0: gain 1 speed unit per frame; at the room's top speed go to state 1 (decide).
; IN:   X = object slot.
; NOTE: Ends when (Flyer_ObjSpeed AND $E0) >= FlyerMaxSpeedFrac. If the speed is still below $20
;       after the increment, the slow-down ending below is taken instead (delay, state 5).
; ----------------------------------------------------------------------------
Flyer_State0SpeedUp:
    INC Flyer_ObjSpeed, X
:
    LDA Flyer_ObjSpeed, X
    AND #$E0
    BNE Flyer_CompareMaxSpeed

    ; (speed AND $E0) = 0: the flyer has stopped.
    ; Set ObjTimer to a random $40-$7F frames (low 6 bits of Random,X).
    LDA Random, X
    AND #$3F
    ORA #$40
    STA ObjTimer, X

    ; Go to flying state 5.
    LDA #$05

; Stores A as the flight state of slot X and returns. Tail of the speed states; also jumped to.
Flyer_SetFlyingState:
    STA Flyer_ObjFlyingState, X
:
    RTS

; Tail of Flyer_State0SpeedUp and Flyer_State4SlowDown: A = speed AND $E0 (nonzero); state 1 once A >= top speed.
Flyer_CompareMaxSpeed:
    CMP FlyerMaxSpeedFrac
    BCC :-                      ; Below the top speed: return, state unchanged.
    LDA #$01                    ; At or above it: go to flight state 1 (decide).
    JMP Flyer_SetFlyingState

; ----------------------------------------------------------------------------
; Moves a flyer 1 pixel along its 8-way direction whenever its position fraction overflows.
; IN:   X = object slot; ObjDir,X = direction bits (two bits for a diagonal).
; OUT:  ObjX,X / ObjY,X and Flyer_ObjOffsetX,X / Flyer_ObjOffsetY,X changed by 1 per set direction bit;
;       Flyer_ObjDistTraveled,X + 1; direction reversed by Flyer_BoundByRoom if a room bound was crossed.
; NOTE: Adds (Flyer_ObjSpeed AND $E0) to Flyer_ObjPosFrac each frame, so the speed is one of 8 steps:
;       $20 = 0.125 px/frame ... $A0 = 0.625, $C0 = 0.75, $E0 = 0.875. Below $20 the flyer does not
;       move. No tile collision: only the room bounds stop a flyer. A diagonal covers 1 pixel on each
;       axis per carry.
; ----------------------------------------------------------------------------
Flyer_Move:
    LDA Flyer_ObjSpeed, X
    AND #$E0
    CLC
    ADC Flyer_ObjPosFrac, X
    STA Flyer_ObjPosFrac, X
    BCC @Exit                   ; No carry: no pixel to move this frame.

    ; [02] walks through the direction bits 1, 2, 4, 8; each bit set in ObjDir moves 1 pixel.
    ;
    ; Right ($01).
    LDA #$01
    STA $02
    LDA ObjDir, X
    BIT $02
    BEQ @Left
    INC ObjX, X
    INC Flyer_ObjOffsetX, X

@Left:
    ; Left ($02).
    ASL $02
    BIT $02
    BEQ @Down
    DEC ObjX, X
    DEC Flyer_ObjOffsetX, X

@Down:
    ; Down ($04).
    ASL $02
    BIT $02
    BEQ @Up
    INC ObjY, X
    INC Flyer_ObjOffsetY, X

@Up:
    ; Up ($08).
    ASL $02
    BIT $02
    BEQ @CountPixelAndBound
    DEC ObjY, X
    DEC Flyer_ObjOffsetY, X

@CountPixelAndBound:
    ; Count the pixel moved, then test the room bounds (reversing the direction if one was crossed).
    INC Flyer_ObjDistTraveled, X
    JSR Flyer_BoundByRoom

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Tests an 8-way mover against the room bounds and reverses its direction if it crossed one.
; IN:   X = object slot; ObjDir,X = direction.
; OUT:  [0F] = ObjDir,X as it was, or 0 when a bound was crossed in a direction the object faces
;       (then ObjDir,X is reversed; for Moldorm see Moldorm_DeferBounce). A = [0F] when not blocked.
; NOTE: The test comes after the move, and the position is not put back. Type $20 (Boulder) is
;       not tested against the top and bottom bounds, so it is never reversed there.
; ----------------------------------------------------------------------------
Flyer_BoundByRoom:
    LDA ObjDir, X
    STA $0F
    JSR Room_BoundDirHorizontally

    ; Test the top and bottom bounds unless the object is a Boulder (type $20).
    LDA ObjType, X
    CMP #$20
    BEQ :+
    JSR Room_BoundDirVertically
:
    ; [0F] still nonzero: no bound crossed in a facing direction. Return.
    LDA $0F
    BNE Flyer_BoundByRoom_Return

; ----------------------------------------------------------------------------
; Turns an 8-way mover around (180 degrees); a Moldorm head defers the turn instead.
; IN:   X = object slot; ObjDir,X = one of the 8 values of Dir_EightWayDirs.
; OUT:  ObjDir,X = opposite direction, except for type $41 (see Moldorm_DeferBounce). Y = its 8-way index.
; NOTE: Flyer_BoundByRoom falls into this; Gohma calls it directly.
; ----------------------------------------------------------------------------
Dir_ReverseObjDir8:
    ; Y = (direction index + 4) mod 8: the opposite direction.
    JSR Dir_GetObjDir8Index
    TYA
    CLC
    ADC #$04
    AND #$07
    TAY

    ; Moldorm (type $41) does not turn here: go store the deferred bounce direction.
    LDA ObjType, X
    CMP #$41
    BEQ Moldorm_DeferBounce

    ; Apply the new direction.
    LDA Dir_EightWayDirs, Y
    STA ObjDir, X

; Shared exit of Flyer_BoundByRoom and Dir_ReverseObjDir8.
Flyer_BoundByRoom_Return:
    RTS

; ----------------------------------------------------------------------------
; Moldorm part of Dir_ReverseObjDir8: a head stores the reversed direction for its next turn.
; IN:   X = object slot; Y = 8-way index of the reversed direction.
; OUT:  Moldorm_ObjBounceDir,X = reversed direction when X is $05 or $0A (the two head slots);
;       nothing for the other segments. ObjDir,X is not changed.
; ----------------------------------------------------------------------------
Moldorm_DeferBounce:
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE Moldorm_DeferBounce_Return
:
    LDA Dir_EightWayDirs, Y
    STA Moldorm_ObjBounceDir, X

; Shared exit of Moldorm_DeferBounce and Flyer_State2Chase.
Moldorm_DeferBounce_Return:
    RTS

; ----------------------------------------------------------------------------
; Flight state 2: every $10 frames turn one step toward the chase target; then state 1.
; IN:   X = object slot; Flyer_ObjTurns,X = turn counter set by the type's decide state.
; NOTE: Runs only when ObjTimer,X = 0. The counter is decremented first, so a counter of N gives
;       N - 1 turns, $10 frames apart, and the Nth expiry goes to state 1 (decide).
; ----------------------------------------------------------------------------
Flyer_State2Chase:
    LDA ObjTimer, X
    BNE Moldorm_DeferBounce_Return             ; Timer still running: return.

    ; Count down the turns. At 0 go to flight state 1.
    DEC Flyer_ObjTurns, X
    BNE Flyer_State2Chase_DelayAndTurn

; Sets flight state 1 (decide) for slot X. Shared by Flyer_State2Chase and Flyer_State3Wander.
Flyer_GoToState1Decide:
    LDA #$01
    JMP Flyer_SetFlyingState

; Part of Flyer_State2Chase: starts the $10-frame delay, then falls into the turn.
Flyer_State2Chase_DelayAndTurn:
    LDA #$10
    STA ObjTimer, X

; ----------------------------------------------------------------------------
; Turns an 8-way mover at most one 45-degree step, toward the chase target when a test allows it.
; IN:   X = object slot; ChaseTargetX, ChaseTargetY = the point to steer toward (not always Link).
; OUT:  ObjDir,X; uses [00] (direction toward the target) and [01].
; NOTE: 1. If the target direction is the facing direction or 45 degrees either side: no change.
;       2. Else the candidates counter-clockwise neighbour, facing, clockwise neighbour are tried in
;          that order; the first that shares a bit with the target direction and has
;          (candidate OR target) < 7 becomes the direction. That needs neither to contain up ($08).
;       3. Else turn one step clockwise, whether or not that is toward the target (always the case
;          when the target is straight behind, or at the same position: target direction 0).
;       Also called by Digdogger, Manhandla and the Boulder/Tektite jump code.
; ----------------------------------------------------------------------------
Dir_TurnObjTowardTarget8:
    ; Build the direction toward the target in [00]. Horizontal part:
    ;   target X = object X: 0;  target X > object X: 1 (right);  else 2 (left).
    LDY #$00
    STY $00
    INY
    LDA ChaseTargetX
    CMP ObjX, X
    BEQ @CalcVertical
    BCS :+
    INY
:
    STY $00

@CalcVertical:
    ; Vertical part, ORed into [00]:
    ;   target Y = object Y: nothing;  target Y > object Y: 4 (down);  else 8 (up).
    ; (Y = 1 or 2, shifted left twice.)
    LDY #$01
    LDA ChaseTargetY
    CMP ObjY, X
    BEQ @SkipVertical
    BCS :+
    INY
:
    TYA
    ASL
    ASL
    ORA $00
    STA $00

@SkipVertical:
    ; Y = 8-way index of the facing direction.
    JSR Dir_GetObjDir8Index

    ; Step 1: compare the target direction with index+1 (clockwise neighbour), index, index-1
    ; (counter-clockwise neighbour). An exact match leaves the direction alone.
    LDA #$03
    STA $01                     ; [01] = candidates left
    INY                         ; Start at the clockwise neighbour.

@CheckNearTarget:
    TYA
    AND #$07                    ; Wrap the index to 0-7.
    TAY
    LDA Dir_EightWayDirs, Y
    CMP $00
    BEQ Dir_TurnObjTowardTarget8_Return             ; Already heading (nearly) at the target: return.
    DEY                         ; Next candidate: one step counter-clockwise.
    DEC $01
    BNE @CheckNearTarget

    ; Step 2: Y is now index-2. Try index-1, index, index+1 in that order: a candidate that
    ; shares a direction bit with the target direction goes to Dir_TurnObjTowardTarget8_TestCandidate.
    LDA #$03
    STA $01
    INY                         ; Y = index-1, the counter-clockwise neighbour.

; Candidate loop of Dir_TurnObjTowardTarget8 (step 2).
Dir_TurnObjTowardTarget8_ScanCandidates:
    TYA
    AND #$07                    ; Wrap the index to 0-7.
    TAY
    LDA Dir_EightWayDirs, Y
    BIT $00
    BNE Dir_TurnObjTowardTarget8_TestCandidate

; Advances to the next candidate (one step clockwise); after the third, turns clockwise.
Dir_TurnObjTowardTarget8_NextCandidate:
    INY
    DEC $01
    BNE Dir_TurnObjTowardTarget8_ScanCandidates

    ; Step 3: no candidate passed. Y is index+2; step back to index+1: turn 45 degrees clockwise.
    DEY

; Sets ObjDir,X to Dir_EightWayDirs[Y]. Also the tail of Dir_TurnObjRandomly8.
Dir_SetObjDir8FromIndex:
    LDA Dir_EightWayDirs, Y
    STA ObjDir, X

; Shared exit of Dir_TurnObjTowardTarget8.
Dir_TurnObjTowardTarget8_Return:
    RTS

; Test of step 2 in Dir_TurnObjTowardTarget8: A = candidate direction, [00] = target direction.
; Accept the candidate only if (candidate OR target) < 7: neither may contain up ($08), and the two
; may not hold left and right between them (down-right with down-left = 7). So a flyer never takes
; this path toward a target above it; it then falls to the clockwise turn of step 3.
Dir_TurnObjTowardTarget8_TestCandidate:
    ORA $00
    CMP #$07
    BCS Dir_TurnObjTowardTarget8_NextCandidate           ; >= 7: rejected, try the next candidate.
    BCC Dir_SetObjDir8FromIndex         ; < 7: take this direction (always taken).

; ----------------------------------------------------------------------------
; Flight state 3: every $10 frames turn at random (Dir_TurnObjRandomly8); then state 1.
; IN:   X = object slot; Flyer_ObjTurns,X = turn counter set by the type's decide state.
; NOTE: Same timing as Flyer_State2Chase: a counter of N gives N - 1 turns $10 frames apart.
; ----------------------------------------------------------------------------
Flyer_State3Wander:
    LDA ObjTimer, X
    BNE Dir_GetObjDir8Index_Return             ; Timer still running: return.

    ; Count down the turns. At 0 go to flight state 1.
    DEC Flyer_ObjTurns, X
    BNE :+
    JMP Flyer_GoToState1Decide
:
    ; Wait $10 frames before the next turn.
    LDA #$10
    STA ObjTimer, X

; ----------------------------------------------------------------------------
; Turns an 8-way mover one 45-degree step clockwise, counter-clockwise, or not at all, at random.
; IN:   X = object slot.
; OUT:  ObjDir,X.
; NOTE: Random+1,X >= $A0 (96 of 256): no turn; $50-$9F (80 of 256): clockwise; < $50 (80 of 256):
;       counter-clockwise. Also called by Digdogger and Manhandla.
; ----------------------------------------------------------------------------
Dir_TurnObjRandomly8:
    JSR Dir_GetObjDir8Index
    LDA Random+1, X
    CMP #$A0
    BCS @SetDir
    INY                         ; One step clockwise.
    CMP #$50
    BCS @SetDir
    DEY                         ; < $50: one step counter-clockwise instead.
    DEY

@SetDir:
    TYA
    AND #$07
    TAY
    JMP Dir_SetObjDir8FromIndex

; ----------------------------------------------------------------------------
; Finds the 8-way index (0-7, clockwise from up) of an object's direction in Dir_EightWayDirs.
; IN:   X = object slot.
; OUT:  Y = index; 0 (up) if ObjDir,X is not one of the 8 values.
; ----------------------------------------------------------------------------
Dir_GetObjDir8Index:
    LDY #$07
:
    LDA ObjDir, X
    CMP Dir_EightWayDirs, Y
    BEQ Dir_GetObjDir8Index_Return
    DEY
    BPL :-
    ; Not found (ObjDir,X is 0 or not an 8-way value): Y is $FF here; make it index 0 (up).
    INY

; Shared exit of Dir_GetObjDir8Index and Flyer_State3Wander.
Dir_GetObjDir8Index_Return:
    RTS
