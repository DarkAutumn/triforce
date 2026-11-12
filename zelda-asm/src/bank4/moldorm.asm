; ============================================================================
; FILE:     bank4/moldorm.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4913-5153
; SUBJECTS: Moldorm
; TOPIC:    Moldorm (object type $41) update: the two heads steer with the flyer chase/wander
;           routines, every $10 frames each segment takes the direction of the one ahead, and a
;           killed segment is replaced by the worm's tail.
; SEE ALSO: bank4/moldorm-init.asm (Moldorm_Init: slots, start values);
;           bank4/fairy-flyer-common.asm (Flyer_State2Chase, Flyer_State3Wander, Flyer_Move, and the room-bound
;           bounce that sets Moldorm_ObjBounceDir); bank1/monster-collisions.asm
;           (Collision_CheckMonsterVsWeaponsAndLink); bank4/aquamentus.asm (Obj_PlayBossDeathCryAndResetShove);
;           bank7/object-stubs-destroy-monster.asm (DeadDummy_StartDeathSparkle, object type $5D).
; KEY FACTS:
;   - Two worms: slots $01-$05 (head $05) and $06-$0A (head $0A). Only a head runs the flight
;     control; the other segments just move along their own ObjDir. Moldorm_Chase, Moldorm_Wander
;   - All segments move 1 pixel every 2 frames (flyer speed $80), 8-way, and stop only while the
;     clock is held. A segment whose ObjDir is 0 is skipped entirely: not moved, not drawn, not
;     hittable and harmless. Moldorm_Update
;   - Whenever the head's ObjTimer is rearmed to $10 (each turn and each state change, so every
;     $10 frames = 8 pixels), slot N takes the old direction of slot N+1: the body follows the
;     head's path one step per segment, and at the start one more segment leaves the start point
;     every $10 frames. Moldorm_PropagateDirs
;   - A head picks a flying state when its timer runs out: chase the chase target (state 2) when
;     Random,X >= $40 (3 in 4), else wander (state 3); 8 turns, so a state lasts $80 frames.
;     Moldorm_DecideFlightState, Moldorm_ChangeFlyingState
;   - A head that reaches the room boundary does not reverse at once: the reversed direction is
;     stored and applied at the next propagation. Moldorm_PropagateDirs
;   - Killing any segment removes the worm's tail instead (lowest slot of that worm still type
;     $41): the tail becomes a dead dummy (type $5D) at the hit segment's position and the hit
;     segment lives on with HP $20. Damage does not carry between segments. Moldorm_Update
;   - A worm dies after 5 kills; the last one kills the head itself (type $41). The 4 before it
;     die as dead dummies (type $5D), which drop nothing and add nothing to RoomKillCount or
;     WorldKillCycle (Obj_EndMetastate, Drop_NoDropTypes): only the 2 head deaths count
;     and can drop an item. Moldorm_Update
;   - Collisions cannot turn or delay a segment: ObjDir and ObjTimer are restored after
;     Collision_CheckMonsterVsWeaponsAndLink, and shove info is cleared only on a kill. Moldorm_Update
; CONTENTS: (generated; L = line in this file)
;   L0059 Moldorm_Update                   code  Updates one Moldorm segment: move, draw, collide; on death hands the death to the worm's tail.
;   L0152 Moldorm_ControlFlight            code  Runs the flight routine for the segment's flying state (0-2 chase, 3 wander).
;   L0157 Moldorm_ControlFlight_JumpTable  data  TABLE: Moldorm flight routines; index = Flyer_ObjFlyingState; value = routine address.
;   L0171 Moldorm_Chase                    code  Flying states 0-2 of a Moldorm head: turn toward the chase target every $10 frames.
;   L0188 Moldorm_ChangeFlyingState        code  Timer = 0, so the flying state has ended: picks a new state and rearms the timer to $10 frames.
;   L0209 Moldorm_Wander                   code  Flying state 3 of a Moldorm head: turn at random every $10 frames.
;   L0228 Moldorm_PropagateDirs            code  Once per $10 frames (head's timer = $10): each segment takes the direction of the one ahead.
;   L0270 Moldorm_PropagateDirs_Return     code  Shared exit of the Moldorm flight routines above.
;   L0280 Moldorm_DecideFlightState        code  Picks a Moldorm head's next flying state: 2 (chase) if Random,X >= $40, else 3 (wander); 8 turns.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates one Moldorm segment: move, draw, collide; on death hands the death to the worm's tail.
; IN:   X = object slot ($01-$0A; heads are $05 and $0A).
; OUT:  on a kill: ObjHP,X = $20; the tail slot gets ObjTimer $11 and this segment's position
;       and invincibility timer; unless the tail is the head itself, the tail becomes type $5D
;       and ObjMetastate,X = 0 (this segment lives on).
; NOTE: Each segment has HP $20 and costs Link $0080 = half a heart on touch, before rings
;       (type $41 in Obj_TypeHpPairs and Damage_ObjTypeDamageToLink). Nothing here tests the stun
;       timer, so a boomerang stun does not stop a segment. The clock stops all movement.
; ----------------------------------------------------------------------------
Moldorm_Update:
    ; A segment with no direction has not left the start point yet: do nothing at all.
    LDA ObjDir, X
    BEQ @Exit

    ; Unless the clock is held: steer (heads only) and move.
    LDA InvClock
    BNE :+
    JSR Moldorm_ControlFlight
    JSR Flyer_Move
:
    ; Draw one sprite: attributes 2 (normal, palette row 6), tile $44 for every segment.
    ; (Anim_WriteSprite cycles the palette instead while the invincibility timer runs.)
    LDA #$02
    STA $03
    LDA #$44
    JSR Anim_WriteSprite

    ; Check collisions, but keep them from changing the direction or the timer:
    ; both are saved and restored around the call.
    LDA ObjDir, X
    PHA
    LDA ObjTimer, X
    PHA
    JSR Collision_CheckMonsterVsWeaponsAndLink
    PLA
    STA ObjTimer, X
    PLA
    STA ObjDir, X

    ; If this segment is still alive (metastate 0), then return.
    LDA ObjMetastate, X
    BEQ @Exit

    ; This segment was killed. The worm loses its tail instead: the tail slot takes over this
    ; segment's death and this slot comes back to life. So the heads stay in slots $05 and $0A
    ; and the live segments stay contiguous.
    ;
    ; Give this slot HP $20 again. This is not a true swap: whatever damage the tail had
    ; taken is forgotten.
    LDA #$20
    STA ObjHP, X
    JSR Obj_PlayBossDeathCryAndResetShove    ; Boss death cry (metastate <> 0); clears shove info.

    ; Look for the tail: the lowest slot of this segment's worm that is still type $41.
    ; Y+1 is the slot: start at slot $01 for slots $01-$05, at slot $06 for slots $06-$0A.
    LDY #$FF
    CPX #$06
    BCC :+
    LDY #$04
:
    INY
    LDA ObjType+1, Y
    CMP #$41
    BNE :-

    ; Set the tail's object timer to $11. No use of this value was found: once the tail is a
    ; dead dummy its metastate becomes $10, and the death sparkle then sets the timer to 6
    ; (purpose unverified).
    LDA #$11
    STA a:ObjTimer+1, Y

    ; Copy the invincibility timer, X and Y of the killed segment to the tail,
    ; so the death sparkle appears where the hit landed.
    LDA ObjInvincibilityTimer, X
    STA ObjInvincibilityTimer+1, Y
    LDA ObjX, X
    STA a:ObjX+1, Y
    LDA ObjY, X
    STA a:ObjY+1, Y

    ; If the tail found is a head (slot $05 or $0A), it is the worm's last segment and the
    ; segment that was killed: return and let it die as type $41.
    CPY #$04
    BEQ @Exit
    CPY #$09
    BEQ @Exit

    ; Else turn the tail into a dead dummy (type $5D), which starts its death on its next update,
    ; and bring the current segment back to life (metastate 0).
    LDA #$5D
    STA ObjType+1, Y
    JSR Obj_ResetMetastate

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Runs the flight routine for the segment's flying state (0-2 chase, 3 wander).
; IN:   X = object slot.
; NOTE: Moldorm only ever holds states 1 (left by the flyer routines when the turns run out),
;       2 and 3 (Moldorm_Init, Moldorm_DecideFlightState); state 1 is replaced in the same update.
; ----------------------------------------------------------------------------
Moldorm_ControlFlight:
    LDA Flyer_ObjFlyingState, X
    JSR Util_TableJump

; TABLE: Moldorm flight routines; index = Flyer_ObjFlyingState; value = routine address.
Moldorm_ControlFlight_JumpTable:
    .ADDR Moldorm_Chase         ; 0 (never set for Moldorm)
    .ADDR Moldorm_Chase         ; 1 (set when a state's turns run out; lasts no longer than the call)
    .ADDR Moldorm_Chase         ; 2 chase
    .ADDR Moldorm_Wander        ; 3 wander

; ----------------------------------------------------------------------------
; Flying states 0-2 of a Moldorm head: turn toward the chase target every $10 frames.
; IN:   X = object slot. Does nothing unless X is a head ($05 or $0A).
; OUT:  may write ObjDir,X, ObjTimer,X, the flying state and turns; falls into the state change
;       when the timer is 0, else goes to the direction propagation.
; NOTE: Flyer_State2Chase does not always turn toward the target (see its direction test), and the
;       chase target is not always Link (ChaseTargetX, ChaseTargetY).
; ----------------------------------------------------------------------------
Moldorm_Chase:
    ; Only a head (slot $05 or $0A) steers; any other segment returns.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE Moldorm_PropagateDirs_Return
:
    ; While ObjTimer <> 0 this does nothing. At 0 it uses up one turn: if turns remain it sets
    ; ObjTimer to $10 and turns; if none remain it sets flying state 1 and leaves ObjTimer 0.
    JSR Flyer_State2Chase

    ; If the timer is running (possibly just rearmed to $10), then
    ; go see if it is time to pass directions down the worm.
    LDA ObjTimer, X
    BNE Moldorm_PropagateDirs

; Timer = 0, so the flying state has ended: picks a new state and rearms the timer to $10 frames.
Moldorm_ChangeFlyingState:
    JSR Moldorm_DecideFlightState
    LDA #$10
    STA ObjTimer, X

    ; If the next lower slot has a direction, then go pass directions down the worm
    ; (the timer is $10 now, so the propagation runs).
    ;
    ; The next lower slot has no direction only before the first propagation, which happens at
    ; the head's first turn. So this test holds the body back for the first $10 frames after
    ; the head's first state choice.
    LDA ObjDir-1, X
    BNE Moldorm_PropagateDirs
    RTS

; ----------------------------------------------------------------------------
; Flying state 3 of a Moldorm head: turn at random every $10 frames.
; IN:   X = object slot. Does nothing unless X is a head ($05 or $0A).
; NOTE: Each turn of Flyer_State3Wander uses Random+1,X: >= $A0 keep the direction, $50-$9F turn one
;       step right, < $50 one step left (8-way steps).
; ----------------------------------------------------------------------------
Moldorm_Wander:
    ; Only a head (slot $05 or $0A) steers; any other segment returns.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE Moldorm_PropagateDirs_Return
:
    ; While ObjTimer <> 0 this does nothing. At 0 it uses up one turn: if turns remain it sets
    ; ObjTimer to $10 and turns at random; if none remain it sets flying state 1 and leaves
    ; ObjTimer 0.
    JSR Flyer_State3Wander

    ; So after the call: timer = $10 means a turn was just made (pass directions down the worm);
    ; timer = 0 means the state ended (go pick a new one, rearm the timer and pass directions);
    ; anything else is the wait between turns.
    LDA ObjTimer, X
    BEQ Moldorm_ChangeFlyingState

; Once per $10 frames (head's timer = $10): each segment takes the direction of the one ahead.
Moldorm_PropagateDirs:
    ; If timer <> $10 (not rearmed in this frame), then return.
    LDA ObjTimer, X
    CMP #$10
    BNE Moldorm_PropagateDirs_Return

    ; If a bounce off the room boundary was deferred (bounce direction <> 0), then
    ; make it the head's direction now and clear it.
    LDA Moldorm_ObjBounceDir, X
    BEQ @SkipBounce
    STA ObjDir, X
    LDA #$00
    STA Moldorm_ObjBounceDir, X

@SkipBounce:
    ; Loop 4 times, counted in [00], once for each segment behind the head.
    LDA #$04
    STA $00

    ; Y+1 is the slot: start at the worm's lowest slot, $01 for head $05, $06 for head $0A,
    ; and work up toward the head.
    LDY #$00
    CPX #$05
    BEQ @Loop
    LDY #$05

@Loop:
    ; The old direction of the next higher slot becomes this slot's old direction
    ; and its facing direction.
    LDA Moldorm_ObjOldDir+2, Y
    STA Moldorm_ObjOldDir+1, Y
    STA a:ObjDir+1, Y

    INY
    DEC $00
    BNE @Loop

    ; The head's current direction becomes its old direction.
    LDA ObjDir, X
    STA Moldorm_ObjOldDir, X

; Shared exit of the Moldorm flight routines above.
Moldorm_PropagateDirs_Return:
    RTS

; ----------------------------------------------------------------------------
; Picks a Moldorm head's next flying state: 2 (chase) if Random,X >= $40, else 3 (wander); 8 turns.
; IN:   X = object slot.
; OUT:  writes Flyer_ObjFlyingState,X and Flyer_ObjTurns,X = 8.
; NOTE: The flyer routines decrement the turns before turning, so 8 gives 7 turns $10 frames
;       apart; with the $10 frames set by the state change a state lasts $80 frames.
; ----------------------------------------------------------------------------
Moldorm_DecideFlightState:
    LDY #$02
    LDA Random, X
    CMP #$40
    BCS :+
    INY
:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$08
    STA Flyer_ObjTurns, X
    RTS
