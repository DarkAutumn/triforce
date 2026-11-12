; ============================================================================
; FILE:     bank4/flying-ghini-peahat.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3976-4104
; SUBJECTS: Flyer
; TOPIC:    Update routines and flight-state tables of the flying Ghini (type $22) and the Peahat (type
;           $1A), both built on the shared flyer code; the "secret found" tune request; padding.
; SEE ALSO: bank4/fairy-flyer-common.asm (Flyer_SpeedUp, Flyer_Chase, Flyer_Wander, Flyer_SlowDown,
;           Flyer_Delay, MoveFlyer); bank4/zora-ow-inits.asm (InitPeahat, EndInitFlyer: start speed);
;           bank4/ghini-armos.asm (InitArmosOrFlyingGhini, DrawGhiniAndCheckCollisions, UpdateGhini
;           kills every flying Ghini); bank1/armos-ghini-spawn.asm (a touched gravestone makes type $22).
; KEY FACTS:
;   - Flight states (Flyer_ObjFlyingState) for both: 0 speed up, 1 decide, 2 chase, 3 wander, 4 slow
;     down, 5 wait without moving. ControlPeahatFlight_JumpTable, ControlFlyingGhiniFlight_JumpTable.
;   - A Peahat can be hit by Link's weapons only in flight state 5 (stopped); in states 0-4 only its
;     touch on Link is tested. UpdatePeahat.
;   - A flying Ghini never tests Link's weapons: it only harms Link (DrawGhiniAndCheckCollisions ends
;     in CheckLinkCollision). UpdateFlyingGhini.
;   - Decide state, Peahat: Random >= $B0 chase (80/256), >= $20 wander (144/256), else slow down
;     (32/256). Flyer_PeahatDecideState.
;   - Decide state, flying Ghini: Random >= $A0 chase (96/256), >= $08 wander (152/256), else slow down
;     (8/256). Flyer_GhiniDecideState.
;   - Chase and wander start with Flyer_ObjTurns = 6. Flyer_SetFlyingStateAnd6Turns.
;   - The clock (InvClock) freezes both; a stun timer or a knockback also stops a Peahat's flight
;     code. UpdateFlyingGhini, UpdatePeahat.
;   - PlaySecretFoundTune requests square-1 tune $04; the dock, Armos and rock/tree secrets call it.
; CONTENTS: (generated; L = line in this file)
;   L0045 UpdateFlyingGhini                   code  Updates a flying Ghini (type $22): flies unless the clock is held, draws, harms Link on touch.
;   L0064 ControlFlyingGhiniFlight            code  Runs the flying Ghini's handler for its flight state (Flyer_ObjFlyingState).
;   L0069 ControlFlyingGhiniFlight_JumpTable  data  TABLE: flying Ghini flight-state handlers; index = Flyer_ObjFlyingState; value = routine address.
;   L0083 Flyer_GhiniDecideState              code  Flying Ghini flight state 1: picks the next flight state at random and sets 6 turns.
;   L0101 UpdatePeahat                        code  Updates a Peahat (type $1A): knockback or flight, draws, and checks collisions by flight state.
;   L0142 ControlPeahatFlight                 code  Runs the Peahat's handler for its flight state (Flyer_ObjFlyingState).
;   L0147 ControlPeahatFlight_JumpTable       data  TABLE: Peahat flight-state handlers; index = Flyer_ObjFlyingState; value = routine address.
;   L0162 Flyer_PeahatDecideState             code  Peahat flight state 1: picks the next flight state at random and sets 6 turns.
;   L0179 Flyer_SetFlyingStateAnd6Turns       code  Sets a flyer's flight state to Y and its turn counter to 6.
;   L0190 PlaySecretFoundTune                 code  Requests the "secret found" tune (square-1 tune request $04).
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a flying Ghini (type $22): flies unless the clock is held, draws, harms Link on touch.
; IN:   X = object slot
; NOTE: Link's weapons are never tested against it; DrawGhiniAndCheckCollisions only checks Link.
;       It is not stopped by ObjStunTimer, only by InvClock.
; ----------------------------------------------------------------------------
UpdateFlyingGhini:
    ; With the clock, don't move.
    LDA InvClock
    BNE :+
    JSR ControlFlyingGhiniFlight
    JSR MoveFlyer
:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; The low bit of the distance flown is loaded but has no effect: the routine jumped to
    ; begins by calling Anim_FetchObjPosForSpriteDescriptor, which overwrites A.
    LDA Flyer_ObjDistTraveled, X
    AND #$01
    JMP DrawGhiniAndCheckCollisions

; ----------------------------------------------------------------------------
; Runs the flying Ghini's handler for its flight state (Flyer_ObjFlyingState).
; IN:   X = object slot
; ----------------------------------------------------------------------------
ControlFlyingGhiniFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

; TABLE: flying Ghini flight-state handlers; index = Flyer_ObjFlyingState; value = routine address.
ControlFlyingGhiniFlight_JumpTable:
    .ADDR Flyer_SpeedUp             ; 0: speed +1 per frame up to FlyerMaxSpeedFrac, then state 1
    .ADDR Flyer_GhiniDecideState    ; 1: pick state 2, 3 or 4 at random
    .ADDR Flyer_Chase               ; 2: turn toward the chase target every $10 frames
    .ADDR Flyer_Wander              ; 3: turn at random every $10 frames
    .ADDR Flyer_SlowDown            ; 4: speed -1 per frame until below $20, then state 5
    .ADDR Flyer_Delay               ; 5: wait for ObjTimer ($40-$7F frames), then state 0

; ----------------------------------------------------------------------------
; Flying Ghini flight state 1: picks the next flight state at random and sets 6 turns.
; IN:   X = object slot
; OUT:  Flyer_ObjFlyingState,X = 2, 3 or 4; Flyer_ObjTurns,X = 6
; NOTE: Random,X >= $A0: 2 chase (96/256). >= $08: 3 wander (152/256). Else: 4 slow down (8/256).
; ----------------------------------------------------------------------------
Flyer_GhiniDecideState:
    LDY #$02
    LDA Random, X
    CMP #$A0
    BCS :+                      ; If >= $A0, go to state 2.
    INY
    CMP #$08
    BCS :+                      ; If >= 8, go to state 3.
    INY                         ; Else go to state 4.
:
    JMP Flyer_SetFlyingStateAnd6Turns    ; Go set the state chosen in Y.

; ----------------------------------------------------------------------------
; Updates a Peahat (type $1A): knockback or flight, draws, and checks collisions by flight state.
; IN:   X = object slot
; NOTE: Link's weapons can hit it only in flight state 5 (stopped after slowing down); in every
;       other state only its touch on Link is tested. The clock or a stun stops its flight code.
; ----------------------------------------------------------------------------
UpdatePeahat:
    ; If the Peahat is being knocked back, then move it that way and go draw
    ; and check for collisions.
    LDA ObjShoveDir, X
    BEQ :+
    JSR Obj_Shove

@L_DrawAndCheckCollisions:
    JMP @DrawAndCheckCollisions
:
    ; With the clock, or while stunned, go draw and check for collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @L_DrawAndCheckCollisions

    ; Else fly.
    JSR ControlPeahatFlight
    JSR MoveFlyer

@DrawAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; The frame image is the low bit of the distance flown (pixels),
    ; so the faster it moves, the faster it animates.
    LDA Flyer_ObjDistTraveled, X
    AND #$01
    JSR DrawObjectMirrored

    ; In flight state 5 (stopped, waiting), check all collisions, so Link's weapons can hurt it.
    ; In all other states, only check for a collision with Link.
    LDA Flyer_ObjFlyingState, X
    CMP #$05
    BEQ :+
    JMP CheckLinkCollision
:
    JMP CheckMonsterCollisions

; ----------------------------------------------------------------------------
; Runs the Peahat's handler for its flight state (Flyer_ObjFlyingState).
; IN:   X = object slot
; ----------------------------------------------------------------------------
ControlPeahatFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

; TABLE: Peahat flight-state handlers; index = Flyer_ObjFlyingState; value = routine address.
ControlPeahatFlight_JumpTable:
    .ADDR Flyer_SpeedUp             ; 0: speed +1 per frame up to FlyerMaxSpeedFrac, then state 1
    .ADDR Flyer_PeahatDecideState   ; 1: pick state 2, 3 or 4 at random
    .ADDR Flyer_Chase               ; 2: turn toward the chase target every $10 frames
    .ADDR Flyer_Wander              ; 3: turn at random every $10 frames
    .ADDR Flyer_SlowDown            ; 4: speed -1 per frame until below $20, then state 5
    .ADDR Flyer_Delay               ; 5: wait for ObjTimer ($40-$7F frames), then state 0

; ----------------------------------------------------------------------------
; Peahat flight state 1: picks the next flight state at random and sets 6 turns.
; IN:   X = object slot
; OUT:  Flyer_ObjFlyingState,X = 2, 3 or 4; Flyer_ObjTurns,X = 6
; NOTE: Random,X >= $B0: 2 chase (80/256). >= $20: 3 wander (144/256). Else: 4 slow down (32/256),
;       the only way to the vulnerable state 5.
; ----------------------------------------------------------------------------
Flyer_PeahatDecideState:
    LDY #$02
    LDA Random, X
    CMP #$B0
    BCS Flyer_SetFlyingStateAnd6Turns    ; If >= $B0, go to state 2.
    INY
    CMP #$20
    BCS Flyer_SetFlyingStateAnd6Turns    ; If >= $20, go to state 3.
    INY                         ; Else go to state 4.

; ----------------------------------------------------------------------------
; Sets a flyer's flight state to Y and its turn counter to 6.
; IN:   X = object slot; Y = flight state
; OUT:  Flyer_ObjFlyingState,X = Y; Flyer_ObjTurns,X = 6
; NOTE: Flyer_Chase and Flyer_Wander decrement the counter before each turn and leave for state 1
;       when it reaches 0, so 6 gives 5 turns, $10 frames apart.
; ----------------------------------------------------------------------------
Flyer_SetFlyingStateAnd6Turns:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$06
    STA Flyer_ObjTurns, X
    RTS

; ----------------------------------------------------------------------------
; Requests the "secret found" tune (square-1 tune request $04).
; OUT:  A = $04; Tune1Request = $04
; ----------------------------------------------------------------------------
PlaySecretFoundTune:
    LDA #$04
    STA Tune1Request
    RTS

; DATA: filler ($FF), 25 bytes of padding. No label, so no code refers to it by name.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF
