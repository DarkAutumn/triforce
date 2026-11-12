; ============================================================================
; FILE:     bank4/flying-ghini-peahat.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3976-4104
; SUBJECTS: Flyer
; TOPIC:    Flying Ghini and Peahat
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 UpdateFlyingGhini                   code
;   L0037 ControlFlyingGhiniFlight            code
;   L0041 ControlFlyingGhiniFlight_JumpTable  data
;   L0049 Flyer_GhiniDecideState              code
;   L0067 UpdatePeahat                        code
;   L0105 ControlPeahatFlight                 code
;   L0109 ControlPeahatFlight_JumpTable       data
;   L0117 Flyer_PeahatDecideState             code
;   L0133 Flyer_SetFlyingStateAnd6Turns       code
;   L0140 PlaySecretFoundTune                 code
; ============================================================================

UpdateFlyingGhini:
    ; If we have the magic clock, then don't move.
    LDA InvClock
    BNE :+
    JSR ControlFlyingGhiniFlight
    JSR MoveFlyer
:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Get the low bit of the flying distance traveled.
    ; But this doesn't seem to be used.
    ; The following call clobbers the A register.
    LDA Flyer_ObjDistTraveled, X
    AND #$01
    JMP DrawGhiniAndCheckCollisions

ControlFlyingGhiniFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlFlyingGhiniFlight_JumpTable:
    .ADDR Flyer_SpeedUp
    .ADDR Flyer_GhiniDecideState
    .ADDR Flyer_Chase
    .ADDR Flyer_Wander
    .ADDR Flyer_SlowDown
    .ADDR Flyer_Delay

Flyer_GhiniDecideState:
    ; Go to the next state randomly:
    ; Random >= $A0: 2
    ; Random >= $08: 3
    ; Else:          4
    ;
    ; Set up 6 turns.
    LDY #$02
    LDA Random, X
    CMP #$A0
    BCS :+                      ; If >= $A0, go to state 2.
    INY
    CMP #$08
    BCS :+                      ; If >= 8, go to state 3.
    INY                         ; Else go to state 4.
:
    JMP Flyer_SetFlyingStateAnd6Turns    ; Go set the state we determined.

UpdatePeahat:
    ; If the object is being shoved, then shove it and go draw
    ; and check for collisions.
    LDA ObjShoveDir, X
    BEQ :+
    JSR Obj_Shove

@L_DrawAndCheckCollisions:
    JMP @DrawAndCheckCollisions
:
    ; If we have the clock or the object is stunned, go draw
    ; and check for collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @L_DrawAndCheckCollisions

    ; Else move around.
    JSR ControlPeahatFlight
    JSR MoveFlyer

@DrawAndCheckCollisions:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Take low bit of the flying distance traveled as the frame image.
    ; So, the faster it moves, the faster it animates.
    LDA Flyer_ObjDistTraveled, X
    AND #$01
    JSR DrawObjectMirrored

    ; In flying state 5, check for all collisions (so it can get hurt).
    ; In all other states, only check for collisions with Link.
    LDA Flyer_ObjFlyingState, X
    CMP #$05
    BEQ :+
    JMP CheckLinkCollision
:
    JMP CheckMonsterCollisions

ControlPeahatFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlPeahatFlight_JumpTable:
    .ADDR Flyer_SpeedUp
    .ADDR Flyer_PeahatDecideState
    .ADDR Flyer_Chase
    .ADDR Flyer_Wander
    .ADDR Flyer_SlowDown
    .ADDR Flyer_Delay

Flyer_PeahatDecideState:
    ; Go to the next state randomly:
    ; Random >= $B0: 2
    ; Random >= $20: 3
    ; Else:          4
    ;
    ; Set up 6 turns.
    LDY #$02
    LDA Random, X
    CMP #$B0
    BCS Flyer_SetFlyingStateAnd6Turns    ; If >= $B0, go to state 2.
    INY
    CMP #$20
    BCS Flyer_SetFlyingStateAnd6Turns    ; If >= $20, go to state 3.
    INY                         ; Else go to state 4.

Flyer_SetFlyingStateAnd6Turns:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$06                    ; Set 6 turns for the next state.
    STA Flyer_ObjTurns, X
    RTS

PlaySecretFoundTune:
    LDA #$04
    STA Tune1Request
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF
