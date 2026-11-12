; ============================================================================
; FILE:     bank4/bubble-keese.asm       BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1095-1237
; SUBJECTS: (none)
; TOPIC:    Bubble and Keese init/update, Keese flight
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 InitBubble                    code
;   L0026 InitBlueKeese                 code
;   L0040 InitRedOrBlackKeese           code
;   L0048 UpdateBubble                  code
;   L0104 AnimateAndDrawCommonObject    code  Params:
;   L0110 UpdateKeese                   code
;   L0130 ControlKeeseFlight            code
;   L0134 ControlKeeseFlight_JumpTable  data
;   L0142 Flyer_KeeseDecideState        code
;   L0158 Flyer_SetStateAndTurns        code
; ============================================================================

InitBubble:
    LDA #$40                    ; QSpeed $40 (1 pixel a frame)
    STA ObjQSpeedFrac, X
    JMP InitWalker

InitBlueKeese:
    ; Choose one of eight random directions.
    LDA Random, X
    AND #$07
    TAY
    LDA Directions8, Y
    STA ObjDir, X
    JSR ResetFlyerState
    LDA #$C0
    STA FlyingMaxSpeedFrac
    LDA #$1F
    STA Flyer_ObjSpeed, X
    RTS

InitRedOrBlackKeese:
    JSR InitBlueKeese

    ; Red and black keeses are like blue ones, but faster.
    LDA #$7F
    STA Flyer_ObjSpeed, X
    RTS

UpdateBubble:
    LDA #$40                    ; Turn rate $40
    JSR UpdateCommonWanderer
    LDA ObjType, X

    ; If the bubble is not type $2B that temporarily blocks the sword;
    ; then subtract $2B from the type to get 1 or 2, which
    ; represent the blue and red palette rows in sprite attributes.
    CMP #$2B
    BEQ @Flash
    SEC
    SBC #$2B
    JMP @Animate

@Flash:
    ; Else make it flash by cycling all the palette rows.
    LDA FrameCounter
    AND #$03

@Animate:
    JSR Anim_SetSpriteDescriptorAttributes

    ; Animation counter 1 to switch animation frames every screen frame.
    LDA #$01
    JSR AnimateAndDrawCommonObject
    JSR CheckLinkCollision

    ; If no collision, then return.
    LDA $06
    BEQ @Exit

    ; If not type $2B (flashing), then go block or unblock the sword
    ; depending on the bubble type.
    LDA ObjType, X
    CMP #$2B
    BNE @BlockOrUnblock

    ; Else the flashing bubble blocks the sword temporarily.
    ; Set a long timer to $10 ($A0 frames) for this purpose.
    LDA #$10
    STA SwordBlockedLongTimer

@Exit:
    RTS

@BlockOrUnblock:
    ; Subtract $2C from the type to get the value for blocking: 0 or 1
    SEC
    SBC #$2C
    STA SwordBlocked
    RTS

; Params:
; A: new value for animation counter
; X: object index
;
AnimateAndDrawCommonObject:
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    JSR Anim_SetObjHFlipForSpriteDescriptor
    LDA #$00
    JMP DrawObjectNotMirrored

UpdateKeese:
    ; If the magic clock is missing and Link is not lifting an item,
    ; then fly.
    LDA InvClock
    ORA ItemLiftTimer
    BNE :+
    JSR ControlKeeseFlight
    JSR MoveFlyer
:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Like peahats, keeses animate as fast as they move.
    ; But keese beat their wings at half the rate.
    LDA Flyer_ObjDistTraveled, X
    AND #$02
    LSR
    JSR DrawObjectMirrored
    JSR CheckMonsterCollisions
    JMP ResetShoveInfo

ControlKeeseFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlKeeseFlight_JumpTable:
    .ADDR Flyer_SpeedUp
    .ADDR Flyer_KeeseDecideState
    .ADDR Flyer_Chase
    .ADDR Flyer_Wander
    .ADDR Flyer_SlowDown
    .ADDR Flyer_Delay

Flyer_KeeseDecideState:
    ; Go to the next state randomly:
    ; Random >= $A0: 2
    ; Random >= $20: 3
    ; Else:          4
    ;
    ; Set up 6 turns.
    LDY #$02
    LDA Random+1, X
    CMP #$A0
    BCS Flyer_SetStateAndTurns  ; If >= $A0, go to state 2.
    INY
    CMP #$20
    BCS Flyer_SetStateAndTurns  ; If >= $20, go to state 3.
    INY                         ; Else go to state 4.

Flyer_SetStateAndTurns:
    TYA
    STA Flyer_ObjFlyingState, X
    LDA #$06                    ; Set 6 turns for the next state.
    STA Flyer_ObjTurns, X
    RTS
