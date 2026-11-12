; ============================================================================
; FILE:     bank4/patra.asm              BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10029-10282
; SUBJECTS: Flyer
; TOPIC:    Patra and its children
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 PatraManeuverTimes             data
;   L0026 UpdatePatra                    code
;   L0081 ControlPatraFlight             code
;   L0085 ControlPatraFlight_JumpTable   data
;   L0091 Flyer_PatraDecideState         code
;   L0109 PatraChildStartAngles          data
;   L0112 PatraChild1RotationCosineBits  data
;   L0115 PatraChild2RotationBits        data
;   L0118 PatraChild1RotationSineBits    data
;   L0121 UpdatePatraChild               code
;   L0201 PatraChild_State1              code
;   L0269 PatraChild_Draw                code
; ============================================================================

PatraManeuverTimes:
    .BYTE $FF, $50

UpdatePatra:
    JSR ControlPatraFlight

    ; Reset flying distance traveled, so that we know what offset to
    ; apply to the children.
    LDA #$00
    STA Flyer_ObjOffsetX, X
    STA Flyer_ObjOffsetY, X
    JSR MoveFlyer
    LDA #$02                    ; 2 animation frames a screen frame
    JSR AnimateAndDrawCommonObject

    ; Loop over the object slots of the 8 children.
    ; If any are found, then go check for collision with Link only.
    LDY #$08

@LoopChildren:
    LDA ObjType+1, Y
    CMP #$25                    ; Patra Child 1 object type
    BEQ @CheckLinkCollision
    CMP #$26                    ; Patra Child 2 object type
    BEQ @CheckLinkCollision
    DEY
    BNE @LoopChildren

    ; If there are no children, then Link can harm Patra.
    JSR CheckMonsterCollisions
    JSR PlayBossHitCryIfNeeded
    JSR PlayBossDeathCryIfNeeded
    JMP @TryChangeManeuver

@CheckLinkCollision:
    ; Else there are child patras left.
    ; Link cannot harm Patra, but it can harm Link.
    JSR CheckLinkCollision

@TryChangeManeuver:
    ; If timer = 0 and the angle of patra child in slot 3 = 0, then
    ; flip the maneuver index.
    LDA ObjTimer+1, X
    ORA ObjAngleWhole+3
    BNE @Exit
    LDA Patra_ObjManeuverIndex, X
    EOR #$01
    STA Patra_ObjManeuverIndex, X
    TYA

    ; Y is only ever 0 at this point, even though there are two elements.
    ; Maybe the TYA instruction above was supposed to be a TAY.
    LDA PatraManeuverTimes, Y
    STA ObjTimer+1, X

@Exit:
    RTS

ControlPatraFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlPatraFlight_JumpTable:
    .ADDR Flyer_SpeedUp
    .ADDR Flyer_PatraDecideState
    .ADDR Flyer_Chase
    .ADDR Flyer_Wander

Flyer_PatraDecideState:
    ; Go to the next state randomly:
    ; Random >= $40: 2
    ; Else:          3
    ;
    ; Set up 8 turns.
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

PatraChildStartAngles:
    .BYTE $14, $10, $0C, $08, $04, $00, $1C

PatraChild1RotationCosineBits:
    .BYTE $06

PatraChild2RotationBits:
    .BYTE $05

PatraChild1RotationSineBits:
    .BYTE $06, $06

UpdatePatraChild:
    ; If state = 1, then go orbit.
    LDA ObjState, X
    BNE PatraChild_State1

    ; State = 0.
    ;
    ; If the current object slot is 2, then go initialize the patra child.
    ; This is the only patra child shown in the first fame.
    CPX #$02
    BEQ @Ready

    ; The patra child in slot 2 is the last one to be updated.
    ; So all other patra children will see slot 2's state = 0, and return.
    ;
    ; This means that in the first frame, slot 2 is the only one to be
    ; shown.
    LDA ObjState+2
    BEQ @Exit

    ; In subsequent frames, patra children in slots >= 3 will reach
    ; here to wait to be shown at the top position (N) one at a time.
    TXA

    ; Subtract 3 from the object slot to get the angle where the
    ; patra child in object slot 2 should be for the current one to appear.
    SEC
    SBC #$03
    TAY

    ; If the patra child in slot 2 has not reached the required angle,
    ; then return.
    LDA ObjAngleWhole+2
    CMP PatraChildStartAngles, Y
    BNE @Exit

@Ready:
    ; Once the patra child in the last slot (9) reaches here, advance
    ; the patra parent's state.
    CPX #$09
    BNE :+
    INC ObjState+1
:
    ; Go to state 1 to orbit.
    INC ObjState, X

    ; UNKNOWN: Nothing seems to read or write the direction after this
    ; -- not even collision detection.
    LDA #$80
    STA ObjDir, X

    ; Set angle $18 (N).
    LDA #$18
    STA ObjAngleWhole, X

    ; This location is right above Patra.
    ; Set X coordinate to Patra's.
    LDA ObjX+1
    STA ObjX, X

    ; If patra child type = $25, then the radius of its orbit is $2C.
    ; Else use radius $18. Store the chosen value in [00].
    LDA #$2C
    LDY ObjType, X
    CPY #$25
    BEQ :+
    LDA #$18
:
    STA $00

    ; This location is right above Patra.
    ; Set Y coordinate to (Patra's Y - [00]).
    LDA ObjY+1
    SEC
    SBC $00
    STA ObjY, X

@Exit:
    RTS

PatraChild_State1:
    ; Add Patra's distance traveled since the last frame to this child's coordinates.
    LDA ObjX, X
    CLC
    ADC Flyer_ObjOffsetX+1
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC Flyer_ObjOffsetY+1
    STA ObjY, X

    ; Reset [0B]. The high byte of the amount to subtract is 0.
    LDA #$00
    STA $0B

    ; If the monster is type $25 Patra Child1, then decrease the
    ; monster's angle by $70, else by $60 to go slower.
    LDA #$70
    LDY ObjType, X
    CPY #$25
    BEQ :+
    LDA #$60
:
    JSR DecreaseObjectAngle

    ; Load Patra's maneuver index.
    ; It determines which counts of bits are looked up below.
    LDY Patra_ObjManeuverIndex+1

    ; If the monster is Patra Child 1 ($25), then look up two separate
    ; counts of bits to use for Y and X increment calculations in rotation.
    ; This leads to a large circle (6,6) and a wobbling ring (5,6).
    LDA ObjType, X
    CMP #$25
    BNE @UsePatraChild2Bits
    LDA PatraChild1RotationCosineBits, Y
    PHA
    LDA PatraChild1RotationSineBits, Y
    TAY
    PLA
    JMP @Rotate

@UsePatraChild2Bits:
    ; Else it's Patra Child 2 ($26). Look up one value that will be used
    ; for both Y and X increment calculations in rotation. This leads to
    ; a small circle (5) and a large circle (6).
    LDA PatraChild2RotationBits, Y
    TAY

@Rotate:
    JSR RotateObjectLocation
    STA ObjY, X
    JSR PatraChild_Draw

    ; If Patra's state still = 0, then the last patra child has not
    ; shown up yet. In this case don't check collisions yet, only return.
    LDA ObjState+1
    BEQ @Exit
    JSR CheckMonsterCollisions

    ; If the monster died, then change to the dead dummy object.
    LDA ObjMetastate, X
    BEQ @Exit
    JSR SetDeadDummyObjType

@Exit:
    RTS

PatraChild_Draw:
    JSR Anim_SetSpriteDescriptorRedPaletteRow

    ; 2 was returned from the routine above in A.
    ; Pass it as the animation counter value to the routine below.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JMP DrawObjectNotMirrored
