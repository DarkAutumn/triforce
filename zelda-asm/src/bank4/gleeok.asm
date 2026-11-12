; ============================================================================
; FILE:     bank4/gleeok.asm             BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 8505-9043
; SUBJECTS: (none)
; TOPIC:    Flying Gleeok head; UpdateGleeok; necks
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0039 UpdateGleeokHead                   code
;   L0071 ControlGleeokHeadFlight            code
;   L0075 ControlGleeokHeadFlight_JumpTable  data
;   L0081 Flyer_GleeokHeadDecideState        code
;   L0095 GleeokNeckXAddrsLo                 data
;   L0098 GleeokNeckXAddrsHi                 data
;   L0101 GleeokNeckYAddrsLo                 data
;   L0104 GleeokNeckYAddrsHi                 data
;   L0107 GleeokNeckMiscAddrsLo              data
;   L0110 GleeokNeckMiscAddrsHi              data
;   L0113 UpdateGleeok                       code
;   L0206 Gleeok_FetchNeckAddrs              code  Returns:
;   L0223 L_Gleeok_UDiv4                     code
;   L0228 Gleeok_MoveNeck                    code
;   L0243 L_Gleeok_StoreRefSegDistance       code
;   L0413 Gleeok_CalcSegmentLimits           code  Params:
;   L0448 Gleeok_StretchNeck                 code  Params:
;   L0492 Gleeok_StretchNeck_JumpTable       data
;   L0503 Gleeok_ExpandSegment               code
;   L0517 L_Gleeok_DecSegmentY               code
;   L0523 L_Gleeok_SetSegmentY               code
;   L0526 Gleeok_IgnoreSegment               code
;   L0529 Gleeok_ContractSegmentY            code
;   L0540 L_Gleeok_ExpandHorizontally        code
;   L0550 Gleeok_ContractSegmentX            code
;   L0561 L_Gleeok_DecSegmentX               code
;   L0567 L_Gleeok_SetSegmentX               code
;   L0571 Gleeok_ContractSegment             code
; ============================================================================

UpdateGleeokHead:
    JSR ControlGleeokHeadFlight
    JSR MoveFlyer

    ; If the low bit of the distance traveled is clear,
    ; and Random < $20,
    ; and there's no object (fireball) in slot $B,
    ; then shoot a fireball $56.
    LDA Flyer_ObjDistTraveled, X
    AND #$01
    BNE :+
    LDA Random+1, X
    CMP #$20
    BCS :+
    LDA ObjType+11
    BNE :+
    LDA #$56
    JSR ShootFireball
:
    LDA #$01                    ; Switch animation frames every screen frame.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirrored
    JSR CheckMonsterCollisions

    ; Reset object state, shove info, and invincibility timer,
    ; because this monster can't get hurt nor die.
    JSR ResetObjMetastate
    JSR SetShoveInfoWith0
    STA ObjInvincibilityTimer, X
    RTS

ControlGleeokHeadFlight:
    LDA Flyer_ObjFlyingState, X
    JSR TableJump

ControlGleeokHeadFlight_JumpTable:
    .ADDR Flyer_SpeedUp
    .ADDR Flyer_GleeokHeadDecideState
    .ADDR Flyer_Chase
    .ADDR Flyer_Wander

Flyer_GleeokHeadDecideState:
    ; Go to the next flying state randomly:
    ; Random < $D0: 2
    ; Else:         3
    ;
    ; Set up 6 turns.
    LDY #$02
    LDA Random+1, X
    CMP #$D0
    BCC :+
    INY
:
    JMP Flyer_SetStateAndTurns

GleeokNeckXAddrsLo:
    .BYTE $38, $52, $6C, $95

GleeokNeckXAddrsHi:
    .BYTE $04, $04, $04, $03

GleeokNeckYAddrsLo:
    .BYTE $45, $5F, $79, $BD

GleeokNeckYAddrsHi:
    .BYTE $04, $04, $04, $03

GleeokNeckMiscAddrsLo:
    .BYTE $20, $2D, $81, $A9

GleeokNeckMiscAddrsHi:
    .BYTE $04, $04, $03, $03

UpdateGleeok:
    JSR Gleeok_DrawBody

    ; Calculate the index of the last neck of this kind of gleeok.
    ; Subtract $42 (Gleeok1) from ObjType[1].
    LDA ObjType+1
    SEC
    SBC #$42

    ; For each neck, counting down:
    STA GleeokCurNeck

@LoopNeck:
    ; If the bit for this neck is in the dead neck mask,
    ; then go loop again.
    LDY GleeokCurNeck
    LDA LevelMasks, Y
    BIT GleeokDeadNeckMask
    BEQ :+
    JMP @NextLoopNeck
:
    ; Get the addresses of this neck's segment data arrays.
    JSR Gleeok_FetchNeckAddrs

@LoadNeckBytes:
    ; Load neck data.
    ;
    ; For each segment from 5 to 0, indexed by Y register:
    ; - load coordinates into object slots 6 to 1
    ; - load miscellany into an array at [0413]
    LDA ($00), Y
    STA a:ObjX+1, Y
    LDA ($02), Y
    STA a:ObjY+1, Y
    LDA ($04), Y
    STA Gleeok_ObjHeadInfo, Y
    DEY
    BPL @LoadNeckBytes

    ; Each screen frame, one neck is chosen to move and try to shoot.
    LDA FrameCounter
    AND #$03
    STA $00
    CPX $00
    BNE @DrawAndCheckCollisions
    JSR Gleeok_MoveNeck
    JSR Gleeok_MoveHead

    ; The head is in object slot 5. Set it for when we want to shoot,
    ; so that it looks like the fireball came from the head.
    LDX #$05

    ; If Random < $20, and there's no object (fireball) in slot $B,
    ; then shoot fireball $56.
    LDA Random, X
    CMP #$20
    BCS @DrawAndCheckCollisions
    LDA ObjType+11
    BNE @DrawAndCheckCollisions
    LDA #$56
    JSR ShootFireball

@DrawAndCheckCollisions:
    ; This is run for all necks.
    JSR Gleeok_DrawHeadAndCheckCollisions

    ; Store the neck data.
    JSR Gleeok_FetchNeckAddrs

@SaveNeckBytes:
    LDA a:ObjX+1, Y
    STA ($00), Y
    LDA a:ObjY+1, Y
    STA ($02), Y
    LDA Gleeok_ObjHeadInfo, Y
    STA ($04), Y
    DEY
    BPL @SaveNeckBytes

@NextLoopNeck:
    ; Bottom of the neck loop.
    ; Decrement the neck index until it < 0.
    DEC GleeokCurNeck
    BPL @LoopNeck
    RTS

; Returns:
; X: neck index
; Y: 5 -- the last segment index
; [00:01]: address of segment X array
; [02:03]: address of segment Y array
; [04:05]: address of segment miscellaneous array
;
Gleeok_FetchNeckAddrs:
    LDX GleeokCurNeck
    LDA GleeokNeckXAddrsLo, X
    STA $00
    LDA GleeokNeckXAddrsHi, X
    STA $01
    LDA GleeokNeckYAddrsLo, X
    STA $02
    LDA GleeokNeckYAddrsHi, X
    STA $03
    LDA GleeokNeckMiscAddrsLo, X
    STA $04
    LDA GleeokNeckMiscAddrsHi, X
    STA $05
    LDY #$05                    ; Start loading data from segment 5.
    RTS

L_Gleeok_UDiv4:
    LSR
    LSR
    JMP L_Gleeok_StoreRefSegDistance

Gleeok_MoveNeck:
    ; Get the H-distance from the head to the base.
    LDA ObjX+5
    SEC
    SBC ObjX+1

    ; If the difference is positive, then go divide by 4 unsigned.
    BPL L_Gleeok_UDiv4

    ; Else divide by 4 signed.
    JSR Negate
    LSR
    LSR
    JSR Negate

L_Gleeok_StoreRefSegDistance:
    ; Store the signed reference segment distance.
    STA GleeokSignedRefSegmentDistance

    ; Get the absolute horizontal reference segment distance.
    ; This is the first tier of reference distances.
    ;
    ; Pass the value here to calculate and store the
    ; second and third tier horizontal distances.
    JSR Abs
    LDX #$00                    ; 0 for horizontal distances.
    JSR Gleeok_CalcSegmentLimits

    ; Get the absolute V-distance from the head to the base,
    ; and divide it be 4.
    LDA ObjY+5
    SEC
    SBC ObjY+1
    JSR Abs
    LSR
    LSR

    ; Calculate and store the second and third tier vertical distances.
    INX
    JSR Gleeok_CalcSegmentLimits

    ; Keep segments within the third tier distance of each other.
    ; Any segment that is too far from the previous one is moved
    ; 2 pixels toward it.
    ;
    ; Loop over the 4 segments above the base, from 0 to 3,
    ; accessing slots 2 to 5.
    LDX #$00

@KeepSegsNearNeighbors:
    ; Get the absolute H-distance between the current segment
    ; and the previous one.
    LDA ObjX+1, X
    SEC
    SBC ObjX+2, X
    JSR Abs

    ; If H-distance < third tier H-distance, then this distance is OK.
    ; Go look at the vertical distance.
    CMP GleeokTertiarySegmentLimits
    BCC @CheckDistanceV

    ; Otherwise, generally, bring this segment closer to
    ; the previous one by 2 pixels.
    LDA ObjX+2, X
    TAY
    INY
    INY
    CMP ObjX+1, X
    BCC :+
    DEY
    DEY
    DEY
    DEY
:
    STY ObjX+2, X

@CheckDistanceV:
    ; Get the absolute V-distance between the current segment
    ; and the previous one.
    LDA ObjY+1, X
    SEC
    SBC ObjY+2, X
    JSR Abs

    ; If V-distance < third tier V-distance, then this distance is OK.
    ; Go loop again.
    CMP GleeokTertiarySegmentLimits+1
    BCC @NextKeepSegsNearNeighbors

    ; Otherwise, generally, bring this segment closer to
    ; the previous one by 2 pixels.
    LDA ObjY+2, X
    TAY
    INY
    INY
    CMP ObjY+1, X
    BCC :+
    DEY
    DEY
    DEY
    DEY
:
    STY ObjY+2, X

@NextKeepSegsNearNeighbors:
    ; Bottom of the loop keeping segments from getting too far apart.
    INX
    CPX #$04
    BNE @KeepSegsNearNeighbors

    ; Stretch or contract depending on the distance of one segment to the next.
    LDX #$00
:
    JSR Gleeok_StretchNeck
    INX
    CPX #$03
    BCC :-

    ; [04D8] is the quarter horizontal distance from the head to the base.
    ; It represents the reference distance from each segment to the base.
    ; Multiples of it are used to mark the reference point of each segment.
    ; Move every segment one pixel closer to its reference point.
    ;
    ; Loop over each middle segment, from 2 to 0 in X register.
    ; Access object slots 4 to 2.
    LDX #$02

@KeepSegsNearRefPoint:
    ; Calculate the reference X coordinate:
    ;   base X + ([04D8] * (loop index + 1))
    TXA
    TAY
    LDA ObjX+1
:
    CLC
    ADC GleeokSignedRefSegmentDistance
    DEY
    BPL :-

    ; If current segment X < reference X, add 1 to it.
    ; Else subtract 1.
    LDY ObjX+2, X
    INY
    CMP ObjX+2, X
    BCS @NextKeepSegsNearRefPoint
    DEY
    DEY

@NextKeepSegsNearRefPoint:
    ; Store the new segment X, and loop again.
    STY ObjX+2, X
    DEX
    BPL @KeepSegsNearRefPoint

    ; Keep the segments in slots 4 and 3 between their neighbors vertically.
    ; So if they're not, then move them 1 pixel toward that goal.
    LDX #$01

@KeepSegsBetweenNeighbors:
    LDA ObjY+3, X
    CMP ObjY+2, X
    BCS :+
    CMP ObjY+4, X
    BCS @NextKeepSegsBetweenNeighbors
    INC ObjY+3, X
    JMP @NextKeepSegsBetweenNeighbors
:
    CMP ObjY+2, X
    BCC @NextKeepSegsBetweenNeighbors
    CMP ObjY+4, X
    BCC @NextKeepSegsBetweenNeighbors
    DEC ObjY+3, X

@NextKeepSegsBetweenNeighbors:
    DEX
    BPL @KeepSegsBetweenNeighbors
    RTS

; Params:
; A: primary reference segment distance
; X: 0 for horizontal, 1 for vertical
;
; Store primary distance, capped at 4.
;
Gleeok_CalcSegmentLimits:
    CMP #$04
    BCC :+
    LDA #$04
:
    STA GleeokPrimarySegmentLimits, X

    ; Add 4 and store secondary distance, capped at 8.
    CLC
    ADC #$04
    CMP #$08
    BCC :+
    LDA #$08
:
    STA GleeokSecondarySegmentLimits, X

    ; Add 4 and store tertiary distance, capped at $B.
    CLC
    ADC #$04
    CMP #$0B
    BCC :+
    LDA #$0B
:
    STA GleeokTertiarySegmentLimits, X
    RTS

; Params:
; X: segment index (0 to 2, corresponding to slots 2 to 4)
;
; Choose the index of a routine that will move the segment
; based on which boundaries are crossed.
;
; 0 will be used, if X and Y are within primary distances of
; the next segment.
;
Gleeok_StretchNeck:
    LDY #$00

    ; If the absolute H-distance >= primary reference segment distance,
    ; then increment the routine index.
    LDA ObjX+3, X
    SEC
    SBC ObjX+3, X               ; UNKNOWN: This should probably be ObjX+2, as with ObjY below.
    JSR Abs
    CMP GleeokPrimarySegmentLimits
    BCC :+
    INY
:
    ; If the absolute H-distance >= secondary reference segment distance,
    ; then increment the routine index.
    CMP GleeokSecondarySegmentLimits
    BCC :+
    ; Begin unverified code 12658
    INY
    ; End unverified code
:
    ; If the absolute V-distance >= primary reference segment distance,
    ; then add 3 to the routine index.
    LDA ObjY+3, X
    SEC
    SBC ObjY+2, X
    JSR Abs
    CMP GleeokPrimarySegmentLimits+1
    BCC :+
    INY
    INY
    INY
:
    ; If the absolute V-distance >= secondary reference segment distance,
    ; then add 3 to the routine index.
    CMP GleeokSecondarySegmentLimits+1
    BCC :+
    INY
    INY
    INY
:
    TYA
    JSR TableJump

Gleeok_StretchNeck_JumpTable:
    .ADDR Gleeok_ExpandSegment
    .ADDR Gleeok_IgnoreSegment
    .ADDR Gleeok_ContractSegmentX
    .ADDR Gleeok_IgnoreSegment
    .ADDR Gleeok_IgnoreSegment
    .ADDR Gleeok_ContractSegmentX
    .ADDR Gleeok_ContractSegmentY
    .ADDR Gleeok_ContractSegmentY
    .ADDR Gleeok_ContractSegment

Gleeok_ExpandSegment:
    ; Randomly, 50% of the time, go move away from the next segment horizontally.
    LDA Random
    BPL L_Gleeok_ExpandHorizontally

    ; Else move away vertically.
    LDA ObjY+2, X
    TAY
    INY
    INY
    CMP ObjY+3, X
    BEQ L_Gleeok_DecSegmentY
    BCS L_Gleeok_SetSegmentY

L_Gleeok_DecSegmentY:
    DEY
    DEY
    DEY
    DEY

L_Gleeok_SetSegmentY:
    STY ObjY+2, X

Gleeok_IgnoreSegment:
    RTS

Gleeok_ContractSegmentY:
    ; Move toward the next segment vertically.
    LDA ObjY+2, X
    TAY
    INY
    INY
    CMP ObjY+3, X
    BEQ L_Gleeok_SetSegmentY
    BCC L_Gleeok_SetSegmentY
    BCS L_Gleeok_DecSegmentY

L_Gleeok_ExpandHorizontally:
    ; Move away from the next segment horizontally.
    LDA ObjX+2, X
    TAY
    INY
    INY
    CMP ObjX+3, X
    BCS L_Gleeok_SetSegmentX
    BCC L_Gleeok_DecSegmentX

Gleeok_ContractSegmentX:
    ; Move toward the next segment horizontally.
    ; Begin unverified code 126B5
    LDA ObjX+2, X
    TAY
    INY
    INY
    CMP ObjX+3, X
    BCC L_Gleeok_SetSegmentX
    ; End unverified code

L_Gleeok_DecSegmentX:
    DEY
    DEY
    DEY
    DEY

L_Gleeok_SetSegmentX:
    STY ObjX+2, X
    RTS

Gleeok_ContractSegment:
    ; Randomly, 50% of the time, go move toward the next segment horizontally.
    ; Else go move vertically.
    ; Begin unverified code 126C5
    LDA Random
    BPL Gleeok_ContractSegmentX
    BMI Gleeok_ContractSegmentY
