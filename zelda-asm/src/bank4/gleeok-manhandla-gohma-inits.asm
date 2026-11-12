; ============================================================================
; FILE:     bank4/gleeok-manhandla-gohma-inits.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 7638-7830
; SUBJECTS: (none)
; TOPIC:    Inits for Gleeok, Manhandla, Gohma, Gleeok head
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 GleeokSegmentYs                   data
;   L0022 InitGleeok                        code
;   L0110 ManhandlaBaseFrameImagesAndAttrs  data
;   L0113 ManhandlaSegmentOffsetsX          data
;   L0116 ManhandlaSegmentOffsetsY          data
;   L0119 InitManhandla                     code
;   L0185 InitGohma                         code
;   L0203 InitGleeokHead                    code
; ============================================================================

GleeokSegmentYs:
    .BYTE $6F, $74, $79, $7E, $83, $88

InitGleeok:
    LDA #$10                    ; Play Aquamentus/Gleeok/Ganon roar.
    STA SampleRequest

    ; Loop 6 times to initialize each segment of 4 necks.
    LDX #$05
:
    ; Set the X of the current segment of each neck to $7C.
    LDA #$7C
    STA Gleeok_NeckXs0, X
    STA Gleeok_NeckXs1, X
    STA Gleeok_NeckXs2, X
    STA Gleeok_NeckXs3, X
    STA ObjX+1, X               ; I don't think this one is needed.

    ; Look up and set the Y of the current segment of each neck.
    LDA GleeokSegmentYs, X
    STA Gleeok_NeckYs0, X
    STA Gleeok_NeckYs1, X
    STA Gleeok_NeckYs2, X
    STA Gleeok_NeckYs3, X
    STA ObjY+1, X               ; I don't think this one is needed.

    ; Set HP of each neck to $A0.
    ;
    ; Because there's only one HP value for each neck,
    ; it's stored right in the object slot -- unlike coordinates.
    ;
    ; Also, note that even though these aren't independent objects
    ; in the object slots -- because object type = 0 -- the values in
    ; these object slots will be useful during collision detection.
    LDA #$A0
    STA ObjHP+1, X

    ; Reset metastate and unintialized flag, so that this neck is ready to go.
    LDA #$00
    STA ObjMetastate+1, X
    STA ObjUninitialized+2, X   ; UNKNOWN: Is this a mistake? (+2 instead of +1)

    ; Invincible to everything but the sword.
    LDA #$FE
    STA ObjInvincibilityMask+1, X

    ; Bottom of the loop.
    DEX
    BPL :-

    ; Set speed flags to $FF (non-0/decelerate):
    ; * neck 0 X
    ; * neck 1 Y
    ; * neck 2 X
    ; * neck 3 Y
    ;
    ; Leave the other other speed flags 0 (accelerate).
    STX Gleeok_HeadInfo0+GleeokHead_SpeedFlagX
    STX Gleeok_HeadInfo1+GleeokHead_SpeedFlagY
    STX Gleeok_HeadInfo2+GleeokHead_SpeedFlagX
    STX Gleeok_HeadInfo3+GleeokHead_SpeedFlagY

    ; Set the V-direction counter of all necks to 3.
    LDA #$03
    STA Gleeok_HeadInfo0+GleeokHead_DirCounterV
    STA Gleeok_HeadInfo1+GleeokHead_DirCounterV
    STA Gleeok_HeadInfo2+GleeokHead_DirCounterV
    STA Gleeok_HeadInfo3+GleeokHead_DirCounterV

    ; Set the H-direction counter of all necks to 6.
    ASL
    STA Gleeok_HeadInfo0+GleeokHead_DirCounterH
    STA Gleeok_HeadInfo1+GleeokHead_DirCounterH
    STA Gleeok_HeadInfo2+GleeokHead_DirCounterH
    STA Gleeok_HeadInfo3+GleeokHead_DirCounterH

    ; Set neck 1 head delay timer to 12.
    ASL
    STA Gleeok_HeadInfo1+GleeokHead_Delay

    ; Set neck 2 head delay timer to 24.
    ASL
    STA Gleeok_HeadInfo2+GleeokHead_Delay

    ; Set neck 3 head delay timer to 36.
    ADC Gleeok_HeadInfo1+GleeokHead_Delay
    STA Gleeok_HeadInfo3+GleeokHead_Delay

    ; Leave neck 0 head delay timer as 0.
    RTS

ManhandlaBaseFrameImagesAndAttrs:
    .BYTE $00, $80, $02, $42, $04

ManhandlaSegmentOffsetsX:
    .BYTE $00, $00

ManhandlaSegmentOffsetsY:
    .BYTE $F0, $10, $00, $00, $00

InitManhandla:
    LDA #$40                    ; Play Digdogger/Manhandla/Patra roar.
    STA SampleRequest

    ; Choose a random 8-way direction.
    LDA Random, X
    AND #$07
    TAY
    LDA Directions8, Y
    STA ObjDir, X

    ; For 5 segments, from 4 to 0, indexed by Y register:
    ; Object slots 5 to 1 are accessed.
    LDY #$04
:
    ; Copy the direction from the base segment.
    LDA ObjDir+1
    STA a:ObjDir+1, Y

    ; All segments are considered Manhandla.
    LDA #$3C
    STA ObjType+1, Y

    ; Set the segment invincible to fire and boomerang.
    LDA #$E2
    STA ObjInvincibilityMask+1, Y

    ; Look up and store the base frame image and sprite attributes byte.
    ;
    ; While updating, the low bit will be flipped on and off
    ; for each animation frame.
    LDA ManhandlaBaseFrameImagesAndAttrs, Y
    STA Manhandla_ObjFrame+1, Y

    ; All segments start out autonomous, and can update immediately.
    LDA #$00
    STA ObjMetastate+1, Y
    STA ObjUninitialized+1, Y

    ; Copy object attributes and HP from the base segment.
    LDA ObjAttr+1
    STA ObjAttr+1, Y
    LDA ObjHP+1
    STA ObjHP+1, Y

    ; Look up the offset of this segment. Add it to the position of
    ; the base segment.
    ;
    ; The position of the base segment in slot 5 is calculated first.
    ; Its base position is the spawn position.
    LDA ObjX+5
    CLC
    ADC ManhandlaSegmentOffsetsX, Y
    STA a:ObjX+1, Y
    LDA ObjY+5
    CLC
    ADC ManhandlaSegmentOffsetsY, Y
    STA a:ObjY+1, Y

    ; Set low speed byte $80.
    LDA #$80
    STA Manhandla_ObjSpeedFrac+1, Y
    DEY
    BPL :-
    RTS

InitGohma:
    LDA #$20                    ; Play Dodongo/Gohma roar.
    STA SampleRequest

    ; Invincible to everything but arrows.
    LDA #$FB
    STA ObjInvincibilityMask, X

    ; Set shoot timer to 1, so that Gohma doesn't shoot right away.
    INC Gohma_ObjShootTimer, X

    ; Start at location ($80, $70).
    LDA #$80
    STA ObjX, X
    LDA #$70
    STA ObjY, X
    JMP ResetObjMetastateAndTimer

InitGleeokHead:
    JSR InitBlueKeese

    ; Set flying speed $BF and maximum $E0.
    LDA #$E0
    STA FlyerMaxSpeedFrac
    LDA #$BF
    STA Flyer_ObjSpeed, X
    RTS
