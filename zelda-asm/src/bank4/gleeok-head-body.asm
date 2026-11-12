; ============================================================================
; FILE:     bank4/gleeok-head-body.asm   BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9044-9454
; SUBJECTS: Gleeok
; TOPIC:    Gleeok head/segment draw and collisions, body
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0037 Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions  code  Description:
;   L0064 Gleeok_DrawHeadAndCheckCollisions               code
;   L0067 Gleeok_DrawSegmentAndCheckCollisions            code
;   L0083 Gleeok_CheckCollisions                          code
;   L0237 Gleeok_MoveHead                                 code
;   L0291 L127FA_Exit                                     code
;   L0294 Gleeok_DecHeadTimer                             code
;   L0305 Gleeok_ChangeCoordinateBySpeedFlag              code  Params:
;   L0316 GleeokBodyTiles0                                data
;   L0319 GleeokBodyTiles1                                data
;   L0322 GleeokBodyTiles2                                data
;   L0325 GleeokBodyBaseTileOffsets                       data
;   L0328 L_Gleeok_DecAnimTimerAndDraw                    code
;   L0333 Gleeok_DrawBody                                 code
;   L0359 L_Gleeok_WriteSprites                           code
; ============================================================================

    ; End unverified code

; Description:
; This block applies to heads and the bottom segment of a neck.
; These are drawn with sprites that come first in order, so that
; they are drawn above all other sprites, including Link.
;
; Params:
; A: tile number
;
; Normal sprite with level palette row
Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions:
    LDY #$03
    STY $03                     ; [03] holds sprite attributes
    PHA                         ; Save the tile number.

    ; To calculate the base offset of the sprite record:
    ; offset := neck index * 8
    ;
    ; This ends up pointing to one of the first 8 eight sprites,
    ; at an even index.
    LDA GleeokCurNeck
    ASL
    ASL
    ASL

    ; But if the segment is not a head (not at object slot 5), then
    ; add $20. This moves the offset to the second set of 8 sprites.
    CPX #$05
    BEQ :+
    CLC
    ADC #$20
:
    TAY
    PLA                         ; Restore the tile number.
    JSR Anim_WriteSpecificSprite
    JMP Gleeok_CheckCollisions

Gleeok_DrawHeadAndCheckCollisions:
    LDX #$05                    ; Head object slot

Gleeok_DrawSegmentAndCheckCollisions:
    LDA #$DA                    ; Neck segment tile
    CPX #$05
    BNE :+
    LDA #$DC                    ; Head tile
:
    CPX #$05
    BEQ Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions
    CPX #$01
    BEQ Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions

    ; The segment is not a head nor bottom. So, we can use a
    ; standard routine to draw. It uses one of the rolling sprites
    ; that is drawn under Link and the heads.
    JSR Anim_WriteLevelPaletteSprite

Gleeok_CheckCollisions:
    ; If this segment is not the head nor the bottom of the neck,
    ; then loop again to process the next segment.
    ;
    ; This kind of segment is only drawn, and is not checked
    ; for object collisions.
    CPX #$05
    BEQ :+
    CPX #$01
    BEQ :+
    JMP @NextLoopSegment
:
    JSR CheckMonsterCollisions

    ; If this segment was harmed, then set the writhing counter,
    ; and the animation counter to its lower value.
    LDA a:ObjShoveDir, X
    BEQ :+
    LDA #$06
    STA GleeokAnimationTimer
    STA Gleeok_WrithingCounter
:
    JSR ResetShoveInfo

    ; If this is the bottom segment, then go reset metastate and loop again.
    ; Gleeok can writhe when it's hit here, but it can't die.
    CPX #$01
    BEQ @ResetMetastate
    JSR PlayBossHitCryIfNeeded

    ; If this neck has not died, then go loop again.
    LDA ObjMetastate, X
    BEQ @NextLoopSegment

    ; This neck died. So, prepare to make a flying head.
    LDA #$60
    STA ObjHP, X                ; I don't see why this is needed, since it won't go in this obj slot.
    TXA                         ; Save the segment object slot in X.
    PHA

    ; If the segment is not a head, then skip making a flying head.
    ;
    ; But how can we get here in this case, since earlier branches
    ; would have been taken to skip all this for all other segments?
    ; This might be defensive coding.
    CPX #$05
    BNE @DestroyNeck

    ; The flying head will go in slot (neck index + 7).
    ; Switch the X register to it.
    LDA GleeokCurNeck
    CLC
    ADC #$07
    TAX

    ; Flag it uninitialized, so that it will be initialized next frame.
    LDA #$FF
    STA ObjUninitialized, X

    ; Copy the original head's coordinates to it.
    LDA ObjX+5
    STA ObjX, X
    LDA ObjY+5
    STA ObjY, X

    ; Set object type $46 (flying head).
    LDA #$46
    STA ObjType, X

@DestroyNeck:
    PLA                         ; Restore the segment object slot in X.
    TAX

    ; Hide the original head's sprite, and the base segment's sprite.
    LDA GleeokCurNeck
    ASL
    ASL
    ASL
    TAY
    LDA #$F8
    STA Sprites, Y
    STA Sprites+32, Y

    ; Add the bit for this neck to the dead neck mask.
    ; Copy it to [00].
    LDY GleeokCurNeck
    LDA LevelMasks, Y
    ORA GleeokDeadNeckMask
    STA GleeokDeadNeckMask
    STA $00

    ; Loop over the four bits in the copy of the dead neck mask.
    ; Count them up, and store the amount in [00].
    LDA #$00
    LDY #$04
:
    LSR $00
    ADC #$00
    DEY
    BNE :-
    STA $00

    ; Compare the number of dead necks to the original number
    ; (object type - $41).
    LDA ObjType+1
    SEC
    SBC #$41
    CMP $00

    ; If they're equal, then go handle the whole boss dying.
    BEQ @BossDied

    ; Else reset object metastate of the segment, and return.
    JMP ResetObjMetastate

@ResetMetastate:
    JSR ResetObjMetastate

@NextLoopSegment:
    ; Bottom of the segment loop.
    ; Decrement segment object slot.
    ;
    ; If >= 1, then go draw the next segment and check collisions.
    ; Else return.
    DEX
    CPX #$01
    BCC L127FA_Exit
    JMP Gleeok_DrawSegmentAndCheckCollisions

@BossDied:
    ; The whole gleeok died, because the last neck died.
    ;
    ; First, hide the first $10 sprites: the attached heads.
    JSR WriteBlankPrioritySprites
    JSR PlayBossDeathCry

    ; We want a death spark. So, set the metastate of the first
    ; monster slot to $11.
    LDA #$11
    STA ObjMetastate+1

    ; Reset the object type of slots 2 to $A.
    ;
    ; Beware a left over fireball in slot $B!
    LDY #$01

@DestroyFireballs:
    LDA #$00
    STA ObjType+1, Y
    INY
    CPY #$0A
    BCC @DestroyFireballs
    RTS

Gleeok_MoveHead:
    ; Don't do anything until the head's initial timer expires.
    LDA Gleeok_CurHeadDelay
    BNE Gleeok_DecHeadTimer

    ; Add 1 or -1 to X as needed.
    LDA ObjX+5
    LDY Gleeok_CurHeadSpeedFlagX
    JSR Gleeok_ChangeCoordinateBySpeedFlag
    STA ObjX+5

    ; Add 1 or -1 to Y as needed.
    LDA ObjY+5
    LDY Gleeok_CurHeadSpeedFlagY
    JSR Gleeok_ChangeCoordinateBySpeedFlag
    STA ObjY+5

    ; Increment the counter to change directions.
    ; If it's still < 4, then return.
    INC Gleeok_CurHeadDirChangeCounter
    LDA Gleeok_CurHeadDirChangeCounter
    CMP #$04
    BCC L127FA_Exit

    ; The counter reached 4. So, reset it.
    ; Then check the individual direction counters.
    LDA #$00
    STA Gleeok_CurHeadDirChangeCounter

    ; Increment the horizontal direction counter.
    ; If >= $C, then reset the counter, and flip direction.
    INC Gleeok_CurHeadDirCounterH
    LDA Gleeok_CurHeadDirCounterH
    CMP #$0C
    BCC @CheckVertical
    LDA #$00
    STA Gleeok_CurHeadDirCounterH
    LDA Gleeok_CurHeadSpeedFlagX
    EOR #$FF
    STA Gleeok_CurHeadSpeedFlagX

@CheckVertical:
    ; Increment the vertical direction counter.
    ; If >= 6, then reset the counter, and flip direction.
    INC Gleeok_CurHeadDirCounterV
    LDA Gleeok_CurHeadDirCounterV
    CMP #$06
    BCC L127FA_Exit
    LDA #$00
    STA Gleeok_CurHeadDirCounterV
    LDA Gleeok_CurHeadSpeedFlagY
    EOR #$FF
    STA Gleeok_CurHeadSpeedFlagY

L127FA_Exit:
    RTS

Gleeok_DecHeadTimer:
    DEC Gleeok_CurHeadDelay
    RTS

; Params:
; A: a value to change
; Y: 0 => increment, 1 => decrement
;
; Returns:
; A: original value +/- 1
;
Gleeok_ChangeCoordinateBySpeedFlag:
    BNE @Subtract
    CLC
    ADC #$01
    RTS

@Subtract:
    SEC
    SBC #$01
    RTS

GleeokBodyTiles0:
    .BYTE $C0, $C4, $C8, $C2, $C6, $CA

GleeokBodyTiles1:
    .BYTE $CC, $C4, $CE, $C2, $C6, $D0

GleeokBodyTiles2:
    .BYTE $D2, $D6, $D8, $D4, $C6, $D0

GleeokBodyBaseTileOffsets:
    .BYTE $06, $00, $06, $0C

L_Gleeok_DecAnimTimerAndDraw:
    ; Decrement animation timer, and go draw.
    DEC GleeokAnimationTimer
    JMP L_Gleeok_WriteSprites

Gleeok_DrawBody:
    ; The animation timer controls when the animation frame changes.
    ; If it has not expired, then go decrement it and draw.
    LDA GleeokAnimationTimer
    BNE L_Gleeok_DecAnimTimerAndDraw

    ; The animation timer has expired. There are two choices for arming it.
    ;
    ; When writhing after a hit, Gleeok animates faster.
    ; So, when the writhing counter <> 0, set a shorter time (6).
    ; Otherwise, set a longer time ($10).
    LDA #$10
    LDY Gleeok_WrithingCounter
    BEQ :+
    DEC Gleeok_WrithingCounter
    LDA #$06
:
    STA GleeokAnimationTimer

    ; Cycle the animation frame from 0 to 3.
    LDA GleeokBodyAnimationFrame
    CLC
    ADC #$01
    AND #$03
    STA GleeokBodyAnimationFrame

L_Gleeok_WriteSprites:
    ; For each row, from 0 to 1, indexed by [06]:
    LDA #$00
    STA $06

    ; Get the base tile offset for the current animation frame.
    LDY GleeokBodyAnimationFrame
    LDX GleeokBodyBaseTileOffsets, Y

@LoopDrawRow:
    ; For each column from 0 to 2, indexed by [07]:
    LDA #$00
    STA $07

@LoopDrawColumn:
    ; Point to the next rolling sprite record.
    LDY RollingSpriteIndex
    LDA SpriteOffsets, Y
    TAY

    ; Set sprite's Y to ($57 + row * $10).
    LDA $06
    ASL
    ASL
    ASL
    ASL
    ADC #$57
    STA Sprites, Y

    ; Look up and set the tile for this spot.
    LDA GleeokBodyTiles0, X
    STA Sprites+1, Y

    ; If temporarily invincible, then set sprite attributes to
    ; cycled palette rows.
    ;
    ; Else use level palette row 7.
    LDA ObjInvincibilityTimer+5
    BNE :+
    LDA #$03
:
    AND #$03
    STA Sprites+2, Y

    ; Set sprite's X to ($74 + column * 8).
    LDA $07
    ASL
    ASL
    ASL
    ADC #$74
    STA Sprites+3, Y

    ; Increment the index for fetching tiles.
    INX
    JSR CycleCurSpriteIndex

    ; Bottom of the column loop.
    ; Increment the column until it = 3.
    INC $07
    LDA $07
    CMP #$03
    BCC @LoopDrawColumn

    ; Bottom of the row loop.
    ; Increment the row until it = 2.
    INC $06
    LDA $06
    CMP #$02
    BCC @LoopDrawRow
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF
