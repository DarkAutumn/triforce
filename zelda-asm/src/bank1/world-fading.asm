; ============================================================================
; FILE:     bank1/world-fading.asm       BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4741-4830
; SUBJECTS: (none)
; TOPIC:    Palette fade cycles, blank priority sprites
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 AnimateWorldFading            code  Params:
;   L0078 BlankPrioritySpriteTemplates  data
;   L0081 WriteBlankPrioritySprites     code
; ============================================================================

; Params:
; [$051C]: cycle number
;
; Returns:
; Z: 1 if done updating
;
AnimateWorldFading:
    ; ObjTimer[12] is the world fade timer. Every 10 frames,
    ; we step to the next half palette. There are 4 steps to fading.
    LDA ObjTimer+12
    BNE @Exit                   ; If the timer hasn't expired, then only delay (return).
    LDA FadeCycle
    BPL :+                      ; If the cycle number is negative, then this is a reverse fade.
    EOR #$83
:
    STA $00                     ; Map cycle value with hex digits XY to offset (X0 + Y*8).
    ASL
    ASL
    ASL
    CLC
    ADC $00
    AND #$FC
    TAY

    ; Start writing in dynamic transfer buf from the next position
    ; available. X holds this offset.
    LDX DynTileBufLen
    LDA #$3F                    ; Write PPU address $3F08, bottom half of background palette.
    STA DynTileBuf, X
    INX
    LDA #$08
    STA DynTileBuf, X
    INX                         ; %A holds 8. Write that as the length of the record.
    STA DynTileBuf, X

    ; Copy 8 bytes of half palette into transfer buf.
    STA $00
    INX

@CopyPalette:
    LDA LevelInfo_PaletteCycles, Y
    STA DynTileBuf, X
    INY
    INX
    DEC $00
    BNE @CopyPalette
    LDA #$FF                    ; Write the end marker.
    STA DynTileBuf, X
    STX DynTileBufLen           ; Store the new position into the dynamic transfer buf as the length.
    INC FadeCycle
    LDA FadeCycle
    AND #$0F
    CMP #$04
    BEQ @ReturnDone
    LDA #$0A                    ; Update every 10 frames.
    STA ObjTimer+12

@Exit:
    RTS

@ReturnDone:
    LDA #$00
    RTS

BlankPrioritySpriteTemplates:
    .BYTE $3D, $1C, $20, $00, $DD, $1C, $20, $00

WriteBlankPrioritySprites:
    ; UNKNOWN: Why are their Y coordinates $3D and $DD?
    ; Those values are the highest and lowest that Link can normally take.
    ;
    ; Repeat two sprites in the first $10 records. They have
    ; transparent tile $1C, priority 1 (behind BG), and X=0.
    ; The first of the pair has Y=$3D, and the second one has Y=$DD.
    ;
    ; These sprites are first in the list. So they are used for things
    ; that must be shown above everything else, including Link.
    LDY #$00
    LDX #$00
:
    LDA BlankPrioritySpriteTemplates, X
    STA Sprites, Y
    INX
    TXA
    AND #$07
    TAX
    INY
    CPY #$40
    BNE :-
    RTS
