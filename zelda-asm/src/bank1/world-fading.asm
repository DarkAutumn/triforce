; ============================================================================
; FILE:     bank1/world-fading.asm       BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4741-4830
; SUBJECTS: Palette, Sprite
; TOPIC:    The dungeon palette fade (cellar, dark room, death): one 8-byte half palette to the PPU every
;           10 frames. Also the 16 blank mask sprites that hide Link in a dungeon's top and bottom doorways.
; SEE ALSO: bank5/mode7-scroll-init.asm (series $40 when scrolling into a dark room); bank5/enter-room.asm
;           (series $C0 entering a light room from a dark one); bank5/cave-cellar-enter.asm (cellar series
;           $00, $A0, $20, $80); bank5/mode11-death-init.asm and bank5/mode11-death.asm (death series $60);
;           bank4/shoot-helpers-candle.asm (candle fire starts $C0); bank4/ganon.asm (reads FadeCycle);
;           bank7/init-mode5-play.asm, bank5/mode3-mode6-init.asm, bank5/submenu-states.asm,
;           bank4/gleeok-head-body.asm (callers of Sprite_WriteBlankPrioritySprites).
; KEY FACTS:
;   - FadeCycle: high nibble = series ($00, $20, $40, $60), low nibble = step 0-3; bit 7 set ($80, $A0,
;     $C0) plays series $00, $20, $40 backwards, step 3 first. Palette_AnimateFade
;   - A step sends 8 bytes, LevelInfo_PaletteCycles[series + step * 8], to PPU $3F08-$3F0F: background
;     palette rows 2 and 3. Sprite palettes and background rows 0 and 1 do not fade. Palette_AnimateFade
;   - A step is written when ObjTimer+12 (the timer of object slot $0C) is 0; the timer is then set to 10
;     frames. The call that writes the fourth step returns Z set: 30 frames after the first step.
;     Palette_AnimateFade
;   - Callers poll the routine once per frame and wait for Z set; it changes no game state besides
;     FadeCycle, the timer and the transfer buffer. Palette_AnimateFade
;   - Sprite_WriteBlankPrioritySprites fills sprite records $00-$0F with 8 blank sprites at Y=$3D and 8 at
;     Y=$DD (Link's Y at the top and bottom screen edges). The PPU draws only the first 8 sprites on a
;     scanline, so nothing else is drawn in those two 16-pixel bands: Link vanishes inside the top and
;     bottom doorways. Every caller runs only in a dungeon. Sprite_BlankPriorityTemplates
; CONTENTS: (generated; L = line in this file)
;   L0044 Palette_AnimateFade               code  Sends the next step of the palette fade FadeCycle to the PPU, one step per 10 frames.
;   L0113 Sprite_BlankPriorityTemplates     data  TABLE: the two mask sprite records copied alternately into sprite records $00-$0F; 4 bytes each =
;   L0129 Sprite_WriteBlankPrioritySprites  code  Fills sprite records $00-$0F with 8 blank sprites at Y=$3D and 8 at Y=$DD (dungeon doorway masks).
; ============================================================================

; ----------------------------------------------------------------------------
; Sends the next step of the palette fade FadeCycle to the PPU, one step per 10 frames.
; IN:   FadeCycle = series + step (bit 7 = backwards); ObjTimer+12 = frames until the next step.
; OUT:  Z set (A = 0) when the fourth and last step has just been written; Z clear otherwise.
;       When a step is written: FadeCycle + 1, ObjTimer+12 = $0A (not after the last step), an 8-byte
;       record for PPU $3F08 appended to DynTileBuf. Uses [00], X, Y.
; NOTE: Series (offsets into LevelInfo_PaletteCycles, 4 steps of 8 bytes each): $00 light level to dark
;       cellar, $20 light cellar to dark, $40 light room to dark room, $60 death (red). Callers store $00,
;       $20, $40, $60 or the reversed $80, $A0, $C0. A step is written on the first call if the timer is 0,
;       so a whole fade takes 30 frames (steps at frames 0, 10, 20, 30).
; ----------------------------------------------------------------------------
Palette_AnimateFade:
    ; ObjTimer+12 (the timer of object slot $0C) spaces the steps. Still counting: return, Z clear.
    LDA ObjTimer+12
    BNE @Exit

    ; Bit 7 of FadeCycle set = backwards: EOR #$83 clears bit 7 and turns step 0-3 into 3-0
    ; ($C0 -> $43, $C1 -> $42, $C2 -> $41, $C3 -> $40).
    LDA FadeCycle
    BPL :+
    EOR #$83
:
    ; Y := series + step * 8, computed as (value * 9) AND $FC. Exact for series $00, $20, $40, $60
    ; and steps 0-3 ($43 -> $58).
    STA $00
    ASL
    ASL
    ASL
    CLC
    ADC $00
    AND #$FC
    TAY

    ; Append a record at the end marker of the dynamic transfer buf (offset X): PPU address $3F08
    ; (background palette rows 2 and 3), byte count 8.
    LDX DynTileBufLen
    LDA #$3F
    STA DynTileBuf, X
    INX
    LDA #$08
    STA DynTileBuf, X
    INX
    STA DynTileBuf, X           ; A is still 8: the record's byte count.

    ; Copy the step's 8 palette bytes; [00] counts them down.
    STA $00
    INX

@CopyPalette:
    LDA LevelInfo_PaletteCycles, Y
    STA DynTileBuf, X
    INY
    INX
    DEC $00
    BNE @CopyPalette
    LDA #$FF                    ; New end marker.
    STA DynTileBuf, X
    STX DynTileBufLen           ; DynTileBufLen := offset of the end marker.

    ; Next step. Low nibble 4 = all four steps written: return done.
    INC FadeCycle
    LDA FadeCycle
    AND #$0F
    CMP #$04
    BEQ @ReturnDone
    LDA #$0A                    ; 10 frames until the next step. Leaves Z clear.
    STA ObjTimer+12

@Exit:
    RTS

@ReturnDone:
    LDA #$00                    ; Z set: the fade is complete.
    RTS

; TABLE: the two mask sprite records copied alternately into sprite records $00-$0F; 4 bytes each =
; Y, tile, attributes, X. Y = $3D (first) and $DD (second), tile $1C (blank: sprites are 8x16, so
; the record shows patterns $1C and $1D of pattern table 0, and all 32 bytes of the two in
; Chr_CommonSpritePatterns are 0), attributes $20 (behind the background, sprite palette row 0),
; X = 0.
Sprite_BlankPriorityTemplates:
    .BYTE $3D, $1C, $20, $00, $DD, $1C, $20, $00

; ----------------------------------------------------------------------------
; Fills sprite records $00-$0F with 8 blank sprites at Y=$3D and 8 at Y=$DD (dungeon doorway masks).
; OUT:  Sprites+$00..+$3F written; X = 0; Y = $40.
; NOTE: $3D and $DD are Link's Y at the top and bottom screen edges (Link_ScreenEdgeBounds). Sprites are
;       8x16 and the PPU draws at most 8 sprites per scanline, lowest record first. These 8 + 8 records
;       come before all others (Link is records $12 and $13), so within the two 16-pixel bands that start
;       at sprite Y $3D and $DD no other sprite is drawn: Link and anything else there is hidden, which
;       is how Link disappears in the top and bottom doorways of a dungeon room.
;       Callers, all on dungeon-only paths (each tests CurLevel first, or is dungeon code): the room
;       tasks of mode 5 init and of room entry in modes 4, 9, $B, $C
;       (Room_RunEntryTasks_MarkVisited), Mode6Leave_Init, the submenu closing, and
;       Gleeok's death (its heads had used these records). The overworld never writes these masks.
; ----------------------------------------------------------------------------
Sprite_WriteBlankPrioritySprites:
    LDY #$00                    ; Y = offset into Sprites, $00-$3F.
    LDX #$00                    ; X = offset into the 8-byte template, wrapping at 8.
@Loop:
    LDA Sprite_BlankPriorityTemplates, X
    STA Sprites, Y
    INX
    TXA
    AND #$07
    TAX
    INY
    CPY #$40
    BNE @Loop
    RTS
