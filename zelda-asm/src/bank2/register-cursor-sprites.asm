; ============================================================================
; FILE:     bank2/register-cursor-sprites.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2081-2222
; SUBJECTS: (none)
; TOPIC:    Mode E/F cursor sprites, slot cycling
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 ModeEandF_SetUpCursorSprites                  code
;   L0059 ModeEandF_WriteNameCursorSpritePosition       code
;   L0072 ModeE_SetNameCursorSpriteX                    code
;   L0077 ModeEandF_WriteCharBoardCursorSpritePosition  code
;   L0104 ModifyFlashingCursorY                         code  Params:
;   L0115 UpdateModeEandF_Idle                          code
; ============================================================================

ModeEandF_SetUpCursorSprites:
    ; Copy almost 3 sprite records ($B bytes) to byte 1 of
    ; Sprites block. Only sprite 0 byte 0 is missing.
    ; These are the cursor sprites.
    LDY #$0A

@CopySprites:
    LDA ModeEandFCursorSprites, Y
    STA Sprites+1, Y
    DEY
    BPL @CopySprites
    LDY CurSaveSlot             ; Set the Y of the slot cursor sprite (#0) according to current save slot.
    LDA ModeEandFSlotCursorYs, Y
    STA ObjY
    STA Sprites
    LDA GameMode
    CMP #$0F

    ; There's more work in mode $E.
    ; The name cursor position is held in ObjX/ObjY[0].
    ; The char-board cursor position is held in ObjX/ObjY[1].
    BEQ @Exit
    LDA #$F3                    ; Use a heart tile for the slot cursor sprite (#0).
    STA Sprites+1

    ; Because the visible part of the cursor block sprite is
    ; in the bottom, move the name cursor's sprite 8 pixels
    ; above its ObjY.
    LDA ObjY
    SEC
    SBC #$08
    STA Sprites+4
    LDA #$70                    ; The base name cursor X is $70.
    STA ObjX
    LDA #$87                    ; The base char-board cursor Y is $87.
    STA ObjY+1
    LDA #$30                    ; The base char-board cursor X is $30.
    STA ObjX+1

@Exit:
    RTS

ModeEandF_WriteNameCursorSpritePosition:
    LDA ObjY
    CMP #$77                    ; If name cursor Y corresponds to a save slot,
    BNE @WriteCursorY           ; then go write the appropriate Y to name cursor sprite.
    LDA #$F8                    ; else hide name cursor.
    STA Sprites+4
    RTS

@WriteCursorY:
    LDA ObjY
    JSR ModifyFlashingCursorY   ; This returns the adjusted coordinate or $F8 to hide it.
    STY Sprites+4               ; Set name cursor sprite Y.

ModeE_SetNameCursorSpriteX:
    LDA ObjX
    STA Sprites+7               ; Set name cursor sprite X.
    RTS

ModeEandF_WriteCharBoardCursorSpritePosition:
    LDA ObjY
    CMP #$77                    ; If name cursor Y corresponds to a save slot,
    BNE @WriteCursorCoords      ; then go write the appropriate Y to char-board sprite.
    LDA #$F8                    ; else hide char-board sprite.
    STA Sprites+8
    RTS

@WriteCursorCoords:
    LDA ObjY+1
    JSR ModifyFlashingCursorY
    STY Sprites+8               ; Set char-board cursor sprite Y.
    LDA ObjX+1
    STA Sprites+11              ; Set char-board cursor sprite X.
    RTS

; Params:
; A: cursor Y
;
; Returns:
; Y: adjusted cursor Y, or $F8 to hide it
;
; Description:
; Adjust the Y coordinate to account for the visible part of
; block cursor being in the bottom half of sprite.
; Also, make the cursor flash.
;
ModifyFlashingCursorY:
    SEC
    SBC #$08
    TAY
    LDA FrameCounter            ; Every 8 frames, put the Y off screen.
    AND #$08
    BNE :+
    LDY #$F8
:
    RTS

UpdateModeEandF_Idle:
    LDA ButtonsPressed
    AND #$20
    BEQ @Exit                   ; If Select was not pressed, then return.

@ChangeSelection:
    LDA #$01                    ; Request to play the selection tune (same as rupee taken).
    STA Tune1Request
    INC CurSaveSlot             ; Choose the next slot.
    LDY CurSaveSlot
    CPY #$04
    BNE :+                      ; If went out of bounds, then wrap around.
    LDY #$00
    STY CurSaveSlot
:
    LDA ModeEandFSlotCursorYs, Y    ; Set sprite Y for new selection.
    STA Sprites
    LDA GameMode
    CMP #$0F
    BEQ @Exit                   ; If in mode $F, then return.
    LDA ObjY                    ; Move name cursor down $18 pixels.
    CLC
    ADC #$18
    STA ObjY
    CMP #$8F                    ; If we're past the "End" option,
    BNE :+
    LDA #$2F                    ; then wrap around.
    STA ObjY
:
    STA Sprites                 ; It seems that this should be Sprites+4.
    LDA #$70                    ; Name cursor X is $70.
    STA Sprites+7
    STA ObjX
    LDA #$00                    ; Reset InitializedNameField and NameCharOffset.
    STA InitializedNameField
    STA NameCharOffset
    LDY CurSaveSlot
    CPY #$03
    BEQ @Exit                   ; If selection is "End" option, then return.
    LDA IsSaveSlotActive, Y
    BNE @ChangeSelection        ; If the slot is not active, then cycle again.

@Exit:
    RTS
