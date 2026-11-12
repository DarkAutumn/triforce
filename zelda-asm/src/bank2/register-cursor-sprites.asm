; ============================================================================
; FILE:     bank2/register-cursor-sprites.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2081-2222
; SUBJECTS: (none)
; TOPIC:    Modes $E and $F: the slot cursor, name cursor and character-board cursor sprites, and
;           Select moving the slot cursor.
; SEE ALSO: bank2/register-eliminate-init.asm (ModeEandFCursorSprites, ModeEandFSlotCursorYs, caller of
;           the setup); bank2/register-eliminate-update.asm (callers); bank2/register-name-input.asm
;           (moves the cursors).
; KEY FACTS:
;   - Sprite 0 is the slot cursor, sprite 1 the name cursor, sprite 2 the character-board cursor
;     (ModeEandF_SetUpCursorSprites).
;   - Select steps CurSaveSlot through 0, 1, 2, 3 ("End") and wraps (UpdateModeEandF_Idle).
;   - In mode $E Select skips slots that are already active, so a registered name cannot be edited;
;     in mode $F every slot can be chosen (UpdateModeEandF_Idle).
;   - Changing the slot in mode $E restarts name entry at the first character of the new slot
;     (UpdateModeEandF_Idle).
;   - The two block cursors blink: shown 8 frames, hidden 8 frames, by FrameCounter bit 3
;     (ModifyFlashingCursorY). Both are hidden while the slot cursor is on "End" (ObjY = $77).
; CONTENTS: (generated; L = line in this file)
;   L0032 ModeEandF_SetUpCursorSprites                  code  Writes the three cursor sprites for modes $E and $F and the starting cursor positions for mode $E.
;   L0077 ModeEandF_WriteNameCursorSpritePosition       code  Writes the name cursor sprite's Y (blinking; hidden on "End") and X.
;   L0091 ModeE_SetNameCursorSpriteX                    code  Copies the name cursor X (ObjX of object slot 0) to its sprite; also the tail of name entry.
;   L0099 ModeEandF_WriteCharBoardCursorSpritePosition  code  Writes the character-board cursor sprite's Y (blinking; hidden on "End") and X.
;   L0121 ModifyFlashingCursorY                         code  Turns a cursor Y into its sprite Y, hiding the sprite on alternate 8-frame periods.
;   L0135 UpdateModeEandF_Idle                          code  Modes $E and $F: on Select, moves the slot cursor to the next slot or "End".
; ============================================================================

; ----------------------------------------------------------------------------
; Writes the three cursor sprites for modes $E and $F and the starting cursor positions for mode $E.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Writes the name cursor sprite's Y (blinking; hidden on "End") and X.
; ----------------------------------------------------------------------------
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

; Copies the name cursor X (ObjX of object slot 0) to its sprite; also the tail of name entry.
ModeE_SetNameCursorSpriteX:
    LDA ObjX
    STA Sprites+7               ; Set name cursor sprite X.
    RTS

; ----------------------------------------------------------------------------
; Writes the character-board cursor sprite's Y (blinking; hidden on "End") and X.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Turns a cursor Y into its sprite Y, hiding the sprite on alternate 8-frame periods.
; IN:   A = cursor Y.
; OUT:  Y = A - 8 (the visible part of the block cursor is the bottom half of the sprite), or $F8
;       (off screen) while FrameCounter bit 3 is clear.
; ----------------------------------------------------------------------------
ModifyFlashingCursorY:
    SEC
    SBC #$08
    TAY
    LDA FrameCounter            ; While bit 3 is clear (8 frames of every 16), put the Y off screen.
    AND #$08
    BNE :+
    LDY #$F8
:
    RTS

; ----------------------------------------------------------------------------
; Modes $E and $F: on Select, moves the slot cursor to the next slot or "End".
; ----------------------------------------------------------------------------
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
    STA Sprites                 ; Writes sprite 0 (slot cursor) Y again, not Sprites+4; same value as above.
    LDA #$70                    ; Name cursor X is $70.
    STA Sprites+7
    STA ObjX
    LDA #$00                    ; Reset IsNameFieldInitialized and NameCharOffset.
    STA IsNameFieldInitialized
    STA NameCharOffset
    LDY CurSaveSlot
    CPY #$03
    BEQ @Exit                   ; If selection is "End" option, then return.
    LDA IsSaveSlotActive, Y
    BNE @ChangeSelection        ; (Mode $E only) If the slot is already active, then cycle again.

@Exit:
    RTS
