; ============================================================================
; FILE:     bank2/register-name-input.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1798-2079
; SUBJECTS: ModeERegister
; TOPIC:    Mode $E name entry: moving the character-board cursor with the direction buttons (with
;           auto-repeat) and typing a character (A) or skipping one (B) in the chosen slot's name.
; SEE ALSO: bank2/register-eliminate-init.asm (ModeERegister_CharMap, ModeERegister_CharBoardYStepsAndBounds,
;           ModeERegister_SlotNameOffsets, transfer header templates); bank2/register-eliminate-update.asm (caller);
;           bank2/register-cursor-sprites.asm (cursor sprites, ModeERegister_SetNameCursorSpriteX).
; KEY FACTS:
;   - The character board is 4 rows of 11 ($2C entries of ModeERegister_CharMap); the cursor is object slot 1's
;     ObjX/ObjY: X $30-$D0 in steps of $10, Y $87-$B7 in steps of $10 (ModeERegister_MoveCharBoardCursor,
;     ModeERegister_StepCharBoardCursorY).
;   - A held direction moves at once, again $11 frames later (timer $10), then every 9 frames (timer
;     8); only a single direction counts (ModeERegister_MoveCharBoardCursor).
;   - Right past the last column moves to the first column of the next row, Left the reverse; all four
;     directions wrap around the board (ModeERegister_MoveCharBoardCursor).
;   - A writes the highlighted character into SaveSlotNames and the nametable and advances; B only
;     advances. There is no delete: the name cursor wraps after 8 characters (ModeERegister_HandleAOrB).
;   - The name cursor is object slot 0's ObjX: $70-$A8 in steps of 8 (ModeERegister_HandleAOrB).
; CONTENTS: (generated; L = line in this file)
;   L0033 ModeERegister_HandleDirections             code  Reads the held direction buttons for the character board; none held resets the repeat state.
;   L0039 ModeERegister_ResetButtonRepeat            code  Clears the button hold/repeat state; IN: A = 0.
;   L0049 ModeERegister_MoveCharBoardCursor          code  Moves the character-board cursor one step in the held direction, with auto-repeat and wrapping.
;   L0187 ModeERegister_StepCharBoardCursorY         code  Moves the character-board cursor one row down or up, wrapping at the bottom or top.
;   L0205 ModeERegister_StepCharBoardCursorY_Return  code  Shared return of the routines above and below.
;   L0211 ModeERegister_HandleAOrB                   code  Sets up the name field for the chosen slot once, then handles A (type a character) and B (skip).
; ============================================================================

; ----------------------------------------------------------------------------
; Reads the held direction buttons for the character board; none held resets the repeat state.
; ----------------------------------------------------------------------------
ModeERegister_HandleDirections:
    LDA ButtonsDown
    AND #$0F                    ; Filter and keep direction buttons.
    BNE ModeERegister_MoveCharBoardCursor    ; If no button is down, then reset repeat state.

; Clears the button hold/repeat state; IN: A = 0.
ModeERegister_ResetButtonRepeat:
    STA IsHoldingButton
    STA IsButtonRepeating
    STA ButtonRepeatTimer
    RTS

; ----------------------------------------------------------------------------
; Moves the character-board cursor one step in the held direction, with auto-repeat and wrapping.
; IN:   A = direction bits held (nonzero).
; ----------------------------------------------------------------------------
ModeERegister_MoveCharBoardCursor:
    TAY
    LDA IsHoldingButton
    BNE @CheckSameButton        ; If we weren't holding a button last frame,
    STY HeldButton              ; Store current buttons down.
    INC IsHoldingButton      ; Now we definitely are holding a button.

@CheckSameButton:
    LDA ButtonsDown
    AND #$0F
    CMP HeldButton
    BEQ @CheckRepeat            ; If it's not the same button as before,
    LDA #$00                    ; then reset repeat state.
    JSR ModeERegister_ResetButtonRepeat

@CheckRepeat:
    LDA ButtonRepeatTimer
    BEQ @ChooseRepeatDelay      ; Once the repeat timer reaches zero, handle the direction button again.
    DEC ButtonRepeatTimer       ; Otherwise, only count down the timer.
    RTS

@ChooseRepeatDelay:
    ; After the first move the timer is $10, after later moves 8. The
    ; timer counts down one per frame and the move happens on the frame
    ; after it reaches 0: $11 frames to the first repeat, then 9.
    LDY #$08
    LDA IsButtonRepeating
    BNE :+
    LDY #$10
:
    STY ButtonRepeatTimer
    LDA ButtonsDown
    AND #$0F
    CMP #$01
    BNE @Left

    ; Pressed Right.
    ;
    ; Increase CharBoardIndex [$041F] to put cursor at character to the right.
    INC CharBoardIndex
    LDA ObjX+1                  ; Move the char board cursor right one spot.
    CLC
    ADC #$10
    STA ObjX+1
    CMP #$E0
    BNE :+                      ; If still on the board, then go finish.
    LDA #$30                    ; Otherwise, move the cursor to the left end.
    STA ObjX+1
    LDX #$00                    ; Cycle down.
    JSR ModeERegister_StepCharBoardCursorY
    LDA ModeE_HasWrappedBoardY
    BEQ :+                      ; If wrapped around to the top,
    LDA #$00                    ; then reset CharBoardIndex.
    STA CharBoardIndex
:
    JMP @FinishInput            ; Go finish.

@Left:
    CMP #$02
    BNE @Down

    ; Pressed Left.
    ;
    ; Decrease CharBoardIndex [$041F] to put cursor at character to the left.
    DEC CharBoardIndex
    LDA ObjX+1                  ; Move the char board cursor left one spot.
    SEC
    SBC #$10
    STA ObjX+1
    CMP #$20
    BNE :+                      ; If still on the board, then go finish.
    LDA #$D0                    ; Otherwise, move the cursor to the right end.
    STA ObjX+1
    LDX #$03                    ; Cycle up.
    JSR ModeERegister_StepCharBoardCursorY
    LDA ModeE_HasWrappedBoardY
    BEQ :+                      ; If wrapped around to the bottom,
    LDA #$2B                    ; then set CharBoardIndex to the last index.
    STA CharBoardIndex
:
    JMP @FinishInput            ; Go finish.

@Down:
    CMP #$04
    BNE @Up

    ; Pressed Down.
    ;
    ; Increase CharBoardIndex [$041F] by $B (one row down).
    LDA CharBoardIndex
    CLC
    ADC #$0B
    STA CharBoardIndex
    LDX #$00                    ; Cycle down.
    JSR ModeERegister_StepCharBoardCursorY
    LDA ModeE_HasWrappedBoardY
    BEQ @Finish                 ; If didn't wrap around, then go finish.
    LDA CharBoardIndex          ; Wrapped around. So, move to top row.
    SEC
    SBC #$2C
    STA CharBoardIndex

@Finish:
    JMP @FinishInput            ; Go finish.

@Up:
    CMP #$08
    BNE @Exit                   ; If no single direction was pressed, then return.

    ; Pressed Up.
    ;
    ; Decrease CharBoardIndex [$041F] by $B (one row up).
    LDA CharBoardIndex
    SEC
    SBC #$0B
    STA CharBoardIndex
    LDX #$03                    ; Cycle up.
    JSR ModeERegister_StepCharBoardCursorY
    LDA ModeE_HasWrappedBoardY
    BEQ @FinishInput            ; If didn't wrap around, then go finish.
    LDA CharBoardIndex          ; Wrapped around. So, move to bottom row.
    CLC
    ADC #$2C
    STA CharBoardIndex

@FinishInput:
    LDA #$01
    STA IsButtonRepeating
    STA Tune1Request            ; Request "selection changed" tune (same as rupee taken).

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Moves the character-board cursor one row down or up, wrapping at the bottom or top.
; IN:   X = 0 for down, 3 for up (offset of a triplet in ModeERegister_CharBoardYStepsAndBounds).
; OUT:  ModeE_HasWrappedBoardY = 1 if the cursor wrapped around, else 0.
; ----------------------------------------------------------------------------
ModeERegister_StepCharBoardCursorY:
    LDY #$00                    ; Assume we don't wrap around.
    LDA ObjY+1                  ; Move the char board cursor Y one spot in given direction.
    CLC
    ADC ModeERegister_CharBoardYStepsAndBounds, X    ; Add $10 or -$10 ($F0), depending on X passed in (0 or 3).
    STA ObjY+1
    INX                         ; Look at boundaries.
    CMP ModeERegister_CharBoardYStepsAndBounds, X
    BNE @ReturnValue            ; If we didn't reach the boundary, then return Y=0.
    INX                         ; Set cursor Y to wrapped around position.
    LDA ModeERegister_CharBoardYStepsAndBounds, X
    STA ObjY+1
    INY                         ; Return ModeE_HasWrappedBoardY [$042A]=1.

@ReturnValue:
    STY ModeE_HasWrappedBoardY

; Shared return of the routines above and below.
ModeERegister_StepCharBoardCursorY_Return:
    RTS

; ----------------------------------------------------------------------------
; Sets up the name field for the chosen slot once, then handles A (type a character) and B (skip).
; ----------------------------------------------------------------------------
ModeERegister_HandleAOrB:
    LDA IsNameFieldInitialized

    ; If IsNameFieldInitialized [$0420] is set, then skip initializing
    ; the name field.
    BNE @CheckAB
    LDY CurSaveSlot
    CPY #$03
    BEQ ModeERegister_StepCharBoardCursorY_Return              ; If at the "End" option, then return.

    ; Set NameCharOffset [$0421] to the offset of first char
    ; in the current slot's name.
    LDA ModeERegister_SlotNameOffsets, Y
    STA NameCharOffset

    ; Get the offset of the end of the initial name character
    ; transfer record header for the current slot.
    LDX ModeERegister_NameCharHeaderEndOffsets, Y
    LDY #$02                    ; Each transfer record header is 3 bytes.

@CopyHeaderTemplate:
    LDA ModeERegister_NameCharTransferHeaders, X
    STA NameInputCharBuf, Y     ; Copy a byte of transfer header for current slot to [$0422][Y].
    DEX
    DEY
    BPL @CopyHeaderTemplate
    INC IsNameFieldInitialized    ; Set IsNameFieldInitialized [$0420] to mark the name field initialized.

@CheckAB:
    ; At this point:
    ; - NameCharOffset [$0421] holds the offset in SaveSlotNames of the current slot's
    ;   first character; it advances as characters are entered.
    ; - [$0422] to [$0424] hold a transfer record header whose VRAM address points to the
    ;   start of the slot's name field in the nametable; it advances the same way.
    ; - IsNameFieldInitialized [$0420] is set to 1.
    LDA ButtonsPressed
    AND #$C0
    BEQ @Exit                   ; If neither A nor B was pressed, then go finish.

    ; A or B was pressed.
    CMP #$80
    BNE @MoveCursor             ; If B was pressed, then go move the name cursor only.

    ; A was pressed.
    ;
    ; Request to play the character click tune (same as bomb set).
    LDY #$20
    STY Tune0Request
    LDY #$02                    ; Copy the 3-byte transfer header ([$0422-0424]) to DynTileBuf.

@CopyHeader:
    LDA NameInputCharBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyHeader
    STY DynTileBuf+4            ; Write the end marker to dynamic transfer buf.
    LDX NameCharOffset
    LDY CharBoardIndex          ; CharBoardIndex in [$041F] will index into character map.
    LDA ModeERegister_CharMap, Y        ; Get the character that's highlighted.
    STA DynTileBuf+3            ; Write the chosen character to dynamic transfer buf.
    STA SaveSlotNames, X                ; Set the character in the name.

@MoveCursor:
    LDA ObjX                    ; Move name cursor right 8 pixels.
    CLC
    ADC #$08
    STA ObjX
    INC NameCharOffset          ; Increment NameCharOffset [$0421].
    INC NameInputCharBuf+1      ; Increment the low VRAM address where next char will go.

    ; If VRAM address still points inside the name field,
    ; then go finish.
    ;
    ; The idea is that each name field in nametable begins at
    ; an address ending in $E. For example, slot 0 has a name
    ; at VRAM addresses $20CE to $20D5.
    ;
    ; Once the $E becomes a 6, we've gone past the end of
    ; the name field.
    ;
    ; Keep in mind that [0423] is the second byte of the transfer
    ; record header.
    LDA NameInputCharBuf+1
    AND #$0F
    CMP #$06
    BNE @Exit

    ; The VRAM address now points outside the name field.
    ; So, wrap around to the beginning of the name field.
    ;
    ; For example, $20D6 -> $20CE.
    LDA NameInputCharBuf+1
    SEC
    SBC #$08
    STA NameInputCharBuf+1

    ; It also means that we went past the end of the save slot
    ; info name. Set the offset to the beginning of the name.
    LDY CurSaveSlot
    LDA ModeERegister_SlotNameOffsets, Y
    STA NameCharOffset

    ; If the name cursor has gone past the end of the field,
    ; then wrap around.
    LDA ObjX
    CMP #$B0
    BNE @Exit
    LDA #$70
    STA ObjX

@Exit:
    JMP ModeERegister_SetNameCursorSpriteX    ; Go set the name cursor sprite X.
