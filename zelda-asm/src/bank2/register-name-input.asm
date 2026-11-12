; ============================================================================
; FILE:     bank2/register-name-input.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1798-2079
; SUBJECTS: (none)
; TOPIC:    Name entry cursor and characters
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 ModeE_HandleDirections       code
;   L0022 ResetButtonRepeatState       code
;   L0028 ModeE_HandleDirectionButton  code
;   L0167 CycleCharBoardCursorY        code  Params:
;   L0184 LA10A_Exit                   code
;   L0187 ModeE_HandleAOrB             code
; ============================================================================

ModeE_HandleDirections:
    LDA ButtonsDown
    AND #$0F                    ; Filter and keep direction buttons.
    BNE ModeE_HandleDirectionButton    ; If no button is down, then reset repeat state.

ResetButtonRepeatState:
    STA IsHoldingButton
    STA IsButtonRepeating
    STA ButtonRepeatTimer
    RTS

ModeE_HandleDirectionButton:
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
    JSR ResetButtonRepeatState

@CheckRepeat:
    LDA ButtonRepeatTimer
    BEQ @ChooseRepeatDelay      ; Once the repeat timer reaches zero, handle the direction button again.
    DEC ButtonRepeatTimer       ; Otherwise, only count down the timer.
    RTS

@ChooseRepeatDelay:
    ; If this is the first button press, then wait $10 frames
    ; to repeat; otherwise wait 8 frames.
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
    JSR CycleCharBoardCursorY
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
    JSR CycleCharBoardCursorY
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
    JSR CycleCharBoardCursorY
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
    JSR CycleCharBoardCursorY
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

; Params:
; X: 0 for down, 3 for up.
;
; Returns:
; ModeE_HasWrappedBoardY [$042A]=1 if wrapped arounnd.
;
; Assume we don't wrap around.
CycleCharBoardCursorY:
    LDY #$00
    LDA ObjY+1                  ; Move the char board cursor Y one spot in given direction.
    CLC
    ADC ModeE_CharBoardYOffsetsAndBounds, X    ; Add $10 or -$10 ($F0), depending on X passed in (0 or 3).
    STA ObjY+1
    INX                         ; Look at boundaries.
    CMP ModeE_CharBoardYOffsetsAndBounds, X
    BNE @ReturnValue            ; If we didn't reach the boundary, then return Y=0.
    INX                         ; Set cursor Y to wrapped around position.
    LDA ModeE_CharBoardYOffsetsAndBounds, X
    STA ObjY+1
    INY                         ; Return ModeE_HasWrappedBoardY [$042A]=1.

@ReturnValue:
    STY ModeE_HasWrappedBoardY

LA10A_Exit:
    RTS

ModeE_HandleAOrB:
    LDA IsNameFieldInitialized

    ; If IsNameFieldInitialized [$0420] is set, then skip initializing
    ; the name field.
    BNE @CheckAB
    LDY CurSaveSlot
    CPY #$03
    BEQ LA10A_Exit              ; If at the "End" option, then return.

    ; Set NameCharOffset [$0421] to the offset of first char
    ; in the current slot's name.
    LDA SlotToNameOffset, Y
    STA NameCharOffset

    ; Get the offset of the end of the initial name character
    ; transfer record header for the current slot.
    LDX SlotToInitialNameCharTransferHeaderEndOffsets, Y
    LDY #$02                    ; Each transfer record header is 3 bytes.

@CopyHeaderTemplate:
    LDA SlotToInitialNameCharTransferHeaders, X
    STA NameInputCharBuf, Y     ; Copy a byte of transfer header for current slot to [$0422][Y].
    DEX
    DEY
    BPL @CopyHeaderTemplate
    INC IsNameFieldInitialized    ; Set IsNameFieldInitialized [$0420] to mark the name field initialized.

@CheckAB:
    ; At this point:
    ; - NameCharOffset [$0421] holds the offset of the first character in the save slot info name for the current slot.
    ;   - This will be changed as the player inputs characters.
    ; - [$0422] to [$0424] hold a transfer record header. The VRAM address points to the beginning of the appropriate name field in the nametable.
    ;   - This will be changed as the player inputs characters.
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
    LDY #$02                    ; Copy our char transfer record header (in [$0422-0424]) to dynamic transfer buf.

@CopyHeader:
    LDA NameInputCharBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyHeader
    STY DynTileBuf+4            ; Write the end marker to dynamic transfer buf.
    LDX NameCharOffset
    LDY CharBoardIndex          ; CharBoardIndex in [$041F] will index into character map.
    LDA ModeE_CharMap, Y        ; Get the character that's highlighted.
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
    LDA SlotToNameOffset, Y
    STA NameCharOffset

    ; If the name cursor has gone past the end of the field,
    ; then wrap around.
    LDA ObjX
    CMP #$B0
    BNE @Exit
    LDA #$70
    STA ObjX

@Exit:
    JMP ModeE_SetNameCursorSpriteX    ; Go set the name cursor sprite X.
