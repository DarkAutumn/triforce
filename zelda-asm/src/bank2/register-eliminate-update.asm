; ============================================================================
; FILE:     bank2/register-eliminate-update.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1612-1796
; SUBJECTS: (none)
; TOPIC:    Register (ZELDA name check) and elimination updates
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 ZeldaString             data
;   L0019 UpdateModeERegister     code
;   L0151 UpdateModeFElimination  code
;   L0167 ModeE_ResetVariables    code
;   L0174 DeleteSlot              code
; ============================================================================

ZeldaString:
    .BYTE $23, $0E, $15, $0D, $0A

UpdateModeERegister:
    LDA ButtonsPressed          ; If didn't press Start,
    AND #$10
    BEQ @GoIdle
    LDA CurSaveSlot             ; or didn't select "End" option,
    CMP #$03
    BEQ @ChoseEnd

@GoIdle:
    JMP @Idle                   ; then go handle idle time.

@ChoseEnd:
    ; Pressed Start over "End" option.
    ;
    ; Silence tune channel 1.
    LDA #$00
    STA CurTune1
    STA $0425                   ; Reset SaveFileNameIndex.
    STA $0423                   ; Reset SaveSlotNameIndex.
    STA CurSaveSlot             ; Reset CurSaveSlot.
    TAX                         ; X holds the current slot number

@LoopSaveSlot:
    LDY CurSaveSlot
    LDA #$FF                    ; Mark save file B committed.
    STA IsSaveFileBCommitted, Y
    TYA
    ASL
    TAY
    LDA #$00                    ; Reset FileBReadyToSave [$0426].
    STA $0426
    STA FileBChecksums, Y       ; Reset checksum for current save file B.
    INY
    STA FileBChecksums, Y
    TXA                         ; Save current save slot number.
    PHA
    JSR FetchFileBAddressSet
    PLA                         ; Restore save slot number.
    TAX

@CopyName:
    LDY $0423                   ; Copy next character from save slot info to file B.
    LDA SaveSlotNames, Y
    LDY $0425                   ; SaveFileNameIndex
    STA ($C4), Y
    CMP #$24
    BEQ @NextChar               ; If we copied a space, then go advance offsets and check things.
    LDA IsSaveSlotActive, X
    BNE @NextChar               ; If the save slot is active, then go advance offsets and check things.

    ; The save slot is not active.
    ;
    ; Initialize file B hearts value to 3 heart containers and 2 hearts.
    LDY #$18
    LDA #$22
    STA ($C0), Y
    INY                         ; Initialize file B heart partial to full.
    LDA #$FF
    STA ($C0), Y
    LDY #$25                    ; Initialize file B's max bombs to 8.
    LDA #$08
    STA ($C0), Y
    TXA                         ; Save the current save slot number.
    PHA
    ASL                         ; Multiply it by 8 to get offset to current name in save slot info.
    ASL
    ASL
    TAY
    LDX #$00                    ; Compare the name to "ZELDA".

@CompareToZelda:
    LDA SaveSlotNames, Y
    CMP ZeldaString, X
    BNE @FlagBReady             ; If there's any mismatch, then skip the rest.
    INY
    INX
    CPX #$05
    BCC @CompareToZelda         ; Go check the next character until the end of "ZELDA".
    PLA                         ; Pop and push the slot number, so we can get it into X.
    PHA
    TAX
    LDY #$00                    ; Set second quest in file B.
    LDA #$01
    STA ($CC), Y

@FlagBReady:
    PLA                         ; Restore save slot number.
    TAX
    LDA #$01                    ; Set FileBReadyToSave [$0426].
    STA $0426
    LDY #$00                    ; Set IsSaveSlotActive in file B.
    STA ($C6), Y

@NextChar:
    INC $0423                   ; Point to the next char in save slot info name.
    INC $0425                   ; Point to the next char in save file B name.
    LDA $0425
    CMP #$08
    BNE @CopyName               ; If we haven't copied 8 characters from save slot info name, then go copy the next one.
    INX                         ; Make X refer to the next slot.
    LDA #$00                    ; Reset the offset to the next save file B char.
    STA $0425                   ; SaveFileNameIndex
    LDA $0426
    BEQ :+                      ; If FileBReadyToSave [$0426] is set, then calculate and store the file B checksum, and mark file B uncommitted.
    JSR CalculateAndStoreFileBChecksumUncommitted
:
    INC CurSaveSlot             ; Advance the slot number.
    LDA CurSaveSlot
    CMP #$03
    BEQ :+                      ; If haven't processed 3 slots,
    JMP @LoopSaveSlot           ; then go process the next one.
:
    LDA #$00                    ; Reset FileBReadyToSave [$0426].
    STA $0426
    STA CurSaveSlot             ; Reset CurSaveSlot.
    JSR ModeE_ResetVariables
    LDA #$01                    ; Make sure we stay updating.
    STA IsUpdatingMode
    JMP UpdateModeDSave_Sub2    ; Go to mode 0 submode 1.

@Idle:
    ; Handle idle time in mode $E.
    LDA CurSaveSlot
    CMP #$03
    BEQ :+                      ; If a slot is chosen,
    JSR ModeE_HandleDirections  ; then we can check the direction buttons.
:
    JSR UpdateModeEandF_Idle
    JSR ModeEandF_WriteNameCursorSpritePosition
    JSR ModeEandF_WriteCharBoardCursorSpritePosition
    JMP ModeE_HandleAOrB

UpdateModeFElimination:
    LDA ButtonsPressed
    CMP #$10
    BEQ :+                      ; If Start wasn't pressed,
    JMP UpdateModeEandF_Idle    ; Then go handle other buttons and idle time.
:
    ; Start was pressed.
    LDA CurSaveSlot
    CMP #$03
    BNE DeleteSlot              ; If a slot was chosen, then go delete it.
    LDA #$0E                    ; "End" was chosen. So, go to mode $E.
    STA GameMode
    LDA #$00
    STA IsUpdatingMode
    STA GameSubmode

ModeE_ResetVariables:
    ; Assumes that zero is passed in A.
    STA CharBoardIndex
    STA IsNameFieldInitialized
    STA NameCharOffset
    RTS

DeleteSlot:
    LDA #$08                    ; "Hurt" sound effect
    STA SampleRequest
    LDY CurSaveSlot
    LDX SlotToBlankNameTransferBufEndOffset, Y

    ; Copy the appropriate transfer buf of a blank name for
    ; current slot to dynamic transfer buf.
    LDY #$04

@CopyBlankBuf:
    LDA DeletedSlotBlankNameTransferBuf, X
    STA DynTileBuf, Y
    DEX
    DEY
    BPL @CopyBlankBuf
    JSR FetchFileAAddressSet
    JSR FormatFileA
    JSR FetchProfileNameAddress
    LDY #$07                    ; Clear the name in save slot info.

@ClearName:
    LDA #$24
    STA ($0C), Y
    DEY
    BPL @ClearName
    RTS
