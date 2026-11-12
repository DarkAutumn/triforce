; ============================================================================
; FILE:     bank2/register-eliminate-update.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1612-1796
; SUBJECTS: ModeERegister, ModeFEliminate
; TOPIC:    Per-frame update of mode $E (register a name) and mode $F (eliminate a save slot): what Start
;           does on a slot or on "End", the ZELDA name test that starts a file in Q2, slot deletion.
; SEE ALSO: bank2/register-eliminate-init.asm (screen setup, tables, Save_FetchFileBAddressSet);
;           bank2/register-name-input.asm (character board and name typing);
;           bank2/register-cursor-sprites.asm (cursors, Select cycling);
;           bank2/mode1-init-save-files.asm (file A/B format and checksum; the commit pass that follows);
;           bank2/moded-save.asm (ModeDSave_Update_Sub2Leave, the exit to mode 0 submode 1).
; KEY FACTS:
;   - Mode $E does nothing permanent until Start is pressed on "End" (CurSaveSlot = 3); then every slot's
;     name is copied from SaveSlotNames into its file B (ModeERegister_Update).
;   - A slot is registered only if it was not active and its name has at least one non-space character:
;     its file B gets HeartValues $22 and HeartPartial $FF (3 containers, 3 full hearts), InvMaxBombs 8,
;     the active flag, a checksum, and is marked uncommitted (ModeERegister_Update).
;   - Q2 start: the first 5 name characters equal to ModeERegister_ZeldaNameChars ("ZELDA"; characters 6-8 are not
;     compared) set the file's quest number to 1 (ModeERegister_Update@CompareToZelda).
;   - Mode $E leaves through GameMode 0 / GameSubmode 1, whose handler copies each valid uncommitted
;     file B over file A (ModeDSave_Update_Sub2Leave, Save_CommitOrRepairFiles).
;   - Mode $F: Start on a slot formats its file A at once and blanks its name; Start on "End" switches
;     to mode $E (ModeFEliminate_Update, ModeFEliminate_DeleteSlot).
; CONTENTS: (generated; L = line in this file)
;   L0033 ModeERegister_ZeldaNameChars  data  TABLE: the name "ZELDA" as 5 character tile codes; compared with the start of a new slot's name.
;   L0042 ModeERegister_Update          code  Mode $E update: Start on "End" writes the new names to the save files; otherwise runs name entry.
;   L0178 ModeFEliminate_Update         code  Mode $F update: Start deletes the chosen slot or, on "End", goes to mode $E; else handles Select.
;   L0195 ModeERegister_ResetNameInput  code  Clears the name-entry state (CharBoardIndex, IsNameFieldInitialized, NameCharOffset); IN: A = 0.
;   L0205 ModeFEliminate_DeleteSlot     code  Eliminates the save slot CurSaveSlot: formats file A and blanks the name on screen and in slot info.
; ============================================================================

; TABLE: the name "ZELDA" as 5 character tile codes; compared with the start of a new slot's name.
ModeERegister_ZeldaNameChars:
    .BYTE $23, $0E, $15, $0D, $0A

; ----------------------------------------------------------------------------
; Mode $E update: Start on "End" writes the new names to the save files; otherwise runs name entry.
; NOTE: SaveSlotNameOffset and SaveFileNameOffset are the name offsets (slot info, file B) and
;       IsFileBReadyToSave is "this file B needs a checksum"; they share bytes with
;       NameInputCharBuf+1 and IsHoldingButton, which are idle here.
; ----------------------------------------------------------------------------
ModeERegister_Update:
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
    STA SaveFileNameOffset      ; Reset the offset into the file B name.
    STA SaveSlotNameOffset      ; Reset the offset into SaveSlotNames.
    STA CurSaveSlot             ; Reset CurSaveSlot.
    TAX                         ; X holds the current slot number

@LoopSaveSlot:
    LDY CurSaveSlot
    LDA #$FF                    ; Mark save file B committed.
    STA IsSaveFileBCommitted, Y
    TYA
    ASL
    TAY
    LDA #$00                    ; Reset IsFileBReadyToSave.
    STA IsFileBReadyToSave
    STA Save_FileBChecksums, Y       ; Reset checksum for current save file B.
    INY
    STA Save_FileBChecksums, Y
    TXA                         ; Save current save slot number.
    PHA
    JSR Save_FetchFileBAddressSet
    PLA                         ; Restore save slot number.
    TAX

@CopyName:
    LDY SaveSlotNameOffset      ; Copy next character from save slot info to file B.
    LDA SaveSlotNames, Y
    LDY SaveFileNameOffset
    STA ($C4), Y
    CMP #$24
    BEQ @NextChar               ; If we copied a space, then go advance offsets and check things.
    LDA IsSaveSlotActive, X
    BNE @NextChar               ; If the save slot is active, then go advance offsets and check things.

    ; The save slot is not active.
    ;
    ; Initialize file B HeartValues to $22: 3 heart containers, all 3 full (with the partial below).
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
    CMP ModeERegister_ZeldaNameChars, X
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
    LDA #$01                    ; Set IsFileBReadyToSave.
    STA IsFileBReadyToSave
    LDY #$00                    ; Set IsSaveSlotActive in file B.
    STA ($C6), Y

@NextChar:
    INC SaveSlotNameOffset      ; Point to the next char in save slot info name.
    INC SaveFileNameOffset      ; Point to the next char in save file B name.
    LDA SaveFileNameOffset
    CMP #$08
    BNE @CopyName               ; Until 8 characters of the name are copied, go copy the next one.
    INX                         ; Make X refer to the next slot.
    LDA #$00                    ; Reset the offset to the next save file B char.
    STA SaveFileNameOffset
    LDA IsFileBReadyToSave
    BEQ :+                      ; If IsFileBReadyToSave is set, store the checksum, mark B uncommitted.
    JSR Save_StoreFileBChecksumUncommitted
:
    INC CurSaveSlot             ; Advance the slot number.
    LDA CurSaveSlot
    CMP #$03
    BEQ :+                      ; If haven't processed 3 slots,
    JMP @LoopSaveSlot           ; then go process the next one.
:
    LDA #$00                    ; Reset IsFileBReadyToSave.
    STA IsFileBReadyToSave
    STA CurSaveSlot             ; Reset CurSaveSlot.
    JSR ModeERegister_ResetNameInput
    LDA #$01                    ; Make sure we stay updating.
    STA IsUpdatingMode
    JMP ModeDSave_Update_Sub2Leave    ; Go to mode 0 submode 1.

@Idle:
    ; Handle idle time in mode $E.
    LDA CurSaveSlot
    CMP #$03
    BEQ :+                      ; If a slot is chosen,
    JSR ModeERegister_HandleDirections  ; then we can check the direction buttons.
:
    JSR ModeERegister_CycleSlotOnSelect
    JSR ModeERegister_WriteNameCursorSprite
    JSR ModeERegister_WriteCharBoardCursorSprite
    JMP ModeERegister_HandleAOrB

; ----------------------------------------------------------------------------
; Mode $F update: Start deletes the chosen slot or, on "End", goes to mode $E; else handles Select.
; NOTE: The Start test is CMP #$10, so Start counts only when no other button is newly pressed.
; ----------------------------------------------------------------------------
ModeFEliminate_Update:
    LDA ButtonsPressed
    CMP #$10
    BEQ :+                      ; If Start wasn't pressed,
    JMP ModeERegister_CycleSlotOnSelect    ; Then go handle other buttons and idle time.
:
    ; Start was pressed.
    LDA CurSaveSlot
    CMP #$03
    BNE ModeFEliminate_DeleteSlot              ; If a slot was chosen, then go delete it.
    LDA #$0E                    ; "End" was chosen. So, go to mode $E.
    STA GameMode
    LDA #$00
    STA IsUpdatingMode
    STA GameSubmode

; Clears the name-entry state (CharBoardIndex, IsNameFieldInitialized, NameCharOffset); IN: A = 0.
ModeERegister_ResetNameInput:
    ; Assumes that zero is passed in A.
    STA CharBoardIndex
    STA IsNameFieldInitialized
    STA NameCharOffset
    RTS

; ----------------------------------------------------------------------------
; Eliminates the save slot CurSaveSlot: formats file A and blanks the name on screen and in slot info.
; ----------------------------------------------------------------------------
ModeFEliminate_DeleteSlot:
    LDA #$08                    ; "Hurt" sound effect
    STA SampleRequest
    LDY CurSaveSlot
    LDX ModeFEliminate_BlankNameBufEndOffsets, Y

    ; Copy the appropriate transfer buf of a blank name for
    ; current slot to dynamic transfer buf.
    LDY #$04

@CopyBlankBuf:
    LDA ModeFEliminate_BlankNameTransferBufs, X
    STA DynTileBuf, Y
    DEX
    DEY
    BPL @CopyBlankBuf
    JSR Save_FetchFileAAddressSet
    JSR Save_FormatFileA
    JSR Save_FetchSlotNameAddress
    LDY #$07                    ; Clear the name in save slot info.

@ClearName:
    LDA #$24
    STA ($0C), Y
    DEY
    BPL @ClearName
    RTS
