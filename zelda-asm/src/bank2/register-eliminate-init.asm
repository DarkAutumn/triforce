; ============================================================================
; FILE:     bank2/register-eliminate-init.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1350-1610
; SUBJECTS: ModeERegister, ModeFEliminate, Save
; TOPIC:    Save file B address sets and their fetch routine; tables for mode $E (register a name) and
;           mode $F (eliminate a save): screen text records, character map, cursor data; and the init
;           routine the two modes share.
; SEE ALSO: bank2/register-eliminate-update.asm, bank2/register-name-input.asm and
;           bank2/register-cursor-sprites.asm (the update code that reads these tables);
;           bank2/mode1-init-save-files.asm and bank2/moded-save.asm (other users of
;           Save_FetchFileBAddressSet, Save_FormatFileB); bank1/modeb-cave-save-sprites.asm (the file A
;           counterpart Save_FetchFileAAddressSet); bank7/init-mode-dispatch.asm (ModeERegister_Init trampoline);
;           bank6/transfer-buf-dispatch.asm (Transfer_BufAddrs: selectors $12 and $16 used here).
; KEY FACTS:
;   - Each save slot has a set of 7 pointers into its file B: items, world flags, name, active flag,
;     an unused byte, death count, quest number. Save_FetchFileBAddressSet copies the set of
;     CurSaveSlot to [C0]..[CD]. Save_FileBAddressSets.
;   - Init runs one submode per frame, 0 to 4: title text, slot names and end option, character board
;     (selector $16), menu palettes (selector $12), cursors and Link sprites; submode 4 sets
;     IsUpdatingMode. ModeERegister_Init_Full.
;   - Submode 0 formats all three file B's (Save_FormatFileB) on entering either mode.
;     ModeERegister_Init_Full@FormatSlotsB.
;   - The screen is built from the mode $F text; in mode $E "ELIMINATION  MODE" and the dash after it
;     are overwritten with "REGISTER YOUR NAME" and "ELIMINATION" with "REGISTER". ModeERegister_TitlePatch,
;     ModeERegister_EndOptionPatch.
;   - Mode $E starts on the first slot whose IsSaveSlotActive is 0; with all three registered
;     CurSaveSlot becomes 3 (the end option) and the character-board cursor is hidden. Mode $F starts
;     on slot 0. ModeERegister_Init_Full@FindInactiveSlot.
;   - The cursor colour patched into the menu palettes is $15 in mode $E and $30 in mode $F.
;     ModeERegister_Init_Full@CheckSub3.
;   - The character board has 44 entries: A-Z, 7 symbols, 0-9, space ($24). ModeERegister_CharMap.
; CONTENTS: (generated; L = line in this file)
;   L0056 Save_FileBAddressSets                   data  TABLE: file B pointer sets, 3 slots x 7 addresses (low byte first): items, world flags, name,
;   L0061 Save_FileBAddressSet1                   data  DATA: file B pointer set of save slot 1 (same 7 fields).
;   L0066 Save_FileBAddressSet2                   data  DATA: file B pointer set of save slot 2 (same 7 fields).
;   L0077 Save_FetchFileBAddressSet               code  Copies the 14-byte file B pointer set of CurSaveSlot to [C0]..[CD].
;   L0102 ModeFEliminate_TitleTransferBuf         data  DATA: PPU transfer record for the screen title at $2064: "ELIMINATION  MODE" between dashes; $1D
;   L0110 ModeERegister_TitlePatch                data  DATA: 18 tiles "REGISTER YOUR NAME", written in mode $E over the 17 tiles "ELIMINATION  MODE" and the
;   L0116 ModeERegister_EndOptionPatch            data  DATA: 11 tiles "REGISTER" plus 3 spaces, written over "ELIMINATION" of the end option in mode $E.
;   L0122 ModeFEliminate_SlotTemplateTransferBuf  data  DATA: PPU transfer records for the 3 slot names ($20CE, $212E, $218E; 8 tiles each, filled in from
;   L0132 ModeERegister_SlotCursorYs              data  TABLE: Y of the slot cursor sprite; index = CurSaveSlot (0-2 slots, 3 the end option).
;   L0136 ModeERegister_CharMap                   data  TABLE: tile code of each character board position; index = CharBoardIndex ($00-$2B).
;   L0146 ModeERegister_NameCharTransferHeaders   data  TABLE: 3-byte PPU transfer headers (address, length 1) of the first name character of each slot;
;   L0151 ModeFEliminate_BlankNameTransferBufs    data  TABLE: three 5-byte PPU transfer records that blank a slot's 8 name tiles with $24 (mode $F).
;   L0157 ModeERegister_CursorSprites             data  DATA: 11 bytes copied to Sprites+1: the slot cursor (tile $F3, attributes $03, X $43) and two
;   L0163 ModeERegister_CharBoardYStepsAndBounds  data  TABLE: character board cursor Y movement, two triplets (step, Y that triggers the wrap, Y after the
;   L0167 ModeFEliminate_BlankNameBufEndOffsets   data  TABLE: offset of the last byte of each slot's record in ModeFEliminate_BlankNameTransferBufs; index = slot.
;   L0171 ModeERegister_SlotNameOffsets           data  TABLE: offset in SaveSlotNames of each slot's 8-character name; index = slot.
;   L0175 ModeERegister_NameCharHeaderEndOffsets  data  TABLE: offset of the last byte of each slot's header in ModeERegister_NameCharTransferHeaders.
;   L0184 ModeERegister_Init_Full                 code  Init of modes $E and $F: builds the register/eliminate screen over 5 frames (GameSubmode 0-4).
; ============================================================================

; TABLE: file B pointer sets, 3 slots x 7 addresses (low byte first): items, world flags, name,
; active flag ($6D90 + slot), unused byte ($6D93 + slot), death count ($6D96 + slot), quest number
; ($6D99 + slot). This label is slot 0's set.
Save_FileBAddressSets:
    .BYTE $98, $68, $10, $69, $80, $68, $90, $6D
    .BYTE $93, $6D, $96, $6D, $99, $6D

; DATA: file B pointer set of save slot 1 (same 7 fields).
Save_FileBAddressSet1:
    .BYTE $C0, $68, $90, $6A, $88, $68, $91, $6D
    .BYTE $94, $6D, $97, $6D, $9A, $6D

; DATA: file B pointer set of save slot 2 (same 7 fields).
Save_FileBAddressSet2:
    .BYTE $E8, $68, $10, $6C, $90, $68, $92, $6D
    .BYTE $95, $6D, $98, $6D, $9B, $6D

; ----------------------------------------------------------------------------
; Copies the 14-byte file B pointer set of CurSaveSlot to [C0]..[CD].
; IN:   CurSaveSlot = 0-2.
; OUT:  [C0:C1] items; [C2:C3] world flags; [C4:C5] name; [C6:C7] active flag; [C8:C9] an unused byte
;       ($6D93 + slot): it is only zeroed on format, added to the checksum and copied to file A;
;       [CA:CB] death count ($6D96 + slot); [CC:CD] quest number.
; ----------------------------------------------------------------------------
Save_FetchFileBAddressSet:
    LDA #$FF                    ; Calculate the end of the address set for current file.
    LDY CurSaveSlot

@AddE:
    CLC
    ADC #$0E
    DEY
    BPL @AddE
    TAY

    ; Copy the file address set (14 bytes) for the current
    ; save file to [$C0] to make it easier to work with.
    LDX #$0D

@CopyAddresses:
    LDA Save_FileBAddressSets, Y
    STA $C0, X
    DEY
    DEX
    BPL @CopyAddresses
    RTS

; DATA: PPU transfer record for the screen title at $2064: "ELIMINATION  MODE" between dashes; $1D
; bytes with its $FF end.
ModeFEliminate_TitleTransferBuf:
    .BYTE $20, $64, $19, $6A, $6A, $6A, $6A, $0E
    .BYTE $15, $12, $16, $12, $17, $0A, $1D, $12
    .BYTE $18, $17, $24, $24, $16, $18, $0D, $0E
    .BYTE $6A, $6A, $6A, $6A, $FF

; DATA: 18 tiles "REGISTER YOUR NAME", written in mode $E over the 17 tiles "ELIMINATION  MODE" and the
; first dash after them.
ModeERegister_TitlePatch:
    .BYTE $1B, $0E, $10, $12, $1C, $1D, $0E, $1B
    .BYTE $24, $22, $18, $1E, $1B, $24, $17, $0A
    .BYTE $16, $0E

; DATA: 11 tiles "REGISTER" plus 3 spaces, written over "ELIMINATION" of the end option in mode $E.
ModeERegister_EndOptionPatch:
    .BYTE $1B, $0E, $10, $12, $1C, $1D, $0E, $1B
    .BYTE $24, $24, $24

; DATA: PPU transfer records for the 3 slot names ($20CE, $212E, $218E; 8 tiles each, filled in from
; SaveSlotNames) and the end option "ELIMINATION END" at $21EA; $34 bytes with the $FF end.
ModeFEliminate_SlotTemplateTransferBuf:
    .BYTE $20, $CE, $08, $24, $24, $24, $24, $24
    .BYTE $24, $24, $24, $21, $2E, $08, $24, $24
    .BYTE $24, $24, $24, $24, $24, $24, $21, $8E
    .BYTE $08, $24, $24, $24, $24, $24, $24, $24
    .BYTE $24, $21, $EA, $0F, $0E, $15, $12, $16
    .BYTE $12, $17, $0A, $1D, $12, $18, $17, $24
    .BYTE $0E, $17, $0D, $FF

; TABLE: Y of the slot cursor sprite; index = CurSaveSlot (0-2 slots, 3 the end option).
ModeERegister_SlotCursorYs:
    .BYTE $2F, $47, $5F, $77

; TABLE: tile code of each character board position; index = CharBoardIndex ($00-$2B).
ModeERegister_CharMap:
    .BYTE $0A, $0B, $0C, $0D, $0E, $0F, $10, $11
    .BYTE $12, $13, $14, $15, $16, $17, $18, $19
    .BYTE $1A, $1B, $1C, $1D, $1E, $1F, $20, $21
    .BYTE $22, $23, $62, $63, $28, $29, $2A, $2B
    .BYTE $2C, $00, $01, $02, $03, $04, $05, $06
    .BYTE $07, $08, $09, $24

; TABLE: 3-byte PPU transfer headers (address, length 1) of the first name character of each slot;
; read backward from the offsets in ModeERegister_NameCharHeaderEndOffsets.
ModeERegister_NameCharTransferHeaders:
    .BYTE $20, $CE, $01, $21, $2E, $01, $21, $8E
    .BYTE $01

; TABLE: three 5-byte PPU transfer records that blank a slot's 8 name tiles with $24 (mode $F).
ModeFEliminate_BlankNameTransferBufs:
    .BYTE $20, $CE, $48, $24, $FF, $21, $2E, $48
    .BYTE $24, $FF, $21, $8E, $48, $24, $FF

; DATA: 11 bytes copied to Sprites+1: the slot cursor (tile $F3, attributes $03, X $43) and two
; sprites of tile $25 parked at Y = $F8 with X $70 and $30.
ModeERegister_CursorSprites:
    .BYTE $F3, $03, $43, $F8, $25, $23, $70, $F8
    .BYTE $25, $23, $30

; TABLE: character board cursor Y movement, two triplets (step, Y that triggers the wrap, Y after the
; wrap): down = +$10, $C7, $87; up = -$10, $77, $B7.
ModeERegister_CharBoardYStepsAndBounds:
    .BYTE $10, $C7, $87, $F0, $77, $B7

; TABLE: offset of the last byte of each slot's record in ModeFEliminate_BlankNameTransferBufs; index = slot.
ModeFEliminate_BlankNameBufEndOffsets:
    .BYTE $04, $09, $0E

; TABLE: offset in SaveSlotNames of each slot's 8-character name; index = slot.
ModeERegister_SlotNameOffsets:
    .BYTE $00, $08, $10

; TABLE: offset of the last byte of each slot's header in ModeERegister_NameCharTransferHeaders.
ModeERegister_NameCharHeaderEndOffsets:
    .BYTE $02, $05, $08

; ----------------------------------------------------------------------------
; Init of modes $E and $F: builds the register/eliminate screen over 5 frames (GameSubmode 0-4).
; OUT:  CurSaveSlot = first unregistered slot (mode $E; 3 if none) or 0 (mode $F); IsUpdatingMode set
;       after submode 4.
; NOTE: Submode 0 formats all three file B's.
; ----------------------------------------------------------------------------
ModeERegister_Init_Full:
    LDA #$00
    STA CurSaveSlot
    JSR ModeERegister_ResetNameInput
    JSR Ppu_TurnOffAllVideo
    LDA GameSubmode
    BNE @CheckSub1              ; Go handle submodes 1 and up.

    ; Submode 0:
    JSR Ppu_TurnOffVideoAndClearNameTables

@FormatSlotsB:
    ; Format all file B's.
    JSR Save_FetchFileBAddressSet
    JSR Save_FormatFileB
    INC CurSaveSlot
    LDA CurSaveSlot
    CMP #$03
    BNE @FormatSlotsB
    LDA #$00                    ; Reset CurSaveSlot.
    STA CurSaveSlot
    LDX #$1C                    ; Copy the title tiles to the dynamic transfer buf.

@CopyTiles:
    LDA ModeFEliminate_TitleTransferBuf, X
    STA DynTileBuf, X
    DEX
    BPL @CopyTiles
    LDA GameMode
    CMP #$0E
    BNE @SetBufLen              ; If in mode E, overwrite "ELIMINATION MODE" with "REGISTER YOUR NAME".
    LDY #$00

@OverwriteTitle:
    LDA ModeERegister_TitlePatch, Y
    STA DynTileBuf+7, Y
    INY
    CPY #$12
    BNE @OverwriteTitle

@SetBufLen:
    LDA #$1D                    ; Record the length of the transfer buf.

@SetTransferBufLenAndIncSubmode:
    STA DynTileBufLen
    INC GameSubmode
    RTS

@CheckSub1:
    CMP #$01
    BNE @CheckSub2

    ; Submode 1:
    ;
    ; Copy ModeFEliminate_SlotTemplateTransferBuf to dynamic transfer buf.
    LDX #$33

@CopySlotTemplates:
    LDA ModeFEliminate_SlotTemplateTransferBuf, X
    STA DynTileBuf, X
    DEX
    BPL @CopySlotTemplates
    LDX #$00                    ; Overwrite payload of a dynamic transfer record with a name.
    LDY #$00

@OverwriteName:
    LDA SaveSlotNames, Y                ; The names are all arranged one after another.
    STA DynTileBuf+3, X
    INX
    INY
    TYA
    AND #$07
    BNE @OverwriteName          ; If copied the whole name,
    INX                         ; Skip the header for the next transfer record.
    INX
    INX
    CPX #$21
    BNE @OverwriteName          ; If there are more names to write, go write the next one.
    LDA GameMode
    CMP #$0E
    BNE @SkipOverwriteEndOption ; If in mode E, overwrite "ELIMINATION" with "REGISTER".
    LDY #$00

@OverwriteEndOption:
    LDA ModeERegister_EndOptionPatch, Y
    STA DynTileBuf+3, X
    INX
    INY
    CPY #$0B
    BNE @OverwriteEndOption

@SkipOverwriteEndOption:
    LDA #$34
    BNE @SetTransferBufLenAndIncSubmode    ; Always taken: record the buf length, advance the submode.

@CheckSub2:
    CMP #$02
    BNE @CheckSub3              ; Go handle submode 3 and 4.

    ; Submode 2:
    ;
    ; Cue a transfer of Transfer_ModeEandFCharBoardBuf.
    LDA #$16

@SelectMenuBuf:
    STA TileBufSelector
    INC GameSubmode
    RTS

@CheckSub3:
    ; Submode 3:
    CMP #$03
    BNE @Sub4                   ; Go handle submode 4.
    LDA #$15                    ; In mode $E, use $15 for cursor color.
    LDY GameMode
    CPY #$0F
    BNE :+
    LDA #$30                    ; In mode $F, use $30 for cursor color.
:
    ; Replace byte 1 of row 3 of sprite palette in transfer buf
    ; TileBufSelector=$12.
    STA Transfer_MenuPalettesBuf+32
    LDA #$12
    BNE @SelectMenuBuf          ; Go cue transfer of menu palettes and advance submode.

@Sub4:
    ; Submode 4:
    LDA GameMode
    CMP #$0F
    BEQ @FoundInactiveSlot      ; If in mode E,
    LDX #$03                    ; Then look for the first slot that's inactive.
    LDY #$FF
    STY CurSaveSlot

@FindInactiveSlot:
    INY
    INC CurSaveSlot
    LDA IsSaveSlotActive, Y
    BEQ @FoundInactiveSlot      ; Found one. Quit the loop.
    DEX
    BPL @FindInactiveSlot       ; Loop again. If not found, then CurSaveSlot will be 3 (out of bounds).

@FoundInactiveSlot:
    JSR ModeERegister_SetUpCursorSprites
    LDA CurSaveSlot
    CMP #$03
    BNE :+                      ; If we're at the "end" option, then hide the char-board cursor.
    LDA #$F8
    STA Sprites+8
:
    LDA #$50                    ; The X of Link objects is $50.
    STA $00
    LDA #$30                    ; The base Y of Link objects is $30.
    STA $01
    INC IsUpdatingMode
    JMP Mode1Menu_WriteLinkSprites
