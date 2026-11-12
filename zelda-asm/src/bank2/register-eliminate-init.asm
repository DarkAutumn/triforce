; ============================================================================
; FILE:     bank2/register-eliminate-init.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 1350-1610
; SUBJECTS: (none)
; TOPIC:    Mode E/F tables and init
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0029 SaveFileBAddressSets                           data
;   L0033 SaveFileBAddressSet1                           data
;   L0037 SaveFileBAddressSet2                           data
;   L0052 FetchFileBAddressSet                           code  TODO: [C8:C9]
;   L0075 ModeFTitleTransferBuf                          data
;   L0081 ModeFTitlePatchRegister                        data
;   L0086 ModeFSaveSlotTemplatePatchRegister             data
;   L0090 ModeFSaveSlotTemplateTransferBuf               data
;   L0099 ModeEandFSlotCursorYs                          data
;   L0102 ModeE_CharMap                                  data
;   L0110 SlotToInitialNameCharTransferHeaders           data
;   L0114 DeletedSlotBlankNameTransferBuf                data
;   L0118 ModeEandFCursorSprites                         data
;   L0122 ModeE_CharBoardYOffsetsAndBounds               data
;   L0125 SlotToBlankNameTransferBufEndOffset            data
;   L0128 SlotToNameOffset                               data
;   L0131 SlotToInitialNameCharTransferHeaderEndOffsets  data
;   L0134 InitModeEandF_Full                             code
; ============================================================================

SaveFileBAddressSets:
    .BYTE $98, $68, $10, $69, $80, $68, $90, $6D
    .BYTE $93, $6D, $96, $6D, $99, $6D

SaveFileBAddressSet1:
    .BYTE $C0, $68, $90, $6A, $88, $68, $91, $6D
    .BYTE $94, $6D, $97, $6D, $9A, $6D

SaveFileBAddressSet2:
    .BYTE $E8, $68, $10, $6C, $90, $68, $92, $6D
    .BYTE $95, $6D, $98, $6D, $9B, $6D

; TODO: [C8:C9]
;
; Returns:
; [C0:C1]: items pointer
; [C2:C3]: world flags pointer
; [C4:C5]: name pointer
; [C6:C7]: IsSaveSlotActive pointer
; [C8:C9]:
; [CA:CB]: death count pointer
; [CC:CD]: quest pointer
;
FetchFileBAddressSet:
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
    LDA SaveFileBAddressSets, Y
    STA $C0, X
    DEY
    DEX
    BPL @CopyAddresses
    RTS

ModeFTitleTransferBuf:
    .BYTE $20, $64, $19, $6A, $6A, $6A, $6A, $0E
    .BYTE $15, $12, $16, $12, $17, $0A, $1D, $12
    .BYTE $18, $17, $24, $24, $16, $18, $0D, $0E
    .BYTE $6A, $6A, $6A, $6A, $FF

ModeFTitlePatchRegister:
    .BYTE $1B, $0E, $10, $12, $1C, $1D, $0E, $1B
    .BYTE $24, $22, $18, $1E, $1B, $24, $17, $0A
    .BYTE $16, $0E

ModeFSaveSlotTemplatePatchRegister:
    .BYTE $1B, $0E, $10, $12, $1C, $1D, $0E, $1B
    .BYTE $24, $24, $24

ModeFSaveSlotTemplateTransferBuf:
    .BYTE $20, $CE, $08, $24, $24, $24, $24, $24
    .BYTE $24, $24, $24, $21, $2E, $08, $24, $24
    .BYTE $24, $24, $24, $24, $24, $24, $21, $8E
    .BYTE $08, $24, $24, $24, $24, $24, $24, $24
    .BYTE $24, $21, $EA, $0F, $0E, $15, $12, $16
    .BYTE $12, $17, $0A, $1D, $12, $18, $17, $24
    .BYTE $0E, $17, $0D, $FF

ModeEandFSlotCursorYs:
    .BYTE $2F, $47, $5F, $77

ModeE_CharMap:
    .BYTE $0A, $0B, $0C, $0D, $0E, $0F, $10, $11
    .BYTE $12, $13, $14, $15, $16, $17, $18, $19
    .BYTE $1A, $1B, $1C, $1D, $1E, $1F, $20, $21
    .BYTE $22, $23, $62, $63, $28, $29, $2A, $2B
    .BYTE $2C, $00, $01, $02, $03, $04, $05, $06
    .BYTE $07, $08, $09, $24

SlotToInitialNameCharTransferHeaders:
    .BYTE $20, $CE, $01, $21, $2E, $01, $21, $8E
    .BYTE $01

DeletedSlotBlankNameTransferBuf:
    .BYTE $20, $CE, $48, $24, $FF, $21, $2E, $48
    .BYTE $24, $FF, $21, $8E, $48, $24, $FF

ModeEandFCursorSprites:
    .BYTE $F3, $03, $43, $F8, $25, $23, $70, $F8
    .BYTE $25, $23, $30

ModeE_CharBoardYOffsetsAndBounds:
    .BYTE $10, $C7, $87, $F0, $77, $B7

SlotToBlankNameTransferBufEndOffset:
    .BYTE $04, $09, $0E

SlotToNameOffset:
    .BYTE $00, $08, $10

SlotToInitialNameCharTransferHeaderEndOffsets:
    .BYTE $02, $05, $08

InitModeEandF_Full:
    LDA #$00
    STA CurSaveSlot
    JSR ModeE_ResetVariables
    JSR TurnOffAllVideo
    LDA GameSubmode
    BNE @CheckSub1              ; Go handle submodes 1 and up.

    ; Submode 0:
    JSR TurnOffVideoAndClearArtifacts

@FormatSlotsB:
    ; Format all file B's.
    JSR FetchFileBAddressSet
    JSR FormatFileB
    INC CurSaveSlot
    LDA CurSaveSlot
    CMP #$03
    BNE @FormatSlotsB
    LDA #$00                    ; Reset CurSaveSlot.
    STA CurSaveSlot
    LDX #$1C                    ; Copy the title tiles to the dynamic transfer buf.

@CopyTiles:
    LDA ModeFTitleTransferBuf, X
    STA DynTileBuf, X
    DEX
    BPL @CopyTiles
    LDA GameMode
    CMP #$0E
    BNE @SetBufLen              ; If in mode E, overwrite "ELIMINATION MODE" with "REGISTER YOUR NAME".
    LDY #$00

@OverwriteTitle:
    LDA ModeFTitlePatchRegister, Y
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
    ; Copy ModeFSaveSlotTemplateTransferBuf to dynamic transfer buf.
    LDX #$33

@CopySlotTemplates:
    LDA ModeFSaveSlotTemplateTransferBuf, X
    STA DynTileBuf, X
    DEX
    BPL @CopySlotTemplates
    LDX #$00                    ; Overwrite payload of a dynamic transfer record with a name.
    LDY #$00

@OverwriteName:
    LDA Names, Y                ; The names are all arranged one after another.
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
    LDA ModeFSaveSlotTemplatePatchRegister, Y
    STA DynTileBuf+3, X
    INX
    INY
    CPY #$0B
    BNE @OverwriteEndOption

@SkipOverwriteEndOption:
    LDA #$34
    BNE @SetTransferBufLenAndIncSubmode    ; Go record the length of the transfer buf, and advance the submode.

@CheckSub2:
    CMP #$02
    BNE @CheckSub3              ; Go handle submode 3 and 4.

    ; Submode 2:
    ;
    ; Cue a transfer of ModeFCharBoardTransferBuf.
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
    STA MenuPalettesTransferBuf+32
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
    JSR ModeEandF_SetUpCursorSprites
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
    JMP Mode1_WriteLinkSprites
