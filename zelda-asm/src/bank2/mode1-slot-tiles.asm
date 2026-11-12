; ============================================================================
; FILE:     bank2/mode1-slot-tiles.asm   BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2556-2733
; SUBJECTS: Mode1Menu
; TOPIC:    Mode 1 (file select) init submodes 1-6: menu palettes with each slot's tunic colour, the
;           menu tiles, one line per save slot (name and hearts), death counts, first cursor position.
; SEE ALSO: bank2/mode1-init-save-files.asm (Mode1Menu_Init_Full_JumpTable, the templates, Mode1Menu_LinkTunicColors);
;           bank2/mode1-menu-update.asm (the menu once init is done);
;           bank1/format-decimal-hearts.asm (Text_FormatHeartRows, Text_FormatDecimalByte).
; KEY FACTS:
;   - Entering the file menu clears the room tile object, IsSwordShotForced and RoomHistory entries 0-4
;     (Mode1Menu_Init_Sub1Palettes).
;   - Submodes 3, 4 and 5 each build one slot's line from SaveSlotNames and SaveSlotHearts; the slot's
;     PPU address is the template's plus $60 * (slot + 1) (Mode1Menu_Init_Sub3SlotLine).
;   - Death counts are shown as 3 decimal characters; an inactive slot with count 0 shows blanks
;     (Mode1Menu_Init_Sub6DeathCounts).
;   - The cursor starts on the first active slot, or on Register (3) when none is active; and
;     LevelEntranceRoomId is set to $FF so the game starts in the level's start room (Mode1Menu_Init_Sub6DeathCounts).
; CONTENTS: (generated; L = line in this file)
;   L0029 Mode1Menu_Init_Sub1Palettes     code  Mode 1 init submode 1: puts each slot's tunic colour in the menu palettes and queues their transfer.
;   L0076 Mode1Menu_Init_Sub2Tiles        code  Mode 1 init submode 2: queues the transfer of tile buffer $14 (the file menu's static tiles).
;   L0086 Mode1Menu_Init_Sub3SlotLine     code  Mode 1 init submodes 3-5: builds the menu line (name, hearts) of save slot CurSaveSlot.
;   L0150 Mode1Menu_Init_Sub6DeathCounts  code  Mode 1 init submode 6: writes the death counts, selects the first active slot, starts updating.
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 1 init submode 1: puts each slot's tunic colour in the menu palettes and queues their transfer.
; ----------------------------------------------------------------------------
Mode1Menu_Init_Sub1Palettes:
    ; Reset CurSaveSlot. Doing this is useful for the
    ; work done here, and for the sequence of submodes
    ; that generate and transfer save slot graphics.
    LDA #$00
    STA CurSaveSlot
    JSR Save_FetchFileAAddressSet
    LDY #$0B                    ; The ring is at this offset in InvItems block.
    LDX #$00                    ; The offset to the byte we want to change in a palette.

@LoopSlot:
    TYA                         ; Save ring offset in InvItems block of current slot.
    PHA
    LDA ($00), Y                ; Get the ring inventory value.
    TAY
    LDA Mode1Menu_LinkTunicColors, Y           ; Get the color for that ring level.

    ; Put the color in the byte 2 of row for current slot in
    ; sprite palette that will be transferred.
    STA Transfer_MenuPalettesBuf+20, X
    PLA                         ; Restore ring offset.
    CLC
    ADC #$28                    ; Point to the ring in the next save slot.
    TAY
    TXA
    CLC
    ADC #$04                    ; Point one row down in palette.
    TAX
    CPX #$0C
    BCC @LoopSlot               ; 3 times.
    JSR Room_ResetTileObjInfo    ; Clear the room tile object's type and position.
    LDA #$12                    ; Cue transfer of menu palettes.
    STA TileBufSelector
    INC GameSubmode
    JSR Ppu_TurnOffVideoAndClearNameTables
    LDY #$04
    LDA #$00
    STA IsSwordShotForced       ; Clear this flag and RoomHistory entries 0-4 (Y = 4 down to 0).
:
    STA RoomHistory, Y
    DEY
    BPL :-
    RTS

; ----------------------------------------------------------------------------
; Mode 1 init submode 2: queues the transfer of tile buffer $14 (the file menu's static tiles).
; ----------------------------------------------------------------------------
Mode1Menu_Init_Sub2Tiles:
    LDA #$14
    STA TileBufSelector
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 1 init submodes 3-5: builds the menu line (name, hearts) of save slot CurSaveSlot.
; NOTE: Increments CurSaveSlot and GameSubmode, so three consecutive submodes cover slots 0-2.
; ----------------------------------------------------------------------------
Mode1Menu_Init_Sub3SlotLine:
    ; Copy the mode 1 line transfer buf template to
    ; dynamic transfer buf.
    LDY #$1F

@CopySlotLineTemplate:
    LDA Mode1Menu_SlotLineTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopySlotLineTemplate
    LDY CurSaveSlot             ; The first time, this is reset in a previous submode.

@OffsetSlotLineAddr:
    LDA DynTileBuf+1            ; Add $60 * (CurSaveSlot + 1) to the PPU address of each record in the buffer.
    CLC
    ADC #$60
    STA DynTileBuf+1
    LDA DynTileBuf+21
    CLC
    ADC #$60
    STA DynTileBuf+21
    LDA DynTileBuf
    ADC #$00
    STA DynTileBuf
    STA DynTileBuf+20
    DEY
    BPL @OffsetSlotLineAddr

    ; Copy name of current slot to beginning of payload
    ; of first record in dynamic transfer buf.
    LDA CurSaveSlot
    ASL
    ASL
    ASL
    TAX
    LDY #$03

@CopyName:
    LDA SaveSlotNames, X
    STA DynTileBuf, Y
    INX
    INY
    CPY #$0B
    BNE @CopyName

    ; Copy heart values from current save slot info
    ; to [$0E:0F] for formatting.
    LDA CurSaveSlot
    ASL
    TAY
    LDA SaveSlotHearts, Y
    STA $0E
    INY
    LDA SaveSlotHearts, Y
    STA $0F
    LDY #$0C
    JSR Text_FormatHeartRows
    INC CurSaveSlot             ; Next time, process the next slot.
    INC GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Mode 1 init submode 6: writes the death counts, selects the first active slot, starts updating.
; ----------------------------------------------------------------------------
Mode1Menu_Init_Sub6DeathCounts:
    ; Copy the mode 1 death counts transfer buf template to
    ; dynamic transfer buf.
    LDY #$12

@CopyTemplate:
    LDA Mode1Menu_DeathCountsTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyTemplate
    LDA #$00
    STA $0A                     ; The save slot.
    LDA #$03
    STA $0B                     ; The offset where the string will be written in dynamic transfer record.

@LoopSlot:
    LDY $0A
    LDA DeathCounts, Y
    JSR Text_FormatDecimalByte
    LDX $0B
    LDA $01
    STA DynTileBuf, X           ; Emit the first character.
    LDA $02
    STA DynTileBuf+1, X         ; Emit the second character.
    LDA $03
    BNE @EmitChar               ; If the third character isn't '0', then go emit it.

    ; If the first or second characters weren't spaces,
    ; then go ahead and emit the '0'.
    LDA $01
    CMP #$24
    BNE @Emit0
    LDA $02
    CMP #$24
    BNE @Emit0
    LDY $0A
    LDA IsSaveSlotActive, Y
    BNE @Emit0                  ; If this slot isn't active,
    LDA #$24
    BNE @EmitChar               ; Go emit a space.

@Emit0:
    LDA #$00                    ; Else, emit a '0'.

@EmitChar:
    STA DynTileBuf+2, X
    TXA                         ; Advance the offset by 6,
    CLC                         ; to the starting position in the next transfer record.
    ADC #$06
    STA $0B
    INC $0A                     ; Increment the save slot.
    LDA $0A
    CMP #$03
    BNE @LoopSlot               ; Go process the next slot, if not done.
    LDY #$FF                    ; Find the first save slot that's active.
    STY CurSaveSlot
    STY LevelEntranceRoomId     ; Room id $FF, so that mode 3 puts Link in the room at StartRoomId.

@FindActiveSlot:
    INY
    INC CurSaveSlot
    LDA IsSaveSlotActive, Y
    BEQ @FindActiveSlot
    LDA #$00
    STA GameSubmode
    INC IsUpdatingMode          ; Start updating.
    RTS
