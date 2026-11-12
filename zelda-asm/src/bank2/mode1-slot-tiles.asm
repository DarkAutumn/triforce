; ============================================================================
; FILE:     bank2/mode1-slot-tiles.asm   BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2556-2733
; SUBJECTS: (none)
; TOPIC:    Mode 1 init submodes: slot name/hearts/deaths tiles
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 InitMode1_Sub1                      code
;   L0059 InitMode1_Sub2                      code
;   L0065 InitMode1_FillAndTransferSlotTiles  code
;   L0126 InitMode1_Sub6                      code
; ============================================================================

InitMode1_Sub1:
    ; Reset CurSaveSlot. Doing this is useful for the
    ; work done here, and for the sequence of submodes
    ; that generate and transfer save slot graphics.
    LDA #$00
    STA CurSaveSlot
    JSR FetchFileAAddressSet
    LDY #$0B                    ; The ring is at this offset in Items block.
    LDX #$00                    ; The offset to the byte we want to change in a palette.

@LoopSlot:
    TYA                         ; Save ring offset in Items block of current slot.
    PHA
    LDA ($00), Y                ; Get the ring inventory value.
    TAY
    LDA LinkColors, Y           ; Get the color for that ring level.

    ; Put the color in the byte 2 of row for current slot in
    ; sprite palette that will be transferred.
    STA MenuPalettesTransferBuf+20, X
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
    JSR ResetRoomTileObjInfo    ; UNKNOWN: Why now?
    LDA #$12                    ; Cue transfer of menu palettes.
    STA TileBufSelector
    INC GameSubmode
    JSR TurnOffVideoAndClearArtifacts
    LDY #$04
    LDA #$00
    STA IsSwordShotForced          ; UNKNOWN: Why now?
:
    STA RoomHistory, Y
    DEY
    BPL :-
    RTS

InitMode1_Sub2:
    LDA #$14
    STA TileBufSelector
    INC GameSubmode
    RTS

InitMode1_FillAndTransferSlotTiles:
    ; Copy the mode 1 line transfer buf template to
    ; dynamic transfer buf.
    LDY #$1F

@CopySlotLineTemplate:
    LDA Mode1SlotLineTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopySlotLineTemplate
    LDY CurSaveSlot             ; The first time, this is reset in a previous submode.

@OffsetSlotLineAddr:
    LDA DynTileBuf+1            ; Add ($60 * CurSaveSlot) to the PPU address of each record in the buffer.
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
    JSR FormatHeartsInTextBuf
    INC CurSaveSlot             ; Next time, process the next slot.
    INC GameSubmode
    RTS

InitMode1_Sub6:
    ; Copy the mode 1 death counts transfer buf template to
    ; dynamic transfer buf.
    LDY #$12

@CopyTemplate:
    LDA Mode1DeathCountsTransferBuf, Y
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
    JSR FormatDecimalByte
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
    STY LevelEntranceRoomId        ; Use room ID $FF, so that mode 3 "Unfurl" will put the player in the room at StartRoomId.

@FindActiveSlot:
    INY
    INC CurSaveSlot
    LDA IsSaveSlotActive, Y
    BEQ @FindActiveSlot
    LDA #$00
    STA GameSubmode
    INC IsUpdatingMode          ; Start updating.
    RTS
