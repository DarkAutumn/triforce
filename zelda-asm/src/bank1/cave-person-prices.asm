; ============================================================================
; FILE:     bank1/cave-person-prices.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 302-568
; SUBJECTS: (none)
; TOPIC:    The cave person's per-frame update: drawing the person and the wares, the medicine shop's
;           wait for the letter, the 9-state machine dispatch, and the price list written under the
;           wares.
; SEE ALSO: bank1/cave-init-person-text.asm (InitCave: CaveFlags, wares, prices); bank1/person-textbox.asm
;           (states 1 and 7); bank1/cave-shop.asm (states 2 and 4); bank1/cave-hint-money-game.asm
;           (states 3, 5, 6, 8; PostDebit, PrependSignToPrice); bank1/format-decimal-hearts.asm
;           (FormatDecimalByte); bank7/room-item-draw.asm (AnimateItemObject);
;           bank7/object-update-table.asm (UpdateObject sends types >= $6A to UpdateCavePerson);
;           bank6/transfer-bufs-static.asm (BlankTextBoxLines, BlankPersonWares).
; KEY FACTS:
;   - Cave person states (ObjState+1): 0 write the price list, 1 type the text, 2 talk / shop / door
;     charge, 3 blank text line 3 and the price row, 4 flicker then vanish, 5 hint or money game or
;     rupee gift, 6 blank text line 3 and the price row, 7 type the hint text, 8 stand and do nothing.
;     UpdateCavePerson_JumpTable
;   - The medicine shop person ($74) does nothing but stand there (no text, no wares, Link not halted)
;     until InvLetter = 2. Pressing B with the letter selected (SelectedItemSlot = $0F) increments
;     InvLetter, moves the selection to the potion slot (7) and starts the shop. UpdateCavePerson
;   - Wares sit at X $58, $78, $98 and Y $98; they are drawn through object slot $13.
;     CaveWareXs, DrawCaveItems
;   - A ware byte whose low 6 bits are $3F is not drawn; taken wares ($FF) therefore vanish.
;     DrawCaveItems
;   - In state 4 the person and wares are drawn only on even frames (flicker). UpdateCavePerson
;   - Prices are 0-255, written right-aligned in 3 cells; a price of 0 is left blank; the minus sign
;     (CaveFlags bit 7) goes in the cell left of the first digit and is dropped for 3-digit prices.
;     WritePricesToDynamicTransferBuf, SwapSpaceAndSign
;   - Persons of type >= $7B (secret rupees) are drawn unmirrored, all others mirrored.
;     DrawCavePerson
; CONTENTS: (generated; L = line in this file)
;   L0050 PriceListTemplateTransferBuf          data  DATA: transfer record template for the price row: nametable address $22C8, 13 tiles: $21 ("X",
;   L0066 UpdateCavePerson                      code  Updates the cave person in object slot 1: draws it and its wares, then runs its state handler.
;   L0125 UpdateCavePerson_JumpTable            data  TABLE: state handlers of a cave person; index = ObjState+1.
;   L0140 DrawCavePerson                        code  Draws the cave person at its position: unmirrored for types >= $7B, mirrored for the others.
;   L0155 CaveWareXs                            data  TABLE: X coordinate of each cave ware; index = ware 0-2 (left to right); value = pixels.
;   L0165 DrawCaveItems                         code  Draws the cave's three wares and, when prices are shown, the rupee icon left of the price row.
;   L0221 UpdateCavePersonState_TransferPrices  code  Cave person state 0: writes the price list if the cave shows prices, then goes to state 1.
;   L0229 WritePricesTransferBuf                code  Second entry: writes the price row with "X" after the rupee icon, sets ObjTimer+1 = $0A and
;   L0245 WritePricesToDynamicTransferBuf       code  Formats the three CavePrices into the price row already copied to DynTileBuf.
;   L0314 CueTransferBufAndAdvanceState         code  Requests a static PPU transfer buffer and advances the person in slot 1 to its next state.
;   L0318 IncCaveState                          code  Advances the person in object slot 1 to its next state (ObjState+1 + 1).
;   L0328 SwapSpaceAndSign                      code  If the given character is a blank, exchanges it with the prefix character in [04].
;   L0340 CopyPriceListTemplate                 code  Copies the 17-byte price row template to the start of DynTileBuf.
; ============================================================================

; DATA: transfer record template for the price row: nametable address $22C8, 13 tiles: $21 ("X",
; replaced by the caller's character) and 12 blanks ($24); then the end marker $FF. 17 bytes.
PriceListTemplateTransferBuf:
    .BYTE $22, $C8, $0D, $21, $24, $24, $24, $24
    .BYTE $24, $24, $24, $24, $24, $24, $24, $24
    .BYTE $FF

; ----------------------------------------------------------------------------
; Updates the cave person in object slot 1: draws it and its wares, then runs its state handler.
; IN:   X = 1 (the person's object slot); ObjState+1 = state 0-8; ObjType+1 = person type $6A-$7D.
; NOTE: Type $74 (medicine shop) returns before the state handler on every frame until the letter
;       has been shown (InvLetter = 2), and unhalts Link meanwhile. The letter is shown by pressing B
;       (ButtonsPressed bit $40) with SelectedItemSlot = $0F: InvLetter is incremented, the selected
;       slot becomes 7 (potion) and the state machine starts at state 0 in the same frame.
;       Link was unhalted before that and nothing here halts him again, so he is not halted while
;       the medicine shop's text is typed.
; NOTE: In state 4, on odd frames (FrameCounter bit 0) nothing is drawn.
; ----------------------------------------------------------------------------
UpdateCavePerson:
    ; In state 4 (vanishing), skip all drawing on odd frames and go straight to the state handler.
    LDA ObjState+1
    CMP #$04
    BNE :+
    LDA FrameCounter
    AND #$01
    BNE @UpdateCavePersonDirect
:
    JSR DrawCavePerson

    ; Any person but the medicine shop ($74), or the medicine shop once the letter has been
    ; shown (InvLetter = 2): go draw the wares and run the state handler.
    LDA ObjType+1
    CMP #$74
    BNE @DrawItems
    LDA InvLetter
    CMP #$02
    BEQ @DrawItems

    ; Medicine shop, letter not shown yet. If the letter is the selected item (slot $0F)
    ; and B was pressed this frame, go show it.
    LDY SelectedItemSlot
    CPY #$0F
    BNE @Unhalt
    LDA ButtonsPressed
    AND #$40
    BNE @UseLetter

@Unhalt:
    ; Otherwise only release Link if he is halted (ObjState = $40), and return without
    ; running the state handler.
    LDA ObjState
    CMP #$40
    BNE :+
    JSR UnhaltLink
:
    RTS

@UseLetter:
    ; Request tune 1 bit $04 (which tune this is, is not established in this file).
    LDA #$04
    STA Tune1Request

    ; Mark the letter shown: InvLetter 1 -> 2.
    INC InvLetter

    ; Select the potion slot (7) for the B button.
    LDA #$07
    STA SelectedItemSlot

@DrawItems:
    JSR DrawCaveItems

@UpdateCavePersonDirect:
    LDA ObjState+1
    JSR TableJump

; TABLE: state handlers of a cave person; index = ObjState+1.
UpdateCavePerson_JumpTable:
    .ADDR UpdateCavePersonState_TransferPrices              ; 0: write the price list
    .ADDR UpdatePersonState_Textbox                         ; 1: type the text, then unhalt Link
    .ADDR UpdateCavePersonState_TalkOrShopOrDoorCharge      ; 2: talk only: finish; else wait for a touch
    .ADDR UpdatePersonState_CueTransferBlankPersonWares     ; 3: blank text line 3 and the price row
    .ADDR UpdatePersonState_DelayThenHide                   ; 4: remove the person when its timer ends
    .ADDR UpdateCavePersonState_HintOrMoneyGame             ; 5: hint, money game or rupee gift
    .ADDR UpdatePersonState_CueTransferBlankPersonWares     ; 6: blank text line 3 and the price row
    .ADDR UpdatePersonState_Textbox                         ; 7: type the hint text
    .ADDR UpdateCavePersonState_DoNothing                   ; 8: final state, person stays

; ----------------------------------------------------------------------------
; Draws the cave person at its position: unmirrored for types >= $7B, mirrored for the others.
; IN:   X = the person's object slot (1).
; ----------------------------------------------------------------------------
DrawCavePerson:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Types $7B-$7D (the secret rupee givers) are drawn not mirrored.
    LDY ObjType+1
    CPY #$7B
    BCS :+

    ; All other cave persons are drawn mirrored.
    JMP DrawObjectMirrored
:
    JMP DrawObjectNotMirrored

; TABLE: X coordinate of each cave ware; index = ware 0-2 (left to right); value = pixels.
; Link must stand at exactly this X to take the ware.
CaveWareXs:
    .BYTE $58, $78, $98

; ----------------------------------------------------------------------------
; Draws the cave's three wares and, when prices are shown, the rupee icon left of the price row.
; OUT:  X = $13 if anything was drawn; ObjX+19 / ObjY+19 (object slot $13) left at the last position
;       set. Uses [0421].
; NOTE: Wares are drawn only with CaveFlags bit $04; the rupee icon (item $18) at ($30,$AB) only
;       with bit $08, which is cleared when a ware is taken or a hint is bought.
; ----------------------------------------------------------------------------
DrawCaveItems:
    ; Without the "show wares" flag, go see about the rupee icon.
    LDA CaveFlags
    AND #$04                    ; Cave flag: show wares
    BEQ @ShowPriceRupee

    ; Loop over the wares from 2 to 0, index kept in [0421].
    LDA #$02
    STA $0421

@LoopWare:
    LDX $0421

    ; Position object slot $13 at the ware: X from the table, Y = $98.
    LDA CaveWareXs, X
    STA ObjX+19
    LDA #$98
    STA ObjY+19

    ; Item id $3F = no ware here (also a taken ware, $FF): skip it.
    LDA CaveItemIds, X
    AND #$3F
    CMP #$3F
    BEQ @NextWare

    ; Draw item A through object slot $13.
    LDX #$13
    JSR AnimateItemObject

@NextWare:
    DEC $0421
    BPL @LoopWare

@ShowPriceRupee:
    ; Without the "show prices" flag, return.
    LDA CaveFlags
    AND #$08                    ; Cave flag: show prices
    BEQ @Exit

    ; Draw a rupee (item $18) at ($30, $AB), left of the "X" of the price row.
    LDA #$30
    STA ObjX+19
    LDA #$AB
    STA ObjY+19
    LDA #$18
    LDX #$13
    JSR AnimateItemObject

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Cave person state 0: writes the price list if the cave shows prices, then goes to state 1.
; OUT:  ObjState+1 = 1; with prices: DynTileBuf holds the price row and ObjTimer+1 = $0A, so the
;       text starts $0A frames later.
; ----------------------------------------------------------------------------
UpdateCavePersonState_TransferPrices:
    ; Without the "show prices" flag, only advance the state.
    LDA CaveFlags
    AND #$08                    ; Cave flag: show prices
    BEQ IncCaveState

; Second entry: writes the price row with "X" after the rupee icon, sets ObjTimer+1 = $0A and
; advances the person's state (the money game reveal calls it and then overrides the state).
WritePricesTransferBuf:
    JSR CopyPriceListTemplate
    LDA #$21                    ; "X"

; ----------------------------------------------------------------------------
; Formats the three CavePrices into the price row already copied to DynTileBuf.
; IN:   A = character for the first cell of the row ($21 "X", or $24 blank); DynTileBuf holds
;       PriceListTemplateTransferBuf (CopyPriceListTemplate).
; OUT:  DynTileBuf+3 = A; price k (0-2) in DynTileBuf+5+4k..+7+4k (hundreds, tens, ones; blanks for
;       leading zeros; all blank for price 0); ObjTimer+1 = $0A; ObjState+1 incremented.
;       Uses [01]-[04], CaveCurPriceIndex, CaveCurPriceOffset.
; NOTE: With CaveFlags bit 7 a "-" ($62) takes the blank cell just left of the first digit (1- and
;       2-digit prices only). The cell before each price (DynTileBuf+4+4k) is never written here.
;       The money game's PrependSignToPrice writes "+" or "-" to DynTileBuf+5+4k, the hundreds cell
;       (blank for its amounts 10-50, or holding the "-" placed here), not to the cell before it.
; ----------------------------------------------------------------------------
WritePricesToDynamicTransferBuf:
    STA DynTileBuf+3

    ; Start with price 0 at text offset 0.
    LDX #$00
    STX CaveCurPriceIndex
    STX CaveCurPriceOffset

@LoopPrice:
    LDA CavePrices, X

    ; Price 0: three blanks in [01], [02], [03], and a blank prefix (X = $24 goes to [04]).
    BNE @Format
    LDX #$24
    STX $01
    STX $02
    STX $03
    JMP @Align

@Format:
    ; [01], [02], [03] = hundreds, tens, ones; leading zeros become blanks ($24).
    JSR FormatDecimalByte

    ; Prefix character for [04]: "-" ($62) if CaveFlags bit 7 (minus sign) is set, else blank.
    LDX #$24
    LDA CaveFlags
    ASL
    BCC @Align
    LDX #$62                    ; "-"

@Align:
    STX $04

    ; Put the prefix in the rightmost blank leading cell: first offer it to the tens cell,
    ; then to the hundreds cell. After the first swap [04] holds a blank, so the prefix is
    ; placed once at most.
    LDY CaveCurPriceOffset
    LDA $02
    JSR SwapSpaceAndSign
    STA DynTileBuf+6, Y
    LDA $01
    JSR SwapSpaceAndSign
    STA DynTileBuf+5, Y
    LDA $03
    STA DynTileBuf+7, Y

    ; Each price takes 4 cells: a blank and 3 digits.
    LDA CaveCurPriceOffset
    CLC
    ADC #$04
    STA CaveCurPriceOffset

    ; Next price, until all 3 are done.
    INC CaveCurPriceIndex
    LDX CaveCurPriceIndex
    CPX #$03
    BNE @LoopPrice

    ; The person's timer delays the first text character of the next state by $0A frames.
    LDA #$0A
    STA ObjTimer+1

    ; Go advance the state (always taken).
    BNE IncCaveState

; ----------------------------------------------------------------------------
; Requests a static PPU transfer buffer and advances the person in slot 1 to its next state.
; IN:   A = transfer buffer selector (offset into TransferBufAddrs), stored in TileBufSelector.
; ----------------------------------------------------------------------------
CueTransferBufAndAdvanceState:
    STA TileBufSelector

; Advances the person in object slot 1 to its next state (ObjState+1 + 1).
IncCaveState:
    INC ObjState+1
    RTS

; ----------------------------------------------------------------------------
; If the given character is a blank, exchanges it with the prefix character in [04].
; IN:   A = a character of the price; [04] = prefix character (sign or blank).
; OUT:  A not blank ($24): A and [04] unchanged. A blank: A = old [04], [04] = $24. X clobbered
;       in the second case.
; ----------------------------------------------------------------------------
SwapSpaceAndSign:
    CMP #$24
    BNE :+
    TAX
    LDA $04
    STX $04
:
    RTS

; ----------------------------------------------------------------------------
; Copies the 17-byte price row template to the start of DynTileBuf.
; ----------------------------------------------------------------------------
CopyPriceListTemplate:
    LDY #$10
:
    LDA PriceListTemplateTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL :-
    RTS
