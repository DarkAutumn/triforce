; ============================================================================
; FILE:     bank1/cave-person-prices.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 302-568
; SUBJECTS: (none)
; TOPIC:    Cave person state machine, letter/potion shop, ware drawing, price list
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 PriceListTemplateTransferBuf          data
;   L0029 UpdateCavePerson                      code
;   L0089 UpdateCavePerson_JumpTable            data
;   L0100 DrawCavePerson                        code
;   L0114 CaveWareXs                            data
;   L0117 DrawCaveItems                         code
;   L0172 UpdateCavePersonState_TransferPrices  code
;   L0179 WritePricesTransferBuf                code
;   L0186 WritePricesToDynamicTransferBuf       code  Params:
;   L0254 CueTransferBufAndAdvanceState         code  Params:
;   L0257 IncCaveState                          code
;   L0273 SwapSpaceAndSign                      code  Params:
;   L0282 CopyPriceListTemplate                 code
; ============================================================================

PriceListTemplateTransferBuf:
    .BYTE $22, $C8, $0D, $21, $24, $24, $24, $24
    .BYTE $24, $24, $24, $24, $24, $24, $24, $24
    .BYTE $FF

UpdateCavePerson:
    ; If in state 4, then every other frame skip drawing the person
    ; and items; and go straight to the state code for each kind of cave.
    LDA ObjState+1
    CMP #$04
    BNE :+
    LDA FrameCounter
    AND #$01
    BNE @UpdateCavePersonDirect
:
    JSR DrawCavePerson

    ; If the cave type is not $74 medicine shop,
    ; or it is but the letter has not been used, then
    ; go draw items and run the state code.
    LDA ObjType+1
    CMP #$74
    BNE @DrawItems
    LDA InvLetter
    CMP #$02
    BEQ @DrawItems

    ; If the letter is selected, and B is pressed, then go use the letter.
    LDY SelectedItemSlot
    CPY #$0F
    BNE @Unhalt
    LDA ButtonsPressed
    AND #$40
    BNE @UseLetter

@Unhalt:
    ; If Link is halted, then unhalt him.
    LDA ObjState
    CMP #$40
    BNE :+
    JSR UnhaltLink
:
    RTS

@UseLetter:
    ; The letter has been used.
    ;
    ; Play the "secret found" tune.
    LDA #$04
    STA Tune1Request

    ; Set the letter state to used (2).
    INC InvLetter

    ; Change the selected item slot to the potion's.
    LDA #$07
    STA SelectedItemSlot

@DrawItems:
    JSR DrawCaveItems

@UpdateCavePersonDirect:
    LDA ObjState+1
    JSR TableJump

UpdateCavePerson_JumpTable:
    .ADDR UpdateCavePersonState_TransferPrices
    .ADDR UpdatePersonState_Textbox
    .ADDR UpdateCavePersonState_TalkOrShopOrDoorCharge
    .ADDR UpdatePersonState_CueTransferBlankPersonWares
    .ADDR UpdatePersonState_DelayThenHide
    .ADDR UpdateCavePersonState_HintOrMoneyGame
    .ADDR UpdatePersonState_CueTransferBlankPersonWares
    .ADDR UpdatePersonState_Textbox
    .ADDR UpdateCavePersonState_DoNothing

DrawCavePerson:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; If the cave/person type >= $7B, then the visual is a moblin.
    ; So, go draw not mirrored.
    LDY ObjType+1
    CPY #$7B
    BCS :+

    ; Else the old man, old woman, and shopkeeper are all mirrored.
    JMP DrawObjectMirrored
:
    JMP DrawObjectNotMirrored

CaveWareXs:
    .BYTE $58, $78, $98

DrawCaveItems:
    ; If the cave flags do not call for showing items, then
    ; go see about showing prices.
    LDA CaveFlags
    AND #$04                    ; Cave flag: Show items
    BEQ @ShowPriceRupee

    ; Loop over the wares to draw, from 2 to 0, indexed by [0421].
    LDA #$02
    STA $0421

@LoopWare:
    LDX $0421

    ; Look up and set the X coordinate for the current item.
    LDA CaveWareXs, X
    STA ObjX+19

    ; Set Y coordinate $98 for the item.
    LDA #$98
    STA ObjY+19

    ; If the item is nothing ($3F), then loop gain.
    LDA CaveItemIds, X
    AND #$3F
    CMP #$3F
    BEQ @NextWare

    ; Switch to the room item object slot, and draw it.
    LDX #$13
    JSR AnimateItemObject

@NextWare:
    ; Bottom of the item drawing loop. Decrement this index until < 0.
    DEC $0421
    BPL @LoopWare

@ShowPriceRupee:
    ; If the cave flags do not call for showing prices, then return.
    LDA CaveFlags
    AND #$08                    ; Cave flag: Show prices
    BEQ @Exit

    ; Show a rupee at ($30, $AB).
    LDA #$30
    STA ObjX+19
    LDA #$AB
    STA ObjY+19
    LDA #$18
    LDX #$13
    JSR AnimateItemObject

@Exit:
    RTS

UpdateCavePersonState_TransferPrices:
    ; If the cave flags do not call for showing prices, then
    ; advance the state and return.
    LDA CaveFlags
    AND #$08                    ; Cave flag: Show prices
    BEQ IncCaveState

WritePricesTransferBuf:
    JSR CopyPriceListTemplate
    LDA #$21                    ; "X"

; Params:
; A: price character
;
WritePricesToDynamicTransferBuf:
    STA DynTileBuf+3

    ; Reset the index of current price being formatted
    ; and offset of price string.
    LDX #$00
    STX CaveCurPriceIndex
    STX CaveCurPriceOffset

@LoopPrice:
    ; Get the price of the current item.
    LDA CavePrices, X

    ; If it's 0, then store a space character in the text field [01], [02], [03].
    BNE @Format
    LDX #$24
    STX $01
    STX $02
    STX $03
    JMP @Align

@Format:
    JSR FormatDecimalByte

    ; If the cave flags call for negative amounts, then
    ; store a dash in prefix byte [04]. Else store a space ($24).
    LDX #$24
    LDA CaveFlags
    ASL
    BCC @Align
    LDX #$62                    ; "-"

@Align:
    STX $04

    ; Move the dash closer to the number.
    LDY CaveCurPriceOffset
    LDA $02
    JSR SwapSpaceAndSign
    STA DynTileBuf+6, Y
    LDA $01
    JSR SwapSpaceAndSign
    STA DynTileBuf+5, Y
    LDA $03
    STA DynTileBuf+7, Y

    ; Add 4 to the base offset for the next string.
    LDA CaveCurPriceOffset
    CLC
    ADC #$04
    STA CaveCurPriceOffset

    ; Bottom of the price loop. Increment the index of the item until = 3.
    INC CaveCurPriceIndex
    LDX CaveCurPriceIndex
    CPX #$03
    BNE @LoopPrice

    ; Delay $A frames at the beginning of the next state.
    LDA #$0A
    STA ObjTimer+1

    ; Go to the next state and return.
    BNE IncCaveState

; Params:
; A: selector
;
CueTransferBufAndAdvanceState:
    STA TileBufSelector

IncCaveState:
    INC ObjState+1
    RTS

; Params:
; A: a character of the price string
; [04]: prefix character
;
; Returns:
;   If original A is not a space:
;     A:    original character
;     [04]: original prefix character
;   Else:
;     A:    original prefix character
;     [04]: original character
;
SwapSpaceAndSign:
    CMP #$24
    BNE :+
    TAX
    LDA $04
    STX $04
:
    RTS

CopyPriceListTemplate:
    ; Copy $11 bytes of the price list template text to the dynamic transfer buf.
    LDY #$10
:
    LDA PriceListTemplateTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL :-
    RTS
