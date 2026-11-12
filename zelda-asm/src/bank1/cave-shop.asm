; ============================================================================
; FILE:     bank1/cave-shop.asm          BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 678-856
; SUBJECTS: (none)
; TOPIC:    Shop purchase, item giveaway, heart requirement, door repair charge
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 UpdateCavePersonState_TalkOrShopOrDoorCharge  code
;   L0179 ClearPricesCaveFlag                           code
;   L0185 Cave_Shop_Return                              code
;   L0188 UpdatePersonState_DelayThenHide               code
;   L0193 L493A_Exit                                    code
; ============================================================================

UpdateCavePersonState_TalkOrShopOrDoorCharge:
    ; If the cave flags call for choosing an item, then
    ; go handle those kinds of caves.
    LDA CaveFlags
    LSR                         ; Cave flag: Choose/Pick up (1)
    BCS @CheckPickUp

    ; Advance to the end state, where the person is still shown.
    LDA #$08
    STA ObjState+1

    ; If this is not the "door charge" old man, then return.
    ; The person only talks.
    LDA ObjType+1
    CMP #$71
    BNE @Exit

    ; Post 20 rupees to subtract.
    LDA RupeesToSubtract
    CLC
    ADC #$14
    STA RupeesToSubtract

    ; Mark this secret taken.
    JSR SetRoomFlagUWItemState

@Exit:
    RTS

@CheckPickUp:
    ; This is a cave where the player chooses or touches an item:
    ; - give item
    ; - shop
    ; - hint
    ; - money game
    ;
    ; If rupees are decreasing, then return.
    LDA RupeesToSubtract
    BNE @Exit

    ; Loop over each item from 2 to 0.
    LDX #$02

@LoopWare:
    ; If this item is nothing, then loop again.
    LDA CaveItemIds, X
    AND #$3F                    ; Mask off the cave flags.
    CMP #$3F                    ; No item
    BEQ @NextWare

    ; If Link's X <> item's X, then loop again.
    LDA ObjX
    CMP CaveWareXs, X
    BNE @NextWare

    ; If the vertical distance between Link and the item < 6, then
    ; go try to take it.
    LDA ObjY
    SEC
    SBC #$98
    JSR Abs
    CMP #$06
    BCC @PickedUp

@NextWare:
    ; Bottom of the item touch loop.
    DEX
    BPL @LoopWare
    RTS

@PickedUp:
    ; Store the index of the item chosen.
    STX CaveChosenIndex

    ; If hint and money game cave flags are missing, then
    ; go take an item given away, or buy it.
    LDA CaveFlags
    AND #$30                    ; Cave flags: Hint and Money game
    BEQ @PickedUpWare

    ; If have cave flag "money game", then go to state 5 and return.
    AND #$10
    BEQ @MadeWager

    ; Else have "hint" cave flag.
    ;
    ; If rupees < price, then return.
    LDA InvRupees
    CMP CavePrices, X
    BCC Cave_Shop_Return

    ; Pass the price to pay for the hint.
    LDA CavePrices, X
    JSR PostDebit

@MadeWager:
    ; Go to state 5. It runs the hint cave and money game.
    LDA #$05
    STA ObjState+1
    RTS

@PickedUpWare:
    ; This is a shop, or something is being given away.
    ;
    ; If "pay" cave flag is missing, then go try to take the item.
    LDA CaveFlags
    AND #$02
    BEQ @TryToTake

    ; If rupees < price, then return.
    LDA InvRupees
    CMP CavePrices, X
    BCC Cave_Shop_Return

    ; Pass the price to pay for the item.
    LDA CavePrices, X
    JSR PostDebit

@TryToTake:
    ; If "heart requirement" cave flag is missing, then
    ; go to take the item without further ado.
    LDA CaveFlags
    AND #$40
    BEQ @Take

    ; Check heart container requirements.
    ;
    ; 5 is the minimum for cave type $6C (white sword)
    ; $C is the minimum otherwise
    LDY #$40
    LDA ObjType+1
    CMP #$6C
    BEQ :+
    LDY #$B0
:
    CPY HeartValues

    ; If the minimum of heart containers is met, then
    ; go take the item without further ado.
    BEQ @Take
    BCC @Take
    ; Begin unverified code 490E
    RTS
    ; End unverified code

@Take:
    JSR SetRoomFlagUWItemState
    LDA CaveItemIds, X
    AND #$3F                    ; Mask off the cave flags.
    PHA                         ; Push the item ID

    ; Flag the item taken by setting its ID to $FF.
    LDA #$FF
    STA CaveItemIds, X
    PLA                         ; Pop the item ID
    JSR TakeItem
    LDA #$1E                    ; Blank textbox lines
    JSR CueTransferBufAndAdvanceState

    ; Set timer to $40 for when we delay in state 4.
    LDA #$40
    STA ObjTimer+1

ClearPricesCaveFlag:
    ; Remove the "show prices" cave flag to get rid of the generic rupee.
    LDA CaveFlags
    AND #$F7
    STA CaveFlags

Cave_Shop_Return:
    RTS

UpdatePersonState_DelayThenHide:
    LDA ObjTimer+1
    BNE L493A_Exit
    STA ObjType+1

L493A_Exit:
    RTS
