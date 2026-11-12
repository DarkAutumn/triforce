; ============================================================================
; FILE:     bank1/cave-hint-money-game.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 858-1009
; SUBJECTS: Cave, Inv, Person
; TOPIC:    Cave person state 5: the paid hint, the money game result and the secret rupee gift; the
;           helpers that post rupees to add or subtract; the shared person state that blanks text
;           line 3 and the price row.
; SEE ALSO: bank1/cave-shop.asm (detects the ware Link touched, charges for a hint, enters state 5);
;           bank1/cave-init-person-text.asm (Cave_Init_SetUpTextAndWares builds CaveFlags and shuffles the money
;           game amounts); bank1/cave-person-prices.asm (state table, price list formatting);
;           bank1/status-bar-rupees.asm (moves the posted amounts into InvRupees).
; KEY FACTS:
;   - State 5 has three cases, tested in this order: CaveFlags bit 4 = paid hint; person type >= $7B =
;     rupee gift; otherwise the money game. Cave_State5HintOrMoneyGame
;   - A hint's text depends on the ware chosen (0-2) and the person type: type $75 uses selectors $14,
;     $14, $16; every other hint person uses $14, $18, $1A. Selector $14 is "THIS AIN'T ENOUGH TO
;     TALK": the price was already charged before state 5 runs, so those choices cost rupees and give
;     nothing. $16 and $18 are directions; $1A is "BOY, YOU'RE RICH!".
;     Cave_HintTextSelectorsType75, Cave_HintTextSelectorsOther
;   - The money game does nothing while InvRupees < 10 (the state repeats every frame).
;     Cave_State5HintOrMoneyGame@HandleMoneyGame
;   - Money game result: the amount behind the chosen rupee is added when it is 20 or 50, otherwise it
;     is subtracted (10 or 40). Cave_State5HintOrMoneyGame@HandleMoneyGame
;   - The rupee gift (person types $7B and up) credits the middle price CavePrices+1 and sets the
;     room's item-taken flag, so it is given once. Cave_State5HintOrMoneyGame
;   - Rupees never change at once: amounts are accumulated in InvRupeesToAdd / InvRupeesToSubtract.
;     Inv_PostRupeeCredit, Inv_PostRupeeDebit
; CONTENTS: (generated; L = line in this file)
;   L0042 Cave_HintTextSelectorsType75      data  TABLE: text selectors of hint person type $75 after Link pays; index = ware chosen (0-2, left to
;   L0048 Cave_HintTextSelectorsOther       data  TABLE: text selectors of the other hint persons after Link pays; index = ware chosen (0-2); value =
;   L0061 Cave_State5HintOrMoneyGame        code  Cave person state 5: answers a paid hint, pays out the money game, or gives the secret rupees.
;   L0165 Inv_PostRupeeCredit               code  Adds A to the rupees still to be credited (InvRupeesToAdd).
;   L0176 Inv_PostRupeeDebit                code  Adds A to the rupees still to be taken away (InvRupeesToSubtract).
;   L0187 Cave_PrependSignToPrice           code  Writes "+" before a money game amount of 20 or 50 and "-" before any other amount.
;   L0199 Cave_State8DoNothing              code  Cave person state 8: does nothing (the person stays shown); also the return of the routine above.
;   L0208 Person_UpdateState_BlankPriceRow  code  Person state shared by caves and dungeon persons: blanks text line 3 and the price row, next state.
; ============================================================================

; TABLE: text selectors of hint person type $75 after Link pays; index = ware chosen (0-2, left to
;        right); value = PersonTextSelector. Cave_HintTextSelectorsOther follows directly and is read through
;        this label with index + 3.
Cave_HintTextSelectorsType75:
    ; $14 "THIS AIN'T ENOUGH TO TALK." (twice), $16 "GO UP,UP, THE MOUNTAIN AHEAD.".
    .BYTE $14, $14, $16

; TABLE: text selectors of the other hint persons after Link pays; index = ware chosen (0-2); value =
;        PersonTextSelector.
Cave_HintTextSelectorsOther:
    ; $14 "THIS AIN'T ENOUGH TO TALK.", $18 "GO NORTH,WEST,SOUTH,WEST TO THE FOREST OF MAZE.",
    ; $1A "BOY, YOU'RE RICH!".
    .BYTE $14, $18, $1A

; ----------------------------------------------------------------------------
; Cave person state 5: answers a paid hint, pays out the money game, or gives the secret rupees.
; IN:   CaveChosenIndex = ware Link touched (0-2); CaveFlags; ObjType+1 = cave person type.
; OUT:  hint: PersonTextSelector set, text restarted, textbox blanked, state 6.
;       gift or money game: InvRupeesToAdd or InvRupeesToSubtract increased, state 8.
; NOTE: The money game is ignored while InvRupees < 10; the state then repeats each frame.
; NOTE: The hint's price was charged by the caller state before this runs, also for selector $14.
; ----------------------------------------------------------------------------
Cave_State5HintOrMoneyGame:
    ; CaveFlags bit 4 = hint cave. Without it, go handle the money game or the gift.
    LDA CaveFlags
    AND #$10
    BEQ @CheckGiftOrMoneyGame

    ; Person type $75 uses the first set of text selectors (base offset 0),
    ; any other type the second set (base offset 3).
    LDA #$00
    LDY ObjType+1
    CPY #$75
    BEQ :+
    LDA #$03
:
    CLC

    ; Base offset + index of the ware chosen = index of the text selector.
    ADC CaveChosenIndex
    TAY
    LDA Cave_HintTextSelectorsType75, Y
    STA PersonTextSelector

    ; Point to the front of the first line of the textbox ($A4).
    LDA Person_TextboxLineAddrsLo+2
    STA PersonTextVramLo

    ; Restart the text at its first character.
    LDA #$00
    STA PersonTextIndex
    JSR Cave_ClearShowPricesFlag     ; Stop showing the rupee beside the prices.
    LDA #$1E                    ; Transfer selector: blank textbox lines.
    JMP Person_CueTransferBufAndAdvanceState   ; To state 6 (blank line 3, price row), then 7 (type the text).

@CheckGiftOrMoneyGame:
    ; Person type < $7B: the money game.
    ; Types $7B and up are the persons that give rupees away.
    LDA ObjType+1
    CMP #$7B
    BCC @HandleMoneyGame
    JSR Cave_CopyPriceListTemplate

    ; Write the price list into the dynamic transfer buf,
    ; with a space ($24) in place of the "X".
    LDA #$24
    JSR Cave_WritePriceListWithChar
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request
    JSR RoomFlags_SetItemTaken  ; Room flag $10: the gift is given only once.

    ; Go to state 8, which only shows the person.
    ; (Cave_WritePriceListWithChar already advanced the state once.)
    LDA #$08
    STA ObjState+1

    ; Credit the middle amount, CavePrices+1.
    LDA CavePrices+1
    JMP Inv_PostRupeeCredit

@HandleMoneyGame:
    ; If rupees < 10, then return and stay in this state.
    LDA InvRupees
    CMP #$0A
    BCC Cave_State4DelayThenHide_Return
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request

    ; Reveal the amounts: copy the three MoneyGameAmounts to CavePrices.
    LDY #$02
:
    LDA MoneyGameAmounts, Y
    STA CavePrices, Y
    DEY
    BPL :-
    JSR Cave_WritePriceList

    ; Go to state 8, which only shows the person.
    LDA #$08
    STA ObjState+1

    ; Put "+" or "-" in front of each amount. Y = 1, 5, 9 addresses the first
    ; (hundreds) cell of amounts 0, 1, 2 in the transfer record.
    LDY #$01
    LDA MoneyGameAmounts+0
    JSR Cave_PrependSignToPrice
    LDY #$05
    LDA MoneyGameAmounts+1
    JSR Cave_PrependSignToPrice
    LDY #$09
    LDA MoneyGameAmounts+2
    JSR Cave_PrependSignToPrice

    ; If the amount chosen is 20 ($14) or 50 ($32), then add it; else subtract it.
    LDX CaveChosenIndex
    LDA MoneyGameAmounts, X
    CMP #$14
    BEQ Inv_PostRupeeCredit
    CMP #$32
    BNE Inv_PostRupeeDebit

; ----------------------------------------------------------------------------
; Adds A to the rupees still to be credited (InvRupeesToAdd).
; IN:   A = amount.
; OUT:  InvRupeesToAdd increased by A (8-bit, no overflow check).
; ----------------------------------------------------------------------------
Inv_PostRupeeCredit:
    CLC
    ADC InvRupeesToAdd
    STA InvRupeesToAdd
    RTS

; ----------------------------------------------------------------------------
; Adds A to the rupees still to be taken away (InvRupeesToSubtract).
; IN:   A = amount.
; OUT:  InvRupeesToSubtract increased by A (8-bit, no overflow check).
; ----------------------------------------------------------------------------
Inv_PostRupeeDebit:
    CLC
    ADC InvRupeesToSubtract
    STA InvRupeesToSubtract
    RTS

; ----------------------------------------------------------------------------
; Writes "+" before a money game amount of 20 or 50 and "-" before any other amount.
; IN:   A = amount; Y = offset of the amount's first cell from DynTileBuf+4 (1, 5 or 9).
; OUT:  DynTileBuf+4,Y = tile $64 ("+") or $62 ("-"); X changed.
; ----------------------------------------------------------------------------
Cave_PrependSignToPrice:
    LDX #$64                    ; "+"
    CMP #$14
    BEQ :+
    CMP #$32
    BEQ :+
    LDX #$62                    ; "-"
:
    TXA
    STA DynTileBuf+4, Y

; Cave person state 8: does nothing (the person stays shown); also the return of the routine above.
Cave_State8DoNothing:
    RTS

; ----------------------------------------------------------------------------
; Person state shared by caves and dungeon persons: blanks text line 3 and the price row, next state.
; OUT:  TileBufSelector = $2A (Transfer_BlankTextLine3AndPricesBuf: 24 blank tiles at $21E4, 13 at $22C8); ObjState+1
;       incremented.
; NOTE: The wares themselves are sprites; this transfer does not remove them.
; ----------------------------------------------------------------------------
Person_UpdateState_BlankPriceRow:
    LDA #$2A                    ; Transfer selector: blank text line 3 and the price row.
    JMP Person_CueTransferBufAndAdvanceState
