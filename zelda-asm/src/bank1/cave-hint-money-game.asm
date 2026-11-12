; ============================================================================
; FILE:     bank1/cave-hint-money-game.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 858-1009
; SUBJECTS: (none)
; TOPIC:    Cave person state 5: the paid hint, the money game result and the secret rupee gift; the
;           helpers that post rupees to add or subtract; the shared person state that blanks text
;           line 3 and the price row.
; SEE ALSO: bank1/cave-shop.asm (detects the ware Link touched, charges for a hint, enters state 5);
;           bank1/cave-init-person-text.asm (InitCaveContinue builds CaveFlags and shuffles the money
;           game amounts); bank1/cave-person-prices.asm (state table, price list formatting);
;           bank1/status-bar-rupees.asm (moves the posted amounts into InvRupees).
; KEY FACTS:
;   - State 5 has three cases, tested in this order: CaveFlags bit 4 = paid hint; person type >= $7B =
;     rupee gift; otherwise the money game. UpdateCavePersonState_HintOrMoneyGame
;   - A hint's text depends on the ware chosen (0-2) and the person type: type $75 uses selectors $14,
;     $14, $16; every other hint person uses $14, $18, $1A. Selector $14 is "THIS AIN'T ENOUGH TO
;     TALK": the price was already charged before state 5 runs, so those choices cost rupees and give
;     nothing. $16 and $18 are directions; $1A is "BOY, YOU'RE RICH!".
;     HintCaveTextSelectors0, HintCaveTextSelectors1
;   - The money game does nothing while InvRupees < 10 (the state repeats every frame).
;     UpdateCavePersonState_HintOrMoneyGame@HandleMoneyGame
;   - Money game result: the amount behind the chosen rupee is added when it is 20 or 50, otherwise it
;     is subtracted (10 or 40). UpdateCavePersonState_HintOrMoneyGame@HandleMoneyGame
;   - The rupee gift (person types $7B and up) credits the middle price CavePrices+1 and sets the
;     room's item-taken flag, so it is given once. UpdateCavePersonState_HintOrMoneyGame
;   - Rupees never change at once: amounts are accumulated in InvRupeesToAdd / InvRupeesToSubtract.
;     PostCredit, PostDebit
; CONTENTS: (generated; L = line in this file)
;   L0042 HintCaveTextSelectors0                         data  TABLE: text selectors of hint person type $75 after Link pays; index = ware chosen (0-2, left to
;   L0048 HintCaveTextSelectors1                         data  TABLE: text selectors of the other hint persons after Link pays; index = ware chosen (0-2); value =
;   L0061 UpdateCavePersonState_HintOrMoneyGame          code  Cave person state 5: answers a paid hint, pays out the money game, or gives the secret rupees.
;   L0165 PostCredit                                     code  Adds A to the rupees still to be credited (InvRupeesToAdd).
;   L0176 PostDebit                                      code  Adds A to the rupees still to be taken away (InvRupeesToSubtract).
;   L0187 PrependSignToPrice                             code  Writes "+" before a money game amount of 20 or 50 and "-" before any other amount.
;   L0199 UpdateCavePersonState_DoNothing                code  Cave person state 8: does nothing (the person stays shown); also the return of the routine above.
;   L0208 UpdatePersonState_CueTransferBlankPersonWares  code  Person state shared by caves and dungeon persons: blanks text line 3 and the price row, next state.
; ============================================================================

; TABLE: text selectors of hint person type $75 after Link pays; index = ware chosen (0-2, left to
;        right); value = PersonTextSelector. HintCaveTextSelectors1 follows directly and is read through
;        this label with index + 3.
HintCaveTextSelectors0:
    ; $14 "THIS AIN'T ENOUGH TO TALK." (twice), $16 "GO UP,UP, THE MOUNTAIN AHEAD.".
    .BYTE $14, $14, $16

; TABLE: text selectors of the other hint persons after Link pays; index = ware chosen (0-2); value =
;        PersonTextSelector.
HintCaveTextSelectors1:
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
UpdateCavePersonState_HintOrMoneyGame:
    ; CaveFlags bit 4 = hint cave. Without it, go handle the money game or the gift.
    LDA CaveFlags
    AND #$10
    BEQ @HandleMoneyGameOrGiveaway

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
    LDA HintCaveTextSelectors0, Y
    STA PersonTextSelector

    ; Point to the front of the first line of the textbox ($A4).
    LDA TextboxLineAddrsLo+2
    STA PersonTextVramLo

    ; Restart the text at its first character.
    LDA #$00
    STA PersonTextIndex
    JSR ClearPricesCaveFlag     ; Stop showing the rupee beside the prices.
    LDA #$1E                    ; Transfer selector: blank textbox lines.
    JMP CueTransferBufAndAdvanceState   ; To state 6 (blank line 3, price row), then 7 (type the text).

@HandleMoneyGameOrGiveaway:
    ; Person type < $7B: the money game.
    ; Types $7B and up are the persons that give rupees away.
    LDA ObjType+1
    CMP #$7B
    BCC @HandleMoneyGame
    JSR CopyPriceListTemplate

    ; Write the price list into the dynamic transfer buf,
    ; with a space ($24) in place of the "X".
    LDA #$24
    JSR WritePricesToDynamicTransferBuf
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request
    JSR SetRoomFlagUWItemState  ; Room flag $10: the gift is given only once.

    ; Go to state 8, which only shows the person.
    ; (WritePricesToDynamicTransferBuf already advanced the state once.)
    LDA #$08
    STA ObjState+1

    ; Credit the middle amount, CavePrices+1.
    LDA CavePrices+1
    JMP PostCredit

@HandleMoneyGame:
    ; If rupees < 10, then return and stay in this state.
    LDA InvRupees
    CMP #$0A
    BCC L493A_Exit
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request

    ; Reveal the amounts: copy the three MoneyGameAmounts to CavePrices.
    LDY #$02
:
    LDA MoneyGameAmounts, Y
    STA CavePrices, Y
    DEY
    BPL :-
    JSR WritePricesTransferBuf

    ; Go to state 8, which only shows the person.
    LDA #$08
    STA ObjState+1

    ; Put "+" or "-" in front of each amount. Y = 1, 5, 9 addresses the first
    ; (hundreds) cell of amounts 0, 1, 2 in the transfer record.
    LDY #$01
    LDA MoneyGameAmounts+0
    JSR PrependSignToPrice
    LDY #$05
    LDA MoneyGameAmounts+1
    JSR PrependSignToPrice
    LDY #$09
    LDA MoneyGameAmounts+2
    JSR PrependSignToPrice

    ; If the amount chosen is 20 ($14) or 50 ($32), then add it; else subtract it.
    LDX CaveChosenIndex
    LDA MoneyGameAmounts, X
    CMP #$14
    BEQ PostCredit
    CMP #$32
    BNE PostDebit

; ----------------------------------------------------------------------------
; Adds A to the rupees still to be credited (InvRupeesToAdd).
; IN:   A = amount.
; OUT:  InvRupeesToAdd increased by A (8-bit, no overflow check).
; ----------------------------------------------------------------------------
PostCredit:
    CLC
    ADC InvRupeesToAdd
    STA InvRupeesToAdd
    RTS

; ----------------------------------------------------------------------------
; Adds A to the rupees still to be taken away (InvRupeesToSubtract).
; IN:   A = amount.
; OUT:  InvRupeesToSubtract increased by A (8-bit, no overflow check).
; ----------------------------------------------------------------------------
PostDebit:
    CLC
    ADC InvRupeesToSubtract
    STA InvRupeesToSubtract
    RTS

; ----------------------------------------------------------------------------
; Writes "+" before a money game amount of 20 or 50 and "-" before any other amount.
; IN:   A = amount; Y = offset of the amount's first cell from DynTileBuf+4 (1, 5 or 9).
; OUT:  DynTileBuf+4,Y = tile $64 ("+") or $62 ("-"); X changed.
; ----------------------------------------------------------------------------
PrependSignToPrice:
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
UpdateCavePersonState_DoNothing:
    RTS

; ----------------------------------------------------------------------------
; Person state shared by caves and dungeon persons: blanks text line 3 and the price row, next state.
; OUT:  TileBufSelector = $2A (BlankPersonWares: 24 blank tiles at $21E4, 13 at $22C8); ObjState+1
;       incremented.
; NOTE: The wares themselves are sprites; this transfer does not remove them.
; ----------------------------------------------------------------------------
UpdatePersonState_CueTransferBlankPersonWares:
    LDA #$2A                    ; Transfer selector: blank text line 3 and the price row.
    JMP CueTransferBufAndAdvanceState
