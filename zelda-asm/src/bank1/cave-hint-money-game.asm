; ============================================================================
; FILE:     bank1/cave-hint-money-game.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 858-1009
; SUBJECTS: (none)
; TOPIC:    Paid hints, money game payout, secret rupee gift, credit/debit
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 HintCaveTextSelectors0                         data
;   L0022 HintCaveTextSelectors1                         data
;   L0025 UpdateCavePersonState_HintOrMoneyGame          code
;   L0129 PostCredit                                     code
;   L0140 PostDebit                                      code  Params:
;   L0152 PrependSignToPrice                             code  Params:
;   L0165 UpdateCavePersonState_DoNothing                code
;   L0168 UpdatePersonState_CueTransferBlankPersonWares  code
; ============================================================================

HintCaveTextSelectors0:
    .BYTE $14, $14, $16

HintCaveTextSelectors1:
    .BYTE $14, $18, $1A

UpdateCavePersonState_HintOrMoneyGame:
    ; If "hint" cave flag is missing, then go handle the money game.
    LDA CaveFlags
    AND #$10
    BEQ @HandleMoneyGameOrGiveaway

    ; If cave type = $75, then use 0 for base offset of
    ; the first set of text selectors, else 3 for the second.
    LDA #$00
    LDY ObjType+1
    CPY #$75
    BEQ :+
    LDA #$03
:
    CLC

    ; Add the base offset and the index of the item chosen
    ; to get the index of a text selector.
    ADC CaveChosenIndex
    TAY

    ; Look up text selector by the index calculated above.
    LDA HintCaveTextSelectors0, Y
    STA PersonTextSelector

    ; Point to the front of the first line of the textbox.
    LDA TextboxLineAddrsLo+2
    STA PersonTextVramLo

    ; Reset the character index.
    LDA #$00
    STA PersonTextIndex
    JSR ClearPricesCaveFlag
    LDA #$1E                    ; Blank textbox lines
    JMP CueTransferBufAndAdvanceState

@HandleMoneyGameOrGiveaway:
    ; If cave type < $7B, then go handle the money game.
    ;
    ; This is a moblin giving away money.
    LDA ObjType+1
    CMP #$7B
    BCC @HandleMoneyGame
    JSR CopyPriceListTemplate

    ; Write a price list into the dynamic transfer buf,
    ; but with a space instead of "X".
    LDA #$24
    JSR WritePricesToDynamicTransferBuf
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request
    JSR SetRoomFlagUWItemState

    ; Go to state 8 to do nothing but show the person.
    LDA #$08
    STA ObjState+1

    ; Post the middle amount in [0431] to add.
    LDA CavePrices+1
    JMP PostCredit

@HandleMoneyGame:
    ; Money game
    ;
    ; If rupees < $A, then return.
    LDA InvRupees
    CMP #$0A
    BCC L493A_Exit
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request

    ; Copy amounts in [0448][Y] for money game to prices [0430][Y].
    LDY #$02
:
    LDA MoneyGameAmounts, Y
    STA CavePrices, Y
    DEY
    BPL :-
    JSR WritePricesTransferBuf

    ; Go to state 8 to do nothing but show the person.
    LDA #$08
    STA ObjState+1

    ; Prepend the sign to each money game amount.
    LDY #$01
    LDA MoneyGameAmounts+0
    JSR PrependSignToPrice
    LDY #$05
    LDA MoneyGameAmounts+1
    JSR PrependSignToPrice
    LDY #$09
    LDA MoneyGameAmounts+2
    JSR PrependSignToPrice

    ; If the amount chosen = 20 or 50, then add it;
    ; else subtract it.
    LDX CaveChosenIndex
    LDA MoneyGameAmounts, X
    CMP #$14
    BEQ PostCredit
    CMP #$32
    BNE PostDebit

PostCredit:
    CLC
    ADC InvRupeesToAdd
    STA InvRupeesToAdd
    RTS

; Params:
; A: amount
;
; Add the amount paid to the rupees to subtract.
;
PostDebit:
    CLC
    ADC InvRupeesToSubtract
    STA InvRupeesToSubtract
    RTS

; Params:
; A: price
; Y: offset from first character after "X"
;
; If the amount is $14 or $32, then use "+", else "-".
;
PrependSignToPrice:
    LDX #$64
    CMP #$14
    BEQ :+
    CMP #$32
    BEQ :+
    LDX #$62
:
    ; Copy the character to the first element after the "X",
    ; offset by the parameter.
    TXA
    STA DynTileBuf+4, Y

UpdateCavePersonState_DoNothing:
    RTS

UpdatePersonState_CueTransferBlankPersonWares:
    LDA #$2A                    ; Also blanks the third line of text.
    JMP CueTransferBufAndAdvanceState
