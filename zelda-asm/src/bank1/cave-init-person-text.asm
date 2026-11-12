; ============================================================================
; FILE:     bank1/cave-init-person-text.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 52-300
; SUBJECTS: Cave, Link, Person
; TOPIC:    The text strings of cave and dungeon persons, the per-cave text selector table, the money
;           game tables, and the cave person's init: person and fires placed, Link halted, wares, prices
;           and the eight cave flags loaded, money game amounts drawn.
; SEE ALSO: bank1/cave-person-prices.asm (Cave_UpdatePerson: the person's state machine, ware and price
;           drawing); bank1/person-textbox.asm (typing the text); bank1/cave-shop.asm (taking or buying
;           a ware, door repair charge); bank1/cave-hint-money-game.asm (paid hints, money game payout,
;           secret rupees); bank1/underworld-person-init.asm (dungeon persons, which share
;           Person_SetUpWithFires and the text tables); bank1/take-item.asm (RoomFlags_GetItemTaken);
;           bank7/object-init-table.asm (Obj_Init sends types >= $6A to Cave_Init).
; KEY FACTS:
;   - A cave person is object type $6A + cave index, always in object slot 1, at ($78,$80); two standing
;     fires (type $40) go in slots 2 and 3 at X $48 and $A8. Cave_Init, Person_SetUpWithFires
;   - Link is halted (ObjState = $40) when the person is set up and stays halted until the text has
;     been typed; the medicine shop ($74) instead releases him on its first update and never halts
;     him again. Person_SetUpWithFires, Link_Unhalt, Cave_UpdatePerson
;   - Person types $6A-$6D, $71, $72 and $7B-$7D are once-only: if bit $10 of the room's flags is set
;     the person is removed (ObjType+1 = 0) and Link is released at once. Types $6E-$70 and $73-$7A
;     (road, hints, money game, shops) always appear. Cave_Init
;   - CaveFlags bits: $01 pick up, $02 pay, $04 show wares, $08 show prices, $10 hint, $20 money game,
;     $40 heart container requirement, $80 minus sign before prices. Cave_Init_SetUpTextAndWares
;   - Wares and prices come from LevelBlockAttrsE: 3 ware bytes at 3 * index, 3 prices 60 bytes later.
;     Cave_Init_SetUpTextAndWares
;   - The money game is decided when the person is initialized, not when Link picks: Random+1 chooses
;     one of 6 orders, Random+2 bit 0 the big loss (10 or 40), bit 1 the win (20 or 50); the third
;     amount is always a loss of 10. Cave_Init_SetUpTextAndWares, Cave_MoneyGamePermutations
;   - The same draw also runs for the secret-rupee persons $7B-$7D (their middle ware byte $98 has
;     bit 7 = flag $20), whose update never reads MoneyGameAmounts. Cave_Init_SetUpTextAndWares
; CONTENTS: (generated; L = line in this file)
;   L0047 Person_TextAddrs                     data  TABLE: address of each person text string in Person_Text; index = text selector (even, 2 bytes per
;   L0054 Person_Text                          data  DATA: the text strings of all cave and dungeon persons (1366 bytes). Each byte: bits 0-5 = character
;   L0071 Cave_PersonTextSelectors             data  TABLE: text selector and two cave flags of each cave person; index = object type - $6A;
;   L0077 Cave_MoneyGameLossAmounts            data  TABLE: the variable loss of the money game; index = bit 0 of Random+2; value = rupees (10, 40).
;   L0083 Cave_MoneyGamePermutations           data  TABLE: the six orders of the three money game amounts, 3 bytes each; value = index of the amount
;   L0089 Cave_MoneyGamePermutationEndIndexes  data  TABLE: offset of the last byte of each order in Cave_MoneyGamePermutations; index = order 0-5.
;   L0101 Cave_Init                            code  Sets up a cave person: places it with its fires, halts Link, removes it if its gift was taken.
;   L0130 Link_Unhalt                          code  Releases Link from the halt: ObjState of slot 0 = 0.
;   L0145 Cave_Init_SetUpTextAndWares          code  Loads the cave's text selector, wares, prices and cave flags; draws the money game amounts.
;   L0304 Person_SetUpWithFires                code  Places a cave or dungeon person with a standing fire on each side, and halts Link.
; ============================================================================

; TABLE: address of each person text string in Person_Text; index = text selector (even, 2 bytes per
; entry; PersonTextSelector); value = address. 38 strings, selectors $00-$4A.
Person_TextAddrs:
    .INCLUDE "dat/PersonTextAddrs.inc"

; DATA: the text strings of all cave and dungeon persons (1366 bytes). Each byte: bits 0-5 = character
; tile ($24 = blank, $25 = blank that is skipped without delay); bits 7-6 = what follows this
; character: $80 continue on textbox line 2, $40 continue on line 3, $C0 end of the text
; (Person_UpdateState_Textbox).
Person_Text:
    .INCBIN "dat/PersonText.dat"

; TABLE: text selector and two cave flags of each cave person; index = object type - $6A;
; value = bits 0-5 text selector (offset into Person_TextAddrs), bit 6 ($40) = wares can be picked up
; (CaveFlags bit 0), bit 7 ($80) = wares must be paid for (CaveFlags bit 1).
;   $6A $40  wooden sword (text $00, pick up)      $74 $D0  medicine shop (text $10, pick up, pay)
;   $6B $60  take any one (text $20, pick up)      $75 $D2  paid hints (text $12, pick up, pay)
;   $6C $42  white sword (text $02, pick up)       $76 $D2  paid hints (text $12, pick up, pay)
;   $6D $42  magic sword (text $02, pick up)       $77 $DC  shop (text $1C, pick up, pay)
;   $6E $04  take any road (text $04)              $78 $DC  shop (text $1C, pick up, pay)
;   $6F $06  hint (text $06)                       $79 $DE  shop (text $1E, pick up, pay)
;   $70 $48  money game (text $08, pick up)        $7A $DE  shop (text $1E, pick up, pay)
;   $71 $0A  door repair charge (text $0A)         $7B $62  secret rupees (text $22, pick up)
;   $72 $4C  letter (text $0C, pick up)            $7C $62  secret rupees (text $22, pick up)
;   $73 $0E  hint (text $0E)                       $7D $62  secret rupees (text $22, pick up)
; The wares themselves are level data (LevelBlockAttrsE), not in this table.
Cave_PersonTextSelectors:
    .BYTE $40, $60, $42, $42, $04, $06, $48, $0A
    .BYTE $4C, $0E, $D0, $D2, $D2, $DC, $DC, $DE
    .BYTE $DE, $62, $62, $62

; TABLE: the variable loss of the money game; index = bit 0 of Random+2; value = rupees (10, 40).
Cave_MoneyGameLossAmounts:
    .BYTE $0A, $28

; TABLE: the six orders of the three money game amounts, 3 bytes each; value = index of the amount
; (0 = loss of 10 or 40, 1 = loss of 10, 2 = win of 20 or 50) placed at the left, middle and right
; ware: 0,1,2 / 1,2,0 / 2,0,1 / 0,2,1 / 2,1,0 / 1,0,2.
Cave_MoneyGamePermutations:
    .BYTE $00, $01, $02, $01, $02, $00, $02, $00
    .BYTE $01, $00, $02, $01, $02, $01, $00, $01
    .BYTE $00, $02

; TABLE: offset of the last byte of each order in Cave_MoneyGamePermutations; index = order 0-5.
Cave_MoneyGamePermutationEndIndexes:
    .BYTE $02, $05, $08, $0B, $0E, $11

; ----------------------------------------------------------------------------
; Sets up a cave person: places it with its fires, halts Link, removes it if its gift was taken.
; IN:   X = the person's object slot (1; the code addresses slot 1 directly); ObjType+1 = person
;       type $6A-$7D.
; OUT:  falls or branches into Cave_Init_SetUpTextAndWares unless the person is removed.
; NOTE: Once-only persons are $6A-$6D (swords, take any one), $71 (door repair), $72 (letter) and
;       $7B-$7D (secret rupees): when bit $10 of the room's flags is set, the person's type becomes 0
;       and Link is unhalted in the same frame. The fires stay.
; ----------------------------------------------------------------------------
Cave_Init:
    ; The person goes at ($78, $80).
    LDA #$78
    LDY #$80
    JSR Person_SetUpWithFires

    ; Types $71, $72, >= $7B and < $6E are once-only: go test the room flag.
    ; Types $6E-$70 and $73-$7A always appear: go set up the text and wares.
    LDA ObjType+1
    CMP #$72
    BEQ @CheckTaken
    CMP #$71
    BEQ @CheckTaken
    CMP #$7B
    BCS @CheckTaken
    CMP #$6E
    BCS Cave_Init_SetUpTextAndWares

@CheckTaken:
    ; Once-only person. If bit $10 of the room's flags is clear (nothing taken or paid yet),
    ; go set up the text and wares.
    JSR RoomFlags_GetItemTaken
    BEQ Cave_Init_SetUpTextAndWares

    ; Already taken: remove the person object and release Link.
    LDA #$00
    STA ObjType+1

; Releases Link from the halt: ObjState of slot 0 = 0.
Link_Unhalt:
    LDA #$00
    STA ObjState
    RTS

; ----------------------------------------------------------------------------
; Loads the cave's text selector, wares, prices and cave flags; draws the money game amounts.
; IN:   ObjType+1 = person type $6A-$7D.
; OUT:  PersonTextSelector, CaveItemIds[3], CavePrices[3], CaveFlags, MoneyGameAmounts[3] (only when
;       flag $20 is set), PersonTextIndex = 0, PersonTextVramLo = start of textbox line 1.
;       Uses [00]-[03], MoneyGameOrder, MoneyGameBaseAmounts.
; NOTE: Money game RNG: Random+1 picks the order (0: $00-$28, 41 of 256; 1-5: 43 of 256 each, in
;       rising bands of $2B); Random+2 bit 0 set makes the variable loss 40 instead of 10, bit 1 set
;       makes the win 50 instead of 20. Both bytes are read once, here.
; ----------------------------------------------------------------------------
Cave_Init_SetUpTextAndWares:
    ; Y = cave index (object type - $6A).
    LDA ObjType+1
    SEC
    SBC #$6A
    TAY

    ; The low 6 bits of the cave's selector byte are the text selector.
    LDA Cave_PersonTextSelectors, Y
    PHA                         ; Save the whole selector byte.
    AND #$3F
    STA PersonTextSelector

    ; The high 2 bits are the "pay" ($80) and "pick up" ($40) cave flags; keep them in [03].
    PLA
    AND #$C0
    STA $03

    ; Y = cave index * 3, the offset of the cave's first ware: start at -3 and add 3
    ; (cave index + 1) times.
    LDA #$FD
:
    CLC
    ADC #$03
    DEY
    BPL :-
    TAY

    ; Loop over the wares from 0 to 2 (left to right).
    LDX #$00

@CopyCaveBytes:
    ; Copy the ware byte: item id in the low 6 bits ($3F = none), cave flags in the high 2 bits.
    LDA LevelBlockAttrsE, Y
    STA CaveItemIds, X

    ; Keep the high 2 bits of wares 0, 1, 2 in [00], [01], [02].
    AND #$C0
    STA $00, X

    ; The price of this ware is 60 bytes after its ware byte.
    LDA LevelBlockAttrsE+60, Y
    STA CavePrices, X

    INY
    INX
    CPX #$03
    BNE @CopyCaveBytes

    ; Rotate the selector's high 2 bits down to bits 1-0 (bit 7 "pay" -> bit 1, bit 6 "pick up"
    ; -> bit 0) and combine them with ware 0's bits 7-6 in [00].
    LDA $03
    ASL
    ROL
    ROL
    ORA $00
    STA $00

    ; Ware 2's high 2 bits go to bits 3-2 (show prices, show wares).
    LDA $02
    LSR
    LSR
    LSR
    LSR
    ORA $00
    STA $00

    ; Ware 1's high 2 bits go to bits 5-4 (money game, hint).
    LDA $01
    LSR
    LSR
    ORA $00
    STA CaveFlags

    ; CaveFlags is complete: $80 minus sign, $40 heart requirement (ware 0); $20 money game,
    ; $10 hint (ware 1); $08 show prices, $04 show wares (ware 2); $02 pay, $01 pick up (selector).
    ;
    ; Without the "money game" flag ($20), skip the draw of the amounts.
    AND #$20
    BEQ @ResetTextbox

    ; Choose the order index Y from Random+1. The reference value starts at $FF and drops by $2B per
    ; step; the first step whose reference is below Random+1 wins. $FF can never be below, so
    ; Y = 6 is never chosen: Y = 5 for $D5-$FF, 4 for $AA-$D4, 3 for $7F-$A9, 2 for $54-$7E,
    ; 1 for $29-$53, and 0 (loop exhausted) for $00-$28.
    LDA #$FF
    LDY #$06

@ChoosePermutation:
    CMP Random+1
    BCC @FoundPermutation
    SEC
    SBC #$2B
    DEY
    BNE @ChoosePermutation

@FoundPermutation:
    ; X = offset of the last byte of the chosen order.
    LDX Cave_MoneyGamePermutationEndIndexes, Y

    ; Copy the order's 3 amount indexes to MoneyGameOrder.
    LDY #$02

@CopyPermutation:
    LDA Cave_MoneyGamePermutations, X
    STA MoneyGameOrder, Y
    DEX
    DEY
    BPL @CopyPermutation

    ; Amount 0 (MoneyGameBaseAmounts[0]): a loss of 10 (Random+2 bit 0 clear) or 40 (set).
    LDA Random+2
    AND #$01
    TAY
    LDA Cave_MoneyGameLossAmounts, Y
    STA MoneyGameBaseAmounts

    ; Amount 1 ($0470 = MoneyGameBaseAmounts[1]): always a loss of 10.
    LDA #$0A
    STA $0470

    ; Amount 2 ($0471 = MoneyGameBaseAmounts[2]): a win of 20 (Random+2 bit 1 clear) or 50 (set).
    LDY #$14
    LDA Random+2
    AND #$02
    BEQ :+
    LDY #$32
:
    STY $0471

    ; MoneyGameAmounts[X] = MoneyGameBaseAmounts[MoneyGameOrder[X]] for wares X = 2 to 0.
    ; Whether an amount is a win or a loss is decided later by its value (20 and 50 win).
    LDX #$02

@ArrangeAmounts:
    LDY MoneyGameOrder, X
    LDA MoneyGameBaseAmounts, Y
    STA MoneyGameAmounts, X
    DEX
    BPL @ArrangeAmounts

@ResetTextbox:
    ; Start the text at its first character.
    LDA #$00
    STA PersonTextIndex

    ; Point to the start of textbox line 1 (entry 2 of Person_TextboxLineAddrsLo, $A4).
    LDA Person_TextboxLineAddrsLo+2
    STA PersonTextVramLo
    RTS

; ----------------------------------------------------------------------------
; Places a cave or dungeon person with a standing fire on each side, and halts Link.
; IN:   X = the person's object slot; A = X coordinate; Y = Y coordinate.
; OUT:  ObjX,X / ObjY,X set; ObjHP,X = 0; ObjAttr,X = $81; ObjState (Link) = $40 (halted);
;       ObjType+2 and ObjType+3 = $40 (standing fire) at X $48 and $A8, same Y as the person.
; NOTE: The fire types go to the fixed slots 2 and 3 but the fire coordinates to slots X+1 and X+2,
;       so the two agree only when X = 1.
; ----------------------------------------------------------------------------
Person_SetUpWithFires:
    STA ObjX, X
    STY ObjY, X
    LDA #$00
    STA ObjHP, X
    LDA #$81
    STA ObjAttr, X

    ; Halt Link.
    LDA #$40
    STA ObjState

    ; Make object slots 2 and 3 standing fires (type $40), left and right of the person.
    LDA #$40
    STA ObjType+2
    STA ObjType+3
    LDA #$48
    STA ObjX+1, X
    LDA #$A8
    STA ObjX+2, X
    STY ObjY+1, X
    STY ObjY+2, X
    RTS
