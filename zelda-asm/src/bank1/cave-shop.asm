; ============================================================================
; FILE:     bank1/cave-shop.asm          BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 678-856
; SUBJECTS: Cave
; TOPIC:    Cave person state 2: a person that only talks (and the door repair charge), or Link
;           touching a ware: purchase, free gift with or without a heart container requirement, entry
;           to the hint and money game state. Also state 4, which removes the person after a ware
;           was taken.
; SEE ALSO: bank1/cave-person-prices.asm (state table, Cave_WareXs, Person_CueTransferBufAndAdvanceState);
;           bank1/cave-init-person-text.asm (CaveFlags, once-only persons);
;           bank1/cave-hint-money-game.asm (state 5, Inv_PostRupeeDebit); bank1/take-item.asm (Item_Take,
;           RoomFlags_SetItemTaken); bank1/status-bar-rupees.asm (how posted rupees are counted off).
; KEY FACTS:
;   - Link takes a ware when his X equals the ware's X exactly ($58, $78 or $98) and his Y is within
;     5 pixels of $98, and only while InvRupeesToSubtract is 0.
;     Cave_State2TalkOrShopOrDoorCharge
;   - Buying (CaveFlags $02) needs InvRupees >= price; the price is posted to InvRupeesToSubtract,
;     not taken at once. With too few rupees nothing happens.
;     Cave_State2TalkOrShopOrDoorCharge
;   - Heart container requirement (CaveFlags $40): HeartValues >= $40 for person $6C (white sword),
;     >= $B0 for any other person. The high nibble is containers - 1 and the low nibble full hearts,
;     so this is exactly "5 containers" and "12 containers" whatever Link's current health.
;     Cave_State2TalkOrShopOrDoorCharge
;   - Taking any ware sets bit $10 of the room's flags, runs Item_Take, and sets the person's timer to
;     $40; from the second frame after (state 4) the person and the other wares flicker, and when
;     the timer reaches 0 the person is removed and the wares with it: one ware per visit.
;     Cave_State2TalkOrShopOrDoorCharge, Cave_State4DelayThenHide
;   - The door repair person ($71) posts a charge of 20 rupees and sets bit $10 of the room's flags
;     as soon as his text ends; the amount of rupees Link has is not tested.
;     Cave_State2TalkOrShopOrDoorCharge
;   - A paid hint (CaveFlags $10) needs InvRupees >= price and goes to state 5; the money game and
;     the secret rupee gift (CaveFlags $20) go to state 5 without any test here.
;     Cave_State2TalkOrShopOrDoorCharge
; CONTENTS: (generated; L = line in this file)
;   L0055 Cave_State2TalkOrShopOrDoorCharge         code  Cave person state 2: talk-only persons finish (door repair charges 20); else Link may take a ware.
;   L0206 Cave_ClearShowPricesFlag                  code  Clears the "show prices" cave flag ($08), which stops the rupee icon being drawn.
;   L0212 Cave_State2TalkOrShopOrDoorCharge_Return  code  Shared return of the state 2 handler.
;   L0221 Cave_State4DelayThenHide                  code  Cave person state 4: when the person's timer runs out, removes the person (ObjType+1 = 0).
;   L0227 Cave_State4DelayThenHide_Return           code  Shared return (also used by the money game when Link has fewer than 10 rupees).
; ============================================================================

; ----------------------------------------------------------------------------
; Cave person state 2: talk-only persons finish (door repair charges 20); else Link may take a ware.
; IN:   CaveFlags; CaveItemIds[3]; CavePrices[3]; ObjType+1 = person type; Link's ObjX, ObjY.
; OUT:  ObjState+1 = 8 (talk only), 5 (hint, money game, rupee gift) or 3 (ware taken, ObjTimer+1 =
;       $40); CaveChosenIndex = ware touched; InvRupeesToSubtract increased by a price or by 20.
; NOTE: Wares are tested from right to left (2 to 0). A ware is touched when ObjX = its X and
;       |ObjY - $98| < 6. Nothing is tested while InvRupeesToSubtract is nonzero.
; NOTE: Heart container minimum: HeartValues >= $40 (5 containers) for person $6C, >= $B0 (12) for
;       all other persons with CaveFlags $40; the low nibble (full hearts) cannot change the result.
; NOTE: The price is posted before the heart container test, so a ware with both the "pay" and the
;       "heart requirement" flag would charge Link without being taken. The sword persons $6C and
;       $6D have no "pay" flag in Cave_PersonTextSelectors.
; ----------------------------------------------------------------------------
Cave_State2TalkOrShopOrDoorCharge:
    ; With the "pick up" flag (bit 0), go see whether Link touches a ware.
    LDA CaveFlags
    LSR                         ; Cave flag: pick up ($01)
    BCS @CheckPickUp

    ; The person only talks: go to the final state 8, where the person stays.
    LDA #$08
    STA ObjState+1

    ; Unless this is the door repair person ($71), that is all.
    LDA ObjType+1
    CMP #$71
    BNE @Exit

    ; Door repair: post 20 rupees to subtract.
    LDA InvRupeesToSubtract
    CLC
    ADC #$14
    STA InvRupeesToSubtract

    ; Set bit $10 of the room's flags, so that the person is gone on the next visit (Cave_Init).
    JSR RoomFlags_SetItemTaken

@Exit:
    RTS

@CheckPickUp:
    ; A cave where Link touches a ware: gift, shop, paid hint, money game or secret rupees.
    ;
    ; While rupees are still being counted off, do nothing.
    LDA InvRupeesToSubtract
    BNE @Exit

    ; Loop over the wares from 2 to 0.
    LDX #$02

@LoopWare:
    ; No ware here (id $3F; a taken ware is $FF): next.
    LDA CaveItemIds, X
    AND #$3F                    ; Mask off the cave flags.
    CMP #$3F                    ; No item
    BEQ @NextWare

    ; Link's X must equal the ware's X.
    LDA ObjX
    CMP Cave_WareXs, X
    BNE @NextWare

    ; And Link's Y must be less than 6 pixels from the ware's Y ($98).
    LDA ObjY
    SEC
    SBC #$98
    JSR Math_Abs
    CMP #$06
    BCC @PickedUp

@NextWare:
    DEX
    BPL @LoopWare
    RTS

@PickedUp:
    ; Remember which ware was touched.
    STX CaveChosenIndex

    ; Neither "hint" ($10) nor "money game" ($20): go take or buy the ware.
    LDA CaveFlags
    AND #$30                    ; Cave flags: hint and money game
    BEQ @PickedUpWare

    ; "Money game" without "hint" (the money game and the secret rupee persons):
    ; go to state 5 without paying anything here.
    AND #$10
    BEQ @GoToState5

    ; "Hint": Link must have at least the price; if not, return.
    LDA InvRupees
    CMP CavePrices, X
    BCC Cave_State2TalkOrShopOrDoorCharge_Return

    ; Post the price of the hint to subtract.
    LDA CavePrices, X
    JSR Inv_PostRupeeDebit

@GoToState5:
    ; State 5 handles the hint, the money game and the rupee gift.
    LDA #$05
    STA ObjState+1
    RTS

@PickedUpWare:
    ; A shop or a gift. Without the "pay" flag ($02), skip the payment.
    LDA CaveFlags
    AND #$02
    BEQ @TryToTake

    ; Link must have at least the price; if not, return.
    LDA InvRupees
    CMP CavePrices, X
    BCC Cave_State2TalkOrShopOrDoorCharge_Return

    ; Post the price of the ware to subtract.
    LDA CavePrices, X
    JSR Inv_PostRupeeDebit

@TryToTake:
    ; Without the "heart requirement" flag ($40), take the ware.
    LDA CaveFlags
    AND #$40
    BEQ @Take

    ; Minimum HeartValues (high nibble = heart containers - 1, low nibble = full hearts): $40 for
    ; person $6C (white sword), $B0 for any other person. The whole byte is compared, but a byte
    ; with high nibble >= 4 (>= $B) is >= $40 ($B0) for every low nibble: 5 and 12 containers.
    LDY #$40
    LDA ObjType+1
    CMP #$6C
    BEQ :+
    LDY #$B0
:
    CPY HeartValues

    ; Minimum <= HeartValues: take the ware.
    BEQ @Take
    BCC @Take

    ; Too few heart containers: nothing happens.
    RTS

@Take:
    ; Set bit $10 of the room's flags (for once-only persons: gone on the next visit).
    JSR RoomFlags_SetItemTaken
    LDA CaveItemIds, X
    AND #$3F                    ; Mask off the cave flags.
    PHA                         ; Push the item id.

    ; Mark the ware taken ($FF): no longer drawn or touchable.
    LDA #$FF
    STA CaveItemIds, X
    PLA                         ; Pop the item id.
    JSR Item_Take
    LDA #$1E                    ; Transfer buffer: Transfer_BlankTextLines12Buf (text lines 1 and 2)
    JSR Person_CueTransferBufAndAdvanceState

    ; Now in state 3. The timer runs out $40 frames from now, in state 4, which then removes
    ; the person.
    LDA #$40
    STA ObjTimer+1

; Clears the "show prices" cave flag ($08), which stops the rupee icon being drawn.
Cave_ClearShowPricesFlag:
    LDA CaveFlags
    AND #$F7
    STA CaveFlags

; Shared return of the state 2 handler.
Cave_State2TalkOrShopOrDoorCharge_Return:
    RTS

; ----------------------------------------------------------------------------
; Cave person state 4: when the person's timer runs out, removes the person (ObjType+1 = 0).
; NOTE: With the person gone its update no longer runs, so the remaining wares are no longer drawn
;       or touchable. During this state the person and wares are drawn on even frames only
;       (Cave_UpdatePerson).
; ----------------------------------------------------------------------------
Cave_State4DelayThenHide:
    LDA ObjTimer+1
    BNE Cave_State4DelayThenHide_Return
    STA ObjType+1

; Shared return (also used by the money game when Link has fewer than 10 rupees).
Cave_State4DelayThenHide_Return:
    RTS
