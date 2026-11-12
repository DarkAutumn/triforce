; ============================================================================
; FILE:     bank1/cave-init-person-text.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 52-300
; SUBJECTS: (none)
; TOPIC:    Person text data, money-game tables, InitCave, cave flags and wares
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 PersonTextAddrs                 data
;   L0024 PersonText                      data
;   L0027 OverworldPersonTextSelectors    data
;   L0032 MoneyGameLossAmounts            data
;   L0035 MoneyGamePermutations           data
;   L0040 MoneyGamePermutationEndIndexes  data
;   L0043 InitCave                        code
;   L0074 UnhaltLink                      code
;   L0079 InitCaveContinue                code
;   L0247 SetUpCommonCaveObjects          code  Params:
; ============================================================================

PersonTextAddrs:
    .INCLUDE "dat/PersonTextAddrs.inc"

PersonText:
    .INCBIN "dat/PersonText.dat"

OverworldPersonTextSelectors:
    .BYTE $40, $60, $42, $42, $04, $06, $48, $0A
    .BYTE $4C, $0E, $D0, $D2, $D2, $DC, $DC, $DE
    .BYTE $DE, $62, $62, $62

MoneyGameLossAmounts:
    .BYTE $0A, $28

MoneyGamePermutations:
    .BYTE $00, $01, $02, $01, $02, $00, $02, $00
    .BYTE $01, $00, $02, $01, $02, $01, $00, $01
    .BYTE $00, $02

MoneyGamePermutationEndIndexes:
    .BYTE $02, $05, $08, $0B, $0E, $11

InitCave:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; If the cave/person does not remember state (give item, door charge),
    ; then go handle other kinds.
    LDA ObjType+1
    CMP #$72
    BEQ @TakeType
    CMP #$71
    BEQ @TakeType
    CMP #$7B
    BCS @TakeType
    CMP #$6E
    BCS InitCaveContinue

@TakeType:
    ; Else the cave/person simply does:
    ; - taken:     no person, no text
    ; - not taken: show a person and text
    ;
    ; So, if the item was not taken, then go set up the text.
    JSR GetRoomFlagUWItemState
    BEQ InitCaveContinue

    ; Otherwise destroy the person object, and let Link continue.
    LDA #$00
    STA ObjType+1

UnhaltLink:
    LDA #$00
    STA ObjState
    RTS

InitCaveContinue:
    ; Get this cave's index (object type - $6A).
    LDA ObjType+1
    SEC
    SBC #$6A
    TAY

    ; Get the person text selector by looking up the text selector byte
    ; for this index, and masking it with $3F.
    LDA OverworldPersonTextSelectors, Y
    PHA                         ; Save the text selector byte before masking.
    AND #$3F
    STA PersonTextSelector

    ; Pop and mask the text selector byte with $C0 to get the
    ; "pay" and "pick up" cave flags and store them in [03].
    PLA
    AND #$C0
    STA $03

    ; At this point, the Y register still has the cave index.
    ; Multiply it by 3 to get the offset of the first item in a set of 3.
    LDA #$FD
:
    CLC
    ADC #$03
    DEY
    BPL :-
    TAY

    ; Loop over the wares from 0 to 2.
    LDX #$00

@CopyCaveBytes:
    ; Copy 3 item ID bytes from the Cave level block info.
    ; Item ID's are in the low 6 bits, cave flags are in the high 2 bits.
    LDA LevelBlockAttrsE, Y
    STA CaveItemIds, X

    ; Copy the cave flags (high 2 bits) of the three items to [00] to [02].
    AND #$C0
    STA $00, X

    ; Copy 3 prices from the Cave level block info.
    LDA LevelBlockAttrsE+60, Y
    STA CavePrices, X

    ; Bottom of the wares copying loop.
    INY
    INX
    CPX #$03
    BNE @CopyCaveBytes

    ; Get the top 2 bits of text selector byte in [03],
    ; shift them to the bottom, and combine them with [00].
    LDA $03
    ASL
    ROL
    ROL
    ORA $00
    STA $00

    ; Get the top 2 bits of item descriptor 2, shift them right 4 times,
    ; and combine them with [00].
    LDA $02
    LSR
    LSR
    LSR
    LSR
    ORA $00
    STA $00

    ; Get the top 2 bits of item descriptor 1, shift them right twice,
    ; and combine them with [00] into CaveFlags.
    LDA $01
    LSR
    LSR
    ORA $00
    STA CaveFlags

    ; The end result is a byte with 8 cave flags.
    ;
    ; If cave flag "money game" is missing, then go finish up,
    ; instead of choosing random amounts for the money game.
    AND #$20
    BEQ @ResetTextbox

    ; Determine an index randomly with increasing probability,
    ; from 6 to 1. Start with the following test:
    ;
    ;   If $FF < Random, then go use index 6.
    LDA #$FF
    LDY #$06

@ChoosePermutation:
    CMP Random+1
    BCC @FoundPermutation

    ; If the test fails, then subtract $2B from the reference number.
    SEC
    SBC #$2B
    DEY
    BNE @ChoosePermutation

@FoundPermutation:
    ; The index chooses the last offset of a permutation of three indexes.
    LDX MoneyGamePermutationEndIndexes, Y

    ; Copy the permutation of 3 indexes to [046C] to [046E].
    LDY #$02

@CopyPermutation:
    LDA MoneyGamePermutations, X
    STA $046C, Y
    DEX
    DEY
    BPL @CopyPermutation

    ; With the first bit of another random number, choose 10 or 40
    ; to put in [046F]. This is a random amount to lose.
    LDA Random+2
    AND #$01
    TAY
    LDA MoneyGameLossAmounts, Y
    STA $046F

    ; Put 10 in [0470]. This is a fixed amount to lose.
    LDA #$0A
    STA $0470

    ; With a different bit of the second random number,
    ; choose 20 or 50 to put in [0471]. This is an amount to win.
    LDY #$14
    LDA Random+2
    AND #$02
    BEQ :+
    LDY #$32
:
    STY $0471

    ; The 3 amounts are now in [046F] to [0471].
    ; Reference these amounts using the permutation of indexes
    ; in [046C] to [046E]; and copy them to [0448] to [044A].
    LDX #$02

@ArrangeAmounts:
    LDY $046C, X
    LDA $046F, Y
    STA MoneyGameAmounts, X
    DEX
    BPL @ArrangeAmounts

@ResetTextbox:
    ; Reset the current character index.
    LDA #$00
    STA PersonTextIndex

    ; Point to the front of the first textbox line.
    LDA TextboxLineAddrsLo+2
    STA PersonTextPtr
    RTS

; Params:
; A: X coordinate
; Y: Y coordinate
;
; Set the person's coordinates, HP, and attributes.
;
SetUpCommonCaveObjects:
    STA ObjX, X
    STY ObjY, X
    LDA #$00
    STA ObjHP, X
    LDA #$81
    STA ObjAttr, X

    ; Halt Link.
    LDA #$40
    STA ObjState

    ; Set up the two bonfires.
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
