; Cave and person variables (bank 1): the textbox, the wares and prices, the money game.
; In a cave the person is the object in slot 1; these bytes reuse the general-purpose object arrays.
; $470 and $471 (entries 1 and 2 of MoneyGameBaseAmounts) are written by raw address.

; ---- Person textbox ----
PersonTextSelector := $415          ; scalar; offset of the person's text address in Person_TextAddrs.
;   2 bytes per entry. In a cave: low 6 bits of the cave's byte in Cave_PersonTextSelectors.
PersonTextIndex := $416             ; scalar; index of the next character of the person's text.
PersonTextVramLo := $45F            ; scalar; low byte of the nametable address of the next character.
;   $A4, $C4, $E4 = start of textbox lines 1, 2, 3 (Person_TextboxLineAddrsLo). +1 per character.

; ---- Cave wares and flags ----
CaveFlags := $413                   ; scalar; cave behaviour bits, built by Cave_Init.
;   bit 0 = wares can be picked up, bit 1 = pay for the ware, bit 2 = show the wares, bit 3 = show the
;   prices, bit 4 = hint cave, bit 5 = money game, bit 6 = heart container requirement, bit 7 = amounts
;   shown negative. Bits 0-1 come from the text selector byte, bits 2-3 from ware 2, bits 4-5 from ware 1,
;   bits 6-7 from ware 0 (high 2 bits of each).
CaveItemIds := $422                 ; [3] cave wares, left to right: item id in bits 0-5 ($3F = none).
;   Bits 6-7 are cave flags. Copied from LevelBlockAttrsE; set to $FF when the ware is taken. Ware X
;   positions: Cave_WareXs.
CavePrices := $430                  ; [3] price or rupee amount of each ware, from LevelBlockAttrsE+60.
;   Money game: overwritten with MoneyGameAmounts when Link picks a rupee.
CaveChosenIndex := $438             ; scalar; index 0-2 of the ware Link picked up.
MoneyGameAmounts := $448            ; [3] money game: rupee amount behind each of the three rupees.
;   A permutation of: 10 or 40 (loss), 10 (loss), 20 or 50 (win). 20 and 50 are added, anything else is
;   subtracted.
MoneyGameOrder := $46C              ; [3] money game: index 0-2 into MoneyGameBaseAmounts for each rupee.
;   One of six orders (Cave_MoneyGamePermutations), chosen by Random+1 in Cave_Init.
MoneyGameBaseAmounts := $46F        ; [3] money game: the amounts before they are arranged.
;   [0] loss of 10 or 40, [1] loss of 10, [2] win of 20 or 50 (Random+2 bits 0 and 1).

; ---- Ware drawing (Cave_DrawWares) ----
CaveDrawWareIndex := $421           ; scalar; index 2 down to 0 of the ware being drawn.

; ---- Price list formatting (Cave_WritePriceListWithChar) ----
CaveCurPriceIndex := $42E           ; scalar; index 0-2 of the price being formatted.
CaveCurPriceOffset := $42F          ; scalar; offset of that price's text in the transfer record; +4 each.
