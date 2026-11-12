; ============================================================================
; FILE:     bank5/submenu-items.asm      BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7844-8197
; SUBJECTS: (none)
; TOPIC:    Submenu items, cursor, selection
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0025 SubmenuItemXs                  data
;   L0029 DrawSubmenuItems               code
;   L0131 SubmenuCursorXs                data
;   L0135 UpdateSubmenuSelection         code
;   L0279 FindAndSelectOccupiedItemSlot  code  Params:
;   L0283 LoopItemSlot                   code
;   L0294 CheckNextItem                  code
;   L0299 FoundSlot                      code
;   L0305 SetSlotFound                   code
;   L0308 L177F1_Exit                    code
;   L0311 CheckBoomerangs                code
;   L0328 CheckLetter                    code
;   L0343 DrawItemInInventoryWithX       code
;   L0359 Cycle9InDirection              code  Params:
; ============================================================================

SubmenuItemXs:
    .BYTE $80, $98, $AC, $B4, $C8, $80, $98, $B0
    .BYTE $C8, $80, $94, $A0, $B0, $C0, $CC, $B0

DrawSubmenuItems:
    ; Look for a magic boomerang, then a wooden boomerang.
    ; If you find one, then draw it.
    LDX #$1E
:
    LDA Items, X
    BNE @FoundBoomerang
    DEX
    CPX #$1C
    BNE :-
    BEQ @DrawOtherItems         ; If you don't find either, skip drawing a boomerang.

@FoundBoomerang:
    LDA #$36
    STA $01                     ; [01] Y
    LDA #$80
    STA $00                     ; [00] X
    TXA
    TAY                         ; Copy the item slot.
    JSR DrawItemInInventory

@DrawOtherItems:
    ; Look at the rest of the items, starting at index 1.
    LDX #$01

@LoopDrawItem:
    LDA Items, X

    ; If at the compass item slot, check this level's compass.
    CPX #$10
    BNE :+
    JSR HasCompass
    LDX #$10                    ; Set compass item slot $10 again.
:
    ; If at the map item slot, check this level's map.
    CPX #$11
    BNE :+
    JSR HasMap
    LDX #$11                    ; Set map item slot $11 again.
:
    ; If there are none of this item, then go advance the index and loop again.
    CMP #$00
    BEQ @NextLoopDrawItem

    ; If at the letter item slot, look at what's in the potion slot.
    ; If there is a potion, then go advance the index and loop again.
    ; That's because we already prepared sprites for the potion.
    CPX #$0F
    BNE :+
    LDA Potion
    BNE @NextLoopDrawItem
:
    TXA                         ; Save the item slot.
    PHA
    TAY                         ; Save item slot to Y register. Maybe this was in anticipation of calling E735?

    ; Look up and set the X coordinate of this item.
    LDA SubmenuItemXs, X
    STA $00                     ; [00] X

    ; If the item goes in the first selectable row (item slot < 5),
    ; then go set Y coordinate to $36 and draw.
    LDA #$36
    CPX #$05
    BCC @Draw

    ; If the item goes in the second selectable row (item slot < 9 or = $F),
    ; then go set Y coordinate to $46 and draw.
    LDA #$46
    CPX #$0F
    BEQ @Draw
    CPX #$09
    BCC @Draw

    ; If the item goes in the unselectable row at the top (item slot < $10),
    ; then go set Y coordinate to $1E and draw.
    LDA #$1E
    CPX #$10
    BCC @Draw

    ; The Y coordinate of the compass is $9E; and the map's is $76.
    ; The X coordinate for both is $2C.
    LDA #$2C
    STA $00
    LDA #$9E
    CPX #$11
    BCC @Draw
    LDA #$76

@Draw:
    STA $01                     ; [01] Y
    JSR DrawItemInInventoryWithX
    PLA                         ; Restore the item slot.
    TAX

@NextLoopDrawItem:
    ; Loop until item slot = $12.
    INX
    CPX #$12
    BCC @LoopDrawItem
    RTS

SubmenuCursorXs:
    .BYTE $80, $98, $B0, $B0, $C8, $80, $98, $B0
    .BYTE $C8

UpdateSubmenuSelection:
    ; Pseudo-item slot 0 is for boomerangs.
    ; If it's not the currently selected item slot, then skip this.
    LDX SelectedItemSlot
    BNE @DrawBreakoutItem

    ; Look for a magic boomerang, then a wooden boomerang.
    ; If you find one, then use its item slot.
    LDX #$1E
:
    LDA Items, X
    BNE @DrawBreakoutItem
    DEX
    CPX #$1C
    BNE :-
    BEQ @AfterBreakoutItem

@DrawBreakoutItem:
    ; Draw the item in the box for the currently selected item,
    ; if we have it.
    ;
    ; There's also the special case for the letter. If the selected item
    ; is the letter, but we have a potion, then skip this.
    ;
    ; If the letter is selected and there's no potion, then make
    ; the item value 1 in [04]. Except that, the call to 05:B81C
    ; below will overwrite [04].
    LDA Items, X
    BEQ @AfterBreakoutItem
    CPX #$0F
    BNE :+
    LDA Potion
    BNE @AfterBreakoutItem
    LDA #$01                    ; Special case for letter selected and no potion.
:
    STA $04                     ; [04] holds the item value.
    LDA #$36
    STA $01                     ; [01] Y
    LDA #$40
    STA $00                     ; [00] X
    JSR DrawItemInInventoryWithX

@AfterBreakoutItem:
    ; If the selected item slot is the letter's and we have potions,
    ; then select the potion item slot.
    LDY SelectedItemSlot
    CPY #$0F
    BNE @DrawCursor
    LDY #$07
    LDA Items, Y
    BEQ @DrawCursor
    ; Begin unverified code 17764
    STY SelectedItemSlot
    ; End unverified code

@DrawCursor:
    ; Look up the X coordinate for the selected item slot.
    ; Set it for the left cursor sprite.
    ; Then add 8 to set the right cursor sprite's X.
    LDA SubmenuCursorXs, Y
    STA Sprites+31
    CLC
    ADC #$08
    STA Sprites+35

    ; The Y coordinate is $36 for item slots < 5, else $46.
    LDA #$36
    CPY #$05
    BCC :+
    LDA #$46
:
    STA Sprites+28
    STA Sprites+32

    ; $1E is the cursor tile.
    LDA #$1E
    STA Sprites+29
    STA Sprites+33

    ; Flash the cursor.
    ; Use palette rows 5 and 6 for 8 frames each.
    LDA FrameCounter
    AND #$08
    LSR
    LSR
    LSR
    ADC #$01
    STA Sprites+30

    ; Flip the right sprite horizontally.
    ORA #$40
    STA Sprites+34

    ; Has the input changed? If input direction = [EF] item search
    ; direction in the previous frame, then return.
    LDA ObjInputDir
    CMP $EF
    BEQ L177F1_Exit
    TAX                         ; Copy input direction to X register.

    ; If input direction = 0, up, or down; then we won't change
    ; the change the selected item slot forward or backward.
    ;
    ; Instead, jump to this routine to make sure an occupied slot
    ; is selected, then return.
    BEQ FindAndSelectOccupiedItemSlot
    CPX #$04
    BCS FindAndSelectOccupiedItemSlot

    ; Cue the "selection changed" tune.
    LDX #$01
    STX Tune1Request

    ; Move the selection in the input direction.
    TAX                         ; Save the input direction.
    LDA SelectedItemSlot        ; Save selected item slot.
    PHA
    TXA                         ; Restore the input direction.
    JSR FindAndSelectOccupiedItemSlot

    ; If the new item slot = old item slot, go cancel the
    ; "selection changed" tune.
    PLA                         ; Pop selected item slot.
    CMP SelectedItemSlot
    BEQ @CancelTune

    ; Else the selection changed. The new slot should have an item.
    ; If it does not, then cancel the "selection changed" tune.
    LDY SelectedItemSlot
    LDA Items, Y
    BNE :+

@CancelTune:
    LSR Tune1Request
:
    RTS

; Params:
; A: direction to search: 0=none, 1=forward, 2=backward
; Y: starting item slot
;
; Returns:
; [EF]: direction to search
;
FindAndSelectOccupiedItemSlot:
    STA $EF
    LDX #$09                    ; Check 9 slots.

LoopItemSlot:
    JSR Cycle9InDirection
    CPY #$00
    BEQ CheckBoomerangs         ; If this is the boomerang pseudo-slot. Go check boomerangs.
    CPY #$03
    BEQ CheckNextItem           ; If it's the bow slot, then skip it. You can't select it.
    LDA Items, Y
    BNE FoundSlot               ; Found an item. Go see if the slot is OK.
    CPY #$07
    BEQ CheckLetter             ; This is the potion slot, but no potion. Go check the letter.

CheckNextItem:
    DEX
    BPL LoopItemSlot            ; If there are more slots, then check the next one.
    LDY #$00                    ; We found no items. Set SelectedItemSlot to zero.

FoundSlot:
    CPY #$02
    BNE SetSlotFound            ; If not the arrow, go set SelectedItemSlot.
    LDA Bow
    BEQ LoopItemSlot            ; If we don't have the bow, then keep looking for a slot.

SetSlotFound:
    STY SelectedItemSlot        ; Set SelectedItemSlot to the slot we found.

L177F1_Exit:
    RTS

CheckBoomerangs:
    ; Check the boomerangs.
    ;
    ; Start with the magical boomerang.
    LDY #$1E
:
    LDA Items, Y
    BNE :+                      ; We have one of the boomerangs. Go use the boomerang pseudo-slot 0.
    DEY
    CPY #$1C
    BNE :-
    LDY #$00                    ; There are no boomerangs. So continue searching where we left off.
    JMP CheckNextItem
:
    LDY #$00                    ; Go finish up with pseudo-slot 0 for boomerangs found.
    JMP FoundSlot

CheckLetter:
    ; Check the letter.
    LDY #$0F
    LDA Items, Y
    BNE :+                      ; If there's a letter, go see if there's a potion.
    LDY #$07
    BNE CheckNextItem           ; There's no letter, so continue searching where we left off.
:
    LDA Potion
    BEQ SetSlotFound            ; If there's no potion, then go set SelectedItemSlot to the letter slot.
    ; Begin unverified code 17818
    LDY #$07
    BNE SetSlotFound
    ; End unverified code

DrawItemInInventoryWithX:
    ; [$00]: X
    ; [$01]: Y
    ; X: item slot
    TXA
    TAY
    JMP DrawItemInInventory

; Params:
; [$EF]: direction to search: 0=none, 1=forward, 2=backward
; Y: value to cycle
;
; If A = 0, does nothing.
; If A = 1, Y := (Y + 1) mod 9
; If A = 2, Y := (Y - 1) mod 9
;
Cycle9InDirection:
    LDA $EF
    AND #$03
    BEQ @Exit
    INY
    LSR
    BCS :+
    DEY
    DEY
:
    CPY #$FF
    BNE :+
    LDY #$08
:
    CPY #$09
    BNE @Exit
    LDY #$00

@Exit:
    RTS
