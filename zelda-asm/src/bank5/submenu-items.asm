; ============================================================================
; FILE:     bank5/submenu-items.asm      BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7844-8197
; SUBJECTS: Inv, Item, Submenu
; TOPIC:    Item submenu: draws the inventory item sprites, the selected item and the flashing cursor;
;           moves the B-button selection left/right; finds an occupied selectable item slot (also used
;           by the status bar every frame).
; SEE ALSO: bank5/submenu-states.asm (Submenu_UpdateActive calls Submenu_DrawItems and
;           Submenu_UpdateSelection each frame the submenu is open); bank7/status-bar-lift-item.asm
;           (StatusBar_DrawItemsAndEnsureSelection: the other caller of Inv_FindAndSelectOccupiedSlot
;           and Item_DrawInInventoryWithX); bank7/room-item-draw.asm (Item_DrawInInventory);
;           bank5/submenu-map.asm (Inv_HasCompass, Inv_HasMap); bank5/wield-items.asm (Link_WieldItem_JumpTable,
;           indexed by SelectedItemSlot).
; KEY FACTS:
;   - Selectable item slots are 0-8 in a ring: 0 boomerangs (pseudo-slot: InvItems+0 is the sword), 1
;     bombs, 2 arrow, 3 bow, 4 candle, 5 flute, 6 food, 7 potion, 8 rod. Inv_StepItemSlotInSearchDir
;   - Slot 3 (bow) can never be selected; slot 2 (arrow) only with the bow. Inv_FindAndSelectOccupiedSlot_Loop, Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow
;   - Slot 0 is selectable when either boomerang (item slots $1D wooden, $1E magic) is held.
;     Inv_FindAndSelectOccupiedSlot_CheckBoomerangs
;   - The letter (item slot $0F) stands in for an empty potion slot: with a letter and no potion
;     SelectedItemSlot becomes $0F; once a potion is held the submenu changes $0F back to 7.
;     Inv_FindAndSelectOccupiedSlot_CheckLetter, Submenu_UpdateSelection
;   - Left = search backward, right = search forward, one search per change of the held direction
;     bits (no auto-repeat): the input is compared with the previous search direction, SubmenuSearchDir.
;     Submenu_UpdateSelection
;   - A search tries at most 10 slots; if none is occupied SelectedItemSlot becomes 0.
;     Inv_FindAndSelectOccupiedSlot, Inv_FindAndSelectOccupiedSlot_Next
;   - With no direction held, or only up/down, the search does not step: it keeps the current slot if
;     it is occupied, else selects 0. A diagonal (up or down held with left or right) does step, without
;     the tune, and counts as a new direction. Inv_StepItemSlotInSearchDir, Submenu_UpdateSelection
;   - The cursor is sprites 7 and 8, tile $1E, alternating sprite palettes 1 and 2 every 8 frames.
;     Submenu_UpdateSelection
;   - Compass and map sprites are shown only inside a level, for that level's compass/map.
;     Submenu_DrawItems
; CONTENTS: (generated; L = line in this file)
;   L0053 Submenu_ItemXs                                    data  TABLE: sprite X of each item in the submenu; index = item slot $00-$0F; value = X (pixels).
;   L0068 Submenu_DrawItems                                 code  Draws the sprites of every held inventory item in the submenu (item slots 1-$11 and a boomerang).
;   L0164 Submenu_CursorXs                                  data  TABLE: sprite X of the selection cursor's left half; index = selectable item slot 0-8; value = X
;   L0183 Submenu_UpdateSelection                           code  Draws the selected item and the cursor in the submenu, and moves the selection on left/right input.
;   L0321 Inv_FindAndSelectOccupiedSlot                     code  Sets SelectedItemSlot to the first occupied selectable item slot, searching from Y in direction A.
;   L0326 Inv_FindAndSelectOccupiedSlot_Loop                code  Loop of Inv_FindAndSelectOccupiedSlot: step Y to the next slot and test it.
;   L0338 Inv_FindAndSelectOccupiedSlot_Next                code  Loop tail of Inv_FindAndSelectOccupiedSlot: count the try; falls through to select slot 0 when out.
;   L0344 Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow  code  An occupied slot Y was found: an arrow (slot 2) is rejected unless the bow is held.
;   L0351 Inv_FindAndSelectOccupiedSlot_Store               code  Stores Y as the selected item slot.
;   L0355 Inv_FindAndSelectOccupiedSlot_Return              code  Shared return of Submenu_UpdateSelection and Inv_FindAndSelectOccupiedSlot.
;   L0359 Inv_FindAndSelectOccupiedSlot_CheckBoomerangs     code  Slot 0 of the search: counts as occupied if the magic ($1E) or wooden ($1D) boomerang is held.
;   L0374 Inv_FindAndSelectOccupiedSlot_CheckLetter         code  Potion slot (7) is empty in the search: select the letter slot ($F) if the letter is held.
;   L0392 Item_DrawInInventoryWithX                         code  Draws the item of item slot X at ([00], [01]) (copies X to Y, then Item_DrawInInventory).
;   L0404 Inv_StepItemSlotInSearchDir                       code  Steps the item slot in Y one place around the ring 0-8 in the direction held in SubmenuSearchDir.
; ============================================================================

; TABLE: sprite X of each item in the submenu; index = item slot $00-$0F; value = X (pixels).
Submenu_ItemXs:
    ; Slots 0-7: boomerang (pseudo-slot 0), bombs, arrow, bow, candle, flute, food, potion.
    .BYTE $80, $98, $AC, $B4, $C8, $80, $98, $B0
    ; Slots 8-$F: rod, raft, book, ring, ladder, magic key, bracelet, letter (drawn at the potion's X).
    .BYTE $C8, $80, $94, $A0, $B0, $C0, $CC, $B0

; ----------------------------------------------------------------------------
; Draws the sprites of every held inventory item in the submenu (item slots 1-$11 and a boomerang).
; OUT:  sprites written through Item_DrawInInventory; A, X, Y, [00], [01], [04] changed.
; NOTE: Rows by sprite Y: $36 = slots 0-4 (boomerang, bombs, arrow, bow, candle); $46 = slots 5-8 and
;       $F (flute, food, potion, rod; the letter at the potion's spot when no potion is held); $1E =
;       slots 9-$E (raft, book, ring, ladder, magic key, bracelet; not selectable). The compass is at
;       ($2C, $9E) and the map at ($2C, $76), shown only for the current level. The arrow is drawn
;       whether or not the bow is held.
; ----------------------------------------------------------------------------
Submenu_DrawItems:
    ; Look for the magic boomerang (item slot $1E), then the wooden one ($1D). Draw the first found.
    LDX #$1E
:
    LDA InvItems, X
    BNE @FoundBoomerang
    DEX
    CPX #$1C
    BNE :-
    BEQ @DrawOtherItems         ; Neither boomerang: nothing to draw in the first spot.

@FoundBoomerang:
    LDA #$36
    STA $01                     ; [01] Y
    LDA #$80
    STA $00                     ; [00] X
    TXA
    TAY                         ; Y = item slot, as Item_DrawInInventory expects.
    JSR Item_DrawInInventory

@DrawOtherItems:
    ; Item slots 1 to $11 (slot 0, the sword, is not drawn here).
    LDX #$01

@LoopDrawItem:
    LDA InvItems, X

    ; Slot $10 (compass): use this level's compass bit instead (0 in the overworld).
    CPX #$10
    BNE :+
    JSR Inv_HasCompass
    LDX #$10                    ; Inv_HasCompass changed X; restore the slot.
:
    ; Slot $11 (map): use this level's map bit instead (0 in the overworld).
    CPX #$11
    BNE :+
    JSR Inv_HasMap
    LDX #$11                    ; Inv_HasMap changed X; restore the slot.
:
    ; Item not held: next slot.
    CMP #$00
    BEQ @NextLoopDrawItem

    ; Slot $F (letter): skip it when a potion is held; the potion was drawn at the same spot (slot 7).
    CPX #$0F
    BNE :+
    LDA InvPotion
    BNE @NextLoopDrawItem
:
    TXA                         ; Save the item slot.
    PHA
    TAY                         ; Not needed: Item_DrawInInventoryWithX copies X to Y itself.

    LDA Submenu_ItemXs, X
    STA $00                     ; [00] X

    ; First selectable row (item slot < 5): Y = $36.
    LDA #$36
    CPX #$05
    BCC @Draw

    ; Second selectable row (item slot 5-8, or $F the letter): Y = $46.
    LDA #$46
    CPX #$0F
    BEQ @Draw
    CPX #$09
    BCC @Draw

    ; Top row, not selectable (item slot 9-$E): Y = $1E.
    LDA #$1E
    CPX #$10
    BCC @Draw

    ; Compass (slot $10) at ($2C, $9E); map (slot $11) at ($2C, $76).
    LDA #$2C
    STA $00
    LDA #$9E
    CPX #$11
    BCC @Draw
    LDA #$76

@Draw:
    STA $01                     ; [01] Y
    JSR Item_DrawInInventoryWithX
    PLA                         ; Restore the item slot.
    TAX

@NextLoopDrawItem:
    ; Loop until item slot = $12.
    INX
    CPX #$12
    BCC @LoopDrawItem
    RTS

; TABLE: sprite X of the selection cursor's left half; index = selectable item slot 0-8; value = X
;        (pixels). Slots 2 (arrow) and 3 (bow) share $B0.
Submenu_CursorXs:
    .BYTE $80, $98, $B0, $B0, $C8, $80, $98, $B0
    .BYTE $C8

; ----------------------------------------------------------------------------
; Draws the selected item and the cursor in the submenu, and moves the selection on left/right input.
; OUT:  SelectedItemSlot; SubmenuSearchDir = search direction used; sprites 7 and 8 (the cursor);
;       Tune1Request = 1 when left or right alone moved the selection to a slot whose InvItems byte
;       is not 0.
; NOTE: A search runs once per change of ObjInputDir (in the submenu: the four direction bits held
;       on controller 1) against SubmenuSearchDir; holding a direction does not repeat. Right (1) searches
;       forward, left (2) backward through slots 0-8. A change to 0, up (8) or down (4) only
;       re-validates the current slot: it stays if occupied, else slot 0 is selected.
; NOTE: A diagonal (value >= 4 with bit 0 or 1 set) takes the no-tune path but still steps, because
;       Inv_StepItemSlotInSearchDir looks only at bits 0-1. So adding or releasing up/down while left or right
;       is held moves the selection one more time.
; NOTE: Moving onto pseudo-slot 0 (boomerang) plays the tune only if Link has a sword: the test reads
;       InvItems+0.
; ----------------------------------------------------------------------------
Submenu_UpdateSelection:
    ; Find the item slot to draw in the selected-item box: the selected slot, or for
    ; pseudo-slot 0 the boomerang held (magic $1E first, then wooden $1D).
    LDX SelectedItemSlot
    BNE @DrawSelectedItem

    LDX #$1E
:
    LDA InvItems, X
    BNE @DrawSelectedItem
    DEX
    CPX #$1C
    BNE :-
    BEQ @AfterSelectedItem      ; No boomerang: nothing to draw in the box.

@DrawSelectedItem:
    ; Draw the selected item in the box at ($40, $36), if it is held.
    ; If the letter slot ($F) is selected but a potion is held, draw nothing this frame: the
    ; selection moves to the potion slot below.
    LDA InvItems, X
    BEQ @AfterSelectedItem
    CPX #$0F
    BNE :+
    LDA InvPotion
    BNE @AfterSelectedItem
    LDA #$01                    ; Letter selected and no potion.
:
    STA $04                     ; No effect: Item_DrawInInventory reloads [04] from InvItems, X.
    LDA #$36
    STA $01                     ; [01] Y
    LDA #$40
    STA $00                     ; [00] X
    JSR Item_DrawInInventoryWithX

@AfterSelectedItem:
    ; If the letter slot ($F) is selected, the cursor goes at the potion slot (Y = 7); and if a
    ; potion is held, the potion slot becomes the selection.
    LDY SelectedItemSlot
    CPY #$0F
    BNE @DrawCursor
    LDY #$07
    LDA InvItems, Y             ; InvPotion
    BEQ @DrawCursor             ; No potion: SelectedItemSlot stays $F, but Y = 7 from here on.
    STY SelectedItemSlot

@DrawCursor:
    ; Y = selected slot 0-8. The cursor is sprite 7 (left half, Sprites+28..31) and sprite 8
    ; (right half, Sprites+32..35), 8 pixels apart.
    LDA Submenu_CursorXs, Y
    STA Sprites+31
    CLC
    ADC #$08
    STA Sprites+35

    ; Sprite Y is $36 for item slots < 5, else $46.
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

    ; Flash the cursor: sprite attribute 1 or 2 (sprite palette 1 or 2), 8 frames each, from bit 3
    ; of FrameCounter. The LSRs leave carry clear.
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

    ; Act only when the input direction differs from SubmenuSearchDir, the direction stored by the
    ; last call of Inv_FindAndSelectOccupiedSlot.
    LDA ObjInputDir
    CMP SubmenuSearchDir
    BEQ Inv_FindAndSelectOccupiedSlot_Return
    TAX                         ; Copy input direction to X register.

    ; Input direction 0, or 4 and above (down, up, or a diagonal): no tune. A pure 0/4/8 does not
    ; step and only makes sure the current slot (Y) is occupied; a diagonal still steps left or
    ; right. The call stores the direction in SubmenuSearchDir and returns to this routine's caller.
    BEQ Inv_FindAndSelectOccupiedSlot
    CPX #$04
    BCS Inv_FindAndSelectOccupiedSlot

    ; Right (1) or left (2): request the "selection changed" tune.
    LDX #$01
    STX Tune1Request

    ; Move the selection in the input direction, starting from slot Y.
    TAX                         ; Save the input direction.
    LDA SelectedItemSlot        ; Save selected item slot.
    PHA
    TXA                         ; Restore the input direction.
    JSR Inv_FindAndSelectOccupiedSlot

    ; If the selection did not change, cancel the tune.
    PLA                         ; Pop selected item slot.
    CMP SelectedItemSlot
    BEQ @CancelTune

    ; The selection changed. If the new slot is empty (slot 0 chosen because nothing was found,
    ; or pseudo-slot 0 with no sword in InvItems+0), cancel the tune too.
    LDY SelectedItemSlot
    LDA InvItems, Y
    BNE :+

@CancelTune:
    LSR Tune1Request            ; 1 -> 0
:
    RTS

; ----------------------------------------------------------------------------
; Sets SelectedItemSlot to the first occupied selectable item slot, searching from Y in direction A.
; IN:   A = search direction: bit 0 set = forward (+1), else bit 1 set = backward (-1), low two bits
;       clear = stay on Y; Y = item slot to start from (the first slot tested is the next one in that
;       direction, or Y itself when not stepping).
; OUT:  SelectedItemSlot = slot found: 0-8, or $F (letter) when the potion slot is empty and the
;       letter is held, or 0 when nothing was found; SubmenuSearchDir = A as passed in.
; NOTE: Tests up to 10 slots (X = 9 down to 0). Slot 0 counts when a boomerang is held; slot 3 (bow)
;       never; slot 2 (arrow) only with InvBow. The status bar code calls this with A = 2 whenever the
;       selected slot is empty, so using up the last of an item moves the selection backward.
; NOTE: An arrow without the bow goes back to the loop without counting a try. With a stepping
;       direction the search still ends; called without a step on slot 2 with arrows and no bow it
;       would never return (the callers reach slot 2 without a step only when it is the selected
;       slot, which this routine never selects without the bow).
; ----------------------------------------------------------------------------
Inv_FindAndSelectOccupiedSlot:
    STA SubmenuSearchDir
    LDX #$09                    ; Loop counter: 10 tries.

; Loop of Inv_FindAndSelectOccupiedSlot: step Y to the next slot and test it.
Inv_FindAndSelectOccupiedSlot_Loop:
    JSR Inv_StepItemSlotInSearchDir
    CPY #$00
    BEQ Inv_FindAndSelectOccupiedSlot_CheckBoomerangs         ; Pseudo-slot 0: occupied if a boomerang is held.
    CPY #$03
    BEQ Inv_FindAndSelectOccupiedSlot_Next           ; Slot 3 (bow) cannot be selected.
    LDA InvItems, Y
    BNE Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow               ; Occupied; go apply the arrow rule.
    CPY #$07
    BEQ Inv_FindAndSelectOccupiedSlot_CheckLetter             ; Potion slot empty: the letter may stand in for it.

; Loop tail of Inv_FindAndSelectOccupiedSlot: count the try; falls through to select slot 0 when out.
Inv_FindAndSelectOccupiedSlot_Next:
    DEX
    BPL Inv_FindAndSelectOccupiedSlot_Loop            ; More tries left.
    LDY #$00                    ; Nothing found: select slot 0.

; An occupied slot Y was found: an arrow (slot 2) is rejected unless the bow is held.
Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow:
    CPY #$02
    BNE Inv_FindAndSelectOccupiedSlot_Store            ; Not the arrow slot: select it.
    LDA InvBow
    BEQ Inv_FindAndSelectOccupiedSlot_Loop            ; Arrow but no bow: keep searching (this try is not counted).

; Stores Y as the selected item slot.
Inv_FindAndSelectOccupiedSlot_Store:
    STY SelectedItemSlot

; Shared return of Submenu_UpdateSelection and Inv_FindAndSelectOccupiedSlot.
Inv_FindAndSelectOccupiedSlot_Return:
    RTS

; Slot 0 of the search: counts as occupied if the magic ($1E) or wooden ($1D) boomerang is held.
Inv_FindAndSelectOccupiedSlot_CheckBoomerangs:
    LDY #$1E
:
    LDA InvItems, Y
    BNE :+                      ; A boomerang is held: select pseudo-slot 0.
    DEY
    CPY #$1C
    BNE :-
    LDY #$00                    ; No boomerang: continue the search from slot 0.
    JMP Inv_FindAndSelectOccupiedSlot_Next
:
    LDY #$00
    JMP Inv_FindAndSelectOccupiedSlot_CheckArrowNeedsBow

; Potion slot (7) is empty in the search: select the letter slot ($F) if the letter is held.
Inv_FindAndSelectOccupiedSlot_CheckLetter:
    LDY #$0F
    LDA InvItems, Y             ; InvLetter
    BNE :+                      ; Letter held.
    LDY #$07
    BNE Inv_FindAndSelectOccupiedSlot_Next           ; No letter: continue the search from slot 7. Always taken.
:
    LDA InvPotion
    BEQ Inv_FindAndSelectOccupiedSlot_Store            ; No potion: select the letter slot ($F).
    ; Not reached from Inv_FindAndSelectOccupiedSlot_Loop, which comes here only when InvPotion is 0.
    LDY #$07
    BNE Inv_FindAndSelectOccupiedSlot_Store

; ----------------------------------------------------------------------------
; Draws the item of item slot X at ([00], [01]) (copies X to Y, then Item_DrawInInventory).
; IN:   X = item slot; [00] = sprite X; [01] = sprite Y
; NOTE: Item_DrawInInventory sets [04] from InvItems, X, so a value the caller put in [04] is lost.
; ----------------------------------------------------------------------------
Item_DrawInInventoryWithX:
    TXA
    TAY
    JMP Item_DrawInInventory

; ----------------------------------------------------------------------------
; Steps the item slot in Y one place around the ring 0-8 in the direction held in SubmenuSearchDir.
; IN:   SubmenuSearchDir: bit 0 set = forward, else bit 1 set = backward; low two bits clear = no step.
;       Y = item slot
; OUT:  Y = (Y + 1) mod 9 forward, (Y - 1) mod 9 backward, or unchanged; A changed.
; NOTE: Only $FF and 9 wrap, so the result is in 0-8 only if Y came in as 0-8.
; ----------------------------------------------------------------------------
Inv_StepItemSlotInSearchDir:
    LDA SubmenuSearchDir
    AND #$03
    BEQ @Exit                   ; Neither right nor left: no step.
    INY
    LSR
    BCS :+                      ; Bit 0 (right): forward, keep Y + 1.
    DEY                         ; Else (left): Y - 1.
    DEY
:
    CPY #$FF
    BNE :+
    LDY #$08                    ; Below 0 wraps to 8.
:
    CPY #$09
    BNE @Exit
    LDY #$00                    ; Above 8 wraps to 0.

@Exit:
    RTS
