; ============================================================================
; FILE:     bank7/status-bar-lift-item.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 974-1128
; SUBJECTS: (none)
; TOPIC:    Status bar items; Link lifting an item
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0025 DrawStatusBarPotion                      code
;   L0031 DrawStatusBarItemsAndEnsureItemSelected  code
;   L0053 DrawStatusBarItemB                       code
;   L0064 DrawStatusBarBoomerang                   code
;   L0076 FindItemOrDrawSword                      code
;   L0084 EnsureSelectedItem                       code
;   L0092 DrawStatusBarSword                       code
;   L0105 CheckMissingItem                         code
;   L0115 CheckLiftItem                            code
;   L0128 SetUpAndDrawLinkLiftingItem              code  Params:
;   L0139 DrawLinkLiftingItem                      code  Params:
;   L0163 L1E847_Exit                              code
;   L0166 EndLinkLiftingItem                       code
;   L0178 L1E859_Exit                              code
; ============================================================================

DrawStatusBarPotion:
    ; We have a potion to draw.
    LDX #$07                    ; Potion item slot.
    STX SelectedItemSlot
    BNE DrawStatusBarItemB      ; Go draw the potion.

DrawStatusBarItemsAndEnsureItemSelected:
    ; Draws the selected item and the sword in the status bar.
    ; Before drawing the item, it ensures that SelectedItemSlot
    ; refers to an occupied, equippable item slot.
    ;
    ; This accounts for the fallback behavior between potions
    ; and letters and the two boomerangs.
    ;
    ; Start at slot zero. Normally, it would be used to
    ; check swords. When it's used as the selected item index;
    ; it's a pseudo-slot used for checking boomerangs.
    LDX SelectedItemSlot
    BEQ DrawStatusBarBoomerang  ; If pseudo-slot zero is chosen, then go check boomerangs.
    LDA InvItems, X
    BEQ CheckMissingItem        ; If there's no item in the current item slot, then go check for potions and letters.
    CPX #$0F                    ; We have an item. See if it's in the letter slot.
    BNE DrawStatusBarItemB      ; If it's not the letter slot, then go draw it.
    LDY InvPotion                  ; It's in the letter slot. See if we have a potion.
    BNE DrawStatusBarPotion     ; If we have a potion, then go set SelectedItemSlot to its slot, and draw it.
    LSR                         ; Otherwise, we only have a letter.
    ORA #$01                    ; Use only 1 as the palette attribute.

DrawStatusBarItemB:
    STA $04                     ; Set palette attribute to inventory value.
    LDA #$1F                    ; Set X and Y coordinates for "B" item in status bar: ($7C, $1F).
    STA $01
    LDA #$7C
    STA $00
    LDA #$05
    JSR SwitchBank
    JSR DrawItemInInventoryWithX
    JMP DrawStatusBarSword      ; Go handle the sword.

DrawStatusBarBoomerang:
    LDX #$1E                    ; Check the magic boomerang first.

@LoopBoomerang:
    LDA InvItems, X
    BNE DrawStatusBarItemB      ; If we have one of the boomerangs, then go draw it.
    DEX
    CPX #$1C
    BNE @LoopBoomerang          ; Go check the next boomerang.
    LDX #$00
    JMP EnsureSelectedItem      ; No boomerangs. Go search for an occupied slot starting at 0.

FindItemOrDrawSword:
    LDA InvItems, X

    ; If there's an item in this slot, then SelectedItemSlot was already
    ; set. Go handle the sword instead of drawing the item.
    ; Next frame, we'll draw this item.
    BNE DrawStatusBarSword

EnsureSelectedItem:
    TXA                         ; Not found. Search for an occupied slot.
    TAY
    LDA #$05
    JSR SwitchBank
    LDA #$02                    ; Go backwards from current slot.
    JSR FindAndSelectOccupiedItemSlot

DrawStatusBarSword:
    ; Handle the sword.
    LDX #$00
    LDA InvItems, X
    BEQ L1E847_Exit             ; If there's no sword, then return.
    LDA #$1F                    ; Set X and Y coordinates for "A" item in status bar: ($94, $1F).
    STA $01
    LDA #$94
    STA $00
    LDA #$05
    JSR SwitchBank
    JMP DrawItemInInventoryWithX

CheckMissingItem:
    ; No item in slot.
    CPX #$07
    BNE FindItemOrDrawSword     ; If the current item slot isn't for potions, then go handle the slot almost as usual.
    LDA InvLetter               ; This slot is for potions but don't have one. Check the letter.
    BEQ EnsureSelectedItem      ; If we don't have a letter, then go look for an occupied slot.
    LDX #$0F                    ; We have a letter. Set SelectedItemSlot to its slot.
    STX SelectedItemSlot
    BNE FindItemOrDrawSword     ; Go handle the sword (indirectly).

CheckLiftItem:
    LDA ItemTypeToLift
    BEQ L1E859_Exit             ; If there's no item to lift, then return.
    DEC ItemLiftTimer
    BEQ EndLinkLiftingItem      ; If the item lift timer has expired, go return to normal.

    ; Set player state to halted.
    LDA #$40
    STA ObjState

; Params:
; [0505]: item type
;
SetUpAndDrawLinkLiftingItem:
    LDA ObjX                    ; Set item X the same as Link's.
    STA ObjX+19
    LDA ObjY                    ; Set item's Y $10 pixels above Link's.
    SEC
    SBC #$10
    STA ObjY+19

; Params:
; [0505]: item type
;
DrawLinkLiftingItem:
    LDX #$00                    ; We're dealing with Link object.
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorAttributes    ; Set 0 sprite attributes. 0 was returned above.
    STA $0C                     ; Set frame 0.
    LDA #$48                    ; Point to Link's sprites.
    STA LeftSpriteOffset
    LDA #$4C
    STA RightSpriteOffset
    LDY #$21
    JSR Anim_WriteSpecificItemSprites
    INC LeftAlignHalfWidthObj   ; Temporarily set left alignment.
    LDA ItemTypeToLift
    LDX #$13                    ; Now deal with item object.
    JSR AnimateItemObject
    DEC LeftAlignHalfWidthObj
    LDA IsDrawnItemNarrow
    BEQ L1E847_Exit             ; If this item is half-width,

    ; Then lift with one hand.
    ; Change the right side to a tile without the arm raised.
    LDA #$08
    STA Sprites+77

L1E847_Exit:
    RTS

EndLinkLiftingItem:
    ; Make Link idle, and reset the item type.
    LDA #$00
    STA ObjState
    STA ItemTypeToLift

    ; If in UW, play the level's song again.
    LDY CurLevel
    BEQ L1E859_Exit
    LDA LevelSongIds, Y
    STA SongRequest

L1E859_Exit:
    RTS
