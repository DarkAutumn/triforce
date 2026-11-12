; ============================================================================
; FILE:     bank7/room-item-draw.asm     BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 805-972
; SUBJECTS: (none)
; TOPIC:    Room item on a monster, item drawing
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 AnimateRoomItemOnMonster  code
;   L0026 PopAndExit                code
;   L0032 MoveAndDrawRoomItem       code
;   L0058 AnimateRoomItemObject     code
;   L0065 AnimateItemObject         code  Params:
;   L0111 DrawItemInInventory       code  Params:
;   L0122 DrawItemBySlot            code  Params:
; ============================================================================

AnimateRoomItemOnMonster:
    ; Put room item object where the first monster is.
    LDA ObjX+1
    STA ObjX+19
    LDA ObjY+1
    STA ObjY+19
    JMP AnimateRoomItemObject   ; Go draw the item at this position.

PopAndExit:
    ; Pop and return.
    PLA
:
    RTS

MoveAndDrawRoomItem:
    ; If the item was taken, or it wasn't active, then return.
    JSR GetRoomFlagUWItemState
    BNE :-
    LDA ObjState+19             ; Room item object state
    BMI :-

    ; If the item type is "none" ($3F), then return.
    LDA RoomItemId
    CMP #$3F
    BEQ :-

    ; If there's a room item, and object 1 is a like-like, stalfos, or gibdo;
    ; then move the item along with the monster.
    LDX #$01
    LDA ObjType+1
    CMP #$17                    ; Like-Like
    BEQ AnimateRoomItemOnMonster
    CMP #$2A                    ; Stalfos
    BEQ AnimateRoomItemOnMonster
    CMP #$30                    ; Gibdo
    BEQ AnimateRoomItemOnMonster

    ; A monster is not carrying the item. So draw the item as usual.
    LDX #$13                    ; Room item is in object slot $13.

AnimateRoomItemObject:
    LDA RoomItemId              ; Pass item type to AnimateItemObject.

; Params:
; A: item type
; X: object index
;
AnimateItemObject:
    PHA                         ; Save the item ID.

    ; If the lifetime timer of the item >= $F0 and even, then
    ; return without drawing. This makes it flash at first.
    ;
    ; [03A8][X] is used to count down the life of the item.
    LDA Item_ObjItemLifetime, X
    CMP #$F0
    BCC :+
    LSR
    BCC PopAndExit              ; Pop and return, if timer >= $F0 and even.
:
    ; Copy room item object position to sprite descriptor.
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Look up the item description for this item ID.
    ; Store the item value part of it in [04].
    PLA                         ; Restore the item ID.
    TAX                         ; X now has item ID.
    LDA ItemIdToDescriptor, X
    CMP #$30
    BEQ @SetItemValueFF         ; If item descriptor is $30, go set item value to $FF.
    AND #$0F                    ; only take the item value part of the descriptor.

@SetItemValue:
    STA $04                     ; [04] holds the item value.

    ; Get item slot for the item ID, and draw the item.
    LDA ItemIdToSlot, X
    TAX
    TAY                         ; Now X and Y have the item slot.
    JMP DrawItemBySlot

@SetItemValueFF:
    ; Begin unverified code 1E731
    LDA #$FF                    ; Use item value $FF.
    BNE @SetItemValue
    ; End unverified code

; Params:
; X: item slot
; Y: item slot
; [00]: X
; [01]: Y
;
DrawItemInInventory:
    LDA Items, X
    STA $04                     ; The item inventory value can also serve as a palette row attribute.

; Params:
; X: item slot
; Y: item slot
; [00]: X
; [01]: Y
; [04]: item value / sprite palette row attribute
;
DrawItemBySlot:
    LDA ItemSlotToPaletteOffsetsOrValues, X
    CPX #$16                    ; (A) If the item flashes (slots $16, $19, $1A, $1B),
    BEQ @Flash
    CPX #$1A
    BEQ @Flash
    CPX #$1B
    BEQ @Flash
    CPX #$19
    BNE @VariesByColor

@Flash:
    LDA FrameCounter            ;  then every 4 frames,
    AND #$08
    LSR
    LSR
    LSR
    ADC #$01                    ; Switch between palettes 5 and 6.

@VariesByColor:
    CPX #$00                    ; (B) If the item varies by color (slots 0, 2, 4, 7, $B),
    BEQ @AddItemAndTableValue
    CPX #$04
    BEQ @AddItemAndTableValue
    CPX #$02
    BEQ @AddItemAndTableValue
    CPX #$07
    BEQ @AddItemAndTableValue
    CPX #$0B
    BEQ @AddItemAndTableValue   ; Go add the palette offset we looked up and sprite palette row attribute.

@WriteSprites:
    ; We got here in one of three ways:
    ;
    ; A. The item flashes. Attribute value 1 or 2 was chosen
    ;    explicitly.
    ; B. The item color varies. ItemSlotToPaletteOffsetsOrValues
    ;    held an offset that was added to base palette attribute
    ;    passed in [$04]. This corresponds to class 2 items.
    ; C. Item doesn't change color. ItemSlotToPaletteOffsetsOrValues
    ;    held absolute attributes.
    ;
    ; At this point, A holds the final sprite attributes.
    LDX #$00
    STX $0C                     ; [0C] refers to frame 0.
    LDX #$0F

    ; Write sprites.
    ; A holds the calculated sprite attributes.
    ; Y holds the item slot.
    JMP Anim_WriteStaticItemSpritesWithAttributes

@AddItemAndTableValue:
    ; Calculate sprite attributes by adding item value / sprite palette
    ; row attribute and the element from ItemSlotToPaletteOffsetsOrValues,
    ; which here represents an offset from the palette row.
    CLC
    ADC $04                     ; Add palette offset to item value / sprite attribute we started with.
    CPX #$00                    ; If item slot is 0 and palette is 6,
    BNE @WriteSprites
    CMP #$02
    BNE @WriteSprites
    LDY #$20                    ; then make the item slot $20, a special slot to differentiate the image of the master sword.
    JMP @WriteSprites           ; Go write the sprites.
