; ============================================================================
; FILE:     bank7/room-item-draw.asm     BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 805-972
; SUBJECTS: Item
; TOPIC:    Drawing items: the room item (object slot $13), including the one carried by the monster in
;           slot 1; any item object by item id; an inventory item by item slot; palette and flashing.
; SEE ALSO: bank1/take-item.asm (Item_IdToSlot, Item_IdToDescriptor, Item_SlotToPaletteOffsetsOrValues,
;           RoomFlags_GetItemTaken, Item_TryTakeRoomItem); bank1/item-sprite-draw.asm (the sprite writers and
;           tile tables); bank7/status-bar-lift-item.asm (status bar and lifted item, callers);
;           bank4/drop-items.asm (dropped items call Item_DrawObject); bank7/mode5-play-update.asm
;           (calls Item_MoveAndDrawRoomItem every frame).
; KEY FACTS:
;   - The room item is object slot $13; its id is RoomItemId ($3F = none). It is not drawn when the room's
;     item-taken flag ($10) is set or ObjState+$13 is negative (Item_MoveAndDrawRoomItem).
;   - If the object in slot 1 is a Like-Like ($17), Stalfos ($2A) or Gibdo ($30), the room item's
;     position is overwritten with that monster's every frame and the item is drawn on it
;     (Item_MoveAndDrawRoomItem, Item_DrawRoomItemOnMonster). Only slot 1 is tested, and only the type.
;   - While carried, the "flash" test reads Item_ObjItemLifetime of slot 1, which for a monster is its
;     position fraction: the item is not drawn on frames where that byte is >= $F0 and even
;     (Item_DrawRoomItemOnMonster, Item_DrawObject).
;   - An item object whose lifetime byte is >= $F0 is drawn only on odd values, so it flashes
;     (Item_DrawObject).
;   - Rupee, heart, Triforce piece and Triforce of Power (item slots $16, $19, $1A, $1B) alternate
;     between sprite palettes 1 and 2 every 8 frames (Item_DrawBySlot@Flash).
;   - Sword, arrow, candle, potion and ring (item slots 0, 2, 4, 7, $B) take their palette from the grade:
;     sword and arrow grade - 1, the others the grade itself (Item_DrawBySlot@AddItemAndTableValue).
;   - The magic sword (slot 0, palette 2) is drawn with the image of pseudo item slot $20
;     (Item_DrawBySlot@AddItemAndTableValue).
; CONTENTS: (generated; L = line in this file)
;   L0046 Item_DrawRoomItemOnMonster    code  Moves the room item (object slot $13) onto the monster in object slot 1 and draws it there.
;   L0054 Item_DrawObject_PopAndReturn  code  Exit of Item_DrawObject when the item is not drawn this frame: drops the saved item id, returns.
;   L0065 Item_MoveAndDrawRoomItem      code  Draws the room item if the room has one that is still there, on the slot 1 monster if it carries it.
;   L0088 Item_DrawRoomItemAtSlot       code  Draws the room item (RoomItemId) at the position of object slot X; falls into Item_DrawObject.
;   L0101 Item_DrawObject               code  Draws item id A at the position of object slot X, flashing while its lifetime byte is >= $F0.
;   L0140 Item_DrawInInventory          code  Draws the item of an item slot as Link owns it: palette base = the inventory value of the slot.
;   L0155 Item_DrawBySlot               code  Picks the sprite palette for an item slot and writes its sprites.
; ============================================================================

; ----------------------------------------------------------------------------
; Moves the room item (object slot $13) onto the monster in object slot 1 and draws it there.
; IN:   X = 1 (set by Item_MoveAndDrawRoomItem).
; NOTE: Enters Item_DrawRoomItemAtSlot with X = 1, so the position and the lifetime byte used for
;       drawing are the monster's (see Item_DrawObject). ObjX+$13/ObjY+$13 keep the monster's last
;       position, which is where the item lies once slot 1 holds another type.
; ----------------------------------------------------------------------------
Item_DrawRoomItemOnMonster:
    LDA ObjX+1
    STA ObjX+19
    LDA ObjY+1
    STA ObjY+19
    JMP Item_DrawRoomItemAtSlot

; Exit of Item_DrawObject when the item is not drawn this frame: drops the saved item id, returns.
Item_DrawObject_PopAndReturn:
    PLA
:
    RTS

; ----------------------------------------------------------------------------
; Draws the room item if the room has one that is still there, on the slot 1 monster if it carries it.
; NOTE: Returns without drawing when the room flag "item taken" ($10) is set, when ObjState+$13 has bit 7
;       set (item not active), or when RoomItemId = $3F (none).
; NOTE: Carriers are tested only by the type in object slot 1: Like-Like $17, Stalfos $2A, Gibdo $30.
; ----------------------------------------------------------------------------
Item_MoveAndDrawRoomItem:
    JSR RoomFlags_GetItemTaken  ; A = room flags AND $10: nonzero = the item was taken.
    BNE :-
    LDA ObjState+19             ; Room item object state: negative = not active.
    BMI :-
    LDA RoomItemId
    CMP #$3F                    ; $3F = no item.
    BEQ :-

    ; If object slot 1 holds a Like-Like, Stalfos or Gibdo, the item moves with it.
    LDX #$01
    LDA ObjType+1
    CMP #$17                    ; Like-Like
    BEQ Item_DrawRoomItemOnMonster
    CMP #$2A                    ; Stalfos
    BEQ Item_DrawRoomItemOnMonster
    CMP #$30                    ; Gibdo
    BEQ Item_DrawRoomItemOnMonster

    ; Not carried: draw it at its own position.
    LDX #$13                    ; Room item object slot.

; Draws the room item (RoomItemId) at the position of object slot X; falls into Item_DrawObject.
Item_DrawRoomItemAtSlot:
    LDA RoomItemId

; ----------------------------------------------------------------------------
; Draws item id A at the position of object slot X, flashing while its lifetime byte is >= $F0.
; IN:   A = item id ($00-$23); X = object slot whose ObjX/ObjY and Item_ObjItemLifetime are used.
; OUT:  sprites written; IsDrawnItemNarrow = 1 if the item is one sprite wide. X, Y not preserved.
; NOTE: With Item_ObjItemLifetime,X >= $F0 the item is drawn only when the value is odd. The byte is
;       an alias of the slot's position fraction, so for a non-item slot (carried room item, slot 1) it
;       is whatever that object keeps there.
; NOTE: [04] = low nibble of the item's descriptor (grade or amount); Item_DrawBySlot uses it as the
;       palette base only for item slots 0, 2, 4, 7, $B.
; ----------------------------------------------------------------------------
Item_DrawObject:
    PHA                         ; Save the item id.

    ; Flash: lifetime >= $F0 and even = skip drawing this time.
    LDA Item_ObjItemLifetime, X
    CMP #$F0
    BCC :+
    LSR
    BCC Item_DrawObject_PopAndReturn
:
    JSR Anim_FetchObjPosForSpriteDescriptor    ; [00] = ObjX,X; [01] = ObjY,X; no horizontal flip.

    ; [04] := value nibble of the item's descriptor.
    PLA                         ; Item id.
    TAX
    LDA Item_IdToDescriptor, X
    CMP #$30
    BEQ @SetItemValueFF         ; Never taken: no entry of Item_IdToDescriptor is $30.
    AND #$0F

@SetItemValue:
    STA $04

    ; X = Y = the item's slot; draw it.
    LDA Item_IdToSlot, X
    TAX
    TAY
    JMP Item_DrawBySlot

@SetItemValueFF:
    ; Not reached (see above): would use item value $FF.
    LDA #$FF
    BNE @SetItemValue

; ----------------------------------------------------------------------------
; Draws the item of an item slot as Link owns it: palette base = the inventory value of the slot.
; IN:   X = Y = item slot (index into InvItems); [00] = X pixel; [01] = Y pixel.
; NOTE: Falls into Item_DrawBySlot with [04] = InvItems,X.
; ----------------------------------------------------------------------------
Item_DrawInInventory:
    LDA InvItems, X
    STA $04

; ----------------------------------------------------------------------------
; Picks the sprite palette for an item slot and writes its sprites.
; IN:   X = Y = item slot; [00] = X pixel; [01] = Y pixel; [04] = item value (grade), used as the
;       palette base for slots 0, 2, 4, 7, $B only.
; OUT:  sprites written through Anim_WriteStaticItemSpritesWithAttributes (frame 0, object index $0F).
; NOTE: Sprite attribute = sprite palette 0-3. (A) Slots $16 rupee, $19 heart, $1A Triforce piece,
;       $1B Triforce of Power: palette 1 or 2, switching every 8 frames. (B) Slots 0 sword, 2 arrow,
;       4 candle, 7 potion, $B ring: [04] + table value ($FF for sword and arrow, 0 for the others).
;       (C) All other slots: the value of Item_SlotToPaletteOffsetsOrValues itself.
; NOTE: Slot 0 with palette 2 (magic sword) is drawn as pseudo slot $20, which has its own tile.
; ----------------------------------------------------------------------------
Item_DrawBySlot:
    LDA Item_SlotToPaletteOffsetsOrValues, X
    CPX #$16                    ; (A) Flashing items: rupee,
    BEQ @Flash
    CPX #$1A                    ; Triforce piece,
    BEQ @Flash
    CPX #$1B                    ; Triforce of Power,
    BEQ @Flash
    CPX #$19                    ; heart.
    BNE @VariesByColor

@Flash:
    LDA FrameCounter            ; Bit 3 of FrameCounter: 8 frames at 0, 8 frames at 1.
    AND #$08
    LSR
    LSR
    LSR
    ADC #$01                    ; Carry is clear after the shifts: palette 1 or 2.

@VariesByColor:
    CPX #$00                    ; (B) Items whose colour follows the grade: sword,
    BEQ @AddItemAndTableValue
    CPX #$04                    ; candle,
    BEQ @AddItemAndTableValue
    CPX #$02                    ; arrow,
    BEQ @AddItemAndTableValue
    CPX #$07                    ; potion,
    BEQ @AddItemAndTableValue
    CPX #$0B                    ; ring.
    BEQ @AddItemAndTableValue

@WriteSprites:
    ; A = final sprite attributes (cases A, B or C of the header). Y = item slot (image).
    LDX #$00
    STX $0C                     ; Frame 0.

    ; X = $0F is the writer's object index argument: nonzero makes it advance the rolling sprite
    ; index after each sprite; it also reads ObjInvincibilityTimer+$0F for palette flashing.
    LDX #$0F
    JMP Anim_WriteStaticItemSpritesWithAttributes

@AddItemAndTableValue:
    ; Palette = item value [04] + table value (an offset here: $FF = -1 for sword and arrow, else 0).
    CLC
    ADC $04
    CPX #$00                    ; Sword slot with palette 2 = magic sword:
    BNE @WriteSprites
    CMP #$02
    BNE @WriteSprites
    LDY #$20                    ; use the image of pseudo item slot $20.
    JMP @WriteSprites
