; ============================================================================
; FILE:     bank1/take-item.asm          BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4322-4739
; SUBJECTS: Inv, Item, Link, RoomFlags, Sfx
; TOPIC:    Picking up items: the item id -> inventory slot and item id -> descriptor (class, value)
;           tables, the "item taken" room flag, the pickup range test for the room item and dropped
;           items, and Item_Take, which applies every item's effect on the inventory, hearts and rupees.
; SEE ALSO: bank4/drop-items.asm (Drop_UpdateItem calls Item_TryTake for dropped items, also from the sword,
;           boomerang and arrow positions); bank7/mode5-play-update.asm (calls Item_TryTakeRoomItem every
;           frame); bank1/cave-shop.asm (calls Item_Take for cave wares); bank7/room-item-draw.asm (draws
;           items with these tables); bank7/status-bar-lift-item.asm (Link_CheckLiftItem: Link holding the
;           item up); bank1/status-bar-rupees.asm (InvRupeesToAdd is credited there). Item_BeginPowerTriforceFanfare
;           (the Triforce of Power fanfare) is in the bank 1 chunk that also holds
;           Link_CheckSlidingTileObjBlocking.
; KEY FACTS:
;   - Item slot n is the byte InvItems+n. Item_IdToSlot gives the slot of an item id; Item_IdToDescriptor
;     gives class (high nibble) and value (low nibble). (Item_IdToSlot, Item_IdToDescriptor)
;   - Class 0 sets the slot byte to the value (always 1); class 1 adds the value; class 2 stores the
;     value as a grade unless the byte already holds a higher grade. (Item_Take, Item_Take_CheckClass1,
;     Item_TakeClass2Grade)
;   - Pickup range: |Link X - item X| < 9 and |Link Y + 3 - item Y| < 9, and the item's lifetime
;     counter must be below $F0. (Item_TryTake)
;   - The room item (object slot $13) cannot be taken while Link is halted, once the room's flag bit
;     $10 is set, or while ObjState[$13] is negative. Taking it sets flag bit $10. (Item_TryTakeRoomItem,
;     Item_TryTake, RoomFlags_SetItemTaken)
;   - Caps: bombs at InvMaxBombs (a pickup adds 4), potion doses at 2, keys at $FF, heart containers at
;     16 (HeartValues >= $F0 refuses). (Item_Take_CheckClass1)
;   - A heart container adds $11 to HeartValues: one container and one full heart; it does not refill
;     the rest. (Item_Take_CheckClass1@TakeHeartContainer)
;   - A heart adds 1 full heart, a fairy 3; once full hearts = containers - 1 the partial heart is set
;     to $FF and the rest is discarded. (Item_TakeHeartsNoSound)
;   - Rupees are not added directly: each rupee increments InvRupeesToAdd (1 or 5). (Item_TakeOneRupee,
;     Item_Take5Rupees)
;   - The clock (id $21) is a plain class 0 item: InvClock = 1. (Item_Take, Item_IdToDescriptor)
;   - Map, compass and Triforce piece set the bit of the current level in their slot and do nothing in
;     the overworld; the Triforce piece also ends the mode and sets GameMode $12. (Item_TakeClass0PerLevel)
;   - Taken in any mode other than 5 (caves, cellars), an item is held up: ItemLiftTimer = $80 and the
;     item song plays. A bomb (id 0) is not held up: ItemTypeToLift = 0 means none. (Item_Take)
;   - Taking a ring that is kept changes Link's palette color at once: $29 no ring, $32 blue, $16 red.
;     (Item_TakeClass2Grade, Link_TunicColorsByRing)
; CONTENTS: (generated; L = line in this file)
;   L0081 Item_IdToSlot                      data  TABLE: item slot of each item; index = item id ($00-$23); value = item slot n (inventory byte InvItems+n).
;   L0111 Item_IdToDescriptor                data  TABLE: item descriptors; index = item id ($00-$23); value = class (high nibble) and value (low nibble).
;   L0128 Item_SlotToPaletteOffsetsOrValues  data  TABLE: sprite palette value per item slot; index = item slot ($00-$1F); value = sprite attribute
;   L0140 RoomFlags_SetItemTaken             code  Sets bit $10 ("item taken") in the current room's flags.
;   L0151 RoomFlags_GetItemTaken             code  Tests bit $10 ("item taken") of the current room's flags.
;   L0164 Link_TunicColorsByRing             data  TABLE: Link's tunic color; index = InvRing (0 none, 1 blue ring, 2 red ring); value = NES palette color.
;   L0173 Item_TryTakeRoomItem               code  Lets Link take the room item (object slot $13) if he is in range; run every frame in play.
;   L0206 Item_TryTake                       code  Takes item object X if Link's position is within range of it, and applies the item.
;   L0259 Item_Take                          code  Gives Link an item: plays its sound, starts the lift outside mode 5, and updates the inventory.
;   L0318 Item_Take_SetValue                 code  Stores A in item slot Y (InvItems+Y); the class 1 code jumps here with its capped sum.
;   L0322 Item_Take_Return                   code  Shared RTS of Item_TryTake, Item_Take and the heart container code.
;   L0328 Item_Take5Rupees                   code  Takes a 5-rupee item: five times Item_TakeOneRupee (InvRupeesToAdd + 5).
;   L0337 Item_Take_CheckClass1              code  Second part of Item_Take: handles class 1 (amount) items and passes classes 2 and 3 on.
;   L0421 Item_TakeClass0PerLevel            code  Class 0 items kept per level: map, compass, Triforce piece; passes the Triforce of Power on.
;   L0463 Item_TakeHearts                    code  Takes a heart: plays the heart/key tune, then adds [0A] + 1 hearts.
;   L0475 Item_TakeHeartsNoSound             code  Adds [0A] + 1 full hearts to Link's health, stopping at his heart containers (fairy: 3 hearts).
;   L0507 Inv_CompareHeartsToContainers      code  Compares Link's full hearts (low nibble of HeartValues) with containers - 1 (high nibble).
;   L0521 Item_Take_GoPowerTriforce          code  Triforce of Power taken in a dungeon: switches bank 1 in and jumps to Item_BeginPowerTriforceFanfare (the f...
;   L0536 Item_TakeClass2Grade               code  Class 2 items (sword, candle, arrow, ring, red potion): keeps the higher grade; a ring recolors Link.
;   L0557 Item_TakeOneRupee                  code  Takes one rupee: plays the rupee tune and adds 1 to InvRupeesToAdd (credited to InvRupees later).
;   L0563 Item_TakeOneRupee_Return           code  Shared RTS of Item_TakeOneRupee and Item_TakeClass2Grade.
;   L0569 Sfx_PlayKeyTakenTune               code  Plays the tune for taking a key or a heart: cancels the Tune1 request, requests Tune0 8.
; ============================================================================

; TABLE: item slot of each item; index = item id ($00-$23); value = item slot n (inventory byte InvItems+n).
;   id  item              slot      id  item              slot      id  item              slot
;   $00 bomb              $01       $0C raft              $09       $18 rupee             $16
;   $01 wooden sword      $00       $0D ladder            $0C       $19 key               $17
;   $02 white sword       $00       $0E Triforce of Power $1B       $1A heart container   $18
;   $03 magic sword       $00       $0F 5 rupees          $1C       $1B Triforce piece    $1A
;   $04 food              $06       $10 rod               $08       $1C magic shield      $1F
;   $05 flute             $05       $11 book              $0A       $1D boomerang         $1D
;   $06 blue candle       $04       $12 blue ring         $0B       $1E magic boomerang   $1E
;   $07 red candle        $04       $13 red ring          $0B       $1F blue potion       $07
;   $08 arrow             $02       $14 bracelet          $0E       $20 red potion        $07
;   $09 silver arrow      $02       $15 letter            $0F       $21 clock             $15
;   $0A bow               $03       $16 compass           $10       $22 heart             $19
;   $0B magic key         $0D       $17 map               $11       $23 fairy             $14
; Slots $12 and $13 (compass and map of level 9) are reached from $10 and $11 by Item_TakeClass0PerLevel.
Item_IdToSlot:
    .BYTE $01, $00, $00, $00, $06, $05, $04, $04
    .BYTE $02, $02, $03, $0D, $09, $0C, $1B, $1C
    .BYTE $08, $0A, $0B, $0B, $0E, $0F, $10, $11
    .BYTE $16, $17, $18, $1A, $1F, $1D, $1E, $07
    .BYTE $07, $15, $19, $14

; TABLE: item descriptors; index = item id ($00-$23); value = class (high nibble) and value (low nibble).
; Class 0 ($0x): Item_Take stores the value (1 in every entry) in the item's slot. Exceptions by slot:
;   compass, map and Triforce piece set a per-level bit instead; the Triforce of Power starts its
;   fanfare (Item_TakeClass0PerLevel).
; Class 1 ($1x): the value is an amount. Default: added to the slot byte. Exceptions by slot: rupees,
;   hearts, the fairy and the heart container (Item_Take_CheckClass1).
; Class 2 ($2x): the value is a grade. Stored only if it is not lower than the grade already held
;   (Item_TakeClass2Grade).
; Class 3 ($3x): no entry uses it. Item_Take would store $FF (then capped as in Item_Take_CheckClass1 for the
;   potion and bomb slots); the item drawing code would use value $FF.
; When an item object is drawn, the value nibble is the sprite palette base for slots 0 sword,
;   2 arrow, 4 candle, 7 potion and $0B ring only (see Item_SlotToPaletteOffsetsOrValues).
;   $00 bomb $14 (adds 4)             $01-$03 swords $21/$22/$23      $04 food $01
;   $05 flute $01                     $06/$07 candles $21/$22         $08/$09 arrows $21/$22
;   $0A bow $01                       $0B magic key $01               $0C raft $01
;   $0D ladder $01                    $0E Triforce of Power $01       $0F 5 rupees $15
;   $10 rod $01                       $11 book $01                    $12/$13 rings $21/$22
;   $14 bracelet $01                  $15 letter $01                  $16 compass $01
;   $17 map $01                       $18 rupee $11                   $19 key $11 (adds 1)
;   $1A heart container $10           $1B Triforce piece $01          $1C magic shield $01
;   $1D boomerang $01                 $1E magic boomerang $01         $1F blue potion $11 (adds 1)
;   $20 red potion $22 (grade 2)      $21 clock $01                   $22 heart $10 (value 0: 1 heart)
;   $23 fairy $12 (value 2: 3 hearts)
Item_IdToDescriptor:
    .BYTE $14, $21, $22, $23, $01, $01, $21, $22
    .BYTE $21, $22, $01, $01, $01, $01, $01, $15
    .BYTE $01, $01, $21, $22, $01, $01, $01, $01
    .BYTE $11, $11, $10, $01, $01, $01, $01, $11
    .BYTE $22, $01, $10, $12

; TABLE: sprite palette value per item slot; index = item slot ($00-$1F); value = sprite attribute
; (palette row 0-2), or for slots $00, $02, $04, $07, $0B an offset added to the item's value/grade.
; Used by Item_DrawBySlot. Offsets: sword $FF (grade - 1), arrow $FF, candle 0, potion 0, ring 0.
; Slots $16 rupee, $19 heart, $1A Triforce piece and $1B Triforce of Power ignore their entry: they
; flash between attributes 1 and 2.
;   Row 1: $00 sword, $01 bomb, $02 arrow, $03 bow, $04 candle, $05 flute, $06 food, $07 potion
;   Row 2: $08 rod, $09 raft, $0A book, $0B ring, $0C ladder, $0D magic key, $0E bracelet, $0F letter
;   Row 3: $10 compass, $11 map, $12 compass 9, $13 map 9, $14 fairy, $15 clock, $16 rupee, $17 key
;   Row 4: $18 heart container, $19 heart, $1A Triforce piece, $1B Triforce of Power, $1C 5 rupees,
;          $1D boomerang, $1E magic boomerang, $1F magic shield
Item_SlotToPaletteOffsetsOrValues:
    .BYTE $FF, $01, $FF, $00, $00, $02, $02, $00
    .BYTE $01, $00, $02, $00, $00, $02, $02, $01
    .BYTE $02, $02, $02, $02, $02, $02, $02, $02
    .BYTE $02, $02, $02, $02, $01, $00, $01, $00

; ----------------------------------------------------------------------------
; Sets bit $10 ("item taken") in the current room's flags.
; OUT:  A = the new flag byte; Y = RoomId; [00:01] = address of the level's room flags.
; NOTE: The same bit serves dungeon room items and overworld cave items and offers (callers in the
;       cave and person code).
; ----------------------------------------------------------------------------
RoomFlags_SetItemTaken:
    JSR RoomFlags_Get
    ORA #$10
    STA ($00), Y
    RTS

; ----------------------------------------------------------------------------
; Tests bit $10 ("item taken") of the current room's flags.
; OUT:  A = $10 and Z clear if the item was taken, else A = 0 and Z set; Y = RoomId;
;       [08:09] = address of the level's room flags.
; ----------------------------------------------------------------------------
RoomFlags_GetItemTaken:
    LDA LevelInfo_WorldFlagsAddr
    STA $08
    LDA LevelInfo_WorldFlagsAddr+1
    STA $09
    LDY RoomId
    LDA ($08), Y
    AND #$10
:
    ; This RTS is also the early exit of Item_TryTakeRoomItem below (its ":-" branches).
    RTS

; TABLE: Link's tunic color; index = InvRing (0 none, 1 blue ring, 2 red ring); value = NES palette color.
Link_TunicColorsByRing:
    .BYTE $29, $32, $16

; ----------------------------------------------------------------------------
; Lets Link take the room item (object slot $13) if he is in range; run every frame in play.
; OUT:  nothing when refused; else falls into Item_TryTake with X = $13, [04] = RoomItemId.
; NOTE: Refused while Link is halted (ObjState & $C0 = $40), when the room's "item taken" flag (bit
;       $10) is set, or while ObjState[$13] is negative (item hidden or absent).
; ----------------------------------------------------------------------------
Item_TryTakeRoomItem:
    ; If Link is halted, then return.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ :-

    ; If the room's item was already taken, then return.
    JSR RoomFlags_GetItemTaken
    BNE :-

    ; Switch to room item object slot $13.
    LDX #$13

    ; If the room item object isn't active (state negative), then return.
    LDA ObjState, X
    BMI :-

    ; ObjRoomItemId,X with X = $13 is RoomItemId (ObjDir[$13]).
    LDA ObjRoomItemId, X
    STA $04                     ; [04] holds the item id.

; ----------------------------------------------------------------------------
; Takes item object X if Link's position is within range of it, and applies the item.
; IN:   X = object slot of the item ($13 room item, or a dropped item's slot); [04] = item id;
;       ObjX/ObjY of slot 0 = the taker's position (Drop_UpdateItem substitutes a weapon's position).
; OUT:  not taken: nothing. Taken: ObjState,X = $FF and ObjY,X = $FF (for a dropped item ObjState is
;       its item id, so $FF marks it taken); for X = $13 the room's "item taken" flag is set; then
;       falls into Item_Take with A = [04].
; NOTE: Range: |ObjX - ObjX,X| < 9 and |ObjY + 3 - ObjY,X| < 9, that is taker Y - item Y in -11..+5.
; NOTE: Refused while Item_ObjItemLifetime,X >= $F0 (a dropped item starts at $FF and loses 1 every
;       second frame, so it cannot be taken while it still flashes).
; ----------------------------------------------------------------------------
Item_TryTake:
    ; If the lifetime counter of the item >= $F0, then return;
    ; so that a new drop can't be picked up right away.
    LDA Item_ObjItemLifetime, X
    CMP #$F0
    BCS Item_Take_Return

    ; If Y distance between Link (Y + 3) and the item >= 9, return.
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    JSR Math_Abs
    CMP #$09
    BCS Item_Take_Return

    ; If X distance between Link and the item >= 9, return.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Math_Abs
    CMP #$09
    BCS Item_Take_Return

    ; Deactivate the item object: state $FF (room item hidden; dropped item id = taken)
    ; and Y = $FF.
    LDA #$FF
    STA ObjState, X
    STA ObjY, X

    ; If it is the room item that is taken, then
    ; mark it taken in room flags.
    CPX #$13
    BNE :+
    JSR RoomFlags_SetItemTaken
:
    LDA $04                     ; Get the item id.

; ----------------------------------------------------------------------------
; Gives Link an item: plays its sound, starts the lift outside mode 5, and updates the inventory.
; IN:   A = item id ($00-$23).
; OUT:  the item's slot byte, hearts, rupees to add or level bit updated by class (see
;       Item_IdToDescriptor); [0A] = the descriptor's value nibble.
; NOTE: Sound: Tune1Request = 8, or 2 for the Triforce of Power (id $0E). Rupees replace it with
;       Tune1 1; keys and hearts cancel it and request Tune0 8 instead; a fairy keeps Tune1 8.
; NOTE: When GameMode <> 5 (caves and cellars run in other modes): ItemLiftTimer = $80, the item song
;       (SongRequest 8) starts and ItemTypeToLift = the id; Link_CheckLiftItem then halts Link while he
;       holds the item up. A bomb (id 0) leaves ItemTypeToLift at 0, so nothing is held up, but the
;       timer and the song are still started.
; NOTE: Class 0 items other than map, compass and the Triforces just get their slot byte set to 1.
;       That includes the clock (id $21, InvClock = 1) and the letter.
; ----------------------------------------------------------------------------
Item_Take:
    LDX #$08
    STX Tune1Request

    ; Unless it's the Triforce of Power; in which case
    ; request Tune1 2 instead.
    CMP #$0E
    BNE :+
    LDX #$02
    STX Tune1Request
:
    ; If not in mode 5 (we're in a cave or cellar), then
    ; prepare to lift the item.
    LDX GameMode
    CPX #$05
    BEQ @SkipLift
    LDX #$80                    ; Lift timer: $80.
    STX ItemLiftTimer
    LDX #$08                    ; Play the "item" song.
    STX SongRequest
    STA ItemTypeToLift          ; Set the item id for lifting.

@SkipLift:
    ; Look up the item slot and descriptor by the item id.
    TAX
    LDA Item_IdToSlot, X
    TAY
    LDA Item_IdToDescriptor, X

    ; [0A] holds the item value from the low nibble of descriptor.
    PHA
    AND #$0F
    STA $0A
    PLA

    ; Check the high nibble which specifies the item's class.
    ;
    ; A: item class
    ; X: item id
    ; Y: item slot
    AND #$F0
    BNE Item_Take_CheckClass1             ; Go check classes 1 and 2.

    ; Class 0. The item is one you have or don't, or a per-level bit.
    ;
    ; First, check the exceptions. If the item is a map, compass, or Triforce; then go handle them.
    CPY #$11                    ; Map item slot
    BEQ Item_TakeClass0PerLevel
    CPY #$10                    ; Compass item slot
    BEQ Item_TakeClass0PerLevel
    CPY #$1A                    ; Triforce pieces item slot
    BEQ Item_TakeClass0PerLevel
    CPY #$1B                    ; Triforce of Power item slot
    BEQ Item_TakeClass0PerLevel

    ; Finally, this is a simple item. Put the item value (1) in the item slot.
    LDA $0A

; Stores A in item slot Y (InvItems+Y); the class 1 code jumps here with its capped sum.
Item_Take_SetValue:
    STA InvItems, Y

; Shared RTS of Item_TryTake, Item_Take and the heart container code.
Item_Take_Return:
    RTS

; ----------------------------------------------------------------------------
; Takes a 5-rupee item: five times Item_TakeOneRupee (InvRupeesToAdd + 5).
; ----------------------------------------------------------------------------
Item_Take5Rupees:
    LDY #$04
:
    JSR Item_TakeOneRupee
    DEY
    BPL :-
    RTS

; Second part of Item_Take: handles class 1 (amount) items and passes classes 2 and 3 on.
Item_Take_CheckClass1:
    ; A: item class
    ; X: item id
    ; Y: item slot
    ; [0A]: item value
    CMP #$10
    BNE @CheckClass2            ; Go check class 2 items.

    ; Class 1. We have a type of item with an amount.
    ; Item value is the amount to add.
    CPY #$18                    ; Heart container item slot
    BEQ @TakeHeartContainer
    CPY #$1C                    ; 5 Rupees item slot
    BEQ Item_Take5Rupees
    CPY #$16                    ; Rupee item slot
    BEQ @TakeRupee
    CPY #$19                    ; Heart item slot
    BEQ Item_TakeHearts

    ; For key item slot, play the tune here.
    ; Add to the amount below.
    CPY #$17
    BNE :+
    JSR Sfx_PlayKeyTakenTune
:
    ; For fairy item slot, go add hearts ([0A] = 2: three hearts) without the heart sound.
    CPY #$14
    BEQ Item_TakeHeartsNoSound

    ; Keys, bombs and the blue potion.
    ; Add item value and value in item slot.
    LDA $0A
    CLC
    ADC InvItems, Y
    BCC :+                      ; If the sum overflowed,

@SetItemValueFF:
    LDA #$FF                    ; then cap it at $FF. (Class 3 also comes here.)
:
    ; For potion item slot (7), cap it at 2.
    CPY #$07
    BNE :+
    CMP #$03
    BCC :+
    LDA #$02
:
    ; For bomb item slot (1), cap it at InvMaxBombs.
    CPY #$01
    BNE :+
    CMP InvMaxBombs
    BCC :+
    LDA InvMaxBombs
:
    JMP Item_Take_SetValue            ; Go set the item value and return.

@TakeHeartContainer:
    ; Slot $18 is HeartValues: high nibble = containers - 1, low nibble = full hearts.
    ; If the high nibble is already $F (16 containers), return.
    ; Else add $11: one more container and one more full heart.
    LDA InvItems, Y
    CMP #$F0
    BCS Item_Take_Return              ; If we already have max hearts, return.
    ADC #$11                    ; Carry is clear here.
    JMP Item_Take_SetValue            ; Go set the item value and return.

@TakeRupee:
    JMP Item_TakeOneRupee

@CheckClass2:
    ; Class $20: grade items. Any other class ($30; no item id has it) stores $FF.
    CMP #$20
    BNE @SetItemValueFF
    BEQ Item_TakeClass2Grade

; ----------------------------------------------------------------------------
; Class 0 items kept per level: map, compass, Triforce piece; passes the Triforce of Power on.
; IN:   Y = item slot ($10 compass, $11 map, $1A Triforce piece, $1B Triforce of Power).
; OUT:  bit (CurLevel - 1) set in the slot byte; in level 9 the bit is 0 and the slot is Y + 2
;       (compass $12, map $13). Map: StatusBarMapTrigger = 1. Triforce piece: the mode ends and
;       GameMode = $12.
; NOTE: Does nothing at all in the overworld (CurLevel = 0).
; NOTE: A Triforce piece taken in level 9 would set bit 0 of slot $1C (not InvTriforce) and would not
;       start mode $12, because the slot index is moved by 2 before the Triforce test.
; ----------------------------------------------------------------------------
Item_TakeClass0PerLevel:
    ; X: item id
    ; Y: item slot
    LDA CurLevel
    BEQ @Exit                   ; If in OW, return.
    CPY #$1B                    ; Triforce of Power item slot
    BEQ Item_Take_GoPowerTriforce
    CPY #$11                    ; Map item slot
    BNE :+                      ; If it's the map item slot,
    LDX #$01                    ; then signal the main loop to draw the status bar map.
    STX StatusBarMapTrigger
:
    ; Choose the right slot for the level. (map/map9, compass/compass9)
    SEC
    SBC #$01                    ; Subtract one from level number to base it on 0.
    CMP #$08
    BCC :+                      ; If the result >= 8, we're in level 9,
    INY                         ; So add 2 to the item slot to use the one for level 9.
    INY
:
    AND #$07                    ; Level index 0 to 7; level 9 gives 0.
    TAX

    ; Combine the mask for this level with the mask in the item slot.
    LDA InvItems, Y
    ORA Util_BitMasks, X
    STA InvItems, Y

    ; If the item taken was a Triforce piece, then go to mode $12.
    CPY #$1A
    BNE @Exit                   ; If it's not a Triforce piece, then return.
    JSR Mode_End
    LDA #$12
    STA GameMode

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Takes a heart: plays the heart/key tune, then adds [0A] + 1 hearts.
; IN:   [0A] = hearts to add minus 1 (0 for a heart).
; ----------------------------------------------------------------------------
Item_TakeHearts:
    JSR Sfx_PlayKeyTakenTune

; ----------------------------------------------------------------------------
; Adds [0A] + 1 full hearts to Link's health, stopping at his heart containers (fairy: 3 hearts).
; IN:   [0A] = hearts to add minus 1.
; OUT:  HeartValues low nibble raised by up to [0A] + 1; [01] = count left; [00] clobbered.
; NOTE: Each step adds one full heart while full hearts < containers - 1 (the last heart is always
;       the partial one). When they are equal, HeartPartial is set to $FF if it is not already and
;       the remaining hearts are discarded. HeartPartial is otherwise left as it is, so a heart
;       raises health by exactly one heart when there is room.
; ----------------------------------------------------------------------------
Item_TakeHeartsNoSound:
    LDA $0A                     ; Move item value in [0A] to [01].
    STA $01                     ; [01] holds the number of hearts to add minus 1.

@CompareHearts:
    ; For that many hearts, add them one by one.
    JSR Inv_CompareHeartsToContainers
    BNE @AddWholeHeart          ; If there's room to add a whole heart, go add it.

    ; All full hearts are filled. If the partial heart is not full ($FF), fill it. Return.
    LDX HeartPartial
    INX
    BNE @FillHeartPartial
    RTS

@AddWholeHeart:
    ; Add a whole heart.
    INC HeartValues
    DEC $01
    BPL @CompareHearts          ; If there are more hearts to add, go compare them to heart containers again.
    RTS

@FillHeartPartial:
    LDA #$FF
    STA HeartPartial
    RTS

; ----------------------------------------------------------------------------
; Compares Link's full hearts (low nibble of HeartValues) with containers - 1 (high nibble).
; OUT:  Z set when they are equal (no room for another full heart); A = high nibble;
;       [00] = low nibble; C set when high nibble >= low nibble.
; ----------------------------------------------------------------------------
Inv_CompareHeartsToContainers:
    LDA HeartValues
    PHA
    AND #$0F
    STA $00
    PLA
    LSR
    LSR
    LSR
    LSR
    CMP $00
    RTS

; Triforce of Power taken in a dungeon: switches bank 1 in and jumps to Item_BeginPowerTriforceFanfare (the fanfare).
Item_Take_GoPowerTriforce:
    LDA #$01
    JSR Mmc1_SwitchBank_Bank5
    JMP Item_BeginPowerTriforceFanfare

; ----------------------------------------------------------------------------
; Class 2 items (sword, candle, arrow, ring, red potion): keeps the higher grade; a ring recolors Link.
; IN:   Y = item slot; [0A] = grade of the item taken.
; OUT:  InvItems,Y = [0A] unless it already held a higher value.
; NOTE: The red potion is grade 2 in the potion slot, so it sets InvPotion to 2.
; NOTE: For the ring slot ($0B), when the grade is stored, Link's color for InvRing is patched into
;       the menu palette row of the save slot and into the level palette, the palette transfer is
;       cued, and GameSubmode is incremented (all in
;       Palette_PatchLinkColorAndCueLevelPalettes).
; ----------------------------------------------------------------------------
Item_TakeClass2Grade:
    ; Y: item slot
    ; [0A]: item value
    LDA $0A
    CMP InvItems, Y
    BCC Item_TakeOneRupee_Return              ; If we have a higher grade of this kind of item, return.
    STA InvItems, Y             ; Set the new item grade.
    CPY #$0B                    ; Ring item slot
    BNE Item_TakeOneRupee_Return              ; If the item is not a ring, return.

    ; We took a ring. Change Link's color.
    LDX CurSaveSlot
    LDY InvRing                 ; Get ring in inventory.
    LDA Link_TunicColorsByRing, Y    ; Get the color for this ring value.
    LDY Palette_SaveSlotRowOffsets, X    ; Get the offset of row 4, 5, or 6, depending on save slot.
    STA Transfer_MenuPalettesBuf+20, Y    ; Patch the color into the menu palette.
    JMP Palette_PatchLinkColorAndCueLevelPalettes    ; Then patch the color into the level's palette.

; ----------------------------------------------------------------------------
; Takes one rupee: plays the rupee tune and adds 1 to InvRupeesToAdd (credited to InvRupees later).
; ----------------------------------------------------------------------------
Item_TakeOneRupee:
    LDA #$01                    ; Play "rupee taken" tune.
    STA Tune1Request
    INC InvRupeesToAdd

; Shared RTS of Item_TakeOneRupee and Item_TakeClass2Grade.
Item_TakeOneRupee_Return:
    RTS

; ----------------------------------------------------------------------------
; Plays the tune for taking a key or a heart: cancels the Tune1 request, requests Tune0 8.
; ----------------------------------------------------------------------------
Sfx_PlayKeyTakenTune:
    LDA #$00                    ; Cancel the "item taken" request made by Item_Take.
    STA Tune1Request
    LDA #$08                    ; Play "key taken" tune.
    STA Tune0Request
    RTS
