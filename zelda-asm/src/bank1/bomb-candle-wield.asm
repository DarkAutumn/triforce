; ============================================================================
; FILE:     bank1/bomb-candle-wield.asm  BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3853-4132
; SUBJECTS: Bomb, Candle, Dir, Link, Obj, Room
; TOPIC:    Using a bomb or a candle (object slots $10/$11), the shared code that puts a new weapon object
;           $10 pixels in front of Link and puts Link in his item-use state, the room's shortcut/item
;           position lookup, the object slot reset, and the grayscale flash of a bomb blast.
; SEE ALSO: bank5/wield-items.asm (Link_WieldItem_JumpTable calls Bomb_Wield and Candle_Wield; the other items
;           enter the Link_PlaceWeaponAhead chain); bank7/bomb.asm (Bomb_Update, Bomb_StateTimes: what the bomb states do);
;           bank7/fire.asm (Fire_Update: what fire states $21/$22 do); bank7/sword-shot-magic-shot.asm
;           (SwordShot_HandleBlocked makes the book's fire through Candle_Wield); bank7/object-animation.asm
;           (Link_AdvanceItemUseState steps Link's item-use state $10 -> $11 -> $31 -> 0).
; KEY FACTS:
;   - Bombs and fires share object slots $10 and $11. A bomb's state is $1x, a fire's $2x. (Bomb_Wield,
;     Candle_Wield)
;   - A bomb needs InvBombs > 0 and a slot that is empty or holds a fire: slot $10 first, then $11. It is
;     refused when both slots hold bombs, or when the other slot holds a bomb in state $11 or $12 (not
;     yet exploding). So a second bomb can be placed only once the first is in state $13 or later.
;     (Bomb_Wield)
;   - A bomb placed in a slot that holds a fire replaces that fire, even if the other slot is empty.
;     (Bomb_Wield@FoundSlot)
;   - Placing a bomb costs 1 InvBombs, starts the bomb in state $11 with ObjTimer 0 and plays tune $20.
;     (Bomb_Wield@Activate)
;   - A candle fire needs an empty slot ($10, else $11); a fire never replaces a bomb or a fire.
;     (Candle_Wield)
;   - Blue candle (InvCandle = 1): refused while HasUsedCandle is set; every fire made sets it. The red
;     candle (2) ignores the flag. (Candle_Wield@NotInUse)
;   - A new fire starts in state $21 (moving) with q-speed $20 (0.5 pixel per frame). (Candle_Wield)
;   - A new weapon object is put $10 pixels from Link's position in the direction Link faces and takes
;     Link's direction. (Link_PlaceWeaponAhead)
;   - Using an item sets Link's state to $10; entries above Link_UseItem_SetState10 also set Link's
;     animation counter to 1 so the state advances on the next animation tick.
;     (Link_UseItem_SetAnimCounter1, Link_UseItem_SetState10)
;   - The bomb blast flashes the screen by setting the grayscale bit of the PPU mask at ObjTimer $16 and
;     $11 and clearing it at $12 and $0D, only in bomb state $13. (Bomb_UpdateFlashEffect)
; CONTENTS: (generated; L = line in this file)
;   L0063 Bomb_Wield                         code  Places a bomb in front of Link in object slot $10 or $11, if the bomb rules allow it.
;   L0119 Link_UseItem_SetWeaponState        code  Sets a new weapon object's state, then places it and puts Link in the item-use state.
;   L0131 Link_UseItem_SetAnimCounter1       code  Sets Link's animation counter to 1, then places the weapon and sets Link's state to $10.
;   L0143 Link_UseItem_SetState10            code  Sets Link's state to $10 (using an item), then places weapon object X in front of him.
;   L0154 Link_PlaceWeaponAhead              code  Puts object X at Link's position plus A pixels in the direction Link faces; copies Link's direction.
;   L0180 Link_PlaceWeaponAhead_Return       code  Shared RTS of Link_PlaceWeaponAhead; Bomb_Wield also returns through it when a bomb is refused.
;   L0188 Dir_ChooseHorizontalOffset         code  Chooses one of two values by the horizontal bits of a direction, or 0 if neither is set.
;   L0204 Dir_ChooseHorizontalOffset_Return  code  Shared RTS of Dir_ChooseHorizontalOffset; Candle_Wield also returns through it when a fire is refused.
;   L0220 Candle_Wield                       code  Makes a candle fire in front of Link in object slot $10 or $11, if the slot and candle rules allow.
;   L0269 Room_GetShortcutOrItemXY           code  Gets the current room's shortcut stairs position (overworld) or room item position (dungeon).
;   L0280 Room_GetShortcutOrItemXYForRoomId  code  Gets the shortcut stairs position (overworld) or room item position (dungeon) of room Y.
;   L0308 Obj_ResetSlotFields                code  Clears an object slot's state, timer, knockback and invincibility and marks it uninitialized.
;   L0328 Bomb_UpdateFlashEffect             code  Flashes the screen during a bomb blast by switching the PPU mask's grayscale bit (bit 0).
; ============================================================================

; ----------------------------------------------------------------------------
; Places a bomb in front of Link in object slot $10 or $11, if the bomb rules allow it.
; OUT:  on success X = the bomb's slot, ObjState,X = $11, ObjTimer,X = 0, InvBombs decremented,
;       Link's state $10 (falls into Link_UseItem_SetWeaponState).
; NOTE: Slot choice: $10 if it is empty or holds a fire, else $11 if it is empty or holds a fire, else
;       fail (two bombs). Then the other slot must be empty, or in a state >= $13: a bomb that is
;       already exploding ($13, $14) or any fire ($21, $22). At most one unexploded bomb exists.
; NOTE: The chosen slot may hold a fire; the bomb overwrites it.
; NOTE: Bomb states (Bomb_Update, Bomb_StateTimes): $11 placed, $12 fuse for $30 frames, $13 blast for $18
;       frames, $14 for $0C frames, then the slot is cleared.
; ----------------------------------------------------------------------------
Bomb_Wield:
    ; If there are no bombs in inventory, return.
    LDA InvBombs
    BEQ Dir_ToTarget_Return     ; An RTS in another chunk.

    ; Look in slot $10. If it's empty or there's a fire
    ; (state = 0, or major state <> $10), then
    ; go activate a bomb.
    LDX #$10
    LDA ObjState, X
    BEQ @FoundSlot
    AND #$F0
    CMP #$10
    BNE @FoundSlot

    ; Slot $10 holds a bomb. Look at slot $11.
    ; If there's a bomb there too (major state = $10), then return.
    INX
    LDA ObjState, X
    BEQ @FoundSlot
    AND #$F0
    CMP #$10
    BEQ Link_PlaceWeaponAhead_Return

@FoundSlot:
    ; Slot X ($10 or $11) is empty or holds a fire. Y = the other slot (X EOR 1).
    ; If the other slot is active with a state < $13 (a bomb in state $11 or $12,
    ; not yet exploding), then return. Fires ($21, $22) and exploding bombs pass.
    TXA
    EOR #$01
    TAY
    LDA a:ObjState, Y
    BEQ @Activate
    CMP #$13
    BCC Link_PlaceWeaponAhead_Return

@Activate:
    ; We're using a bomb. Decrement the count.
    DEC InvBombs

    ; Play the "set a bomb" tune.
    LDA #$20
    STA Tune0Request

    ; Reset the object timer, so that Bomb_Update advances the state on its first run.
    LDA #$00
    STA ObjTimer, X

    ; Start in state $11.
    LDA #$11

; ----------------------------------------------------------------------------
; Sets a new weapon object's state, then places it and puts Link in the item-use state.
; IN:   A = initial state for the weapon ($11 bomb, $80 food); X = weapon object slot.
; OUT:  ObjState,X = A; then as Link_UseItem_SetAnimCounter1.
; ----------------------------------------------------------------------------
Link_UseItem_SetWeaponState:
    STA ObjState, X

; ----------------------------------------------------------------------------
; Sets Link's animation counter to 1, then places the weapon and sets Link's state to $10.
; IN:   X = weapon object slot.
; NOTE: With the counter at 1, the next call of Link_AnimateWalking takes it to 0, which runs
;       Link_AdvanceItemUseState (state $10 -> $11) and reloads the counter with 6. The next two times it
;       reaches 0 make the state $31 and then clear its low 6 bits (Link idle again). An item use
;       started here therefore lasts 1 + 6 + 6 = 13 animation calls, unless the weapon's own code
;       changes Link's counter or state meanwhile.
; ----------------------------------------------------------------------------
Link_UseItem_SetAnimCounter1:
    LDA #$01
    STA ObjAnimCounter

; ----------------------------------------------------------------------------
; Sets Link's state to $10 (using an item), then places weapon object X in front of him.
; IN:   X = weapon object slot.
; OUT:  ObjState (Link) = $10; as Link_PlaceWeaponAhead with A = $10.
; NOTE: Link's animation counter is left as it is; callers entering here (boomerang, candle) set it
;       themselves or not at all. While Link's state is not 0 his animation counter runs every time
;       he is animated (Link_AnimateWalking), and Link_AdvanceItemUseState advances the state when it hits 0.
; ----------------------------------------------------------------------------
Link_UseItem_SetState10:
    LDA #$10
    STA ObjState

; ----------------------------------------------------------------------------
; Puts object X at Link's position plus A pixels in the direction Link faces; copies Link's direction.
; IN:   A = distance in pixels ($10 from every caller); X = object slot.
; OUT:  ObjDir,X = ObjDir (Link); ObjX,X and ObjY,X set; [00] = 0, [01] = A, [02] = $F0.
; NOTE: The offset for left and up is always $F0 (-$10), whatever A is.
; NOTE: Facing right: X + A; left: X - $10; down: Y + A; up: Y - $10. The other axis is unchanged.
; ----------------------------------------------------------------------------
Link_PlaceWeaponAhead:
    ; Set the offset choices to:
    ;  A ($10) for right and down in [01]
    ; -$10 for left and up in [02]
    LDY #$F0
    STA $01
    STY $02

    ; Set object's direction to Link's.
    LDA ObjDir
    STA ObjDir, X

    ; Set object's X to Link's X + offset for direction.
    JSR Dir_ChooseHorizontalOffset
    ADC ObjX
    STA ObjX, X

    ; Set object's Y to Link's Y + offset for direction.
    LDA ObjDir
    LSR                         ; Shift down (4) to bit 0 and up (8) to bit 1: selects like right/left.
    LSR
    JSR Dir_ChooseHorizontalOffset
    ADC ObjY
    STA ObjY, X

; Shared RTS of Link_PlaceWeaponAhead; Bomb_Wield also returns through it when a bomb is refused.
Link_PlaceWeaponAhead_Return:
    RTS

; ----------------------------------------------------------------------------
; Chooses one of two values by the horizontal bits of a direction, or 0 if neither is set.
; IN:   A = direction bits; [01] = value for right (bit 0); [02] = value for left (bit 1).
; OUT:  A = [01] if bit 0 set, else [02] if bit 1 set, else 0; C clear; [00] = 0; Y = 0, 1 or 2.
; ----------------------------------------------------------------------------
Dir_ChooseHorizontalOffset:
    ; [00] = 0 is the value used when the direction is not horizontal.
    LDY #$00
    STY $00
    AND #$03
    BEQ @GetValue               ; If vertical, go choose [00] (value 0).
    INY                         ; Direction is horizontal. Increment index to 1.
    AND #$01
    BNE @GetValue               ; If direction is right, go choose [01].
    INY                         ; Direction is left. Increment index to 2.

@GetValue:
    LDA $0000, Y                ; Load the offset from [00], [01] or [02].
    CLC                         ; Clear carry in preparation for adding offset to another value.

; Shared RTS of Dir_ChooseHorizontalOffset; Candle_Wield also returns through it when a fire is refused.
Dir_ChooseHorizontalOffset_Return:
    RTS

; ----------------------------------------------------------------------------
; Makes a candle fire in front of Link in object slot $10 or $11, if the slot and candle rules allow.
; OUT:  X = last slot tested ($10 or $11), whether or not a fire was made. On success ObjState,X = $21,
;       HasUsedCandle = 1, Link's state = $10.
; NOTE: Needs an empty slot: $10, else $11. Unlike a bomb, a fire never replaces anything. The fire
;       starts moving (state $21) at q-speed $20 = 0.5 pixel per frame, $10 pixels ahead of Link.
; NOTE: Blue candle (InvCandle = 1) fails when HasUsedCandle is already set. Any other InvCandle value
;       skips the test (red candle: no limit). The routine does not test that a candle is owned.
; NOTE: The magic book's fire comes through here too (SwordShot_HandleBlocked): that caller zeroes
;       HasUsedCandle first and restores it and Link's state afterwards, so a book fire neither needs
;       nor spends the blue candle's use and does not put Link in the item-use state. It tells success
;       by ObjState,X = $21, which is why X must be the last slot tested.
; ----------------------------------------------------------------------------
Candle_Wield:
    ; Look in slots $10 and $11. If none are empty, return.
    ; Otherwise, X will have the empty slot found.
    ;
    ; Note that when no slot is empty, X = $11, and that slot may hold a fire made
    ; earlier that is still in state $21.
    LDX #$10
    LDA ObjState, X
    BEQ @NotInUse
    INX
    LDA ObjState, X
    BNE Dir_ChooseHorizontalOffset_Return

@NotInUse:
    ; If we have the blue candle and it was used, then return.
    LDA InvCandle
    CMP #$01
    BNE :+
    LDA HasUsedCandle
    BNE Dir_ChooseHorizontalOffset_Return
:
    ; Set the candle used.
    LDA #$01
    STA HasUsedCandle

    ; Reset the new fire object's movement info.
    ; Set q-speed to $20 (0.5 pixel per frame).
    LDA #$00
    STA ObjGridOffset, X
    STA ObjPosFrac, X
    LDA #$20
    STA ObjQSpeedFrac, X

    ; Activate the fire object. Initial state = $21 (moving fire).
    LDA #$21
    STA ObjState, X
    LDA #$04                    ; Flame sound effect
    JSR Sfx_Play

    ; Each animation frame lasts 4 frames.
    LDA #$04
    STA ObjAnimCounter, X
    JSR Link_UseItem_SetState10
    RTS

; ----------------------------------------------------------------------------
; Gets the current room's shortcut stairs position (overworld) or room item position (dungeon).
; OUT:  A = X coordinate; Y = Y coordinate (both multiples of $10).
; ----------------------------------------------------------------------------
Room_GetShortcutOrItemXY:
    LDY RoomId

; ----------------------------------------------------------------------------
; Gets the shortcut stairs position (overworld) or room item position (dungeon) of room Y.
; IN:   Y = room id.
; OUT:  A = X coordinate (high nibble of the position byte, low nibble 0);
;       Y = Y coordinate (low nibble of the position byte * $10).
; NOTE: Bits 5-4 of LevelBlockAttrsF for the room choose one of the level's 4 positions in
;       LevelInfo_ShortcutOrItemPosArray.
; ----------------------------------------------------------------------------
Room_GetShortcutOrItemXYForRoomId:
    LDA LevelBlockAttrsF, Y
    AND #$30                    ; Isolate shortcut or item position index.
    LSR
    LSR
    LSR
    LSR
    TAY
    LDA LevelInfo_ShortcutOrItemPosArray, Y    ; Get one of the 4 positions.
    PHA                         ; Save position.
    AND #$0F                    ; Low nibble * $10 = Y coordinate.
    ASL
    ASL
    ASL
    ASL
    TAY
    PLA                         ; Restore position.
    AND #$F0                    ; High nibble * $10 = X coordinate.
    RTS

; ----------------------------------------------------------------------------
; Clears an object slot's state, timer, knockback and invincibility and marks it uninitialized.
; IN:   A = 0 (the value stored); X = object slot.
; OUT:  ObjShoveDir, ObjShoveDistance, ObjTimer, ObjState, ObjInvincibilityTimer of X = A;
;       ObjUninitialized,X = $FF; ObjMetastate,X = 1.
; NOTE: Does not touch ObjType: the caller (Obj_SetTypeAndClear) stores the type first. Metastate 1
;       is the first spawn-cloud step.
; ----------------------------------------------------------------------------
Obj_ResetSlotFields:
    STA ObjShoveDir, X
    STA ObjShoveDistance, X
    STA ObjTimer, X
    STA ObjState, X
    STA ObjInvincibilityTimer, X
    LDA #$FF
    STA ObjUninitialized, X
    LDA #$01
    STA ObjMetastate, X
    RTS

; ----------------------------------------------------------------------------
; Flashes the screen during a bomb blast by switching the PPU mask's grayscale bit (bit 0).
; IN:   X = bomb object slot.
; OUT:  CurPpuMask bit 0 set when ObjTimer,X is $16 or $11, cleared when it is $12 or $0D; unchanged
;       at every other timer value.
; NOTE: Acts only in bomb state $13, whose timer starts at $18 (Bomb_StateTimes). Grayscale is therefore on
;       for timer values $16-$13 and $11-$0E: two flashes of 4 frames.
; ----------------------------------------------------------------------------
Bomb_UpdateFlashEffect:
    ; If state <> $13, return.
    LDA ObjState, X
    CMP #$13
    BNE @Exit

    ; Shift the grayscale bit (bit 0) out of the mask. One of the two paths below
    ; shifts a new bit 0 back in.
    LDA CurPpuMask
    LSR

    LDY ObjTimer, X

    ; At times $16 and $11, rotate carry=1 into the low bit.
    ; Carry is set by the CPY that compared equal.
    CPY #$16
    BEQ :+
    CPY #$11
    BNE @Clear
:
    ROL
    JMP @SetMask

@Clear:
    ; At times $12 and $D, shift left to make the low bit 0.
    CPY #$12
    BEQ :+
    CPY #$0D
    BNE @Exit
:
    ASL

@SetMask:
    ; Only at times $16, $11, $12, $D do we commit the mask
    ; we calculated.
    STA CurPpuMask

@Exit:
    RTS
