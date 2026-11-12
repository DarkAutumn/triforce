; ============================================================================
; FILE:     bank1/bomb-candle-wield.asm  BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3853-4132
; SUBJECTS: (none)
; TOPIC:    Place bomb, candle fire, weapon placement, bomb flash
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 WieldBomb                                       code
;   L0079 PlaceWeaponForPlayerStateAndAnimAndWeaponState  code  Params:
;   L0082 PlaceWeaponForPlayerStateAndAnim                code
;   L0095 PlaceWeaponForPlayerState                       code  TRIFORCE: This sets ObjState[0] (Link) = $10, which is the animation lock state.
;   L0106 PlaceWeapon                                     code  Places the weapon $10 pixels away from the player in the
;   L0131 L69AB_Exit                                      code
;   L0143 ChooseOffsetForDirectionH                       code  Params:
;   L0158 L69BE_Exit                                      code
;   L0166 WieldCandle                                     code  Returns:
;   L0215 GetShortcutOrItemXY                             code  Returns:
;   L0225 GetShortcutOrItemXYForRoom                      code  Params:
;   L0249 DestroyObject_WRAM                              code  Params:
;   L0261 UpdateBombFlashEffect                           code
; ============================================================================

WieldBomb:
    ; If there are no bombs in inventory, return.
    LDA InvBombs
    BEQ Dir_ToTarget_Return

    ; Look in slot $10. If it's empty or there's a fire
    ; (state = 0, or major state <> $10), then
    ; go activate a bomb.
    LDX #$10
    LDA ObjState, X
    BEQ @FoundSlot
    AND #$F0
    CMP #$10
    BNE @FoundSlot

    ; Else look at slot $11.
    ; If there's a bomb there (major state = $10), then return.
    INX
    LDA ObjState, X
    BEQ @FoundSlot
    AND #$F0
    CMP #$10
    BEQ L69AB_Exit

@FoundSlot:
    ; We found a slot that's empty or has a fire. It could be $10 or $11.
    ; If the other slot has a bomb that's not yet detonating (state < $13),
    ; then return.
    TXA
    EOR #$01
    TAY
    LDA a:ObjState, Y
    BEQ @Activate
    CMP #$13
    BCC L69AB_Exit

@Activate:
    ; We're using a bomb. Decrement the count.
    DEC InvBombs

    ; Play the "set a bomb" tune.
    LDA #$20
    STA Tune0Request

    ; Reset the object timer.
    LDA #$00
    STA ObjTimer, X

    ; Start in state $11.
    LDA #$11

; Params:
; A: initial state
; X: object index
;
PlaceWeaponForPlayerStateAndAnimAndWeaponState:
    STA ObjState, X

PlaceWeaponForPlayerStateAndAnim:
    ; Set player's animation counter to 1, so that it will roll over
    ; as soon as possible, causing movement frame to become 0 (legs apart),
    ; which is how we want to end item-use animations.
    ;
    ; Set player's state to $10 (wielding sword/item).
    LDA #$01
    STA ObjAnimCounter

; TRIFORCE: This sets ObjState[0] (Link) = $10, which is the animation lock state.
; TRIFORCE: Link state values during melee: $11 (swing, 12f) → $31 (finish, 1f) → $00 (controllable)
; TRIFORCE: Link state values during item use: $11 (6f) → $31 (6f) → $00 (controllable)
; TRIFORCE: Python: frame_skip_wrapper polls link_status==0 AND sword_animation==0
PlaceWeaponForPlayerState:
    LDA #$10
    STA ObjState

; Places the weapon $10 pixels away from the player in the
; player's direction.
;
; Params:
; A: $10 for right and down
; X: object index
;
PlaceWeapon:
    ; Set the offset choices to:
    ;  $10 for right and down in [01]
    ; -$10 for left and up in [02]
    LDY #$F0
    STA $01
    STY $02

    ; Set object's direction to the player's.
    LDA ObjDir
    STA ObjDir, X

    ; Set object's X to player's X + offset for direction.
    JSR ChooseOffsetForDirectionH
    ADC ObjX
    STA ObjX, X

    ; Set object's Y to player's Y + offset for direction.
    LDA ObjDir
    LSR                         ; Shift player's direction, so that it behaves like a horizontal if it's vertical.
    LSR
    JSR ChooseOffsetForDirectionH
    ADC ObjY
    STA ObjY, X

L69AB_Exit:
    RTS

; Params:
; A: direction
; [01]: value for right
; [02]: value for left
;
; Returns:
; A: chosen value, or 0 if not horizontal
; C: 0
;
ChooseOffsetForDirectionH:
    ; Set default value 0 in default address 0.
    LDY #$00
    STY $00
    AND #$03
    BEQ @GetValue               ; If vertical, go choose slot 0 (value 0).
    INY                         ; Direction is horizontal. Increment index to 1.
    AND #$01
    BNE @GetValue               ; If direction is right, go choose slot 1.
    INY                         ; Direction is left. Increment index to 2.

@GetValue:
    LDA $0000, Y                ; Load the offset for the calculated index.
    CLC                         ; Clear carry in preparation for adding offset to another value.

L69BE_Exit:
    RTS

; Returns:
; X: last object slot tested, regardless of whether it's empty
;
; Fails if no empty slot was found, or the blue candle was already used.
;
WieldCandle:
    ; Look in slots $10 and $11. If none are empty, return.
    ; Otherwise, X will have the empty slot found.
    ;
    ; Note that it's possible that we fail to find an empty slot,
    ; but the caller might find slot X occupied by a fire made earlier.
    LDX #$10
    LDA ObjState, X
    BEQ @NotInUse
    INX
    LDA ObjState, X
    BNE L69BE_Exit

@NotInUse:
    ; If we have the blue candle and it was used, then return.
    LDA InvCandle
    CMP #$01
    BNE :+
    LDA HasUsedCandle
    BNE L69BE_Exit
:
    ; Set the candle used.
    LDA #$01
    STA HasUsedCandle

    ; Reset the new fire object's movement info.
    ; Set quarter speed (q-speed) to $20 (half a pixel a frame).
    LDA #$00
    STA ObjGridOffset, X
    STA ObjPosFrac, X
    LDA #$20
    STA ObjQSpeedFrac, X

    ; Activate the fire object. Initial state = $21 (moving fire).
    LDA #$21
    STA ObjState, X
    LDA #$04                    ; Flame sound effect
    JSR PlayEffect

    ; Each animation frame lasts 4 frames.
    LDA #$04
    STA ObjAnimCounter, X
    JSR PlaceWeaponForPlayerState
    RTS

; Returns:
; A: X coordinate of shortcut (OW) or item (UW)
; Y: Y coordinate of shortcut (OW) or item (UW)
;
GetShortcutOrItemXY:
    LDY RoomId

; Params:
; Y: room ID
;
; Returns:
; A: X coordinate of shortcut (OW) or item (UW)
; Y: Y coordinate of shortcut (OW) or item (UW)
;
GetShortcutOrItemXYForRoom:
    LDA LevelBlockAttrsF, Y
    AND #$30                    ; Isolate shortcut or item position index.
    LSR
    LSR
    LSR
    LSR
    TAY
    LDA LevelInfo_ShortcutOrItemPosArray, Y    ; Get one of the 4 positions.
    PHA                         ; Save position.
    AND #$0F
    ASL
    ASL
    ASL
    ASL
    TAY
    PLA                         ; Restore position.
    AND #$F0
    RTS

; Params:
; A: 0
; X: object index
;
DestroyObject_WRAM:
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

UpdateBombFlashEffect:
    ; If state <> $13, return.
    LDA ObjState, X
    CMP #$13
    BNE @Exit

    ; The effect involves turning grayscale rendering on and off
    ; in PPUMASK.
    ;
    ; Start by setting A register to the PPU mask shifted right to
    ; throw away the grayscale bit.
    LDA CurPpuMask
    LSR

    ; Then, based on the timer value, shift in a 1 or 0 in the low bit.
    LDY ObjTimer, X

    ; At times $16 and $11, rotate carry=1 to the low bit.
    ; Carry will have been set, because of the comparison (CPY).
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
