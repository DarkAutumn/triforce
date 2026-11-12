; ============================================================================
; FILE:     bank7/sword-and-rod.asm      BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4362-4679
; SUBJECTS: (none)
; TOPIC:    The swing of the sword (object slot $0D) and of the magic rod (slot $12, shared with the
;           arrow): one state machine for both, their position and drawing, and the creation of the sword
;           shot or magic shot in slot $0E on the third state.
; SEE ALSO: bank5/wield-items.asm (WieldSword, WieldRod: initial states 1 and $31, first timer 5);
;           bank5/link-handle-input.asm (Link_HandleInput: A and B are read only while Link's state is 0);
;           bank7/object-animation.asm (AnimateLinkObjState: Link's state $10 -> $11 -> $31 -> 0);
;           bank7/sword-shot-magic-shot.asm (flight, spread, the book's fire); bank1/bomb-candle-wield.asm
;           (PlaceWeapon); bank1/sword-bomb-arrow-collision.asm (sword and rod damage);
;           bank7/boomerang-arrow-food.asm (UpdateArrowOrBoomerang, weapon frame and attribute tables).
; KEY FACTS:
;   - Sword states 1-5, rod states $31-$35 (same low nibble). Counting the frame of the button press as
;     frame 0: state 1 frames 0-3 (not drawn), state 2 frames 4-11 (fully extended), state 3 frame 12,
;     state 4 frame 13, state 5 frame 14 (not drawn), slot free (state 0) on frame 15. UpdateSwordOrRod
;   - The shot is made on the frame the weapon enters state 3 (frame 12). UpdateSwordOrRod
;   - The sword hits monsters only in state 2 (frames 4-11); the rod hits in every state $31-$35
;     (frames 0-14), damage $20. CheckMonsterSwordCollision, CheckMonsterArrowOrRodCollision
;     (bank1/sword-bomb-arrow-collision.asm)
;   - Sword shot: slot $0E must be empty, full hearts (low nibble of HeartValues) must equal containers
;     - 1 (high nibble) and HeartPartial must be >= $80, so up to just under half a heart may be
;     missing. Starts in state $10 at q-speed $C0 (3 pixels per frame). MakeSwordShot
;   - Magic shot (rod): no health test and no cost. Refused only while a magic shot (state bit 7) is in
;     slot $0E; it replaces a sword shot that is flying or spreading. State $80, q-speed $A0 (2.5
;     pixels per frame). UpdateSwordOrRod@MakeMagicShot, SetUpWeaponWithState
;   - Either shot starts $10 pixels ahead of Link and is dropped at once when it faces left or right
;     with X < $14 or X >= $EC. SetUpWeaponWithState
;   - Sword and rod have separate slots and neither wield routine tests the other. They are kept apart
;     only by Link's state: both set it to $10 and A/B are read only at state 0. A and B pressed on the
;     same frame start both; the rod's magic shot then replaces the sword shot (slot $12 is updated
;     after $0D and $0E). UpdateSwordOrRod, WieldSword, WieldRod, Link_HandleInput
;   - Each state change also writes Link's animation counter (8 entering state 2, else 1), which times
;     Link's own state: $11 through frame 11, $31 on frame 12, 0 on frame 13. UpdateSwordOrRod
;   - The weapon is repositioned from Link's position and direction every frame, so it turns and moves
;     with him. PlayerToWeaponOffsetsX, PlayerToWeaponOffsetsY
;   - Sword palette row = 3 + InvSword (4 wooden, 5 white, 6 magic); the rod uses row 5.
;     UpdateSwordOrRod@CalcSwordAttrs
;   - IsSwordShotForced would allow the sword shot at any health, but no code ever sets it. MakeSwordShot
; CONTENTS: (generated; L = line in this file)
;   L0057 UpdateRodOrArrow        code  Updates object slot $12: a rod (state $3x) goes to the sword routine, an arrow to the arrow one.
;   L0068 PlayerToWeaponOffsetsX  data  TABLE: X offset from Link to the sword or rod; index = (state - 1) * 4 + reverse direction index
;   L0074 PlayerToWeaponOffsetsY  data  TABLE: Y offset from Link to the sword or rod; index = (state - 1) * 4 + reverse direction index
;   L0093 UpdateSwordOrRod        code  Runs the swing of the sword (slot $0D) or rod (slot $12): states 1 to 5, drawing, and the shot.
;   L0265 SetUpWeaponWithState    code  Starts the shot in slot $0E: state from A, $10 pixels ahead of Link, speed by kind of shot.
;   L0299 L1F854_Exit             code  Shared return of UpdateSwordOrRod, SetUpWeaponWithState and MakeSwordShot.
;   L0307 ResetObjState           code  Sets ObjState,X to 0, which frees a weapon slot or puts a monster back in its state 0.
;   L0320 MakeSwordShot           code  Makes a sword shot in slot $0E if the slot is empty and Link's health is full.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates object slot $12: a rod (state $3x) goes to the sword routine, an arrow to the arrow one.
; IN:   X = $12.
; NOTE: The rod and the arrow share this slot, so neither can be used while the other is active.
; ----------------------------------------------------------------------------
UpdateRodOrArrow:
    ; The rod uses states $31-$35. The arrow uses $1x (flying) and $2x (spark); state 0 (empty)
    ; also goes to the arrow routine, which returns.
    LDA ObjState, X
    AND #$F0
    CMP #$30
    BCS UpdateSwordOrRod
    JMP UpdateArrowOrBoomerang

; TABLE: X offset from Link to the sword or rod; index = (state - 1) * 4 + reverse direction index
; (0 up, 1 down, 2 left, 3 right); value = signed pixels added to Link's X.
PlayerToWeaponOffsetsX:
    .BYTE $FF, $01, $00, $F8, $FF, $01, $F5, $0B    ; State 1: -1, +1, 0, -8.  State 2: -1, +1, -11, +11
    .BYTE $FF, $01, $F9, $07, $FF, $01, $FD, $03    ; State 3: -1, +1, -7, +7. State 4: -1, +1, -3, +3

; TABLE: Y offset from Link to the sword or rod; index = (state - 1) * 4 + reverse direction index
; (0 up, 1 down, 2 left, 3 right); value = signed pixels added to Link's Y.
PlayerToWeaponOffsetsY:
    .BYTE $F7, $F2, $F5, $F5, $F6, $0D, $03, $03    ; State 1: -9, -14, -11, -11. State 2: -10, +13, +3, +3
    .BYTE $F7, $09, $03, $03, $FF, $05, $03, $03    ; State 3: -9, +9, +3, +3.    State 4: -1, +5, +3, +3

; ----------------------------------------------------------------------------
; Runs the swing of the sword (slot $0D) or rod (slot $12): states 1 to 5, drawing, and the shot.
; IN:   X = $0D (sword, states 1-5) or $12 (rod, states $31-$35). ObjAnimCounter,X = frames left in
;       the state (5 when wielded).
; OUT:  Advances ObjState,X; writes ObjX/ObjY/ObjDir,X and Link's ObjAnimCounter; entering state 3
;       makes a sword shot (MakeSwordShot) or a magic shot in slot $0E.
; NOTE: Frames from the button press (frame 0). The wield routine stores timer 5 during Link's
;       update, and this routine runs later in the same frame and already counts it down to 4. So
;       state 1 is seen at the end of 4 frames (0-3): the 5th decrement, on frame 4, enters state
;       2 with timer 8. State 2 frames 4-11, state 3 frame 12, state 4 frame 13, state 5 frame
;       14, state 0 on frame 15. Drawn in states 2-4 only (frames 4-13). Monsters are updated
;       after the weapons, so their collision tests see these end-of-frame states.
;       Link's state, driven by the animation counter written here, is 0 again on frame 13, so A
;       or B can start the other weapon on frame 14 and the same weapon on frame 16.
; ----------------------------------------------------------------------------
UpdateSwordOrRod:
    ; Low nibble of state = 0: the slot is empty (or, for slot $12, holds nothing in a rod state).
    LDA ObjState, X
    AND #$0F
    BEQ @Exit

    ; Count down the state's timer. While it runs, only reposition and draw.
    DEC ObjAnimCounter, X
    BNE @DrawSwordOrRod

    ; The timer expired. The next state lasts 8 frames if the current one is state 1 (so state 2
    ; lasts 8), else 1 frame.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$08
    DEY
    BEQ :+
    LDA #$01
:
    ; Give the same count to Link's animation counter, which times his state ($11 until state 2
    ; ends, then $31 for one frame, then 0; see AnimateLinkObjState), and to the weapon's timer.
    STA ObjAnimCounter
    STA ObjAnimCounter, X

    ; Go to the next state.
    INC ObjState, X

    ; State 6 does not exist: free the slot (state 0). Otherwise go draw.
    LDA ObjState, X
    AND #$0F
    CMP #$06
    BCC @DrawSwordOrRod
    JSR ResetObjState

@Exit:
    RTS

@DrawSwordOrRod:
    ; [0F] = 0: no horizontal flip.
    LDA #$00
    STA $0F

    ; State 5: return without moving or drawing the weapon.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$FC
    CPY #$05
    BEQ @Exit

@Add4:
    ; [00] = (state - 1) * 4: the start of the state's set in the offset tables.
    CLC
    ADC #$04
    DEY
    BNE @Add4
    STA $00

    ; The weapon takes Link's direction every frame; he may have turned.
    LDA ObjDir
    STA ObjDir, X

    ; Y = [00] + reverse direction index: the entry in the offset tables.
    JSR GetOppositeDir
    TYA
    CLC
    ADC $00
    TAY

    ; Weapon X = Link's X + offset; copy to [00] for drawing.
    LDA ObjX
    CLC
    ADC PlayerToWeaponOffsetsX, Y
    STA ObjX, X
    STA $00

    ; Weapon Y = Link's Y + offset; copy to [01] for drawing.
    LDA ObjY
    CLC
    ADC PlayerToWeaponOffsetsY, Y
    STA ObjY, X
    STA $01

    ; Direction for choosing the image: up (8) in state 1, else the weapon's direction.
    ; (State 1 is never drawn; see the test below.)
    LDA ObjState, X
    AND #$0F
    TAY
    LDA #$08
    DEY
    BEQ @CalcAttrs
    LDA ObjDir, X

@CalcAttrs:
    ; [0C] = the weapon's frame for that direction: 0 vertical, 1 horizontal.
    JSR GetOppositeDir
    LDA RDirectionToWeaponFrame, Y
    STA $0C

    ; Sprite attributes. The base attribute flips the image vertically for "down".
    ; Rod (any slot but $0D): base OR 1 = palette row 5.
    LDA RDirectionToWeaponBaseAttribute, Y
    CPX #$0D
    BEQ @CalcSwordAttrs
    ORA #$01
    JMP @SetAttrs

@CalcSwordAttrs:
    ; Sword: base + InvSword - 1 = palette row 4 wooden, 5 white, 6 magic.
    CLC
    ADC InvSword
    SEC
    SBC #$01

@SetAttrs:
    JSR Anim_SetSpriteDescriptorAttributes

    ; Facing left (reverse direction index 2): [0F] = 1 to flip the image horizontally.
    CPY #$02
    BNE :+
    INC $0F
:
    ; State 1: return. The weapon has been positioned but is not shown.
    LDA ObjState, X
    AND #$0F
    CMP #$01
    BEQ L1F854_Exit

    ; Y = item slot whose image is drawn.
    LDY #$00                    ; Item slot 0: sword
    CPX #$0D
    BEQ :+
    LDY #$08                    ; Item slot 8: rod
:
    JSR Anim_WriteItemSprites

    ; Only state 3 goes on to make a shot. The state lasts one frame (frame 12 of the swing), so
    ; this happens once per swing.
    LDA ObjState, X
    AND #$0F
    CMP #$03
    BNE L1F854_Exit

    ; The sword (slot $0D) goes to MakeSwordShot; the rod (slot $12) makes a magic shot here.
    CPX #$12
    BNE MakeSwordShot

    ; Magic shot, in the shot slot $0E.
    LDX #$0E

    ; If the slot holds a magic shot (state bit 7 set), return: one magic shot at a time.
    ; An empty slot or a sword shot (flying $10 or spreading $11) is overwritten.
    LDA ObjState, X
    BEQ @MakeMagicShot
    ASL
    BCS L1F854_Exit

@MakeMagicShot:
    ; Play the "magic shot" tune.
    LDA #$04
    STA Tune0Request

    ; State $80 = magic shot flying.
    LDA #$80

; ----------------------------------------------------------------------------
; Starts the shot in slot $0E: state from A, $10 pixels ahead of Link, speed by kind of shot.
; IN:   A = state ($10 sword shot, $80 magic shot); X = $0E.
; OUT:  ObjState,X = A, or 0 when the shot starts too close to the left or right screen edge.
; NOTE: Q-speed $C0 (3 pixels per frame) for the sword shot, $A0 (2.5) for the magic shot.
; ----------------------------------------------------------------------------
SetUpWeaponWithState:
    STA ObjState, X

    ; Place the shot $10 pixels from Link in his direction, and give it his direction.
    LDA #$10
    JSR PlaceWeapon

    ; A shot facing left or right with X < $14 or X >= $EC is dropped at once (state 0).
    ; Its sound has already been requested.
    LDA ObjDir, X
    AND #$03
    BEQ @ChooseSpeed
    LDA ObjX, X
    CMP #$14
    BCC ResetObjState
    CMP #$EC
    BCS ResetObjState

@ChooseSpeed:
    ; Q-speed $A0 if state bit 7 is set (magic shot), else $C0 (sword shot).
    LDY #$C0
    LDA ObjState, X
    ASL
    BCC :+
    LDY #$A0
:
    TYA
    STA ObjQSpeedFrac, X

    ; The shot starts with Link's grid offset.
    LDA ObjGridOffset
    STA ObjGridOffset, X

; Shared return of UpdateSwordOrRod, SetUpWeaponWithState and MakeSwordShot.
L1F854_Exit:
    RTS

; ----------------------------------------------------------------------------
; Sets ObjState,X to 0, which frees a weapon slot or puts a monster back in its state 0.
; IN:   X = object slot.
; OUT:  A = 0.
; ----------------------------------------------------------------------------
ResetObjState:
    LDA #$00
    STA ObjState, X
    RTS

; ----------------------------------------------------------------------------
; Makes a sword shot in slot $0E if the slot is empty and Link's health is full.
; IN:   Called (by branch) from UpdateSwordOrRod when the sword enters state 3.
; OUT:  X = $0E. On success ObjState+$0E = $10 and the sword shot sample is requested.
; NOTE: "Full" = full hearts equal to heart containers - 1 and HeartPartial >= $80: Link may be
;       short by less than half a heart. Any shot still in slot $0E (flying or spreading sword
;       shot, or a magic shot) blocks a new sword shot.
; ----------------------------------------------------------------------------
MakeSwordShot:
    ; Switch to the shot slot $0E.
    LDX #$0E

    ; If the slot is in use (state <> 0), return.
    LDA ObjState, X
    BNE L1F854_Exit

    ; If IsSwordShotForced is set, skip the health test. Its three writers (FormatHeartsInTextBuf,
    ; ClearRoomHistory and the file menu's init) all store 0, so the test always applies.
    LDA IsSwordShotForced
    BNE @SetUp

    ; HeartValues: high nibble = heart containers - 1, low nibble = full hearts.
    ; If the two nibbles differ, a whole heart or more is missing: return.
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
    BNE L1F854_Exit

    ; HeartPartial is the fill of the last heart ($FF full). Below $80 (half), return.
    LDA HeartPartial
    CMP #$80
    BCC L1F854_Exit

@SetUp:
    LDA #$01                    ; Sample 1: sword shot
    JSR PlaySample

    ; Start the shot in state $10 (flying; $11 is the spread, see SetShotSpreadingState).
    LDA #$10
    BNE SetUpWeaponWithState
