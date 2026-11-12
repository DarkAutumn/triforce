; ============================================================================
; FILE:     bank7/sword-shot-magic-shot.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 3450-3761
; SUBJECTS: (none)
; TOPIC:    Link's sword shot (beam) and the rod's magic shot in object slot $0E: flight, what happens when
;           the shot is stopped (the beam bursts into four corners, the magic shot lights a fire if Link
;           has the book), and drawing (also used to draw monsters' sword and magic shots).
; SEE ALSO: bank7/sword-and-rod.asm (MakeSwordShot, SetUpWeaponWithState: when a shot is made, its speed);
;           bank1/shot-aim-boomerang-speed.asm (MoveShot); bank1/room-bounds.asm (BoundByRoom: where a
;           shot stops); bank1/sword-bomb-arrow-collision.asm (CheckMonsterSwordShotOrMagicShotCollision:
;           damage, and the call to HandleShotBlocked on a hit); bank1/bomb-candle-wield.asm (WieldCandle);
;           bank7/fire.asm (the fire the book makes); bank4/monster-shots-fireballs.asm (monsters' shots).
; KEY FACTS:
;   - Slot $0E state: 0 = no shot; $10 = sword shot flying; $11 = sword shot spreading; $80 = magic shot
;       flying. Bit 7 = magic shot, bit 0 = spreading. UpdateSwordShotOrMagicShot, HandleShotBlocked
;   - Only one shot exists at a time (one slot), and the slot stays busy while the beam spreads: 22 frames
;       after the frame on which it was stopped. SpreadShot
;   - A flying shot stops only at the room bounds (BoundByRoom inside MoveShot; tiles are not tested) or
;       when the collision code reports a hit on a monster. UpdateSwordShotOrMagicShot, HandleShotBlocked
;   - A stopped sword shot spreads: four corner sprites move diagonally 1 pixel a frame, from 2 to 23
;       pixels away from the point of impact. SetShotSpreadingState, SpreadShot
;   - A stopped magic shot disappears at once. With the book (InvBook nonzero) it first makes a standing
;       fire (state $22, ObjTimer $4F) at its position in fire slot $10 or $11. HandleShotBlocked
;   - The book's fire needs no candle and does not use up the blue candle's once-per-room use
;       (HasUsedCandle is saved and restored). HandleShotBlocked
;   - While the shot is spreading, the slot's ObjDir holds the corner offset ($FE down to $E9), not a
;       direction. SetShotSpreadingState, SpreadShot
; CONTENTS: (generated; L = line in this file)
;   L0043 SwordShotSpreadBaseAttr     data  TABLE: sprite attributes of the four corners of a spreading sword shot; index = corner loop index in
;   L0056 UpdateSwordShotOrMagicShot  code  Updates Link's sword shot or magic shot each frame: moves it, stops it at the room bounds, draws it.
;   L0091 DrawSwordShotOrMagicShot    code  Draws a flying sword shot or magic shot, flashing through the four sprite palettes.
;   L0166 HandleShotBlocked           code  Stops Link's shot: a sword shot starts spreading; a magic shot vanishes (leaving a fire with the book).
;   L0219 DeactivateLinkShot          code  Deactivates Link's shot in slot $0E; second entry of HandleShotBlocked and the end of SpreadShot.
;   L0223 DeactivateShot              code  Deactivates the shot in slot X (state = 0).
;   L0233 SetShotSpreadingState       code  Puts a stopped sword shot into the spreading state ($10 becomes $11) with corner offset -2.
;   L0249 SpreadShot                  code  Draws the four corners of a spreading sword shot, moves them 1 pixel outward, and ends the shot.
;   L0379 L1F49F_Exit                 code  Shared return (RTS) of SpreadShot, UpdateBoomerangOrFood and UpdateArrowOrBoomerang.
; ============================================================================

; TABLE: sprite attributes of the four corners of a spreading sword shot; index = corner loop index in
;   SpreadShot; value = OAM flip bits ($40 horizontal, $80 vertical).
;   0: top-right ($40), 1: bottom-right ($C0), 2: bottom-left ($80), 3: top-left ($00, the image as stored).
SwordShotSpreadBaseAttr:
    .BYTE $40, $C0, $80, $00

; ----------------------------------------------------------------------------
; Updates Link's sword shot or magic shot each frame: moves it, stops it at the room bounds, draws it.
; IN:   X = object slot ($0E; called once a frame by UpdateMode5Play)
; NOTE: State 0 = inactive; bit 0 set = sword shot spreading (SpreadShot); bit 7 set = magic shot.
;       Speed is ObjQSpeedFrac (set by SetUpWeaponWithState: $C0 = 3 pixels a frame for the sword shot,
;       $A0 = 2.5 for the magic shot; MoveObject adds the q-speed to the position fraction 4 times a frame).
;       The shot is stopped here only by the room bounds (MoveShot calls BoundByRoomWithA, which zeroes
;       [0F]); walls and other tiles inside the bounds do not stop it. A hit on a monster stops it from
;       the collision code, which calls HandleShotBlocked itself.
; ----------------------------------------------------------------------------
UpdateSwordShotOrMagicShot:
    ; If the shot is not active, return.
    LDA ObjState, X
    BEQ L1F36A_Exit

    ; If bit 0 of the state is set, the sword shot is spreading out.
    LSR
    BCC :+
    JMP SpreadShot
:
    ; This test has no effect: both outcomes continue at the next instruction.
    LDA ObjGridOffset, X
    BNE :+
:
    ; Move the shot in its direction. [0E] is not set up here, so whether MoveShot accumulates
    ; ObjGridOffset depends on the value left in [0E] by earlier code.
    LDA ObjDir, X
    JSR MoveShot

    ; [0F] = 0 means the shot was at a room bound and did not move: go handle it being blocked.
    LDA $0F
    BEQ HandleShotBlocked

    ; If the low 3 bits of the grid offset are 0, reset the grid offset to 0.
    LDA ObjGridOffset, X
    AND #$07
    BNE DrawSwordShotOrMagicShot
    STA ObjGridOffset, X

; ----------------------------------------------------------------------------
; Draws a flying sword shot or magic shot, flashing through the four sprite palettes.
; IN:   X = object slot: $0E for Link's shot, or a monster slot holding a monster's shot
;       (object type $57 sword shot, $58/$59 magic shot; called by the monster shot update in bank 4)
; NOTE: The palette is (FrameCounter AND 3), so it changes every frame.
; ----------------------------------------------------------------------------
DrawSwordShotOrMagicShot:
    ; Prepare the sprite position: [00] = X, [01] = Y, [0F] = 0 (no horizontal flip).
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; If the direction is horizontal, draw the sprites 3 pixels lower.
    LDA ObjDir, X
    PHA
    AND #$03
    BEQ @Flash
    LDA $01
    CLC
    ADC #$03
    STA $01

@Flash:
    PLA

    ; Set the sprite attributes to (base attribute OR (frame counter AND 3)).
    ; This makes the shot flash by cycling all the palettes, one each frame.
    JSR GetOppositeDir          ; Only for Y = reverse direction index: 0 up, 1 down, 2 left, 3 right.
    LDA FrameCounter
    AND #$03
    ORA RDirectionToWeaponBaseAttribute, Y
    JSR Anim_SetSpriteDescriptorAttributes

    ; Set [0C] to the frame for the direction: 0 vertical, 1 horizontal.
    LDA RDirectionToWeaponFrame, Y
    STA $0C

    ; If the direction is left, flip the right-facing image by setting [0F] to 1.
    CPY #$02
    BNE :+
    INC $0F
:
    ; Choose the item sprite slot in Y: $22 = sword shot image, $23 = magic shot image.
    ; A monster's shot (slot < $0D) chooses by object type: $57 (sword shot) => $22; any other
    ; type => $23 (the only other types the caller sends here are $58 and $59, magic shots).
    LDY #$22
    CPX #$0D
    BCS @PlayerShot
    LDA ObjType, X
    CMP #$57
    BEQ @WriteSprites
    BNE @WriteMagicSprites

@PlayerShot:
    ; Link's shot chooses by state: bit 7 set (magic shot) => $23; clear (sword shot) => $22.
    LDA ObjState, X
    ASL
    BCC @WriteSprites

@WriteMagicSprites:
    LDY #$23

@WriteSprites:
    JMP Anim_WriteItemSprites

; ----------------------------------------------------------------------------
; Stops Link's shot: a sword shot starts spreading; a magic shot vanishes (leaving a fire with the book).
; IN:   X = shot object slot (always $0E: from UpdateSwordShotOrMagicShot at a room bound, and from
;       CheckMonsterSwordShotOrMagicShotCollision when the shot hits a monster)
; OUT:  X = $0E if the magic shot path ran with the book; else unchanged. A = 0 after a deactivation.
; NOTE: Without the book (InvBook = 0) the magic shot is simply deactivated.
;       With the book, WieldCandle is called to claim fire slot $10 or $11 (it plays the flame sound
;       effect when it makes a fire). HasUsedCandle is cleared
;       for the call and restored after it, so the fire needs no candle, is not refused after the blue
;       candle was used in this room, and does not count as that use. Link's state is also restored,
;       so he is not put into the item-use state.
;       The new fire becomes a standing fire (state $22) at the shot's position, with the shot's
;       direction and ObjTimer = $4F (79 frames); a candle's standing fire gets $3F in UpdateFire.
;       Quirk: if both fire slots are busy, WieldCandle returns with X = $11. If slot $11 then holds
;       a moving candle fire (state $21), the test below passes and that fire is turned into the
;       standing fire at the shot's position. Any other state in slot $11 (a bomb, a standing fire):
;       no fire is made.
; ----------------------------------------------------------------------------
HandleShotBlocked:
    ; Bit 7 of the state clear: a sword shot. Go start spreading it.
    LDA ObjState, X
    ASL
    BCC SetShotSpreadingState

    ; This is a magic shot.
    ;
    ; If Link does not have the book, go deactivate the shot object.
    LDA InvBook
    BEQ DeactivateShot

    ; Try to activate a fire object. Temporarily mark the candle unused, so that success depends
    ; only on finding an empty fire slot.
    ;
    ; WieldCandle also sets Link's state to $10 (using an item), which is not wanted here.
    ;
    ; For both these reasons, save Link's state and HasUsedCandle, and restore them afterward.
    LDA ObjState
    PHA
    LDA HasUsedCandle
    PHA
    LDA #$00
    STA HasUsedCandle
    JSR WieldCandle             ; Returns X = the fire slot it used, or $11 if it found none.
    PLA
    STA HasUsedCandle
    PLA
    STA ObjState

    ; State $21 (moving fire) in slot X means WieldCandle made a fire, or that slot $11 already held
    ; a moving fire (see NOTE above). Any other value: go deactivate the shot object only.
    LDA ObjState, X
    CMP #$21
    BNE DeactivateLinkShot

    ; Make the state $22: a standing fire.
    INC ObjState, X

    ; Copy the shot's position and direction to the fire.
    LDY #$0E
    LDA a:ObjX, Y
    STA ObjX, X
    LDA a:ObjY, Y
    STA ObjY, X
    LDA a:ObjDir, Y
    STA ObjDir, X

    ; The fire's timer is $4F frames.
    LDA #$4F
    STA ObjTimer, X

; Deactivates Link's shot in slot $0E; second entry of HandleShotBlocked and the end of SpreadShot.
DeactivateLinkShot:
    LDX #$0E

; Deactivates the shot in slot X (state = 0).
DeactivateShot:
    JMP ResetObjState

; ----------------------------------------------------------------------------
; Puts a stopped sword shot into the spreading state ($10 becomes $11) with corner offset -2.
; IN:   X = shot object slot
; NOTE: From here on the slot's ObjDir is not a direction: it holds the negative offset of the
;       left and top corners from the shot's position (the centre of the spread). The right and
;       bottom corners use the same distance with the opposite sign.
; ----------------------------------------------------------------------------
SetShotSpreadingState:
    INC ObjState, X             ; Set bit 0: spreading.
    LDA #$FE                    ; Offset -2 pixels.
    STA ObjDir, X
    RTS

; ----------------------------------------------------------------------------
; Draws the four corners of a spreading sword shot, moves them 1 pixel outward, and ends the shot.
; IN:   X = shot object slot; ObjDir,X = negative corner offset ($FE on the first frame)
; NOTE: Runs on 22 frames, with offsets $FE down to $E9 (2 to 23 pixels on each axis). After the
;       frame drawn at $E9 the offset becomes $E8 and the slot is deactivated, so a new sword shot
;       can be made from the next frame on. The shot does no damage in this state
;       (CheckMonsterSwordShotOrMagicShotCollision returns when bit 0 of the state is set).
;       A corner is not drawn when its X would wrap around the screen, or in the underworld when
;       its Y is less than $3E or at least $E8.
; ----------------------------------------------------------------------------
SpreadShot:
    ; Copy the negative offset in ObjDir to [02] (X offset) and [03] (Y offset).
    ; These will be negated in turn for each corner.
    LDA ObjDir, X
    STA $02
    STA $03

    ; Reset [0F] to not flip horizontally.
    LDA #$00
    STA $0F

    ; Loop 4 times to draw each corner of the spreading sword shot.
    ; The corners are: top-left (index 3), bottom-left (2), bottom-right (1), top-right (0).
    LDY #$03

@DrawShotCorner:
    TYA                         ; Save the loop index.
    PHA

    ; Save X offset [02] and Y offset [03].
    LDA $02
    PHA
    LDA $03
    PHA

    ; Set the sprite attributes for this corner: flip bits OR (frame counter AND 3).
    ;
    ; The flip bits mirror the one corner image. The frame counter makes the shot flash by
    ; cycling through the four sprite palettes, one each frame.
    LDA FrameCounter
    AND #$03
    ORA SwordShotSpreadBaseAttr, Y
    JSR Anim_SetSpriteDescriptorAttributes

    ; Add the object's X (the center of the spread) and the
    ; current offset in [02] to set sprite X in [00].
    LDA ObjX, X
    CLC
    ADC $02
    STA $00

    ; If sprite X >= object X and sprite X >= $FC, skip this corner.
    ; Else calculate the distance from object X to sprite X.
    CMP ObjX, X
    BCC @ReverseSub
    CMP #$FC
    BCS @NextCorner
    SEC
    SBC ObjX, X
    JMP @CheckDistance

@ReverseSub:
    LDA ObjX, X
    SEC
    SBC $00

@CheckDistance:
    ; If the distance >= $20, skip this corner. The offset never exceeds 23 pixels, so this
    ; happens only when sprite X wrapped around the screen edge.
    CMP #$20
    BCS @NextCorner

    ; Add the object's Y (the center of the spread) and the
    ; current offset in [03] to set sprite Y in [01].
    LDA ObjY, X
    CLC
    ADC $03
    STA $01

    ; If in the underworld (CurLevel <> 0), and (sprite Y < $3E or >= $E8), then skip this corner.
    LDY CurLevel
    BEQ @Draw
    CMP #$3E
    BCC @NextCorner
    CMP #$E8
    BCS @NextCorner

@Draw:
    ; Draw the corner.
    ; Pass [0C] = frame 2 (the spread corner image), Y = item sprite slot $23 (the magic shot's
    ; slot; its third frame is the corner image).
    LDA #$02
    STA $0C
    LDY #$23
    JSR Anim_WriteItemSprites

@NextCorner:
    ; Prepare to process the next corner.
    ;
    ; First, restore X offset [02] and Y offset [03].
    PLA
    STA $03
    PLA
    STA $02

    ; Take turns negating the X offset or the Y offset for the next round.
    ; - loop index 3: negate Y offset [03]
    ; - loop index 2: negate X offset [02]
    ; - loop index 1: negate Y offset [03]
    ; - loop index 0: negates [00], which no longer matters
    PLA                         ; Get the loop index.
    PHA
    TAY
    CPY #$01                    ; Loop indexes 3 and 2 are also the zero-page addresses to negate.
    BNE :+                      ; But for loop index 1,
    LDY #$03                    ; use address 3 to negate the Y offset.
:
    LDA $0000, Y
    EOR #$FF
    CLC
    ADC #$01
    STA $0000, Y

    ; Restore the loop index, decrement it, and loop again if >= 0.
    PLA
    TAY
    DEY
    BPL @DrawShotCorner

    ; Decrease the negative offset (held in ObjDir), so that the corners are 1 pixel farther
    ; from the center point next frame.
    DEC ObjDir, X

    ; Once the negative offset reaches $E8, go deactivate the shot object.
    LDA ObjDir, X
    CMP #$E8
    BNE L1F49F_Exit
    JMP DeactivateLinkShot

; Shared return (RTS) of SpreadShot, UpdateBoomerangOrFood and UpdateArrowOrBoomerang.
L1F49F_Exit:
    RTS
