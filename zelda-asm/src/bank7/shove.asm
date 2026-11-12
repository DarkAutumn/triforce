; ============================================================================
; FILE:     bank7/shove.asm              BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2323-2494
; SUBJECTS: Shove
; TOPIC:    Knockback movement: moves a shoved object (Link or a monster) up to 4 pixels per frame in
;           its shove direction until the distance runs out or something blocks it.
; SEE ALSO: bank1/shove-knockback.asm (Shove_Begin: sets direction, distance and invincibility);
;           bank7/tile-collision.asm (Tile_GetAheadOfObj); bank1/room-bounds.asm (Room_BoundDirA);
;           bank1/modeb-cave-save-sprites.asm (Person_BlockLinkMovingUp); bank7/mode5-play-update.asm
;           (Obj_SnapToGrid); bank4/walker-wanderer-common.asm and other monster files (callers).
; KEY FACTS:
;   - Speed: 4 pixels per frame, one pixel at a time, each pixel tested (Shove_MovePixels). Shove_Begin gives
;     Link $20 pixels (8 frames) and a monster $40 pixels (16 frames) when nothing stops the shove.
;   - The first call of a shove (bit 7 of ObjShoveDir set) only clears bit 7 and validates the direction;
;     it does not move the object (Shove_Update).
;   - A shove perpendicular to the object's facing axis is allowed only at grid offset 0. Off the grid
;     a monster's shove is cancelled; Link's is redirected to the opposite of his facing direction
;     (Shove_Update@CheckPerpendicularShove).
;   - Tiles are tested only at grid offset 0, after snapping the object to the grid; an unwalkable tile
;     ahead ends the shove (Shove_MovePixels@LoopShovePixel).
;   - The room bounds end the shove at any grid offset (Shove_MovePixels@CheckBoundary).
;   - If object slot 1 holds the Grumble Goriya ($36) or a dungeon person ($4B-$52), an upward shove
;     ends while Link's Y < $8E; the test reads Link's Y whichever object is shoved
;     (Shove_MovePixels@CheckBlocked).
;   - Each pixel also moves ObjGridOffset by 1; it is reset to 0 at a multiple of $10, and for Link also
;     at a multiple of 8 (Shove_MovePixels@ChooseSpeed).
;   - When the shove ends, ObjShoveDir and ObjShoveDistance are both cleared (Shove_Reset).
;   - Bit 6 of ObjShoveDir is never tested here: every use masks with $0F, $0C, $05 or $03
;     (Shove_Update, Shove_MovePixels).
; CONTENTS: (generated; L = line in this file)
;   L0050 Shove_Update             code  Advances an object's knockback by one frame: validate on the first call, then move up to 4 pixels.
;   L0103 Shove_Reset              code  Ends an object's knockback: clears ObjShoveDir,X and ObjShoveDistance,X.
;   L0107 Shove_SetDirAndDistance  code  Second entry: stores A (callers pass 0) in ObjShoveDir,X and ObjShoveDistance,X. A is unchanged.
;   L0124 Shove_MovePixels         code  Moves a shoved object up to 4 pixels, one at a time, stopping the shove at the first obstacle.
; ============================================================================

; ----------------------------------------------------------------------------
; Advances an object's knockback by one frame: validate on the first call, then move up to 4 pixels.
; IN:   X = object slot; ObjShoveDir,X (bit 7 = first call, low nibble = direction);
;       ObjShoveDistance,X = pixels left.
; OUT:  ObjX/ObjY and ObjGridOffset moved; ObjShoveDir and ObjShoveDistance cleared when the shove ends.
;       Uses [02], [03], [0F].
; NOTE: Callers call this while ObjShoveDir,X is nonzero. The first call does not move the object.
; NOTE: First call: if the shove axis differs from the axis the object faces and its grid offset is
;       not 0, a monster's shove is cancelled and Link's direction is replaced by the opposite of ObjDir.
;       Shove_Begin chooses Link's shove axis from ObjDir when he is off the grid, so Link's replacement
;       is reached only if ObjDir or the grid offset changed since Shove_Begin ran (unverified whether
;       that ever happens; upstream marked the block as never seen executing).
; ----------------------------------------------------------------------------
Shove_Update:
    ; Bit 7 clear: not the first call of this shove. Go move.
    LDA ObjShoveDir, X
    ASL
    BCC @MoveIfNotDone

    ; First call: clear bit 7. A = the shove direction (bit 6, if set, stays in).
    LSR
    STA ObjShoveDir, X

    ; ObjDir 1 or 2 (also 0): the object faces horizontally.
    LDY ObjDir, X
    CPY #$03
    BCC @FacingHorizontally

    ; The object faces vertically. A vertical shove is fine: return.
    AND #$03
    BEQ @Exit

@CheckPerpendicularShove:
    ; The shove is perpendicular to the facing axis. It is allowed at grid offset 0.
    LDA ObjGridOffset, X
    BEQ @Exit

    ; Off the grid: cancel the shove of any object but Link.
    CPX #$00
    BNE Shove_Reset

    ; Link: shove him opposite to the direction he faces instead.
    ; (Upstream marked these three instructions as never seen executing; see the routine header.)
    LDA ObjDir
    JSR Dir_GetOpposite
    STA ObjShoveDir

@Exit:
    RTS

@FacingHorizontally:
    ; A horizontal shove is fine: return. A vertical one is perpendicular: check the grid offset.
    AND #$0C
    BNE @CheckPerpendicularShove
    RTS

@MoveIfNotDone:
    ; If pixels are left, move. Otherwise the shove is over: fall through and clear it.
    LDA ObjShoveDistance, X
    BNE Shove_MovePixels

; ----------------------------------------------------------------------------
; Ends an object's knockback: clears ObjShoveDir,X and ObjShoveDistance,X.
; IN:   X = object slot.
; OUT:  A = 0.
; ----------------------------------------------------------------------------
Shove_Reset:
    LDA #$00

; Second entry: stores A (callers pass 0) in ObjShoveDir,X and ObjShoveDistance,X. A is unchanged.
Shove_SetDirAndDistance:
    STA ObjShoveDir, X
    STA ObjShoveDistance, X
    RTS

; ----------------------------------------------------------------------------
; Moves a shoved object up to 4 pixels, one at a time, stopping the shove at the first obstacle.
; IN:   X = object slot; ObjShoveDir,X low nibble = direction; ObjShoveDistance,X = pixels left (not 0).
; OUT:  position, ObjGridOffset,X and ObjShoveDistance,X updated; shove cleared if blocked.
;       Uses [02] (step +1 or -1), [03] (pixel counter), [0F] (direction for the tests).
; NOTE: Per pixel: at grid offset 0 the object is snapped to the grid and the tile ahead is tested
;       (blocked when the tile >= RoomFirstUnwalkableTile); the room bounds are tested always; with the
;       Grumble Goriya ($36) or a dungeon person ($4B-$52) in slot 1, an upward shove stops while
;       Link's Y < $8E.
; NOTE: The distance is not tested inside the loop: if fewer than 4 pixels were left, the loop still
;       moves 4 and ObjShoveDistance wraps below 0. Shove_Begin's distances ($20, $40) are multiples of 4.
; ----------------------------------------------------------------------------
Shove_MovePixels:
    ; [03] counts the 4 pixels of this frame.
    LDA #$04
    STA $03

@LoopShovePixel:
    ; On a grid point: snap to the grid, then stop the shove if the tile ahead is unwalkable.
    LDA ObjGridOffset, X
    BNE @CheckBoundary
    JSR Obj_SnapToGrid
    LDA ObjShoveDir, X
    AND #$0F
    STA $0F
    JSR Tile_GetAheadOfObj
    CMP RoomFirstUnwalkableTile
    BCS Shove_Reset

@CheckBoundary:
    ; At any grid offset: stop the shove if the object is at the room bounds in that direction
    ; (Room_BoundDirA returns 0 then).
    LDA ObjShoveDir, X
    AND #$0F
    JSR Room_BoundDirA
    BEQ Shove_Reset

    ; If slot 1 holds the Grumble Goriya ($36) or a dungeon person ($4B-$52), apply the person's
    ; barrier: Person_BlockLinkMovingUp zeroes [0F] when the direction is up and Link's Y < $8E. It reads
    ; Link's Y even when the shoved object is a monster.
    LDA ObjType+1
    CMP #$36
    BEQ @CheckBlocked
    CMP #$4B
    BCC @ChooseSpeed
    CMP #$53
    BCS @ChooseSpeed

@CheckBlocked:
    JSR Person_BlockLinkMovingUp
    LDA $0F
    BEQ Shove_Reset

@ChooseSpeed:
    ; [02] = +1 for right or down, -1 for left or up: the change of the coordinate.
    LDY #$01
    LDA ObjShoveDir, X
    AND #$05
    BNE :+
    LDY #$FF
:
    STY $02

    ; The pixel will be moved: one less to go.
    DEC ObjShoveDistance, X

    ; Move the grid offset by the same step.
    LDA ObjGridOffset, X
    CLC
    ADC $02
    STA ObjGridOffset, X

    ; Back on a grid point: reset the grid offset to 0 when it is a multiple of $10, or for Link a
    ; multiple of 8. (A is 0 in both cases when it is stored.)
    AND #$0F
    BEQ @SetGridOffset
    CPX #$00
    BNE @ApplySpeed
    AND #$07
    BNE @ApplySpeed

@SetGridOffset:
    STA ObjGridOffset, X

@ApplySpeed:
    ; Horizontal direction: add the step to X.
    LDA ObjShoveDir, X
    AND #$03
    BEQ @AddToY
    LDA ObjX, X
    CLC
    ADC $02
    STA ObjX, X
    JMP @NextShovePixel

@AddToY:
    ; Vertical direction: add the step to Y.
    LDA ObjY, X
    CLC
    ADC $02
    STA ObjY, X

@NextShovePixel:
    DEC $03
    BNE @LoopShovePixel
    RTS
