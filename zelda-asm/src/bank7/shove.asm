; ============================================================================
; FILE:     bank7/shove.asm              BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2323-2494
; SUBJECTS: Obj
; TOPIC:    Knockback movement: moves a shoved object (Link or a monster) up to 4 pixels per frame in
;           its shove direction until the distance runs out or something blocks it.
; SEE ALSO: bank1/shove-knockback.asm (BeginShove: sets direction, distance and invincibility);
;           bank7/tile-collision.asm (GetCollidingTileMoving); bank1/room-bounds.asm (BoundByRoomWithA);
;           bank1/modeb-cave-save-sprites.asm (CheckPersonBlocking); bank7/mode5-play-update.asm
;           (EnsureObjectAligned); bank4/walker-wanderer-common.asm and other monster files (callers).
; KEY FACTS:
;   - Speed: 4 pixels per frame, one pixel at a time, each pixel tested (ShoveMoveMin). BeginShove gives
;     Link $20 pixels (8 frames) and a monster $40 pixels (16 frames) when nothing stops the shove.
;   - The first call of a shove (bit 7 of ObjShoveDir set) only clears bit 7 and validates the direction;
;     it does not move the object (Obj_Shove).
;   - A shove perpendicular to the object's facing axis is allowed only at grid offset 0. Off the grid
;     a monster's shove is cancelled; Link's is redirected to the opposite of his facing direction
;     (Obj_Shove@CheckPerpendicularShove).
;   - Tiles are tested only at grid offset 0, after snapping the object to the grid; an unwalkable tile
;     ahead ends the shove (ShoveMoveMin@LoopShovePixel).
;   - The room bounds end the shove at any grid offset (ShoveMoveMin@CheckBoundary).
;   - If object slot 1 holds the Grumble Goriya ($36) or a dungeon person ($4B-$52), an upward shove
;     ends while Link's Y < $8E; the test reads Link's Y whichever object is shoved
;     (ShoveMoveMin@CheckBlocked).
;   - Each pixel also moves ObjGridOffset by 1; it is reset to 0 at a multiple of $10, and for Link also
;     at a multiple of 8 (ShoveMoveMin@ChooseSpeed).
;   - When the shove ends, ObjShoveDir and ObjShoveDistance are both cleared (ResetShoveInfo).
;   - Bit 6 of ObjShoveDir is never tested here: every use masks with $0F, $0C, $05 or $03
;     (Obj_Shove, ShoveMoveMin).
; CONTENTS: (generated; L = line in this file)
;   L0050 Obj_Shove          code  Advances an object's knockback by one frame: validate on the first call, then move up to 4 pixels.
;   L0103 ResetShoveInfo     code  Ends an object's knockback: clears ObjShoveDir,X and ObjShoveDistance,X.
;   L0107 SetShoveInfoWith0  code  Second entry: stores A (callers pass 0) in ObjShoveDir,X and ObjShoveDistance,X. A is unchanged.
;   L0124 ShoveMoveMin       code  Moves a shoved object up to 4 pixels, one at a time, stopping the shove at the first obstacle.
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
;       BeginShove chooses Link's shove axis from ObjDir when he is off the grid, so Link's replacement
;       is reached only if ObjDir or the grid offset changed since BeginShove ran (unverified whether
;       that ever happens; upstream marked the block as never seen executing).
; ----------------------------------------------------------------------------
Obj_Shove:
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
    BNE ResetShoveInfo

    ; Link: shove him opposite to the direction he faces instead.
    ; (Upstream marked these three instructions as never seen executing; see the routine header.)
    LDA ObjDir
    JSR GetOppositeDir
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
    BNE ShoveMoveMin

; ----------------------------------------------------------------------------
; Ends an object's knockback: clears ObjShoveDir,X and ObjShoveDistance,X.
; IN:   X = object slot.
; OUT:  A = 0.
; ----------------------------------------------------------------------------
ResetShoveInfo:
    LDA #$00

; Second entry: stores A (callers pass 0) in ObjShoveDir,X and ObjShoveDistance,X. A is unchanged.
SetShoveInfoWith0:
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
;       moves 4 and ObjShoveDistance wraps below 0. BeginShove's distances ($20, $40) are multiples of 4.
; ----------------------------------------------------------------------------
ShoveMoveMin:
    ; [03] counts the 4 pixels of this frame.
    LDA #$04
    STA $03

@LoopShovePixel:
    ; On a grid point: snap to the grid, then stop the shove if the tile ahead is unwalkable.
    LDA ObjGridOffset, X
    BNE @CheckBoundary
    JSR EnsureObjectAligned
    LDA ObjShoveDir, X
    AND #$0F
    STA $0F
    JSR GetCollidingTileMoving
    CMP RoomFirstUnwalkableTile
    BCS ResetShoveInfo

@CheckBoundary:
    ; At any grid offset: stop the shove if the object is at the room bounds in that direction
    ; (BoundByRoomWithA returns 0 then).
    LDA ObjShoveDir, X
    AND #$0F
    JSR BoundByRoomWithA
    BEQ ResetShoveInfo

    ; If slot 1 holds the Grumble Goriya ($36) or a dungeon person ($4B-$52), apply the person's
    ; barrier: CheckPersonBlocking zeroes [0F] when the direction is up and Link's Y < $8E. It reads
    ; Link's Y even when the shoved object is a monster.
    LDA ObjType+1
    CMP #$36
    BEQ @CheckBlocked
    CMP #$4B
    BCC @ChooseSpeed
    CMP #$53
    BCS @ChooseSpeed

@CheckBlocked:
    JSR CheckPersonBlocking
    LDA $0F
    BEQ ResetShoveInfo

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
