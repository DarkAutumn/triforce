; ============================================================================
; FILE:     bank1/qspeed-direction-math.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3511-3636
; SUBJECTS: Dir, Math, Obj
; TOPIC:    The two routines that turn an object's q-speed into whole-pixel steps and count them in the
;           grid offset (with its limits); opposite direction and direction index; Math_Abs and Math_Negate.
; SEE ALSO: bank7/walker-move.asm (Obj_MoveByQSpeed: sets the limits, calls these four times a frame and moves
;           the coordinate by the returned carry); bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate
;           sets Link's grid offset back to 0 at a multiple of 8); bank5/link-handle-input.asm (turning
;           rules that read and rewrite the grid offset); bank7/walker-collision-screen-edge.asm
;           (Link_CheckScreenEdge needs grid offset 0); bank1/shot-aim-boomerang-speed.asm (Obj_MoveShot).
; KEY FACTS:
;   - One call adds (or subtracts) ObjQSpeedFrac to ObjPosFrac and yields at most one pixel; Obj_MoveByQSpeed
;     calls four times a frame, so pixels per frame = q-speed / 64 (Obj_AddQSpeedToPosFrac).
;   - ObjGridOffset counts the pixels: +1 per pixel right or down (Obj_AddQSpeedToPosFrac), -1 per
;     pixel left or up (Obj_SubQSpeedFromPosFrac).
;   - Limits: when ObjGridOffset equals PositiveGridCellSize or NegativeGridCellSize (+8/-8 for Link,
;     +$10/-$10 for every other object, set by Obj_MoveByQSpeed) the pixel is withheld, in BOTH routines: an
;     object at either limit cannot move either way until other code changes the offset.
;   - 0 is not a limit: the offset counts through 0 freely (+1, 0, -1 = $FF, ...); nothing here stops
;     or snaps an object at a grid point (Obj_AddQSpeedToPosFrac, Obj_SubQSpeedFromPosFrac).
;   - At a limit the fraction still accumulates; only the pixel is dropped.
;   - The test is for equality with a limit before the step, so an offset already beyond a limit is
;     not stopped (Obj_AddQSpeedToPosFrac).
;   - Dir_GetOpposite uses the lowest set direction bit (right, then left, then down, then up) and also
;     returns the direction index in Y: right 3, left 2, down 1, up 0. Direction 0 is not handled.
;   - Math_Abs and Math_Negate leave $80 as $80.
; CONTENTS: (generated; L = line in this file)
;   L0052 Obj_AddQSpeedToPosFrac    code  Adds the q-speed to the position fraction for a step right or down; a carry is one pixel.
;   L0094 Obj_SubQSpeedFromPosFrac  code  Subtracts the q-speed from the position fraction for a step left or up; a borrow is one pixel.
;   L0125 Dir_Opposites             data  TABLE: opposite of each direction; index = direction index (0 up, 1 down, 2 left, 3 right);
;   L0137 Dir_GetOpposite           code  Gives the opposite of a direction in A and the direction's index (0 up .. 3 right) in Y.
;   L0155 Math_Abs                  code  Takes the absolute value of a signed byte (falls into Math_Negate when A is negative).
;   L0163 Math_Negate               code  Negates a byte (two's complement).
; ============================================================================

; ----------------------------------------------------------------------------
; Adds the q-speed to the position fraction for a step right or down; a carry is one pixel.
; IN:   X = object slot; PositiveGridCellSize, NegativeGridCellSize = grid offset limits (Obj_MoveByQSpeed
;       stores 8 and $F8 for Link, $10 and $F0 for any other object)
; OUT:  C = 1 if the object moves one pixel (the caller adds C to ObjX or ObjY); ObjPosFrac,X
;       updated; ObjGridOffset,X = itself + C
; NOTE: If ObjGridOffset,X equals either limit before the step, C is forced to 0: no pixel and no
;       change to the grid offset, although ObjPosFrac still advances. The positive AND the negative
;       limit both block this (positive-going) step.
; NOTE: 0 is not a limit. An offset of -1 ($FF) steps to 0 and on to +1 like any other value, so an
;       object heading right or down with a negative offset walks through the grid point (offset 0)
;       and continues up to the positive limit.
; NOTE: Obj_MoveByQSpeed calls this four times per frame, each call giving 0 or 1 pixel: q-speed $60 gives
;       1 and 2 pixels on alternate frames (1.5 per frame); q-speed $40 gives 1 pixel every frame.
; ----------------------------------------------------------------------------
Obj_AddQSpeedToPosFrac:
    LDA ObjPosFrac, X
    CLC
    ADC ObjQSpeedFrac, X
    STA ObjPosFrac, X
    PHP                         ; Save the carry: 1 = the fraction passed a whole pixel.

    ; If the grid offset equals the positive or the negative limit, withhold the pixel.
    LDA ObjGridOffset, X
    CMP PositiveGridCellSize
    BEQ @ClearCarry
    CMP NegativeGridCellSize
    BNE :+

@ClearCarry:
    PLP                         ; Replace the saved carry with 0.
    CLC
    PHP
:
    PLP                         ; Carry for the addition below,
    PHP                         ; saved again to be the return value.

    ; Grid offset += 1 if a pixel is being moved.
    LDA ObjGridOffset, X
    ADC #$00
    STA ObjGridOffset, X
    PLP                         ; Return C = 1 for a pixel moved; 0 for none or at a limit.
    RTS

; ----------------------------------------------------------------------------
; Subtracts the q-speed from the position fraction for a step left or up; a borrow is one pixel.
; IN:   X = object slot; PositiveGridCellSize, NegativeGridCellSize = grid offset limits (Obj_MoveByQSpeed
;       stores 8 and $F8 for Link, $10 and $F0 for any other object)
; OUT:  C = 0 if the object moves one pixel (the caller subtracts the borrow from ObjX or ObjY);
;       ObjPosFrac,X updated; ObjGridOffset,X = itself - 1 when C = 0
; NOTE: If ObjGridOffset,X equals either limit before the step, C is forced to 1: no pixel and no
;       change to the grid offset, although ObjPosFrac still changes. The negative AND the positive
;       limit both block this (negative-going) step.
; NOTE: 0 is not a limit. An offset of +1 steps to 0 and on to -1 ($FF), so an object heading left or
;       up with a positive offset walks through the grid point (offset 0) and continues down to the
;       negative limit.
; ----------------------------------------------------------------------------
Obj_SubQSpeedFromPosFrac:
    LDA ObjPosFrac, X
    SEC
    SBC ObjQSpeedFrac, X
    STA ObjPosFrac, X
    PHP                         ; Save the carry: 0 = the fraction borrowed a whole pixel.

    ; If the grid offset equals the positive or the negative limit, withhold the pixel.
    LDA ObjGridOffset, X
    CMP PositiveGridCellSize
    BEQ @SetCarry
    CMP NegativeGridCellSize
    BNE :+

@SetCarry:
    PLP                         ; Replace the saved carry with 1 (no borrow).
    SEC
    PHP
:
    PLP                         ; Carry for the subtraction below,
    PHP                         ; saved again to be the return value.

    ; Grid offset -= 1 if a pixel is being moved.
    LDA ObjGridOffset, X
    SBC #$00
    STA ObjGridOffset, X
    PLP                         ; Return C = 0 for a pixel moved; 1 for none or at a limit.
    RTS

; TABLE: opposite of each direction; index = direction index (0 up, 1 down, 2 left, 3 right);
; value = direction bits.
Dir_Opposites:
    .BYTE $04, $08, $01, $02    ; up -> down, down -> up, left -> right, right -> left

; ----------------------------------------------------------------------------
; Gives the opposite of a direction in A and the direction's index (0 up .. 3 right) in Y.
; IN:   A = direction bits (1 right, 2 left, 4 down, 8 up)
; OUT:  A = opposite direction bit; Y = direction index of the input: 3 right, 2 left, 1 down, 0 up
; NOTE: With several bits set the lowest wins: right, then left, then down, then up. Many callers use
;       this only to reduce a diagonal input to one direction or to get the index in Y.
; NOTE: With none of the low four bits set the loop ends with Y = $FF and A is read from
;       Dir_Opposites+$FF, a byte outside the table: callers must pass a direction.
; ----------------------------------------------------------------------------
Dir_GetOpposite:
    LDY #$03

@Loop:
    LSR                         ; Shift out the next direction bit, lowest (right) first.
    BCS :+
    DEY
    BPL @Loop
:
    LDA Dir_Opposites, Y
    RTS

; ----------------------------------------------------------------------------
; Takes the absolute value of a signed byte (falls into Math_Negate when A is negative).
; IN:   A = signed value
; OUT:  A = |A| ($80 stays $80)
; NOTE: The branch for a non-negative value goes to the RTS at the end of Math_Negate.
; ----------------------------------------------------------------------------
Math_Abs:
    BPL :+

; ----------------------------------------------------------------------------
; Negates a byte (two's complement).
; IN:   A = value
; OUT:  A = -A (0 stays 0, $80 stays $80); C = 1 only when A was 0
; ----------------------------------------------------------------------------
Math_Negate:
    EOR #$FF
    CLC
    ADC #$01
:
    RTS
