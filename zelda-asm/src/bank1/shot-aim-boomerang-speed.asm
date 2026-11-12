; ============================================================================
; FILE:     bank1/shot-aim-boomerang-speed.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 3638-3851
; SUBJECTS: Boomerang, Dir, Obj
; TOPIC:    Moving a shot one axis at a time inside the room bounds; direction and distance from one
;           object to another; choosing one of nine aim angles for fireballs and returning boomerangs;
;           the returning boomerang's speed and its slow-to-fast switch.
; SEE ALSO: bank1/room-bounds.asm (Room_BoundDirA); bank7/walker-move.asm (Obj_MoveByQSpeed);
;           bank7/boomerang-arrow-food.asm (arrow and boomerang flight, boomerang speed tables);
;           bank7/boomerang-return-draw.asm (boomerang return states $40/$50);
;           bank7/sword-shot-magic-shot.asm (sword shot and magic shot use Obj_MoveShot);
;           bank4/monster-shots-fireballs.asm (Fireball_Update aims with these routines).
; KEY FACTS:
;   - A shot that would cross a room bound does not move; Obj_MoveShot returns [0E] = $80 and [0F] = 0.
;   - Obj_MoveShot moves with the grid offset zeroed, so the grid offset limits never stop a shot; with
;     [0E] = 0 the grid offset accumulates the signed pixels flown (the boomerang's range counter).
;   - Aim has nine steps from horizontal (index 0) through diagonal (4) to vertical (8). With
;     g = larger and l = smaller axis distance: g - l < 8 gives 4; else the index moves
;     min(4, floor(g / l)) steps from 4 toward the longer axis (Dir_CalcAimSpeedIndex).
;   - An axis counts as "reached" when its distance is < 9 pixels; [00] counts such axes
;     (Dir_GetAxisDirAndDistanceToTarget). A returning boomerang is caught when both are.
;   - Equal coordinates give direction left (2) or up (8) (Dir_GetAxisDirAndDistanceToTarget).
;   - Boomerang return state $4x runs at half the table q-speed; state $5x at the full q-speed
;     (Boomerang_SetReturnSpeed).
;   - Boomerang_SetReturnSpeed decrements ObjMovingLimit once per call in state $4x and sets state $50 at 0;
;     the return code calls it twice a frame, so a limit of $20 lasts $10 frames.
; CONTENTS: (generated; L = line in this file)
;   L0049 Obj_MoveShot                       code  Moves a shot or boomerang along one direction unless that would cross a room bound.
;   L0086 Dir_GetDirsAndDistancesToTarget    code  Finds the horizontal and vertical direction and distance from one object to another.
;   L0122 Dir_CalcAimSpeedIndex              code  Picks one of nine aim angles (speed table index 0-8) from the two axis distances to a target.
;   L0186 Boomerang_SetReturnSpeed           code  Sets a returning boomerang's q-speed: half in state $4x (which it also times), full otherwise.
;   L0215 Dir_GetAxisDirAndDistanceToTarget  code  Finds direction and distance from an origin to a target coordinate on one axis.
;   L0235 Dir_ToTarget_Return                code  Shared RTS; Bomb_Wield (bank1/bomb-candle-wield.asm) also branches here to return.
; ============================================================================

; ----------------------------------------------------------------------------
; Moves a shot or boomerang along one direction unless that would cross a room bound.
; IN:   A = direction to move (one axis); X = object slot; [0E] = 0 to add the pixels moved to the grid
;       offset, nonzero to leave the grid offset as it was; ObjQSpeedFrac,X = speed
; OUT:  moved: [0F] = A, coordinate and ObjPosFrac,X updated, [0E] unchanged.
;       blocked by a room bound: [0E] = $80, [0F] = 0, nothing moved.
; NOTE: The grid offset is set to 0 for the move itself, so Obj_MoveByQSpeed's grid offset limits (+-$10)
;       are never reached and cannot stop the shot. With [0E] = 0 the offset afterwards is the old value
;       plus the signed pixels just moved (+ right/down, - left/up): a running distance. The boomerang
;       code compares its absolute value with ObjMovingLimit as the range.
; NOTE: The arrow/boomerang code calls this for the horizontal part of ObjDir, increments [0E], then
;       calls it for the vertical part, so on a diagonal only the horizontal distance is counted.
; ----------------------------------------------------------------------------
Obj_MoveShot:
    ; If the direction leads past a room bound, return [0E] = $80 instead of moving.
    JSR Room_BoundDirA
    BEQ @ReturnBlocked

    ; Move the object with its grid offset at 0.
    LDA ObjGridOffset, X
    PHA
    LDA #$00
    STA ObjGridOffset, X
    JSR Obj_MoveByQSpeed
    PLA                         ; A = grid offset before the move.

    ; [0E] <> 0: put the old grid offset back. [0E] = 0: old offset + pixels just moved.
    LDY $0E
    BNE :+
    CLC
    ADC ObjGridOffset, X
:
    STA ObjGridOffset, X
    RTS

@ReturnBlocked:
    LDA #$80
    STA $0E
    RTS

; ----------------------------------------------------------------------------
; Finds the horizontal and vertical direction and distance from one object to another.
; IN:   A = object slot of the target; X = object slot of the origin; [00] = count to add to (callers
;       that read it set it to 0 first)
; OUT:  [03] = horizontal distance in pixels; [04] = vertical distance (also in A);
;       [0B] = horizontal direction to the target (1 right, 2 left); [0A] = vertical direction
;       (4 down, 8 up); [00] += number of axes whose distance is < 9; uses [01], [02]
; NOTE: Coordinates are compared as they are (top-left corners, unsigned). Equal coordinates give
;       left and up.
; ----------------------------------------------------------------------------
Dir_GetDirsAndDistancesToTarget:
    ; Horizontal.
    PHA
    TAY
    LDA #$02                    ; Start from left; becomes right (1) if the target's X is greater.
    STA $0A
    LDA a:ObjX, Y
    LDY ObjX, X
    JSR Dir_GetAxisDirAndDistanceToTarget
    STA $03
    LDA $0A
    STA $0B

    ; Vertical.
    PLA
    TAY
    LDA #$08                    ; Start from up; becomes down (4) if the target's Y is greater.
    STA $0A
    LDA a:ObjY, Y
    LDY ObjY, X
    JSR Dir_GetAxisDirAndDistanceToTarget
    STA $04
    RTS

; ----------------------------------------------------------------------------
; Picks one of nine aim angles (speed table index 0-8) from the two axis distances to a target.
; IN:   Y = starting index (callers pass 4, the diagonal); [03] = horizontal distance;
;       [04] = vertical distance
; OUT:  Y = [00] = index: 0 = purely horizontal, 4 = diagonal (equal X and Y speed), 8 = purely
;       vertical; [03], [04] are overwritten; uses [01]
; NOTE: With g = the larger and l = the smaller distance: if g - l < 8 the index stays 4. Otherwise it
;       moves one step toward the longer axis, then one more step for each further time l fits in what
;       is left of g, stopping at 0 or 8: min(4, floor(g / l)) steps (4 steps when l = 0).
; NOTE: The index selects a pair of q-speeds from the caller's X and Y tables (fireballs and
;       returning boomerangs). The directions in [0A] and [0B] are not read here.
; ----------------------------------------------------------------------------
Dir_CalcAimSpeedIndex:
    ; [00] = the index being adjusted, starting at the caller's value.
    ; Lower indexes mean more X speed and less Y speed; higher ones the reverse.
    STY $00

    ; [01] = step: -1 (toward horizontal) while horizontal distance >= vertical distance.
    LDA #$FF
    STA $01

    ; If horizontal distance [03] < vertical distance [04], swap them and step +1 (toward vertical).
    ; Afterwards [03] = larger distance, [04] = smaller distance.
    LDA $03
    CMP $04
    BCS @Sorted
    PHA
    LDA $04
    STA $03
    PLA
    STA $04
    LDA #$01
    STA $01

@Sorted:
    ; If the distances differ by less than 8 pixels, keep the starting index (diagonal).
    LDA $03
    SEC
    SBC $04
    CMP #$08
    BCC @Return

@Turn:
    ; Step the index toward the longer axis.
    LDA $00
    CLC
    ADC $01
    STA $00

    ; At 0 or 8 the aim is along an axis: stop.
    BEQ @Return
    CMP #$08
    BEQ @Return

    ; Take the smaller distance out of the larger; while the remainder is still >= the smaller
    ; distance, step again.
    LDA $03
    SEC
    SBC $04
    STA $03
    CMP $04
    BCS @Turn

@Return:
    LDY $00
    RTS

; ----------------------------------------------------------------------------
; Sets a returning boomerang's q-speed: half in state $4x (which it also times), full otherwise.
; IN:   A = q-speed from the speed table; X = object slot; ObjMovingLimit,X = calls left in state $4x
; OUT:  ObjQSpeedFrac,X = A, or A / 2 in state $4x; ObjMovingLimit,X decremented in state $4x; when
;       it reaches 0, ObjState,X = $50 (the low nibble, the animation step, restarts at 0)
; NOTE: The return code calls this twice per frame (vertical speed, then horizontal speed), so the
;       limit of $20 set on entering state $40 lasts $10 frames. The table's largest value is $80
;       (2 pixels per frame on one axis); halved it is $40 (1 pixel per frame).
; ----------------------------------------------------------------------------
Boomerang_SetReturnSpeed:
    STA ObjQSpeedFrac, X

    ; Only state $4x (slow return) changes anything further.
    LDA ObjState, X
    AND #$F0
    CMP #$40
    BNE @Exit

    ; State $4x: half speed.
    LSR ObjQSpeedFrac, X

    ; In state $4x ObjMovingLimit counts down to the switch to state $50 (fast return).
    DEC ObjMovingLimit, X
    BNE @Exit
    LDA #$50
    STA ObjState, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Finds direction and distance from an origin to a target coordinate on one axis.
; IN:   A = target coordinate; Y = origin coordinate; [0A] = the axis's decreasing direction
;       (2 left or 8 up); [00] = a count
; OUT:  A = distance in pixels; [0A] = direction from origin to target (unchanged if origin >= target,
;       else shifted right once: 2 -> 1 right, 8 -> 4 down); [00] incremented if distance < 9;
;       [01] = smaller coordinate, [02] = larger coordinate
; ----------------------------------------------------------------------------
Dir_GetAxisDirAndDistanceToTarget:
    ; If origin >= target, the direction in [0A] already points at the target (or they are equal).
    ; Otherwise swap the two so that [02] >= [01], and flip the direction by shifting it right.
    STA $01
    STY $02
    CPY $01
    BCS @Subtract
    STA $02
    STY $01
    LSR $0A

@Subtract:
    ; A = distance. If it is < 9, count this axis in [00].
    LDA $02
    SEC
    SBC $01
    CMP #$09
    BCS Dir_ToTarget_Return
    INC $00
; Shared RTS; Bomb_Wield (bank1/bomb-candle-wield.asm) also branches here to return.
Dir_ToTarget_Return:
    RTS
