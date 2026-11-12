; ============================================================================
; FILE:     bank1/object-overlap-test.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 6351-6493
; SUBJECTS: Collision
; TOPIC:    The hit point of a thin weapon (sword, rod, arrow, sword shot, magic shot) and the box test
;           between two points that every collision check uses.
; SEE ALSO: bank1/sword-bomb-arrow-collision.asm (callers that set damage and thresholds);
;           bank1/weapon-damage-death.asm (Collision_CheckWeaponPointVsMonster: what happens on overlap);
;           bank1/link-collision-harm.asm (Link against monsters, threshold 9);
;           bank1/monster-collisions.asm (Obj_GetMiddle: the monster's middle in [02],[03]).
; KEY FACTS:
;   - Two points collide when the horizontal distance is less than [0D] and the vertical distance less
;     than [0E]; the bounds are exclusive (Collision_TestPointsWithThresholds).
;   - Distances are the absolute value of an 8-bit difference, so a true distance over $80 wraps and reads
;     as 256 minus the distance (Collision_TestPointsWithThresholds).
;   - The hit point of a thin weapon is (X+6, Y+8) when Link faces up or down and (X+8, Y+6) when he faces
;     left or right: Link's direction is used, not the weapon's (Collision_CheckThinWeaponVsMonster_Thresholds).
;   - Collision_TestPoints uses one threshold for both axes (Collision_TestPoints).
; CONTENTS: (generated; L = line in this file)
;   L0033 Collision_CheckThinWeaponVsMonster             code  Tests a thin weapon against a monster using one threshold for both axes.
;   L0047 Collision_CheckThinWeaponVsMonster_Thresholds  code  Sets the hit point of a thin weapon from Link's facing direction, then tests it against a monster.
;   L0085 Collision_TestPoints                           code  Tests whether two points are closer than A pixels on both axes.
;   L0098 Collision_TestPointsWithThresholds             code  Tests whether two points are closer than [0D] pixels horizontally and [0E] pixels vertically.
; ============================================================================

; ----------------------------------------------------------------------------
; Tests a thin weapon against a monster using one threshold for both axes.
; IN:   A = damage points; X = monster slot; [00] = weapon slot; [02],[03] = monster's middle;
;       [09] = damage type; [0D] = threshold.
; OUT:  [06] = 1 when they overlap; [07] = damage points; [0E] = [0D]; Y = weapon slot.
; NOTE: Falls into Collision_CheckThinWeaponVsMonster_Thresholds.
; ----------------------------------------------------------------------------
Collision_CheckThinWeaponVsMonster:
    STA $07                     ; [07] := damage points
    LDA $0D
    STA $0E                     ; vertical threshold := horizontal threshold

; ----------------------------------------------------------------------------
; Sets the hit point of a thin weapon from Link's facing direction, then tests it against a monster.
; IN:   X = monster slot; [00] = weapon slot; [02],[03] = monster's middle; [07] = damage points;
;       [09] = damage type; [0D],[0E] = horizontal and vertical thresholds.
; OUT:  [06] = 1 when they overlap; Y = weapon slot; [04],[05] = the weapon's hit point.
; NOTE: The point is chosen by Link's direction (ObjDir of slot 0), not the weapon's own. An arrow or
;       shot in flight therefore uses Link's current facing: its hit point moves by 2 pixels on each
;       axis when Link turns to the other axis.
; ----------------------------------------------------------------------------
Collision_CheckThinWeaponVsMonster_Thresholds:
    LDY $00

    ; Link faces up or down: [04] := weapon X + 6; A := weapon Y + 8.
    LDA ObjDir
    AND #$0C
    BEQ @CheckHorizontal
    LDA a:ObjX, Y
    CLC
    ADC #$06
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$08
    JMP @Test

@CheckHorizontal:
    ; Link faces left or right: [04] := weapon X + 8; A := weapon Y + 6.
    LDA a:ObjX, Y
    CLC
    ADC #$08
    STA $04
    LDA a:ObjY, Y
    CLC
    ADC #$06
@Test:
    ; A (the point's Y) is stored in [05] by the routine jumped to.
    JMP Collision_CheckWeaponPointVsMonster

; ----------------------------------------------------------------------------
; Tests whether two points are closer than A pixels on both axes.
; IN:   A = threshold for both axes; [00] = an object slot, only copied to Y; [02],[03] = point 1 (X, Y);
;       [04],[05] = point 2 (X, Y).
; OUT:  A = [06] = 1 when they collide, else 0 (Z flag set accordingly); Y = [00]; [0D] = [0E] = the
;       threshold; [0A] = horizontal distance; [0B] = vertical distance (written only when the horizontal
;       test passed).
; NOTE: Falls into Collision_TestPointsWithThresholds.
; ----------------------------------------------------------------------------
Collision_TestPoints:
    STA $0D
    STA $0E

; ----------------------------------------------------------------------------
; Tests whether two points are closer than [0D] pixels horizontally and [0E] pixels vertically.
; IN:   [00] = an object slot, only copied to Y; [02],[03] = point 1 (X, Y); [04],[05] = point 2 (X, Y);
;       [0D] = horizontal threshold; [0E] = vertical threshold.
; OUT:  A = [06] = 1 when they collide, else 0 (Z flag set accordingly); Y = [00]; [0A] = horizontal
;       distance; [0B] = vertical distance (written only when the horizontal test passed).
; NOTE: Collision needs distance < threshold on both axes. Each distance is the absolute value of the
;       8-bit difference taken as a signed byte.
; ----------------------------------------------------------------------------
Collision_TestPointsWithThresholds:
    LDA #$00
    STA $06                     ; [06] := 0: no collision
    LDY $00

    ; [0A] := |point 1 X - point 2 X|. No collision if it is >= [0D].
    LDA $02
    SEC
    SBC $04
    JSR Math_Abs
    STA $0A

    CMP $0D
    BCS @ReturnValue

    ; [0B] := |point 1 Y - point 2 Y|. No collision if it is >= [0E].
    LDA $03
    SEC
    SBC $05
    JSR Math_Abs
    STA $0B

    CMP $0E
    BCS @ReturnValue

    ; Close enough on both axes.
    INC $06

@ReturnValue:
    LDA $06
    RTS
