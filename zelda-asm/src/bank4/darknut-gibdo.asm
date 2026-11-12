; ============================================================================
; FILE:     bank4/darknut-gibdo.asm      BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6449-6514
; SUBJECTS: Darknut, Gibdo
; TOPIC:    Darknut (types $0B red, $0C blue) init and update, and Gibdo (type $30) update. Both walk
;           the grid as wanderers and hurt Link by touch only.
; SEE ALSO: bank4/walker-wanderer-common.asm (Walker_Init, Wanderer_Update, Wanderer_MoveAndTurn:
;           the shared walking and turning); bank1/weapon-damage-death.asm and
;           bank1/sword-bomb-arrow-collision.asm (the Darknut shield: Damage_HandleWeaponHit,
;           Damage_ParryOrShove); bank7/object-update-table.asm (HP and attribute tables);
;           bank1/monster-collisions.asm (Damage_ObjTypeDamageToLink); bank4/drop-items.asm (drop groups).
; KEY FACTS:
;   - A Darknut's invincibility mask is $F6: only damage types $01 (sword, sword shot, rod swing)
;     and $08 (bomb blast) can hurt it; boomerang, arrow, magic shot and fire cannot (Darknut_Init).
;   - Red Darknut walks at q-speed $20 (0.5 pixel per frame), blue at $28 (0.625) (Darknut_Init).
;   - Darknut_Update zeroes ObjStunTimer every frame, so a Darknut is never held by a stun; the
;     boomerang, which stuns other monsters, is refused by the mask anyway (Darknut_Update).
;   - The shield is not in this file: a hit is parried when the weapon's direction and the Darknut's
;     are exact opposites (Damage_HandleWeaponHit, Damage_ParryOrShove in bank 1). That holds for a
;     bomb too (its direction is Link's when he placed it), but a parried bomb still knocks the
;     Darknut back (Bomb_CheckBombOrFireVsMonster).
;   - Darknut and Gibdo both use turn rate $80: at a grid point they may turn toward the chase
;     target when Random+1,X <= $80, 129 chances in 256 (Darknut_Update, Gibdo_Update).
;   - Gibdo has no init routine of its own (the init table gives it Walker_Init): it keeps the room
;     entry q-speed $20 (0.5 pixel per frame) and a mask of 0, so every weapon hurts it and the
;     boomerang stuns it (Gibdo_Update).
;   - With InvClock set or a nonzero ObjStunTimer the walk is skipped, but collisions are still
;     checked and the monster is still drawn (Gibdo_Update, Darknut_Update).
; CONTENTS: (generated; L = line in this file)
;   L0044 Darknut_Init    code  Initializes a Darknut (type $0B red, $0C blue): weapon immunities, speed, facing direction.
;   L0069 Gibdo_Update    code  Updates a Gibdo (type $30): wanders with turn rate $80, checks collisions, draws itself.
;   L0089 Darknut_Update  code  Updates a Darknut (types $0B, $0C): wanders with turn rate $80, checks collisions, draws itself.
; ============================================================================

; ----------------------------------------------------------------------------
; Initializes a Darknut (type $0B red, $0C blue): weapon immunities, speed, facing direction.
; IN:   X = object slot; ObjType,X = $0B or $0C.
; OUT:  ObjInvincibilityMask,X = $F6; ObjQSpeedFrac,X = $20 (red) or $28 (blue); ObjDir,X set by
;       Walker_Init if it was 0.
; NOTE: Mask $F6 leaves damage types $01 (sword, sword shot, rod swing) and $08 (bomb) clear: those
;       are the only ones that can hurt a Darknut. Boomerang ($02), arrow ($04), magic shot ($10)
;       and fire ($20) are refused. HP comes from Obj_TypeHpPairs: red $40, blue $80.
; ----------------------------------------------------------------------------
Darknut_Init:
    LDA #$F6                    ; Immune to all damage types but $01 (sword class) and $08 (bomb).
    STA ObjInvincibilityMask, X

    ; Red Darknut ($0B): q-speed $20 (0.5 pixel per frame).
    LDA #$20
    LDY ObjType, X
    CPY #$0B
    BEQ :+

    ; Blue Darknut: q-speed $28 (0.625 pixel per frame).
    LDA #$28
:
    STA ObjQSpeedFrac, X
    JMP Walker_Init

; ----------------------------------------------------------------------------
; Updates a Gibdo (type $30): wanders with turn rate $80, checks collisions, draws itself.
; IN:   X = object slot.
; NOTE: Gibdo's init is Walker_Init, so its q-speed is the room entry value $20 (0.5 pixel per frame)
;       and its invincibility mask is 0: every weapon hurts it and the boomerang stuns it.
;       Wanderer_Update plays out knockback, and skips the walk while InvClock is set or
;       ObjStunTimer,X is nonzero. It has one frame image, flipped horizontally on every other
;       animation frame (8 frames each); the animation runs even when the Gibdo does not move.
; ----------------------------------------------------------------------------
Gibdo_Update:
    LDA #$80                    ; Turn rate $80.
    JSR Wanderer_Update
    JSR Collision_CheckMonsterVsWeaponsAndLink
    LDA #$08                    ; 8 frames per animation frame.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    JSR Anim_SetObjHFlipForSpriteDescriptor     ; [0F] = ObjAnimFrame,X: flip on frame 1.
    LDA #$00                    ; Frame image 0.
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Updates a Darknut (types $0B, $0C): wanders with turn rate $80, checks collisions, draws itself.
; IN:   X = object slot.
; OUT:  ObjStunTimer,X = 0.
; NOTE: The stun timer is zeroed every frame after the collision check, so no stun carries into
;       the next frame's walk. The shield is applied by the collision code, not here: a hit is
;       parried when the weapon's direction and ObjDir,X are exact opposites
;       (Damage_HandleWeaponHit, Damage_ParryOrShove).
;       Frame images: 0 left/right, 1 down, 2 up, plus 3 on animation frame 1 (8 frames each).
; ----------------------------------------------------------------------------
Darknut_Update:
    LDA #$80                    ; Turn rate $80.
    JSR Wanderer_Update
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Cancel any stun.
    LDA #$00
    STA ObjStunTimer, X
    LDA #$08                    ; 8 frames per animation frame.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; Facing left (2): flip horizontally ([0F] was just set to 0).
    LDA ObjDir, X
    CMP #$02
    BNE :+
    INC $0F
:
    ; A = direction >> 2 = base frame image: 0 right or left, 1 down, 2 up.
    LSR
    LSR

    ; Animation frame 0: draw the base image.
    LDY ObjAnimFrame, X
    BEQ @Draw

    ; Animation frame 1: images 3-5.
    CLC
    ADC #$03

    ; Facing up on animation frame 1: flip horizontally. Images 2 and 5 are the same tiles, so the
    ; flip is what animates the walk upward.
    LDY ObjDir, X
    CPY #$08
    BNE @Draw
    INC $0F

@Draw:
    JSR Anim_DrawObjectNotMirrored
    RTS
