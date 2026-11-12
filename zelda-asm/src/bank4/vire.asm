; ============================================================================
; FILE:     bank4/vire.asm               BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6915-7025
; SUBJECTS: Vire
; TOPIC:    Vire (object type $12): wanders toward Link, hopping while it moves sideways; a hit that
;           does not kill it sends it flying and turns it into two red Keese (type $1C).
; SEE ALSO: bank4/zol-gel.asm (Zol_State1Slide and Gel_MoveSplitting: the shared state 1);
;           bank1/weapon-damage-death.asm (Damage_HandleWeaponHit turns a hit Vire to face the
;           weapon's direction); bank1/shove-knockback.asm (Shove_Begin: Vire keeps its direction after
;           touching Link); bank4/shoot-helpers-candle.asm (Shoot creates the Keese);
;           bank4/walker-wanderer-common.asm (Wanderer_Update); bank7/boomerang-arrow-food.asm
;           (food attracts Vires); bank4/drop-items.asm (drop group lists).
; KEY FACTS:
;   - HP $40 (nibble 4 in Obj_TypeHpPairs). A hit of $40 or more kills it outright (magic sword,
;     bomb, silver arrow); a weaker hit that harms it makes it split (Vire_CheckCollisions).
;   - The boomerang only stuns it (stun timer $10): it starts no invincibility timer, so no split
;     (Vire_CheckCollisions tests ObjInvincibilityTimer; for a Vire only Shove_Begin, bank 1, starts it).
;   - Split sequence: state 0 -> 1 when hit and still alive; state 1 = Zol_State1Slide: it flies
;     at q-speed $FF in its facing direction (the hit set it to the weapon's direction) until a
;     tile or the room bound blocks it; then state 2, and Vire_Update splits it the same frame.
;   - The split destroys the Vire and makes up to two red Keese (type $1C) at its position, each only
;     if Obj_FindEmptyMonsterSlot finds a free slot; RoomObjCount is incremented by 1 regardless
;     (Vire_Update@SplitUp). A split Vire drops nothing and is not counted as a kill here.
;   - While in state 1 it checks no collisions: it cannot be hit and cannot hurt Link
;     (Vire_CheckCollisions).
;   - Hop: while facing left or right, Vire_HopYOffsets[abs(grid offset)] is added to Y every frame; the
;     16 entries sum to 0 (Vire_State0Wander). No hop while it faces up or down, while InvClock is set
;     or while it is stunned.
;   - Touch damage byte $01 = 1 heart before rings (Damage_ObjTypeDamageToLink); q-speed $20 = 0.5 pixels per
;     frame (room-entry default); turn rate $80. Invincibility mask is never set (0).
;   - Type $12 is in Drop_Group2Types (drop group 2), which applies when it is killed outright.
; CONTENTS: (generated; L = line in this file)
;   L0051 Vire_Update                 code  Updates a Vire: runs its state, then checks collisions and draws it, or splits it into two red Keese.
;   L0090 Vire_UpdateState            code  Runs the handler for the Vire's state (0 wander and hop, 1 fly away after a hit).
;   L0096 Vire_UpdateState_JumpTable  data  TABLE: Vire state handlers; index = ObjState,X (0-1); value = routine address. State 2 is never
;   L0102 Vire_HopYOffsets            data  TABLE: Vire hop; index = abs(ObjGridOffset,X) (0-$F); value = signed pixels added to Y on each frame
;   L0114 Vire_State0Wander           code  Vire state 0: wanders toward Link (turn rate $80) and hops while facing left or right.
;   L0152 Vire_CheckCollisions        code  Checks a wandering Vire's collisions and starts the split if it was hit and survived.
;   L0175 Vire_Draw                   code  Animates and draws a Vire: frame images 0-1 normally, 2-3 when it faces up.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Vire: runs its state, then checks collisions and draws it, or splits it into two red Keese.
; IN:   X = object slot.
; OUT:  on a split: the Vire's slot is freed, RoomObjCount += 1, up to two type $1C objects created.
; NOTE: The split happens on the frame the state reaches 2 (set by Zol_State1Slide when the
;       flying Vire is blocked). Each Keese needs a free slot in $01-$0B; the Vire's own slot was just
;       freed, so the first always finds one. If the second finds none, only one Keese appears but
;       RoomObjCount was still raised by 1.
; ----------------------------------------------------------------------------
Vire_Update:
    JSR Vire_UpdateState

    ; State 2 or above: split into two red Keese.
    LDA ObjState, X
    CMP #$02
    BCS @SplitUp
    JSR Vire_CheckCollisions
    JMP Vire_Draw

@SplitUp:
    ; One monster is destroyed and two are made: one more object in the room (as with Zol).
    INC RoomObjCount
    JSR Obj_DestroyMonster

    ; Twice: look for a free monster slot ($0B down to $01); if there is one, create a red Keese in
    ; it with the "shoot" operation (the Vire's position and direction, state $10).
    LDY #$01

@LoopMakeKeese:
    TYA                         ; Save loop counter.
    PHA
    JSR Obj_FindEmptyMonsterSlot
    BEQ @NextLoopMakeKeese      ; Z set = no free slot.
    LDA #$1C                    ; Red Keese object type
    STA $00
    JSR Shooter_Shoot

@NextLoopMakeKeese:
    PLA                         ; Restore loop counter.
    TAY
    DEY
    BPL @LoopMakeKeese
    RTS

; ----------------------------------------------------------------------------
; Runs the handler for the Vire's state (0 wander and hop, 1 fly away after a hit).
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Vire_UpdateState:
    LDA ObjState, X
    JSR Util_TableJump

; TABLE: Vire state handlers; index = ObjState,X (0-1); value = routine address. State 2 is never
;        dispatched: the state only becomes 2 inside this call, and Vire_Update then destroys the Vire.
Vire_UpdateState_JumpTable:
    .ADDR Vire_State0Wander              ; 0: wander toward Link, hopping
    .ADDR Zol_State1Slide         ; 1: fly straight until blocked (shared with Zol), then state 2

; TABLE: Vire hop; index = abs(ObjGridOffset,X) (0-$F); value = signed pixels added to Y on each frame
;        spent at that offset. The entries sum to 0; the running sum is lowest (-8) at offsets 6-8.
Vire_HopYOffsets:
    .BYTE $00, $FD, $FE, $FF, $FF, $00, $FF, $00   ; 0, -3, -2, -1, -1, 0, -1, 0
    .BYTE $00, $01, $00, $01, $01, $02, $03, $00   ; 0, +1, 0, +1, +1, +2, +3, 0

; ----------------------------------------------------------------------------
; Vire state 0: wanders toward Link (turn rate $80) and hops while facing left or right.
; IN:   X = object slot.
; NOTE: The hop offset is added on every frame, not once per pixel. At the default q-speed $20 the
;       Vire spends 2 frames on each pixel, so each entry applies twice and the hop peaks 16 pixels
;       up (derived from the speed; not traced). Wanderer_Update never applies knockback to
;       a Vire: the hit that sets ObjShoveDir also moves it to state 1 (or kills it).
; ----------------------------------------------------------------------------
Vire_State0Wander:
    LDA #$80                    ; Turn rate $80 (see Wanderer_ObjTurnRate)
    JSR Wanderer_Update

    ; If Link has the clock or the Vire is stunned, no hop.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @Exit

    ; Facing up or down: no hop.
    LDA ObjDir, X
    AND #$03
    BEQ @Exit

    ; Index the hop table by the distance from the last grid point, 0-$F pixels, either sign.
    LDA ObjGridOffset, X
    JSR Math_Abs
    TAY

    ; Add this offset to Y, so the Vire rises and falls once per square walked sideways.
    LDA ObjY, X
    CLC
    ADC Vire_HopYOffsets, Y
    STA ObjY, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Checks a wandering Vire's collisions and starts the split if it was hit and survived.
; IN:   X = object slot.
; OUT:  ObjState,X = 1 when the Vire was harmed but not killed.
; NOTE: In state 1 (flying off to split) no collisions are checked at all: it cannot be hit and
;       cannot hurt Link. A hit that kills it outright (damage >= its HP) gives a normal death.
;       A boomerang hit only stuns it (stun timer $10, 0 damage): the boomerang path never reaches
;       Shove_Begin (bank 1), the code that starts a Vire's invincibility timer on a hit, so the
;       boomerang cannot start a split. The knockback Shove_Begin sets is never applied.
; ----------------------------------------------------------------------------
Vire_CheckCollisions:
    ; State 1: already on its way to split. No collision checks.
    LDA ObjState, X
    BNE @Exit
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Dying (metastate nonzero): no split.
    LDA ObjMetastate, X
    BNE @Exit

    ; A nonzero invincibility timer means a weapon harmed it on this frame: go to state 1.
    ; (The damage code has already turned it to face the weapon's direction.)
    LDA ObjInvincibilityTimer, X
    BEQ @Exit
    INC ObjState, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Animates and draws a Vire: frame images 0-1 normally, 2-3 when it faces up.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Vire_Draw:
    LDA #$0A                    ; $0A frames per animation frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; Frame image = animation frame (0 or 1), plus 2 when facing up (direction bit 8 shifted to 2).
    LDA ObjDir, X
    AND #$08
    LSR
    LSR
    CLC
    ADC ObjAnimFrame, X
    JMP Anim_DrawObjectMirrored
