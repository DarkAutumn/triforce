; ============================================================================
; FILE:     bank4/stalfos.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4670-4769
; SUBJECTS: Stalfos
; TOPIC:    Stalfos (object type $2A): a common wanderer that, in Q2 only, stops and fires a
;           sword shot (object type $57) in its facing direction.
; SEE ALSO: bank4/walker-wanderer-common.asm (Wanderer_Update: walking, turning toward the chase
;           target, the "wants to shoot" flag; Walker_Init is the Stalfos init routine);
;           bank4/shoot-helpers-candle.asm (Shooter_ShootIfWanted: slot search and the limit of 4 shots);
;           bank4/moblin-lynel-shooting.asm (Shooter_RunShootTimer: the same shoot timer logic for Moblin,
;           Lynel, Octorok); bank4/monster-shots-fireballs.asm (the sword shot's flight);
;           bank7/object-update-table.asm (Obj_TypeHpPairs, Obj_TypeAttributes).
; KEY FACTS:
;   - Stalfos walks as a common wanderer with turn rate $80 and never sets its own q-speed in Q1:
;     it keeps the room-entry default $20 (0.5 pixel per frame). Stalfos_Update
;   - Q1: Stalfos never shoots; Stalfos_Update returns right after drawing. Stalfos_Update
;   - Q2: while the shoot timer is 0, a shot sequence may start only on frames where Random,X >= $F8
;     (8 in 256), the Stalfos is not flashing from a hit, and its "wants to shoot" flag is set (it
;     turned toward the chase target on its last grid square). Stalfos_Update@PrepareToShoot
;   - Q2 shot sequence: ObjShootTimer = $30, then -1 per frame; q-speed is 0 (Stalfos stands still)
;     while the timer is nonzero; the sword shot is fired on the frame the timer reaches $10, that is
;     $20 frames after the start; walking resumes when it reaches 0. Stalfos_Update@SetShootTimer
;   - The shot is skipped (the timer keeps running) if the clock is held or the Stalfos is stunned
;     on the frame the timer is $10. A hit (invincibility timer nonzero) cancels the sequence:
;     the timer is zeroed. Stalfos_Update@PrepareToShoot, Stalfos_Update@SetShootTimer
;   - The shot fails when no monster slot is free or 4 monster shots are already in flight
;     (Shooter_ShootIfWanted); its q-speed is then $20 for that one frame and the timer still runs out.
;     Stalfos_Update@CheckResult
; CONTENTS: (generated; L = line in this file)
;   L0043 Stalfos_Update  code  Updates a Stalfos: wanders, checks collisions, draws; in Q2 also runs its sword-shot timer.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Stalfos: wanders, checks collisions, draws; in Q2 also runs its sword-shot timer.
; IN:   X = object slot.
; OUT:  Q2 only: writes ObjShootTimer,X and ObjQSpeedFrac,X (0 while the shoot timer runs, else $20);
;       may create a sword shot (type $57) in a free monster slot.
; NOTE: Q1 leaves ObjQSpeedFrac,X alone, so Stalfos walks at the default $20 given to every monster
;       slot on room entry. HP is $20 and the attribute byte $81 (type $2A in Obj_TypeHpPairs and
;       Obj_TypeAttributes); touching it costs Link $0080 = half a heart before rings
;       (Damage_ObjTypeDamageToLink).
; ----------------------------------------------------------------------------
Stalfos_Update:
    LDA #$80                    ; Turn rate $80 for the wanderer logic.
    JSR Wanderer_Update    ; Shoved: moves by the shove. Clock or stun: does not move.
    JSR Collision_CheckMonsterVsWeaponsAndLink
    LDA #$08                    ; Reload value 8 for the animation counter.
    JSR Obj_AnimateAndDrawCommon
    LDA #$20                    ; Walking q-speed $20 (0.5 pixel per frame),
    STA $01                     ; kept in [01] for the Q2 code below.

    ; Q1 (quest number 0 for the current save slot): return. Stalfos does not shoot.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    BEQ @Exit

    ; Q2. A shot sequence is running while shoot timer <> 0: go continue it.
    ; Otherwise only 8 of 256 values of Random,X ($F8-$FF) let a new one start; else return.
    LDA ObjShootTimer, X
    BNE @PrepareToShoot
    LDA Random, X
    CMP #$F8
    BCC @Exit

@PrepareToShoot:
    LDA $01                     ; Push the walking q-speed; it is popped at @SetSpeed
    PHA                         ; or replaced with 0 at @ZeroSpeed.

    ; If flashing from a hit (invincibility timer <> 0), then go set shoot timer to 0:
    ; a running sequence is cancelled.
    LDY #$00
    LDA ObjInvincibilityTimer, X
    BNE @SetShootTimer

    ; If shoot timer >= 1, then go store it decremented.
    LDY ObjShootTimer, X
    DEY
    BPL @SetShootTimer

    ; Shoot timer = 0. If the "wants to shoot" flag is clear, then go keep walking.
    ; (The wanderer code sets the flag when Stalfos turns toward the chase target.)
    LDA Shooter_ObjWantsToShoot, X
    BEQ @SetSpeed

    ; Start a shot sequence: shoot timer = $30 frames.
    LDY #$30

@SetShootTimer:
    TYA
    STA ObjShootTimer, X

    ; If shoot timer = 0 (ran out, or cancelled by a hit), then go walk.
    BEQ @SetSpeed

    ; Shoot timer <> 0: stand still. The shot leaves only on the frame the timer is $10,
    ; and not if the clock is held or the Stalfos is stunned on that frame.
    CPY #$10
    BNE @ZeroSpeed
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @ZeroSpeed

    ; Try to make a sword shot (object type $57) at Stalfos's position, facing its direction.
    ; C = 0 if the flag is clear, no monster slot is free, or 4 monster shots are in flight.
    LDA #$57
    STA $00
    JSR Shooter_ShootIfWanted

    ; If it failed, then go walk (through @CheckResult).
    BCC @CheckResult

    ; Succeeded: set the object timer to $80 and decrement Shooter_ObjShotCounter,X.
    ; Neither has an effect seen in Stalfos code: this routine and the wanderer and walker movement
    ; code it calls do not read ObjTimer for a Stalfos, and no walking-monster code reads the counter
    ; (it is only ever decremented; the same two instructions are in Shooter_ShootAndSetTimer).
    LDA #$80
    STA ObjTimer, X
    DEC Shooter_ObjShotCounter, X

@CheckResult:
    ; C is still the result of Shooter_ShootIfWanted. Failed: go walk at q-speed $20 this frame.
    BCC @SetSpeed

    ; Shot made: clear the "wants to shoot" flag and stand still.
    LDA #$00
    STA Shooter_ObjWantsToShoot, X

@ZeroSpeed:
    PLA                         ; Replace the q-speed on the stack with 0.
    LDA #$00
    PHA

@SetSpeed:
    PLA                         ; Q-speed: 0 (standing) or $20 (walking).
    STA ObjQSpeedFrac, X

@Exit:
    RTS
