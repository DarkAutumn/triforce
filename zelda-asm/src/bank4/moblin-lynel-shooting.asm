; ============================================================================
; FILE:     bank4/moblin-lynel-shooting.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1950-2165
; SUBJECTS: Lynel, Moblin, MonsterArrow, Shooter
; TOPIC:    Moblin and Lynel updates; the shoot timer shared by walking shooters (Moblin, Lynel,
;           Octorok): when they stop, when the shot leaves, what cancels it; the monster arrow's update.
; SEE ALSO: bank4/walker-wanderer-common.asm (Wanderer_MoveAndTurn: movement, turning, "wants to
;           shoot"); bank4/goriya.asm (Goriya_Update, used by the Lynel); bank4/octorok.asm (third
;           caller of Shooter_RunShootTimer); bank4/shoot-helpers-candle.asm (Shooter_ShootIfWanted,
;           Shooter_ShootLimited: the 4-shot limit, Shooter_Shoot); bank4/monster-shots-fireballs.asm
;           (MonsterShot_Update for the Lynel's
;           sword shot; MonsterShot_CheckLinkCollision, MonsterShot_Bounce, MonsterShot_DestroyCounted);
;           bank7/boomerang-arrow-food.asm (Boomerang_UpdateOrArrow, Arrow_Draw);
;           bank1/link-collision-harm.asm (which shots the shields block; damage to Link);
;           bank7/object-update-table.asm (Obj_TypeHpPairs, Obj_TypeAttributes).
; KEY FACTS:
;   - Moblin (types $03, $04) shoots monster arrows (type $5B); Lynel (types $01, $02) shoots sword
;     shots (type $57). Both walk at q-speed $20 (0.5 px/frame). Moblin_Update, Lynel_Update
;   - A shot takes $30 (48) frames during which the monster stands still (q-speed 0); the shot is
;     created when the timer reaches $10, 32 frames after it stopped. Shooter_RunShootTimer
;   - The sequence starts only when ObjShootTimer is 0 and Shooter_ObjWantsToShoot is set (the monster
;     has just turned toward the chase target, or for the Lynel the target is within $50 pixels on
;     the longer axis). Shooter_RunShootTimer
;   - Types $01, $03, $09, $0A test this every frame; types $02, $04, $07, $08 only on frames where
;     Random,X >= $F8 (8 of 256 values), so they shoot much less often. On the other frames the
;     q-speed is not written either: a fast Octorok of type $08 keeps its init speed $30 until the
;     gate first passes, then walks at $40. Shooter_RunShootTimer
;   - Hitting the monster (ObjInvincibilityTimer nonzero) resets the shoot timer to 0: a pending shot
;     is cancelled and the monster walks again. Shooter_RunShootTimer@TryShootingNow
;   - With the clock, or while the monster is stunned, the timer runs but no shot is created.
;     Shooter_RunShootTimer@SetShootTimer
;   - The shot is also lost when no monster slot is empty or 4 counted shots are already in flight
;     (MonsterShotCount; Shooter_ShootLimited in bank4/shoot-helpers-candle.asm). Shooter_ShootAndSetTimer
;   - Monster arrow: q-speed $80 (2 px/frame), ignores tiles, sparks for 3 frames at the room bounds,
;     bounces for 16 frames when Link's shield blocks it, is destroyed when it harms Link.
;     MonsterArrow_Update
;   - Moblin and Lynel do not draw or test collisions themselves (attribute byte $00 in
;     Obj_TypeAttributes): the object loop of the play update does both after the update.
;     Moblin_Update, Lynel_Update
; CONTENTS: (generated; L = line in this file)
;   L0055 Moblin_Update                    code  Updates a Moblin (types $03, $04): wanders toward Link with turn rate $A0 and shoots arrows.
;   L0069 Lynel_Update                     code  Updates a Lynel (types $01, $02): moves and aims with the Goriya routine and shoots sword shots.
;   L0096 Shooter_RunShootTimer            code  Runs a walking shooter's shoot timer: stop for $30 frames, fire at $10, and set its q-speed.
;   L0216 Shooter_ShootAndSetTimer         code  Shoots through Shooter_ShootIfWanted and, on success, sets the shooter's object timer to $80.
;   L0224 Shooter_ShootAndSetTimer_Return  code  Shared RTS of Shooter_ShootAndSetTimer and MonsterArrow_Update.
;   L0236 MonsterArrow_Update              code  Updates a monster's arrow (type $5B): flies at 2 pixels a frame, sparks out, or bounces off the shield.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Moblin (types $03, $04): wanders toward Link with turn rate $A0 and shoots arrows.
; IN:   X = object slot.
; NOTE: Walks at q-speed $20 (0.5 pixels a frame); shot = monster arrow (type $5B) through
;       Shooter_RunShootTimer. Drawing and collision tests are done by the object loop, not here.
; ----------------------------------------------------------------------------
Moblin_Update:
    LDA #$A0                    ; Turn rate $A0
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_MoveAndTurn   ; Move; turn toward the chase target; set or clear "wants to shoot".
    LDY #$5B                    ; Monster arrow object type
    JMP :+

; ----------------------------------------------------------------------------
; Updates a Lynel (types $01, $02): moves and aims with the Goriya routine and shoots sword shots.
; IN:   X = object slot.
; NOTE: Walks at q-speed $20 (0.5 pixels a frame); shot = sword shot (type $57) through
;       Shooter_RunShootTimer. Goriya_Update throws no boomerang for these types; it sets "wants to shoot"
;       when the chase target is less than $51 pixels away on the longer axis.
; ----------------------------------------------------------------------------
Lynel_Update:
    JSR Goriya_Update
    LDY #$57                    ; Sword shot object type
:
    LDA #$20                    ; Q-speed $20 (0.5 pixels a frame)

; ----------------------------------------------------------------------------
; Runs a walking shooter's shoot timer: stop for $30 frames, fire at $10, and set its q-speed.
; IN:   A = the monster's walking q-speed; Y = shot object type; X = object slot.
; OUT:  ObjQSpeedFrac,X = A (walking) or 0 (standing to shoot); ObjShootTimer,X updated; a shot may
;       be created. Uses [00] (shot type), [01] (q-speed).
; NOTE: Called by Moblin, Lynel (this file) and Octorok (bank4/octorok.asm, shot type $53).
; NOTE: Sequence once started: timer $30 -> 0, one step per frame; the monster stands still for
;       all $30 (48) frames and the shot leaves when the timer reaches $10, 32 frames after the
;       start. It starts when the timer is 0 and Shooter_ObjWantsToShoot,X is set.
; NOTE: Types $01, $03, $09, $0A (the Lynel, Moblin and Octoroks with the higher HP) run this logic
;       every frame. The other types run it with the timer at 0 only on frames where Random,X >=
;       $F8 (8 of 256 values), so they start a shot far less often; on the other frames this routine
;       returns without writing the q-speed, and ObjQSpeedFrac,X keeps its last value. For types
;       $02, $04, $07 that is $20 either way (the room-entry default or the init value). A type
;       $08 Octorok keeps its init q-speed $30 (0.75 px/frame) until the gate first passes; from
;       then on it has the $40 (1 px/frame) its update passes in A.
; NOTE: Being hit cancels a pending shot: while ObjInvincibilityTimer,X is nonzero the timer is
;       forced to 0. With the clock or while stunned the timer still counts but the shot at $10 is
;       skipped. If the shot cannot be made at $10 (no empty slot, 4 counted shots in flight, or
;       "wants to shoot" cleared meanwhile) it is skipped too.
; ----------------------------------------------------------------------------
Shooter_RunShootTimer:
    STA $01

    ; Types $01, $03, $09, $0A always run the timer logic. So does any type whose shoot timer is
    ; already running.
    LDA ObjType, X
    CMP #$03                    ; Moblin type $03
    BEQ @TryShootingNow
    CMP #$01                    ; Lynel type $01
    BEQ @TryShootingNow
    CMP #$09                    ; Octorok type $09 (slow, 2 HP)
    BEQ @TryShootingNow
    CMP #$0A                    ; Octorok type $0A (fast, 2 HP)
    BEQ @TryShootingNow
    LDA ObjShootTimer, X
    BNE @TryShootingNow

    ; Other types with the timer at 0: return unless Random,X >= $F8.
    LDA Random, X
    CMP #$F8
    BCC @Exit

@TryShootingNow:
    ; Summary
    ;
    ; if temporarily invincible then
    ;   shoot timer := 0
    ; elif shoot timer > 0 then
    ;   decrement shoot timer
    ; elif wants to shoot then
    ;   shoot timer := $30
    ;
    ; qspeed := [01]
    ;
    ; if shoot timer <> 0 then
    ;   if shoot timer = $10 and no clock and not stunned then
    ;     shoot
    ;     if succeeded then
    ;       wants to shoot := 0
    ;       qspeed := 0
    ;   else
    ;     qspeed := 0
    ;
    ; Push the walking q-speed; it is replaced by 0 below if the monster has to stand still.
    LDA $01
    STY $00                     ; [00] := shot object type, for the shooting routine.
    PHA
    LDY #$00

    ; If temporarily invincible (just hit), go set the shoot timer to 0.
    LDA ObjInvincibilityTimer, X
    BNE @SetShootTimer

    ; Else Y := shoot timer - 1.
    LDY ObjShootTimer, X
    DEY

    ; If Y >= 0, the timer was running: go store the decremented value.
    ; Else the timer was 0: if "wants to shoot" is set, start it at $30;
    ; if not, go set the walking speed and return.
    BPL @SetShootTimer
    LDA Shooter_ObjWantsToShoot, X
    BEQ @SetSpeed
    LDY #$30

@SetShootTimer:
    ; shoot timer := Y
    TYA
    STA ObjShootTimer, X

    ; If the new shoot timer = 0, go set the walking speed.
    BEQ @SetSpeed

    ; The timer is running. Unless it is exactly $10, stand still.
    CPY #$10
    BNE @ZeroSpeed

    ; Timer = $10: time to shoot. If we have the clock, or the monster is stunned, then skip the
    ; shot and stand still.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @ZeroSpeed

    ; Shoot.
    LDA $00
    JSR Shooter_ShootAndSetTimer

    ; If it failed (C clear), go set the walking speed for this frame; the timer keeps running.
    BCC @SetSpeed

    ; It succeeded: clear "wants to shoot", and stand still.
    LDA #$00
    STA Shooter_ObjWantsToShoot, X

@ZeroSpeed:
    PLA                         ; Replace the speed that was pushed with 0.
    LDA #$00
    PHA

@SetSpeed:
    ; Set the speed that was decided: 0 or the value passed in [01].
    PLA
    STA ObjQSpeedFrac, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Shoots through Shooter_ShootIfWanted and, on success, sets the shooter's object timer to $80.
; IN:   A = shot object type; X = shooter's object slot (must be CurObjIndex).
; OUT:  C set when the shot was made; then Y = the shot's slot, ObjTimer,X = $80 and
;       Shooter_ObjShotCounter,X is decremented. C clear when "wants to shoot" is 0, no slot is empty
;       or 4 shots are in flight.
; NOTE: The shot starts in state $10 with timer 0 at the shooter's position, facing as the shooter.
; NOTE: Only Shooter_RunShootTimer calls this. Shooter_ObjShotCounter,X is a write-only byte: nothing
;       reads it for a Moblin, Lynel or Octorok (the other RAM symbols at $437 belong to fireballs,
;       flyers, Digdogger, Dodongo and Manhandla). The $80 timer has no reader either:
;       Shooter_RunShootTimer, Wanderer_MoveAndTurn, Goriya_Update (for these types), Octorok_Update and
;       Walker_Move do not read ObjTimer.
; ----------------------------------------------------------------------------
Shooter_ShootAndSetTimer:
    JSR Shooter_ShootIfWanted
    BCC Shooter_ShootAndSetTimer_Return             ; Not shot.
    LDA #$80
    STA ObjTimer, X
    DEC Shooter_ObjShotCounter, X

; Shared RTS of Shooter_ShootAndSetTimer and MonsterArrow_Update.
Shooter_ShootAndSetTimer_Return:
    RTS

; ----------------------------------------------------------------------------
; Updates a monster's arrow (type $5B): flies at 2 pixels a frame, sparks out, or bounces off the shield.
; IN:   X = the arrow's object slot.
; NOTE: States (high nibble of ObjState): $10 flying; $20 spark after Obj_MoveShot refused the move at
;       the room bounds (3 frames, then the arrow is destroyed); $30 bouncing off Link's shield (16
;       frames, then destroyed: MonsterShot_Bounce). Tiles do not stop it: no tile test is made here.
; NOTE: An arrow that harms Link is destroyed at once. Every way out goes through
;       MonsterShot_DestroyCounted or MonsterShot_Bounce, which decrement MonsterShotCount.
; ----------------------------------------------------------------------------
MonsterArrow_Update:
    ; Q-speed $80 (2 pixels a frame).
    LDA #$80
    STA ObjQSpeedFrac, X

    ; If the arrow's timer has not expired, then only draw it, and see if the shooter still exists.
    ; (The shooting code in this file creates arrows with timer 0.)
    LDA ObjTimer, X
    BNE @CheckShooter

    ; If the arrow is sparking (state $2x), then go update the base arrow.
    LDA a:ObjState, X
    AND #$F0
    CMP #$20
    BEQ @UpdateBase

    ; If the arrow is not flying (state $1x), then go handle it bouncing off of Link's shield.
    CMP #$10
    BNE @CheckBounce

    ; Flying: [0F] := facing direction, then update the base arrow.
    ; The assignment has no effect: Obj_MoveShot (called by Boomerang_UpdateOrArrow) stores the direction
    ; it is given in [0F] itself (Room_BoundDirA).
    LDA ObjDir, X
    STA $0F
    JMP @UpdateBase

@CheckShooter:
    ; If the shooter's slot (ObjRefId,X) is empty, then zero the arrow's timer, so that it flies on
    ; the next frame. Either way, draw the arrow.
    ; Not reached: only the Moblin code above makes type $5B, Shooter_Shoot zeroes the arrow's
    ; timer, Obj_Init sets no timer for types from $53 up, and no other code writes the timer
    ; of an arrow's slot. (trace) Six Moblin rooms, 21,976 arrow frames: the timer was always 0.
    LDY ObjRefId, X
    LDA ObjType, Y
    BNE :+
    STA ObjTimer, X
:
    JMP Arrow_Draw

@UpdateBase:
    JSR Boomerang_UpdateOrArrow  ; Move (state $1x) or count down the spark (state $2x); draw.

    ; If the arrow is no longer flying, then go see whether it is bouncing.
    LDA a:ObjState, X
    AND #$F0
    CMP #$10
    BNE @CheckBounce
    JSR MonsterShot_CheckLinkCollision  ; A blocked arrow gets state $30 here.

    ; If [06] = 0, then Link was not harmed: there was no collision, or the shield blocked the arrow.
    ; So, return.
    LDA $06
    BEQ Shooter_ShootAndSetTimer_Return

    ; Else the arrow harmed Link. Destroy it.
    JMP MonsterShot_DestroyCounted

@CheckBounce:
    ; A = high nibble of the state. $30: bounce off the shield. Anything else: return.
    CMP #$30
    BNE Shooter_ShootAndSetTimer_Return
    JMP MonsterShot_Bounce
