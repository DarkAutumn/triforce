; ============================================================================
; FILE:     bank4/goriya.asm             BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 434-624
; SUBJECTS: (none)
; TOPIC:    Update of Goriya (types $05, $06), also the movement step of Lynel and Armos: walk the
;           grid, face the chase target along the longer axis when it is near. Shared tail of all
;           walking monsters: copy facing to input direction; a Goriya throws its boomerang there.
; SEE ALSO: bank4/walker-wanderer-common.asm (Wanderer_TargetPlayer, which ends here; InitWalker);
;           bank4/shoot-helpers-candle.asm (_ShootIfWanted, ShootLimited: free slot, 4-shot limit);
;           bank7/boomerang-arrow-food.asm and bank7/boomerang-return-draw.asm (the boomerang's
;           flight and the catch that releases the Goriya); bank1/monster-collisions.asm (a dying
;           Goriya removes its boomerang); bank4/moblin-lynel-shooting.asm (UpdateLynel);
;           bank4/ghini-armos.asm (UpdateArmos); bank7/walker-move.asm (Walker_Move).
; KEY FACTS:
;   - On every grid point ($10 pixels) a Goriya, Lynel or Armos compares the horizontal and vertical
;     distances to the chase target; if the longer one is < $51 pixels it faces the target along
;     that axis and sets "wants to shoot". UpdateGoriya
;   - Type $05 (blue Goriya, HP 5, sprite palette 1) throws whenever its ObjTimer is 0 and it wants
;     to shoot; type $06 (red Goriya, HP 3, sprite palette 2) in addition needs Random,X to be $23
;     or $77 on that frame (2 chances in 256). L_Walker_SetInputDirAndTryShootingBoomerang
;   - No throw with the clock, while stunned, without a free monster slot, or with 4 counted monster
;     shots already in flight. L_Walker_SetInputDirAndTryShootingBoomerang, ShootLimited
;   - After a throw the Goriya's state is $80 and its update returns at once: it stands still until
;     the boomerang comes back and resets the state. UpdateGoriya
;   - The boomerang (type $5C) starts at the Goriya's position and facing, state $10, q-speed $A0
;     (2.5 pixels a frame), range $51 pixels. L_Walker_SetInputDirAndTryShootingBoomerang
;   - The "not while state bit 7 is set" test is skipped when the object in the NEXT slot (X+1) is
;     type $1E (Armos): the code reads ObjType+1,X. UpdateGoriya
; CONTENTS: (generated; L = line in this file)
;   L0045 UpdateGoriya                                 code  Moves a Goriya (also Lynel, Armos) and on a grid point turns it toward a near chase target.
;   L0154 L_Walker_SetInputDirAndTryShootingBoomerang  code  Sets a walking monster's input direction to its facing; a Goriya may then throw its boomerang.
; ============================================================================

; ----------------------------------------------------------------------------
; Moves a Goriya (also Lynel, Armos) and on a grid point turns it toward a near chase target.
; IN:   X = object slot; ChaseTargetX/Y
; OUT:  ObjDir,X, Shooter_ObjWantsToShoot,X; uses [00]-[03], [0E]
; NOTE: Update routine of types $05 and $06; called first by UpdateLynel and UpdateArmos.
;       While ObjState,X has bit 7 set (a Goriya whose boomerang is out) nothing happens: no
;       movement, no turn. The exemption from that test reads the type of slot X+1, not X.
;       On a grid point with q-speed <> 0: [00] = |dy|, [01] = |dx| to the chase target; the longer
;       one (dy on a tie) is used; if it is < $51 pixels the monster faces the target along that
;       axis and "wants to shoot" = 1, else "wants to shoot" = 0 and the facing stays.
; ----------------------------------------------------------------------------
UpdateGoriya:
    ; A = type of the object in the next slot (X+1).
    LDA ObjType+1, X

    ; If that is not an Armos ($1E) and the high bit of this monster's state is set, then return:
    ; the monster is waiting for its boomerang.
    CMP #$1E
    BEQ :+
    LDA ObjState, X
    BMI @Exit
:
    JSR Walker_Move

    ; If not being shoved, then go handle other activities.
    LDA ObjShoveDir, X
    BEQ @AfterMove

@Exit:
    RTS

@AfterMove:
    ; If q-speed = 0 or the monster is between grid points, then go try shooting.
    LDA ObjQSpeedFrac, X
    BEQ L_Walker_SetInputDirAndTryShootingBoomerang
    LDA ObjGridOffset, X
    AND #$0F
    BNE L_Walker_SetInputDirAndTryShootingBoomerang

    ; On a grid point (offset 0, $10 or $F0): reset the grid offset to 0.
    STA ObjGridOffset, X

    ; If Link's state = $FF, go try shooting.
    ; What puts Link in state $FF is (unverified).
    LDA ObjState
    CMP #$FF
    BEQ L_Walker_SetInputDirAndTryShootingBoomerang

    ; [00] = absolute vertical distance between the chase target and the monster.
    ; [02] = vertical direction from the monster to the chase target:
    ;        down (4) if the target is at the same Y or below, else up (8).
    LDA #$04
    STA $02
    LDA ChaseTargetY
    LDY ObjY, X
    CMP ObjY, X
    BCS :+
    LDA ObjY, X                 ; Target is above: swap the operands, direction becomes 8.
    LDY ChaseTargetY
    ASL $02
:
    STY $0E
    SEC
    SBC $0E
    STA $00

    ; [01] = absolute horizontal distance between the chase target and the monster.
    ; [03] = horizontal direction from the monster to the chase target:
    ;        right (1) if the target is at the same X or to the right, else left (2).
    LDA #$01
    STA $03
    LDA ChaseTargetX
    LDY ObjX, X
    CMP ObjX, X
    BCS :+
    LDA ObjX, X                 ; Target is to the left: swap the operands, direction becomes 2.
    LDY ChaseTargetX
    ASL $03
:
    STY $0E
    SEC
    SBC $0E
    STA $01

    ; Y = 0 to use the vertical distance and direction ([00], [02]) when the vertical distance is
    ; >= the horizontal one; else 1 for the horizontal pair ([01], [03]).
    LDY #$00
    LDA $00
    CMP $01
    BCS :+
    INY
:
    ; Reset the "wants to shoot" flag.
    LDA #$00
    STA Shooter_ObjWantsToShoot, X

    ; If the distance chosen < $51 pixels, then
    ; set the "wants to shoot" flag to 1,
    ; and face in the chosen direction.
    LDA $0000, Y
    CMP #$51
    BCS L_Walker_SetInputDirAndTryShootingBoomerang
    INC Shooter_ObjWantsToShoot, X
    LDA $0002, Y
    STA ObjDir, X

; ----------------------------------------------------------------------------
; Sets a walking monster's input direction to its facing; a Goriya may then throw its boomerang.
; IN:   X = object slot; Shooter_ObjWantsToShoot,X
; OUT:  ObjInputDir,X = ObjDir,X. After a throw: ObjState,X = $80, ObjRefId links thrower and
;       boomerang both ways, Shooter_ObjWantsToShoot,X = 0, ObjTimer,X = Random,X AND $3F.
; NOTE: Jumped to every frame by UpdateGoriya and by Wanderer_TargetPlayer (all wanderers); for any
;       type other than $05 and $06 it returns after setting the input direction.
;       Throw conditions, all required: type $05 (blue), or type $06 (red) with Random,X = $23 or
;       $77; ObjTimer,X = 0; no clock (InvClock) and not stunned; "wants to shoot"; a free monster
;       slot; fewer than 4 counted monster shots (_ShootIfWanted, ShootLimited).
;       The boomerang: type $5C, at the thrower's position and facing, state $10, q-speed $A0
;       (2.5 pixels a frame), range $51 pixels. When it is caught, the boomerang code sets the
;       thrower's state to 0 and its ObjTimer to $30, $50 or $70 frames, replacing the timer set here.
; ----------------------------------------------------------------------------
L_Walker_SetInputDirAndTryShootingBoomerang:
    ; Set the input direction to the facing direction.
    LDA ObjDir, X
    STA ObjInputDir, X

    ; A = $5C, the monster boomerang object type.
    LDA #$5C

    ; If the monster is type $05 (blue Goriya), go check the timer: it needs no random roll.
    LDY ObjType, X
    CPY #$05
    BEQ @CheckTimerToShoot

    ; If it's not type $06 (red Goriya) either, then return.
    CPY #$06
    BNE @Exit

    ; Type $06 (red): if the random byte is neither $23 nor $77, then return.
    LDA Random, X
    CMP #$23
    BEQ @CheckTimerToShootBoomerang
    CMP #$77
    BNE @Exit

@CheckTimerToShootBoomerang:
    ; A = $5C, the monster boomerang object type.
    LDA #$5C

@CheckTimerToShoot:
    ; If object timer <> 0, return.
    LDY ObjTimer, X
    BNE @Exit

    ; Store in [00] the type of object the monster will throw.
    STA $00

    ; If Link has the clock, or the monster is stunned, then return.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @Exit

    ; Try to shoot (Y = the new object's slot). Return, if it failed (C = 0).
    LDA $00
    JSR _ShootIfWanted
    BCC @Exit

    ; Set the monster's state to $80: waiting for the boomerang.
    LDA #$80
    STA ObjState, X

    ; Reset the "wants to shoot" flag.
    LDA #$00
    STA Shooter_ObjWantsToShoot, X

    ; The boomerang remembers the thrower's slot.
    TXA
    STA ObjRefId, Y

    ; The thrower remembers the boomerang's slot.
    TYA
    STA ObjRefId, X

    ; Set the boomerang's state to $10 (flying out).
    LDA #$10
    STA a:ObjState, Y

    ; Fly at q-speed $A0 (2.5 pixels a frame).
    LDA #$A0
    STA ObjQSpeedFrac, Y

    ; Fly out up to $51 pixels.
    LDA #$51
    STA ObjMovingLimit, Y

    ; Metastate 0: no spawn cloud, the boomerang updates at once.
    LDA #$00
    STA ObjMetastate, Y

    ; The first animation frame lasts 3 frames.
    LDA #$03
    STA ObjAnimCounter, Y

    ; Set the monster's timer to a random 0-$3F frames.
    LDA Random, X
    AND #$3F
    STA ObjTimer, X

@Exit:
    RTS
