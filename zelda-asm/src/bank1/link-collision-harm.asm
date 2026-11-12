; ============================================================================
; FILE:     bank1/link-collision-harm.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5592-5828
; SUBJECTS: (none)
; TOPIC:    Link-vs-monster collision, shield, HarmLink, ring, death
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0027 CheckLinkCollision         code  Params:
;   L0048 CheckLinkCollisionPreinit  code
;   L0145 L73A9_Exit                 code
;   L0148 HarmLink                   code
;   L0171 Link_BeHarmed              code  Params:
;   L0252 Link_BeHarmed_Return       code
; ============================================================================

; Params:
; X: monster object index
;
; Returns:
; [00]: 0 for Link slot
; [06]: 1 if objects collide
; [09]: 0 for Link damage type (none)
; [0C]: 1 if objects collide
; [034B]: ShotCollidesWithLink
;
CheckLinkCollision:
    JSR GetObjectMiddle

    ; Reset variables to return.
    LDA #$00
    STA ShotCollidesWithLink
    STA $06
    STA $09
    STA $0C
    LDY #$00
    STY $00

    ; If Link is invincible, or we have the magic clock,
    ; or Link is stunned, or the monster is stunned;
    ; then return no collision.
    LDA ObjInvincibilityTimer
    ORA InvClock
    ORA ObjStunTimer
    ORA ObjStunTimer, X
    BNE L73A9_Exit

CheckLinkCollisionPreinit:
    ; If Link is halted or paralyzed, then return no collision.
    LDA ObjState
    CMP #$40
    BEQ L73A9_Exit
    LDA LinkParalyzed
    BNE L73A9_Exit

    ; If the object is a monster's shot/projectile, and it's not in a
    ; state that it can do damage; then return no collision.
    LDA ObjType, X
    CMP #$53
    BCC :+
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BNE L73A9_Exit
:
    ; Store Link's midpoint coordinates in [04] and [05].
    LDA ObjX
    CLC
    ADC #$08
    STA $04
    LDA ObjY
    CLC
    ADC #$08
    STA $05

    ; If both objects are 9 pixels apart or more, then return no collision.
    LDA #$09
    JSR DoObjectsCollide
    BEQ L73A9_Exit

    ; They collide. Now find out how they collide.
    ;
    ; If it's a regular monster, go harm Link.
    LDA ObjType, X
    CMP #$53
    BCC HarmLink

    ; Here, the attacker is a monster's shot/projectile.
    ; Flag this kind of collision.
    INC ShotCollidesWithLink

    ; If object type = Fireball2 ($56) or $5A, or Link is not idle;
    ; then go harm Link.
    CMP #$56
    BEQ HarmLink
    CMP #$5A
    BEQ HarmLink
    LDA ObjState
    AND #$F0
    BNE HarmLink

    ; If Link and the shot are not facing opposite directions,
    ; then go harm Link.
    ;
    ; Note the bitwise AND in the test. It picks out one direction
    ; component of a monster that can move diagonally.
    LDA ObjDir
    ORA ObjDir, X
    AND #$0C
    CMP #$0C
    BEQ @FacingOpposite
    LDA ObjDir
    ORA ObjDir, X
    AND #$03
    CMP #$03
    BNE HarmLink

@FacingOpposite:
    ; Link and the shot are facing opposite directions.
    ; Link can parry certain weapons/shots.
    ;
    ; For flying rock, $54, arrow, and boomerang, this means
    ; no harmful collision; but it still counts as a shot collision.
    ; So, go treat this as a parry.
    LDA ObjType, X
    CMP #$55
    BCC @Parry
    CMP #$5B
    BCS @Parry

    ; If the magic shield is missing, then the rest of the shots do damage.
    ; So, go harm Link.
    LDA InvMagicShield
    BEQ HarmLink

@Parry:
    ; Else Link parries.
    ;
    ; Play the "parry" tune, and cancel the collision (reset [06]).
    LDA #$01
    STA Tune0Request
    LDA #$00
    STA $06

L73A9_Exit:
    RTS

HarmLink:
    JSR BeginShove

    ; Flag collision with [0C], even though [06] was already set.
    INC $0C

    ; Unpack the damage points for the monster's type.
    ; The low nibble is the high byte.
    ; (Damage points byte AND $F0) is the low byte.
    LDY ObjType, X
    LDA ObjTypeToDamagePoints, Y
    PHA
    AND #$0F
    STA $0D
    PLA
    AND #$F0
    STA $0E

; Params:
; X: object index of the attacker
; [0D]: damage points high byte
; [0E]: damage points low byte
;
Link_BeHarmed:
    ; If object type is not whirlwind, then make the "hurt" sound effect.
    LDY ObjType, X
    CPY #$2E
    BEQ :+
    LDA #$08
    JSR PlaySample
:
    ; For every level of ring (1 or 2), divide the 16-bit damage amount in [0E:0D] by 2.
    LDY InvRing
    BEQ @ResetHelp
:
    LSR $0D
    ROR $0E
    DEY
    BNE :-

@ResetHelp:
    ; Reset values used in special item drops.
    LDA #$00
    STA WorldKillCount
    STA HelpDropCount
    STA HelpDropValue

    ; If heart partial >= low damage points byte, then simply
    ; subtract the damage byte from heart partial.
    ; Else go borrow from full hearts.
    LDA HeartPartial
    CMP $0E
    BCC @BorrowHeart
    SEC
    SBC $0E
    STA HeartPartial

    ; If full hearts >= high damage points byte, then simply
    ; subtract the damage byte from full hearts.
    ; Else go handle Link dying.
    LDA HeartValues
    AND #$0F
    CMP $0D
    BCC @HandleDied
    LDA HeartValues
    SEC
    SBC $0D
    STA HeartValues
    RTS

@BorrowHeart:
    ; This isn't a simple case of borrowing. For example, supposing:
    ;   full heart=4 partial heart=$70
    ; The straightforward way to subtract $80 would yield $3F0.
    ; But the method here yields $3EF, because partial heart is
    ; considered full when = $FF instead of $100.
    LDA $0E
    SEC
    SBC HeartPartial
    STA $0E
    LDA HeartValues
    AND #$0F
    BEQ @HandleDied             ; If we can't borrow from full hearts, then Link died.
    DEC HeartValues
    LDA #$FF
    STA HeartPartial
    BNE @ResetHelp

@HandleDied:
    ; Link died.
    ;
    ; Set full hearts, partial heart, and Link's object state to 0.
    ; Make Link face down.
    ; Go to mode $11.
    LDA HeartValues
    AND #$F0
    STA HeartValues
    JSR EndGameMode
    STA HeartPartial
    STA ObjState
    LDA #$11
    STA GameMode
    LDA #$04
    STA ObjDir
Link_BeHarmed_Return:
    RTS
