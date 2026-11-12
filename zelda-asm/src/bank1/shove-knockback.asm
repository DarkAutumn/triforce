; ============================================================================
; FILE:     bank1/shove-knockback.asm    BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 6495-6749
; SUBJECTS: Filler
; TOPIC:    BeginShove: starts a knockback (direction, distance, invincibility timer) for Link hit by a
;           monster, shot or fire, or for a monster hit by one of Link's weapons. Then the tail of bank 1:
;           padding, the bank's copy of the reset stub, its bank-switch routine and its vectors.
; SEE ALSO: bank7/shove.asm (Obj_Shove: moves the object along the knockback set up here);
;           bank1/link-collision-harm.asm (HarmLink calls BeginShove, then takes the hearts);
;           bank1/sword-bomb-arrow-collision.asm (weapon hits that end in BeginShove; Darknut parry);
;           bank1/object-overlap-test.asm (DoObjectsCollideWithThresholds leaves the distance in [0B]);
;           bank7/fire.asm (Link burned by his own fire); bank4/gohma.asm (Gohma's part counter [0F]);
;           bank7/object-stubs-destroy-monster.asm (DecrementInvincibilityTimer);
;           bank7/reset-mmc1-vectors.asm (the reset code the stub here duplicates).
; KEY FACTS:
;   - Link hit: ObjInvincibilityTimer = $18 (about 48 frames, it drops every second frame) and
;     ObjShoveDistance = $20 pixels. Nothing happens while his timer is still nonzero. BeginShove
;   - Monster hit by a weapon: ObjInvincibilityTimer = $10 (about 32 frames) and ObjShoveDistance = $40
;     pixels. Nothing happens while its timer is still nonzero. BeginShove@SetShoveDir
;   - A monster whose ObjInvincibilityMask has the weapon's damage type bit [09] is never shoved and gets
;     no invincibility timer. BeginShove
;   - Link is pushed away from the attacker along one axis: off the grid (ObjGridOffset <> 0) the axis he
;     faces; on the grid, horizontal when the vertical distance [0B] between the midpoints is < 4 pixels,
;     else vertical. BeginShove@UseDistance
;   - A monster is pushed in the direction the weapon faces (ObjDir of the weapon slot), not away from
;     Link. BeginShove@MonsterDefender
;   - After hitting Link a monster turns around (ObjDir reversed), unless its ObjAttr has bit $80 or it
;     is a Vire (type $12). Not done when Link was invincible. BeginShove
;   - Gohma (types $33, $34) is shoved only when the part hit is 3 or 4 of its 5 collision parts (the
;     ones at Gohma's ObjX and ObjX - 8) and Gohma_ObjEyeState is 3 (fully open). BeginShove@Gohma
;   - ObjShoveDir gets bit $80 ("knockback not started yet", cleared by Obj_Shove); bit $40 is added for
;     monsters with ObjAttr bit $80, and no code reads that bit. BeginShove@MonsterDefender
;   - The reset stub at $BF50 has the same bytes as the fixed bank's reset code except that its two JSR
;     targets are $BFxx instead of $FFxx: it ends by switching bank 7 in and jumping to $E440.
;     SwitchBank_Local1
; CONTENTS: (generated; L = line in this file)
;   L0065 BeginShove         code  Starts a knockback: sets the shove direction, distance and invincibility timer of the object hit.
;   L0233 Filler_7751        data  DATA: filler ($FF), 31 bytes: the end of the BANK_01_CODE segment (the code that runs from RAM).
;   L0283 SwitchBank_Local1  code  Bank 1's copy of SwitchBank: maps program bank A at $8000-$BFFF through the MMC1.
; ============================================================================

; ----------------------------------------------------------------------------
; Starts a knockback: sets the shove direction, distance and invincibility timer of the object hit.
; IN:   X = the monster's object slot: the attacker when Link is hit, the object hit when a weapon
;           hits a monster. X >= $0D is one of Link's own fires burning Link.
;       [00] = 0 when Link is the one hit, else the object slot of the weapon that hit monster X.
;       [09] = damage type bit of the weapon (as in ObjInvincibilityMask); 0 when Link is hit.
;       [0B] = vertical distance in pixels between the two midpoints (left there by
;              DoObjectsCollideWithThresholds); used only when Link is hit while on the grid.
;       [0F] = Gohma only: collision part counter 5..1 of Gohma_CheckCollisions.
; OUT:  Link hit:    ObjShoveDir = direction away from X with bit $80, ObjShoveDistance = $20,
;                    ObjInvincibilityTimer = $18; ObjDir,X reversed for most monsters.
;       Monster hit: ObjShoveDir,X = weapon's direction with bit $80, ObjShoveDistance,X = $40,
;                    ObjInvincibilityTimer,X = $10.
;       Uses [04], [05], [08]. X is kept; Y = [00].
; NOTE: Does nothing when the object hit still has a nonzero ObjInvincibilityTimer, or when the
;       monster's ObjInvincibilityMask contains [09]. It deals no damage: the callers do that.
;       The timer drops once every second frame (DecrementInvincibilityTimer), so $18 = about 48
;       frames for Link and $10 = about 32 frames for a monster.
;       Obj_Shove does the moving, up to 4 pixels per call. Its first call moves nothing and may
;       cancel the knockback: for an object between grid points (ObjGridOffset <> 0) a shove across
;       the axis it faces is dropped for a monster (the invincibility timer set here stays) and
;       turned into the direction opposite Link's facing for Link.
; ----------------------------------------------------------------------------
BeginShove:
    ; A monster (X < $0D) that is invincible to the damage type in [09] is not shoved.
    ; When Link is the one hit, [09] is 0 and this test never returns.
    ; Slots $0D and up (Link's fire) have no ObjInvincibilityMask entry and skip the test.
    LDY $00
    CPX #$0D
    BCS @CalcDir
    LDA ObjInvincibilityMask, X
    AND $09
    BNE @Exit

@CalcDir:
    ; Work out the direction from object X to object Y: the direction that pushes Link away
    ; from the monster. The result in [08] is used only when Link is the one hit; for a monster
    ; it is replaced by the weapon's direction at @MonsterDefender.
    ;
    ; Start with the vertical axis: [08] = 8 (up), [04] = Y coordinate of X, [05] = Y coordinate of Y.
    LDA #$08
    STA $08
    LDA ObjY, X
    STA $04
    LDA a:ObjY, Y
    STA $05

    ; Link hit while between grid points (ObjGridOffset <> 0): take the axis he faces.
    ; ObjDir bits 1 and 2 (right, left) select the horizontal axis.
    CPY #$00
    BNE @UseDistance
    LDA ObjGridOffset
    BEQ @UseDistance
    LDA ObjDir
    AND #$03
    BNE @CheckHorizontal
    BEQ @CheckVertical

@UseDistance:
    ; Link on a grid point (or a monster hit, where the result is not used): [0B] < 4 selects the
    ; horizontal axis, else the vertical one. For Link, [0B] is the vertical distance between his
    ; midpoint and the attacker's, left by DoObjectsCollideWithThresholds: both paths that hit Link
    ; (HarmLink, and Link's own fire) run that test just before coming here. So an attacker within
    ; 3 pixels of Link's height pushes him sideways, and any other pushes him up or down.
    LDA $0B
    CMP #$04
    BCS @CheckVertical

@CheckHorizontal:
    ; Horizontal axis: [08] = 2 (left), [04] = X coordinate of X, [05] = X coordinate of Y.
    LDA #$02
    STA $08
    LDA ObjX, X
    STA $04
    LDA a:ObjX, Y
    STA $05

@CheckVertical:
    ; Attacker's coordinate >= the other's (attacker to the right or below, or level with it):
    ; keep [08] (2 left or 8 up). Else shift it right into the opposite direction (1 right or
    ; 4 down). Either way [08] points from the attacker toward the object hit.
    LDA $04
    CMP $05
    BCS @CheckDefender
    LSR $08

@CheckDefender:
    ; [00] <> 0: a weapon hit monster X.
    CPY #$00
    BNE @MonsterDefender

    ; Link is the one hit. No new knockback while he is still invincible.
    LDA ObjInvincibilityTimer
    BNE @Exit

    ; Bit $80 of ObjShoveDir marks a knockback that Obj_Shove has not started yet.
    LDA $08
    ORA #$80
    STA ObjShoveDir

    ; Link: invincible for timer value $18 (48 frames), pushed $20 pixels.
    LDA #$18
    STA ObjInvincibilityTimer
    LDA #$20
    STA ObjShoveDistance

    ; Attacker in slot $0D or above (Link's own fire): it has no direction to reverse.
    CPX #$0D
    BCS @Exit

    ; A monster or monster's shot hit Link. It keeps its direction when its ObjAttr has bit $80
    ; ("does not turn around after hitting Link") ...
    LDA ObjAttr, X
    AND #$80
    BNE @Exit

    ; ... or when it is a Vire (type $12).
    LDA ObjType, X
    CMP #$12
    BEQ @Exit

    ; Every other attacker turns around.
    LDA ObjDir, X
    JSR GetOppositeDir
    STA ObjDir, X

@Exit:
    RTS

@MonsterDefender:
    ; A weapon hit monster X: the knockback direction is the direction the weapon faces
    ; (ObjDir of weapon slot Y), whatever was calculated above.
    LDA a:ObjDir, Y
    STA $08

    ; For a monster with ObjAttr bit $80, bit $40 is added to the shove direction.
    ; Obj_Shove masks the direction with $0F, $03, $0C or $05 wherever it reads it, and every other
    ; reader of ObjShoveDir only tests it for zero, so the bit has no effect.
    LDA ObjAttr, X
    AND #$80
    BEQ @CheckInvincible
    LDA $08
    ORA #$40
    STA $08

@CheckInvincible:
    ; No new knockback (and no new invincibility time) while the monster is still invincible.
    LDA ObjInvincibilityTimer, X
    BNE @Exit2

    ; Gohma (types $33 and $34) has extra conditions.
    LDA ObjType, X
    CMP #$33
    BEQ @Gohma
    CMP #$34
    BNE @SetShoveDir

@Gohma:
    ; [0F] is the part counter of Gohma_CheckCollisions (5 parts, 8 pixels apart, counted 5 down
    ; to 1 from left to right: part 4 is at Gohma's ObjX - 8, part 3 at ObjX). Only parts 3 and 4
    ; can be shoved; a hit on any other part starts no knockback and no invincibility time.
    LDA $0F
    CMP #$03
    BEQ @CheckEyeOpen
    CMP #$04
    BNE @Exit2

@CheckEyeOpen:
    ; The eye must show image 3 (fully open). $046B is Gohma_ObjEyeState (this bank does not include
    ; the file that defines the name).
    LDA $046B, X
    CMP #$03
    BNE @Exit2

@SetShoveDir:
    ; Bit $80 of ObjShoveDir marks a knockback that Obj_Shove has not started yet.
    LDA $08
    ORA #$80
    STA ObjShoveDir, X

    ; Monster: pushed $40 pixels, invincible for timer value $10 (32 frames).
    LDA #$40
    STA ObjShoveDistance, X
    LDA #$10
    STA ObjInvincibilityTimer, X

@Exit2:
    RTS

; DATA: filler ($FF), 31 bytes: the end of the BANK_01_CODE segment (the code that runs from RAM).
; The linker fills the rest of the bank up to the reset stub at $BF50 with $FF as well.
Filler_7751:
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF

.SEGMENT "BANK_01_ISR"




; Bank 1's copy of the reset handler, at $BF50, where this bank's reset vector points. The CPU
; reads vectors at $FFFA-$FFFF, so the vector is seen only if bank 1 is mapped at $C000-$FFFF,
; while the vector and the two JSRs below name $BFxx addresses (bank 1 at $8000-$BFFF); whether
; the stub can ever run is (unverified). The bytes are 6502 code, the same instructions as
; IsrReset in the fixed bank:
;   $BF50  SEI / CLD / LDA #$00 / STA $2000    interrupts off, PPU control 0 (no NMI)
;   $BF57  LDX #$FF / TXS                      stack pointer $FF
;   $BF5A  LDA $2002 / AND #$80 / BEQ $BF5A    wait for a vertical blank
;   $BF61  LDA $2002 / AND #$80 / BEQ $BF61    wait for a second one
;   $BF68  ORA #$FF / STA $8000 / STA $A000 / STA $C000 / STA $E000
;                                              bit 7 set: reset the MMC1 shift register
;   $BF76  LDA #$0F / JSR $BF98                MMC1 control = $0F
;   $BF7B  LDA #$00 / STA $A000 / (LSR / STA $A000) x 4
;                                              MMC1 CHR bank 0 register = 0
;   $BF90  LDA #$07 / JSR $BFAC                switch program bank 7 in (SwitchBank_Local1)
;   $BF95  JMP $E440                           the main program's entry in the fixed bank, the same
;                                              address IsrReset jumps to
;   $BF98  STA $8000 / (LSR / STA $8000) x 4 / RTS
;                                              unlabeled copy of SetMMC1Control: writes the low 5 bits
;                                              of A, lowest first, to the MMC1 control register
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60

; ----------------------------------------------------------------------------
; Bank 1's copy of SwitchBank: maps program bank A at $8000-$BFFF through the MMC1.
; IN:   A = bank number (low 5 bits).
; NOTE: Writes the 5 bits, lowest first, to the MMC1 program bank register ($E000). At $BFAC. Called
;       by the reset stub above; this file's own code does not otherwise use it.
; ----------------------------------------------------------------------------
SwitchBank_Local1:
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    LSR
    STA $E000
    RTS

.SEGMENT "BANK_01_VEC"




; Bank 1's interrupt vectors at $BFFA: NMI $E484, reset $BF50 (the stub above), IRQ $BFF0.
; $E484 is the same NMI address as in the fixed bank's vectors; $BFF0 lies in the $FF fill
; between the bank-switch routine and these vectors.
    .BYTE $84, $E4, $50, $BF, $F0, $BF
