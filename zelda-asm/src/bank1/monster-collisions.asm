; ============================================================================
; FILE:     bank1/monster-collisions.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5457-5590
; SUBJECTS: Collision, Damage, Obj, Person
; TOPIC:    A monster's collision pass for one frame: Link's six weapon slots in a fixed order, then the
;           monster's touch on Link; the capture flag of Wallmaster and Like-Like; the object midpoint;
;           and the table of damage each object type does to Link.
; SEE ALSO: bank1/weapon-damage-death.asm (boomerang test, immunity mask, Damage_DealToMonster, monster death);
;           bank1/sword-bomb-arrow-collision.asm (the other weapon tests); bank1/link-collision-harm.asm
;           (Link_CheckMonsterTouch; Link_HarmByObjType reads Damage_ObjTypeDamageToLink; ring halving);
;           bank1/shove-knockback.asm (Shove_Begin sets the invincibility timers);
;           bank7/mode5-play-update.asm (the object loop that calls Collision_CheckMonsterVsWeaponsAndLink);
;           bank1/trap-rupee-stash.asm (the blade trap jumps to Person_DrawAndCheckCollisions_Common);
;           bank4/statues.asm (Statue_Update reads IsPersonFireballsEnabled); bank4/goriya.asm (state
;           $80 and ObjRefId of a Goriya whose boomerang is out).
; KEY FACTS:
;   - Weapon slots are tested in this order: $0F boomerang, $0E sword shot or magic shot, $10 and $11
;     bomb or fire, $0D sword, $12 arrow or rod; then the monster's touch on Link.
;     Collision_CheckMonsterVsWeaponsAndLink
;   - ObjAttr bit $20: no weapon is tested, only the touch on Link. Collision_CheckMonsterVsWeaponsAndLink
;   - While the monster's ObjInvincibilityTimer is nonzero (set to $10 = 32 frames by a weapon hit)
;     nothing is tested: it cannot be hit again and its touch does not hurt Link. Collision_CheckMonsterVsWeaponsAndLink
;   - The timer is tested once, before the first weapon; the weapon tests check neither it nor the
;     metastate. Two weapons overlapping a monster in the same frame both deal damage, and a monster
;     killed by the first can be killed again by the second (kill counters + 1 twice).
;     Collision_CheckMonsterVsWeaponsAndLink
;   - A monster killed by a weapon in this pass still gets its touch test on Link in the same call.
;     Collision_CheckMonsterVsWeaponsAndLink@CheckLink
;   - Wallmaster ($27) and Like-Like ($17): when the touch harmed Link ([0C] nonzero), the monster's
;     ObjCaptureTimer is incremented, which starts the capture. Collision_CheckMonsterVsWeaponsAndLink@CheckCaptors
;   - A Goriya ($05, $06) that dies while its boomerang is out (state bit 7 set) has the boomerang's
;     slot (ObjRefId) emptied at once. Collision_CheckMonsterVsWeaponsAndLink
;   - Object middle = (X + 8, Y + 8); with ObjAttr bit $40 it is (X + 4, Y + 8). Obj_GetMiddle
;   - Damage to Link per object type: low nibble = whole hearts, $80 = half a heart, before rings.
;     Highest: 4 hearts (Ganon $3E, shots $59 and $5A). 0: Bubbles, whirlwind, pond fairy, rupee stash,
;     Grumble, Zelda, dungeon persons. Damage_ObjTypeDamageToLink
;   - Obj_GetMiddle has no RTS of its own: it ends on the first byte of Damage_ObjTypeDamageToLink, $60,
;     which is the RTS opcode. Obj_GetMiddle
; CONTENTS: (generated; L = line in this file)
;   L0061 Person_DrawAndCheckCollisions_Common    code  Runs the collision pass for object X, turns a "kill" of the slot-1 person into fireballs, and draws.
;   L0072 Person_Draw                             code  Draws object X at its position with frame 0 as a mirrored pair. IN: X = object slot.
;   L0093 Collision_CheckMonsterVsWeaponsAndLink  code  Tests the monster in slot X against Link's six weapon slots, then against Link himself.
;   L0165 Obj_GetMiddle                           code  Computes the middle point of object X for the collision tests.
;   L0192 Damage_ObjTypeDamageToLink              data  TABLE: damage an object does to Link when it touches him; index = object type $00-$5C; value: low
; ============================================================================

; ----------------------------------------------------------------------------
; Runs the collision pass for object X, turns a "kill" of the slot-1 person into fireballs, and draws.
; IN:   X = object slot.
; NOTE: The only jump to this label is the blade trap's update (Trap_DrawAndCheckCollisions). A trap
;       has ObjAttr bit $20, so for it the pass tests only its touch on Link.
;       The test below reads ObjMetastate+1 (object slot 1) whatever X is: a nonzero value is copied to
;       IsPersonFireballsEnabled and the metastate is cleared, so the object in slot 1 does not die.
;       While the flag is set, Statue_Update shoots fireballs with pattern 2, from ($48,$82) and
;       ($A8,$86). In a trap room slot 1 holds a trap (Trap_Init_Full), which Link's weapons cannot
;       kill (bit $20), and a trap's update does not run during its own spawn cloud (Obj_Update);
;       whether slot 1's metastate is ever nonzero while another trap updates, so that this branch
;       is taken, is (unverified). Dungeon persons use their own copy of this test
;       (Person_CheckCollisions).
; ----------------------------------------------------------------------------
Person_DrawAndCheckCollisions_Common:
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Slot 1's metastate nonzero: set the fireball flag and clear the metastate.
    LDA ObjMetastate+1
    BEQ Person_Draw
    STA IsPersonFireballsEnabled
    LDA #$00
    STA ObjMetastate+1

; Draws object X at its position with frame 0 as a mirrored pair. IN: X = object slot.
Person_Draw:
    JSR Anim_FetchObjPosForSpriteDescriptor    ; [00], [01] := position; [0F] := 0; returns A = 0.
    JMP Anim_DrawObjectMirrored      ; A = 0: frame 0.

; ----------------------------------------------------------------------------
; Tests the monster in slot X against Link's six weapon slots, then against Link himself.
; IN:   X = object slot of the monster.
; OUT:  X kept; [02], [03] = the monster's middle; [0C] = 1 when its touch harmed Link.
;       The monster may be damaged, stunned, shoved, turned (ObjDir) or put in the dying metastate;
;       ObjCaptureTimer,X + 1 for a Wallmaster or Like-Like that harmed Link.
; NOTE: Order: $0F boomerang (food never hits), $0E sword shot or magic shot, $10 and $11 bomb or fire,
;       $0D sword, $12 arrow or rod, then Link_CheckMonsterTouch.
;       ObjAttr bit $20 skips all the weapons and the timer test below, going straight to the touch
;       on Link. Otherwise a nonzero ObjInvincibilityTimer (Shove_Begin sets $10 = about 32 frames
;       when a weapon hits) skips everything, the touch on Link included.
;       The timer is tested only here, and the weapon routines test neither it nor the metastate: two
;       weapons that overlap the monster in the same frame both deal their damage, and when the first
;       kills it the second runs the death code again if its damage is at least the HP left (the kill
;       counters then count twice). A monster killed here still gets its touch test in this call.
;       The capture test runs only while the monster's metastate is 0 (not dying).
; ----------------------------------------------------------------------------
Collision_CheckMonsterVsWeaponsAndLink:
    JSR Obj_GetMiddle         ; [02], [03] := the monster's middle, used by every test below.

    ; ObjAttr bit $20: Link's weapons cannot touch it. Only test its touch on Link.
    LDA ObjAttr, X
    AND #$20
    BNE @CheckLink

    ; Still flashing from an earlier hit: no test at all this frame.
    LDA ObjInvincibilityTimer, X
    BNE @Exit

    ; Each weapon slot in turn; Y = weapon slot.
    LDY #$0F                    ; Boomerang (food in this slot never collides)
    JSR Boomerang_CheckMonsterCollision
    LDY #$0E                    ; Sword shot or magic shot
    JSR SwordShot_CheckMonsterCollision
    LDY #$10                    ; Bomb or fire, first slot
    JSR Bomb_CheckBombOrFireVsMonster
    LDY #$11                    ; Bomb or fire, second slot
    JSR Bomb_CheckBombOrFireVsMonster
    LDY #$0D                    ; Sword
    JSR Sword_CheckMonsterCollision
    LDY #$12                    ; Arrow or rod
    JSR Arrow_CheckArrowOrRodVsMonster

@CheckLink:
    JSR Link_CheckMonsterTouch      ; [0C] := 1 if the touch harmed Link.

    ; Metastate 0 = alive: go see about the monsters that capture Link.
    LDA ObjType, X
    LDY ObjMetastate, X
    BEQ @CheckCaptors

    ; The monster is dying (a weapon above killed it). If it is a Goriya ($05 or $06) whose state has
    ; bit 7 set (it threw its boomerang and waits for it), empty the boomerang's slot.
    CMP #$05
    BEQ :+
    CMP #$06
    BNE @Exit
:
    LDA ObjState, X
    BPL @Exit
    LDY ObjRefId, X             ; Y := object slot of this Goriya's boomerang.
    LDA #$00
    STA ObjType, Y

@Exit:
    RTS

@CheckCaptors:
    ; Wallmaster ($27) or Like-Like ($17) whose touch harmed Link this call ([0C] nonzero):
    ; increment its capture timer. Nonzero means Link is captured.
    CMP #$27
    BEQ :+
    CMP #$17
    BNE @Exit2
:
    LDA $0C
    BEQ @Exit2
    INC ObjCaptureTimer, X

@Exit2:
    RTS

; ----------------------------------------------------------------------------
; Computes the middle point of object X for the collision tests.
; IN:   X = object slot.
; OUT:  [02] = ObjX + 8 (ObjX + 4 when ObjAttr bit $40 is set); [03] = ObjY + 8.
; NOTE: There is no RTS instruction here. Execution runs into the first byte of
;       Damage_ObjTypeDamageToLink, $60, which is the RTS opcode.
; ----------------------------------------------------------------------------
Obj_GetMiddle:
    ; [02] := [03] := 8, half of a 16-pixel object.
    LDA #$08
    STA $02
    STA $03

    ; ObjAttr bit $40 (half width for collisions): the X offset becomes 4.
    LDA ObjAttr, X
    AND #$40
    BEQ :+
    LSR $02
:
    LDA ObjX, X
    CLC
    ADC $02
    STA $02

    LDA ObjY, X
    CLC
    ADC $03
    STA $03

; TABLE: damage an object does to Link when it touches him; index = object type $00-$5C; value: low
; nibble = whole hearts, high nibble = sixteenths of a heart ($80 = half a heart), before the rings
; halve it (Link_HarmByObjType, Link_BeHarmed). Rows below give hearts per type; 0 = no hearts lost.
; Entry $00 is not data: it is the RTS opcode ($60) that ends Obj_GetMiddle. No object has type 0.
; Types $5D and up have no entry.
Damage_ObjTypeDamageToLink:
    ; $00 RTS opcode; Lynel $01 2, $02 1; Moblin $03 1/2, $04 1/2; Goriya $05 1, $06 1/2;
    ; Octorok $07 1/2
    .BYTE $60, $02, $01, $80, $80, $01, $80, $80
    ; Octorok $08 1/2, $09 1/2, $0A 1/2; Darknut $0B 1, $0C 2; Tektite $0D 1/2, $0E 1/2; Leever $0F 1
    .BYTE $80, $80, $80, $01, $02, $80, $80, $01
    ; Leever $10 1/2; Zora $11 1/2; Vire $12 1; Zol $13 1; Gel $14 1/2, $15 1/2; Pols Voice $16 2;
    ; Like-Like $17 1
    .BYTE $80, $80, $01, $01, $80, $80, $02, $01
    ; Small Digdogger $18 2; type $19 0; Peahat $1A 1/2; Keese $1B 1/2, $1C 1/2, $1D 1/2; Armos $1E 1;
    ; boulder spawner $1F 1/2
    .BYTE $02, $00, $80, $80, $80, $80, $01, $80
    ; Boulder $20 1/2; Ghini $21 1, $22 1; Wizzrobe $23 2, $24 1; Patra child $25 2, $26 2;
    ; Wallmaster $27 1/2
    .BYTE $80, $01, $01, $02, $01, $02, $02, $80
    ; Rope $28 1/2; type $29 1/2; Stalfos $2A 1/2; Bubble $2B 0, $2C 0, $2D 0; Whirlwind $2E 0;
    ; Pond fairy $2F 0
    .BYTE $80, $80, $80, $00, $00, $00, $00, $00
    ; Gibdo $30 2; Dodongo $31 1, $32 1; Gohma $33 2, $34 2; Rupee stash $35 0; Grumble $36 0;
    ; Zelda $37 0
    .BYTE $02, $01, $01, $02, $02, $00, $00, $00
    ; Digdogger $38 2, $39 2; Lanmola $3A 2, $3B 2; Manhandla $3C 1; Aquamentus $3D 1; Ganon $3E 4;
    ; Guard fire $3F 1/2
    .BYTE $02, $02, $02, $02, $01, $01, $04, $80
    ; Standing fire $40 1/2; Moldorm $41 1/2; Gleeok $42-$45 1; Gleeok head $46 1; Patra $47 2
    .BYTE $80, $80, $01, $01, $01, $01, $01, $02
    ; Patra $48 2; Trap $49 1, $4A 1; dungeon persons $4B-$4F 0
    .BYTE $02, $01, $01, $00, $00, $00, $00, $00
    ; Dungeon persons $50-$52 0; flying rock $53 1/2; type $54 1/2; fireball $55 1/2, $56 1;
    ; sword shot $57 2
    .BYTE $00, $00, $00, $80, $80, $80, $01, $02
    ; Magic shot $58 2, $59 4; shot $5A 4; monster arrow $5B 1/2; monster boomerang $5C 1
    .BYTE $02, $04, $04, $80, $01
