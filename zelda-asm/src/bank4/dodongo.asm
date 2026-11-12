; ============================================================================
; FILE:     bank4/dodongo.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 5855-6447
; SUBJECTS: Anim, Dodongo
; TOPIC:    Dodongo (types $31, $32): walking, swallowing a bomb and bloating, the bomb-cloud stun, the
;           two ways it dies, its two-box collision check and its drawing.
; SEE ALSO: bank4/aquamentus-digdogger-dodongo-inits.asm (Dodongo_Init); bank4/walker-wanderer-common.asm
;           (Wanderer_MoveAndTurn: walking and turning); bank7/bomb.asm (bomb states $12-$14 and their
;           timers); bank1/monster-collisions.asm, bank1/sword-bomb-arrow-collision.asm and
;           bank1/shove-knockback.asm (the sword hit that sets ObjInvincibilityTimer);
;           bank4/drop-items.asm (Drop_SetUpItem: what the help-drop bytes do);
;           bank7/object-update-table.asm (HP and attribute tables, death sparkle to drop).
; KEY FACTS:
;   - Death by bombs: the second bomb it swallows kills it. Each swallowed bomb adds 1 to
;     Dodongo_ObjBombHits (never reset in this file); the bloat runs on to the Die substate only when
;     the count is 2 or more (Dodongo_State1Bloated_Wait).
;   - Death by sword: a Dodongo stunned by a bomb blast (state 2) dies from one touch of the extended
;     sword in slot $0D, whatever the sword. Nothing else can hurt it: its mask is $FF except for that
;     one test (Dodongo_CheckCollisionsStandardSize, Dodongo_CheckCollisions).
;   - The sword death sets HelpDropCount and IsHelpDropBomb to $0A, which turns the next drop that is
;     set up (normally this Dodongo's own) into a bomb. The bomb death sets neither: its drop is the
;     ordinary random one (Dodongo_CheckCollisions, Dodongo_State1Bloated_Die).
;   - Only the bomb in slot $10 counts, for eating and for the stun; a bomb in slot $11 does nothing
;     to a Dodongo (Dodongo_CheckBombHit).
;   - A bomb is eaten while its fuse burns (bomb state $12) and the Dodongo walks (state 0), when the
;     bomb's centre is in the half of the body ahead of the Dodongo's centre: up to 16 pixels ahead
;     and within 4 pixels sideways (8 when it faces up or down) (Dodongo_TryEatBomb).
;   - Stun: a blast or cloud (bomb states $13, $14) whose centre is within 12 pixels of the Dodongo's
;     centre on both axes puts a walking Dodongo in state 2 (Dodongo_CheckBombHit).
;   - The stun lasts while ObjStunTimer counts $20 down to 1 in 10-frame steps: 301-310 frames. During
;     it the Dodongo stands still and its touch does not hurt Link (Dodongo_State2Stunned).
;   - A swallowed bomb that does not kill holds the Dodongo still for $20 + $40 frames plus 1 frame to
;     return to walking; the killing bomb takes $20 + $40 + $40 frames (Dodongo_BloatedWaitTimes).
;   - It walks at the q-speed every slot gets on room entry, $20 (0.5 pixel per frame), with turn rate
;     $20 (Dodongo_State0Move).
;   - Facing left or right it is 32 pixels wide: ObjX is its left end, and collisions are checked for
;     the 16-pixel boxes at ObjX and ObjX + $10 (Dodongo_CheckCollisions, Dodongo_Draw).
;   - HP ($F0) plays no part: the sword hit subtracts its damage, but the Dodongo dies because the
;     hit started knockback invincibility (Dodongo_CheckCollisions).
; CONTENTS: (generated; L = line in this file)
;   L0083 Dodongo_Update                           code  Updates a Dodongo (types $31, $32) for one frame: state, collisions, bomb test, drawing.
;   L0096 Dodongo_BloatedWaitTimes                 data  TABLE: length of each bloated wait substate; index = Dodongo_ObjBloatedSubstate 0-2; value = frames.
;   L0103 Dodongo_UpdateState                      code  Runs the handler of the Dodongo's current state.
;   L0108 Dodongo_UpdateState_JumpTable            data  TABLE: Dodongo state handlers; index = ObjState; value = routine address.
;   L0125 Dodongo_State0Move                       code  State 0: walks the Dodongo for one frame as a wanderer with turn rate $20.
;   L0171 Dodongo_State2Stunned                    code  State 2: keeps the Dodongo stunned until its stun timer runs down, then returns it to walking.
;   L0187 Dodongo_State1Bloated                    code  State 1: runs the bloated substate of a Dodongo that swallowed a bomb.
;   L0192 Dodongo_State1Bloated_JumpTable          data  TABLE: bloated substate handlers; index = Dodongo_ObjBloatedSubstate; value = routine address.
;   L0211 Dodongo_State1Bloated_Wait               code  Bloated substates 0-2: times the wait, counts the swallowed bomb and picks the next substate.
;   L0258 Dodongo_State1Bloated_DecTimer           code  Shared tail of the bloated wait substates: counts the bloated timer down by 1.
;   L0271 Dodongo_State1Bloated_Die                code  Bloated substate 3: starts the Dodongo's death (sparkle, boss death cry) and resets its state.
;   L0276 Dodongo_State1Bloated_End                code  Bloated substate 4, and the end of the stun: returns the Dodongo to state 0 and substate 0.
;   L0296 Dodongo_CheckCollisions                  code  Checks collisions for one or both 16-pixel halves of the Dodongo and kills it if the sword hit.
;   L0332 Dodongo_CheckCollisions_Return           code  Shared exit of Dodongo_CheckCollisions and Dodongo_CheckCollisionsStandardSize.
;   L0347 Dodongo_CheckCollisionsStandardSize      code  Checks weapons and Link against the 16-pixel box at ObjX; the sword can hit only a stunned Dodongo.
;   L0366 Dodongo_MouthNegativeLimitsX             data  TABLE: lower limit (inclusive) of dx for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
;   L0371 Dodongo_MouthPositiveLimitsX             data  TABLE: upper limit (exclusive) of dx for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
;   L0376 Dodongo_MouthNegativeLimitsY             data  TABLE: lower limit (inclusive) of dy for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
;   L0381 Dodongo_MouthPositiveLimitsY             data  TABLE: upper limit (exclusive) of dy for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
;   L0400 Dodongo_CheckBombHit                     code  Tests the bomb in slot $10 against a walking Dodongo: a ticking bomb may be eaten, a blast stuns.
;   L0469 Dodongo_TryEatBomb                       code  Makes the Dodongo swallow the ticking bomb in slot $10 if the bomb lies just ahead of its centre.
;   L0522 Dodongo_BombPositiveLimits               data  TABLE: upper limit (exclusive) of the centre distance for Dodongo_IsBombInRange; index = range 0-1.
;   L0527 Dodongo_BombNegativeLimits               data  TABLE: lower limit (inclusive) of the centre distance for Dodongo_IsBombInRange; index = range 0-1.
;   L0538 Dodongo_IsBombInRange                    code  Tests whether the bomb's centre is within a square range of the Dodongo's centre.
;   L0579 Dodongo_FrameImages                      data  TABLE: frame image of the half drawn at ObjX, walking or stunned; index = (ObjDir >> 1) + 5 * frame.
;   L0586 Dodongo_FrameHFlips                      data  TABLE: horizontal flip for each entry of Dodongo_FrameImages; same index; value = $40 flip, 0 none.
;   L0592 Dodongo_BloatedFrameImages               data  TABLE: frame images of the bloated Dodongo; index = ObjDir >> 1; value = frame image number.
;   L0598 Dodongo_BloatedFrameHFlips               data  TABLE: horizontal flip for each entry of Dodongo_BloatedFrameImages; same index; $40 flip, 0 none.
;   L0611 Dodongo_Draw                             code  Draws the Dodongo: one 16-pixel image facing up or down, two side by side facing left or right.
;   L0720 Anim_SetSpriteDescriptorLevelPaletteRow  code  Sets both sprite attribute bytes of the sprite descriptor to 3 (sprite palette 3).
;   L0725 Dodongo_Draw_Return                      code  Shared exit of Dodongo_Draw; the routine above also returns through it.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Dodongo (types $31, $32) for one frame: state, collisions, bomb test, drawing.
; IN:   X = object slot.
; NOTE: ObjState: 0 walking, 1 bloated (swallowed a bomb), 2 stunned (bomb blast or cloud).
;       The bomb test runs after the state handler, so a state it sets is acted on next frame.
;       Both deaths leave state 0, and the bomb test still runs in the death frame: a bomb in
;       range in slot $10 is then swallowed or sets state 2 with no further effect (the death
;       sparkle has begun and this routine is not called again for the slot).
; ----------------------------------------------------------------------------
Dodongo_Update:
    JSR Dodongo_UpdateState
    JSR Dodongo_CheckCollisions
    JSR Dodongo_CheckBombHit
    JMP Dodongo_Draw

; Five unused bytes. They have no label, no label expression in any bank reaches them (the table
; below is indexed from 0 up, and the code above is not indexed), and no instruction addresses bank 4
; by a raw number. They have the shape of this file's 5-entry direction tables (index =
; direction >> 1, entry 3 = $FF unused).
    .BYTE $10, $F0, $10, $FF, $F0

; TABLE: length of each bloated wait substate; index = Dodongo_ObjBloatedSubstate 0-2; value = frames.
Dodongo_BloatedWaitTimes:
    .BYTE $20, $40, $40         ; 0: 32 frames; 1: 64 frames; 2: 64 frames (second bomb only)

; ----------------------------------------------------------------------------
; Runs the handler of the Dodongo's current state.
; IN:   X = object slot; ObjState,X = 0 walking, 1 bloated, 2 stunned.
; ----------------------------------------------------------------------------
Dodongo_UpdateState:
    LDA ObjState, X
    JSR Util_TableJump

; TABLE: Dodongo state handlers; index = ObjState; value = routine address.
Dodongo_UpdateState_JumpTable:
    .ADDR Dodongo_State0Move      ; 0: walking
    .ADDR Dodongo_State1Bloated   ; 1: bloated (swallowed a bomb)
    .ADDR Dodongo_State2Stunned   ; 2: stunned (bomb blast or cloud)

; ----------------------------------------------------------------------------
; State 0: walks the Dodongo for one frame as a wanderer with turn rate $20.
; IN:   X = object slot.
; OUT:  ObjX, ObjY, ObjDir, ObjInputDir, ObjGridOffset,X as left by Wanderer_MoveAndTurn.
; NOTE: ObjX is the left end of the body. Unless the Dodongo faces left, the walk is made with
;       ObjX + $10 and the $10 is taken off again afterwards, so the tile and room boundary tests are
;       made for a point $10 pixels right of ObjX.
;       Speed is the q-speed $20 (0.5 pixel per frame) given to every slot on room entry; nothing
;       here changes it. Turn rate $20: at a grid point the Dodongo may turn toward the chase target
;       only when Random+1,X <= $20 (33 chances in 256); see Wanderer_MoveAndTurn.
;       With InvClock set, or while ObjStunTimer,X is nonzero, Walker_Move does not move it.
; ----------------------------------------------------------------------------
Dodongo_State0Move:
    ; Facing left (direction 2): no shift, and Y = 0 is pushed below.
    ; Facing right, down or up (bits 1, 4, 8): add $10 to ObjX for the walk, and Y = -$10 to undo it.
    LDY #$00
    LDA ObjDir, X
    AND #$0D
    BEQ @Move
    LDA ObjX, X
    CLC
    ADC #$10
    STA ObjX, X
    LDY #$F0

@Move:
    TYA
    PHA
    LDA #$20                    ; Turn rate $20.
    STA Wanderer_ObjTurnRate, X
    JSR Wanderer_MoveAndTurn

    ; Undo the shift: add the saved 0 or -$10 to ObjX (the one chosen before the walk, even if the
    ; walk changed the direction).
    PLA
    CLC
    ADC ObjX, X
    STA ObjX, X

    ; If ObjX < $20, face right. Only ObjDir is set here; Wanderer_MoveAndTurn copies it to
    ; ObjInputDir at the end of its next call.
    CMP #$20
    BCS :+
    LDA #$01
    STA ObjDir, X
:
    RTS

; ----------------------------------------------------------------------------
; State 2: keeps the Dodongo stunned until its stun timer runs down, then returns it to walking.
; IN:   X = object slot.
; OUT:  ObjStunTimer,X = $20 if it was 0. If it was 1: ObjState,X = 0 and bloated substate 0.
; NOTE: ObjStunTimer drops by 1 every 10 frames, so from $20 to 1 takes 301-310 frames. Only in this
;       state can the sword kill (Dodongo_CheckCollisionsStandardSize). While the timer is nonzero
;       the Dodongo's touch does not hurt Link (Link_CheckMonsterTouch). The state ends with the timer
;       still at 1; Walker_Move then holds the Dodongo in place until it reaches 0 (up to 10 frames).
;       The Dodongo does not eat bombs in this state (Dodongo_CheckBombHit).
; ----------------------------------------------------------------------------
Dodongo_State2Stunned:
    ; ObjStunTimer: 0 -> start the stun at $20; 1 -> the stun is over, back to state 0;
    ; any other value -> wait.
    LDY a:ObjStunTimer, X
    DEY
    BEQ Dodongo_State1Bloated_End
    BPL :+
    LDA #$20
    STA a:ObjStunTimer, X
:
    RTS

; ----------------------------------------------------------------------------
; State 1: runs the bloated substate of a Dodongo that swallowed a bomb.
; IN:   X = object slot; Dodongo_ObjBloatedSubstate,X = 0-4.
; ----------------------------------------------------------------------------
Dodongo_State1Bloated:
    LDA Dodongo_ObjBloatedSubstate, X
    JSR Util_TableJump

; TABLE: bloated substate handlers; index = Dodongo_ObjBloatedSubstate; value = routine address.
Dodongo_State1Bloated_JumpTable:
    .ADDR Dodongo_State1Bloated_Wait  ; 0: wait $20 frames; counts the bomb
    .ADDR Dodongo_State1Bloated_Wait  ; 1: wait $40 frames, drawn bloated
    .ADDR Dodongo_State1Bloated_Wait  ; 2: wait $40 frames, drawn flickering (2nd bomb only)
    .ADDR Dodongo_State1Bloated_Die   ; 3: die (2nd bomb only)
    .ADDR Dodongo_State1Bloated_End   ; 4: back to state 0

; ----------------------------------------------------------------------------
; Bloated substates 0-2: times the wait, counts the swallowed bomb and picks the next substate.
; IN:   X = object slot.
; OUT:  Dodongo_ObjBloatedTimer,X counted down by 1; Dodongo_ObjBombHits,X + 1 on the first frame of
;       substate 0; Dodongo_ObjBloatedSubstate,X changed on the frame the timer reads 1.
; NOTE: Substate 0 lasts $20 frames and substate 1 $40 frames. With fewer than 2 bombs swallowed the
;       substate then becomes 4, which takes 1 frame to return to state 0: 97 frames in all without
;       moving. With 2 or more it goes on to substate 2 ($40 frames) and 3: the Dodongo starts to
;       die on the 161st frame of state 1.
;       The first frame of substate 0 empties slot $10 a second time, so a bomb or fire placed there
;       on that frame is lost.
; ----------------------------------------------------------------------------
Dodongo_State1Bloated_Wait:
    ; Dodongo_ObjBloatedTimer: 0 -> start this substate's wait (below); 1 -> the wait is over, go
    ; advance the substate; any other value -> go count down.
    LDY Dodongo_ObjBloatedTimer, X
    DEY
    BEQ @AdvanceSubstate
    BPL Dodongo_State1Bloated_DecTimer

    ; Timer = 0: load the wait for this substate.
    LDY Dodongo_ObjBloatedSubstate, X
    LDA Dodongo_BloatedWaitTimes, Y
    STA Dodongo_ObjBloatedTimer, X

    ; Substates 1 and 2 only wait.
    CPY #$00
    BNE Dodongo_State1Bloated_DecTimer

    ; Substate 0, first frame: clear slot $10 (Dodongo_TryEatBomb already removed the swallowed
    ; bomb from it).
    LDY #$10
    LDA #$00
    STA a:ObjState, Y

    ; Count the swallowed bomb.
    LDA Dodongo_ObjBombHits, X
    CLC
    ADC #$01
    STA Dodongo_ObjBombHits, X
    JMP Dodongo_State1Bloated_DecTimer

@AdvanceSubstate:
    INC Dodongo_ObjBloatedSubstate, X

    ; New substate 1: go on waiting.
    LDA Dodongo_ObjBloatedSubstate, X
    CMP #$02
    BCC Dodongo_State1Bloated_DecTimer

    ; New substate 2 or 3. With 2 or more bombs swallowed keep it: substate 2 waits $40 frames and
    ; substate 3 kills the Dodongo. With fewer, skip to substate 4, which returns to state 0.
    LDY Dodongo_ObjBombHits, X
    CPY #$02
    BCS Dodongo_State1Bloated_DecTimer
    LDA #$04
    STA Dodongo_ObjBloatedSubstate, X

; Shared tail of the bloated wait substates: counts the bloated timer down by 1.
Dodongo_State1Bloated_DecTimer:
    DEC Dodongo_ObjBloatedTimer, X
    RTS

; ----------------------------------------------------------------------------
; Bloated substate 3: starts the Dodongo's death (sparkle, boss death cry) and resets its state.
; IN:   X = object slot.
; OUT:  ObjMetastate,X = $10; ObjState,X = 0; Dodongo_ObjBloatedSubstate,X = 0; A = 0.
; NOTE: Also called by Dodongo_CheckCollisions for the sword death. Neither death goes through
;       Damage_KillMonster, so WorldKillCount and HelpDropCount are not incremented. When the sparkle
;       ends (Obj_EndMetastate, about 19 frames later) WorldKillCycle and RoomKillCount go up by
;       1 and Drop_SetUpItem chooses the drop.
; ----------------------------------------------------------------------------
Dodongo_State1Bloated_Die:
    JSR DeadDummy_StartDeathSparkle         ; ObjMetastate,X = $10 (death sparkle) and the "monster died" tune.
    JSR Sample_PlayBossDeathCry

; Bloated substate 4, and the end of the stun: returns the Dodongo to state 0 and substate 0.
Dodongo_State1Bloated_End:
    JSR Obj_ResetState           ; ObjState,X = 0; returns A = 0.
    STA Dodongo_ObjBloatedSubstate, X
    RTS

; ----------------------------------------------------------------------------
; Checks collisions for one or both 16-pixel halves of the Dodongo and kills it if the sword hit.
; IN:   X = object slot.
; OUT:  ObjX,X unchanged. If a hit registered: death started, HelpDropCount = IsHelpDropBomb = $0A.
; NOTE: ObjInvincibilityTimer,X <> 0 after a check means Shove_Begin accepted a hit, and only the
;       sword on a stunned Dodongo gets that far (Dodongo_CheckCollisionsStandardSize). The Dodongo
;       then dies at once, whatever its HP ($F0) and the sword's damage.
;       With both help bytes at $0A, Drop_SetUpItem turns the next drop into a bomb (item $00)
;       without a random roll; that is this Dodongo's own drop unless another monster's death ends
;       first. Exceptions: WorldKillCount = $10 at that moment gives a fairy instead, and if Link is
;       harmed before the sparkle ends both bytes are zeroed (Link_BeHarmed) and the drop is the
;       ordinary random one.
;       Facing up or down only the box at ObjX is checked; facing left or right also the box at
;       ObjX + $10. Link can be hurt by either box.
; ----------------------------------------------------------------------------
Dodongo_CheckCollisions:
    ; Check the 16-pixel box at ObjX.
    JSR Dodongo_CheckCollisionsStandardSize

    ; A nonzero invincibility timer means the sword hit: go die.
    LDA ObjInvincibilityTimer, X
    BNE @Die

    ; Facing up or down (direction >= 4): the body is one box wide. Return.
    LDA ObjDir, X
    CMP #$04
    BCS Dodongo_CheckCollisions_Return

    ; Facing left or right: check the right half too, with ObjX moved $10 to the right meanwhile.
    LDA ObjX, X
    PHA
    CLC
    ADC #$10
    STA ObjX, X
    JSR Dodongo_CheckCollisionsStandardSize
    PLA
    STA ObjX, X

    ; No hit on this half either: return.
    LDA ObjInvincibilityTimer, X
    BEQ Dodongo_CheckCollisions_Return

@Die:
    JSR Dodongo_State1Bloated_Die

    ; Force the next drop to be a bomb: HelpDropCount at its cap ($0A), IsHelpDropBomb nonzero.
    LDA #$0A
    STA HelpDropCount
    STA IsHelpDropBomb

; Shared exit of Dodongo_CheckCollisions and Dodongo_CheckCollisionsStandardSize.
Dodongo_CheckCollisions_Return:
    RTS

; ----------------------------------------------------------------------------
; Checks weapons and Link against the 16-pixel box at ObjX; the sword can hit only a stunned Dodongo.
; IN:   X = object slot; ObjX,X = left edge of the box to test.
; OUT:  ObjInvincibilityTimer,X = $10 if the sword hit a stunned Dodongo.
;       ObjInvincibilityMask,X = $FF, or $FE if stunned.
; NOTE: The first pass (Collision_CheckMonsterVsWeaponsAndLink) runs with mask $FF: no weapon damages, stuns or
;       shoves; every weapon but bomb and fire plays the parry tune. That pass also tests Link's
;       touch (damage byte $01 in Damage_ObjTypeDamageToLink = 1 heart before rings).
;       In state 2 a second pass tests slot $0D only, and Sword_CheckMonsterCollision acts only in
;       sword state 2 (the extended step, 8 frames of a swing). The rod (slot $12), the sword shot
;       and every other weapon are never tested with the sword bit cleared.
; ----------------------------------------------------------------------------
Dodongo_CheckCollisionsStandardSize:
    LDA #$FF                    ; Immune to every damage type during the general check.
    STA ObjInvincibilityMask, X
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Not stunned (state <> 2): done.
    LDA ObjState, X
    CMP #$02
    BNE Dodongo_CheckCollisions_Return

    ; Stunned: clear the sword bit (1) of the mask and test the sword in slot $0D again.
    LDA #$FE
    STA ObjInvincibilityMask, X
    JSR Obj_GetMiddle         ; [02], [03] = centre of the box (ObjX + 8, ObjY + 8).
    LDY #$0D
    JMP Sword_CheckMonsterCollision

; TABLE: lower limit (inclusive) of dx for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
; Signed; compared with dx = [04] = Dodongo centre X minus bomb centre X.
Dodongo_MouthNegativeLimitsX:
    .BYTE $F0, $00, $F8, $FF, $F8       ; right -16, left 0, down -8, (unused), up -8

; TABLE: upper limit (exclusive) of dx for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
; Signed; compared with dx = [04].
Dodongo_MouthPositiveLimitsX:
    .BYTE $00, $10, $08, $FF, $08       ; right 0, left +16, down +8, (unused), up +8

; TABLE: lower limit (inclusive) of dy for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
; Signed; compared with dy = [05] = Dodongo centre Y minus bomb centre Y.
Dodongo_MouthNegativeLimitsY:
    .BYTE $FC, $FC, $F0, $FF, $00       ; right -4, left -4, down -16, (unused), up 0

; TABLE: upper limit (exclusive) of dy for a bomb in the mouth; index = ObjDir >> 1; value = pixels.
; Signed; compared with dy = [05].
Dodongo_MouthPositiveLimitsY:
    .BYTE $04, $04, $00, $FF, $10       ; right +4, left +4, down 0, (unused), up +16

; ----------------------------------------------------------------------------
; Tests the bomb in slot $10 against a walking Dodongo: a ticking bomb may be eaten, a blast stuns.
; IN:   X = object slot.
; OUT:  ObjState,X = 2 if a blast or cloud is in range; for a ticking bomb see Dodongo_TryEatBomb.
;       [00], [01] = Dodongo centre X, Y; [02], [03] = bomb centre X, Y.
; NOTE: Acts only in state 0 and only on slot $10: a bomb in slot $11 is never looked at. Bomb_Wield
;       uses slot $10 unless it already holds a bomb, so the bomb a Dodongo ignores is one placed
;       while the bomb before it is still in its blast or cloud (states $13, $14) in slot $10.
;       Bomb states (Bomb_Update, Bomb_StateTimes): $12 fuse, $30 frames; $13 blast, $18 frames; $14 cloud,
;       $0C frames. In $12 the bomb can only be eaten. In $13 and $14 the Dodongo is stunned if the
;       two centres are within -12..+11 pixels of each other on both axes. (The stun branch also
;       takes bomb state $11, which lasts only from placement to the bomb's own update in the same
;       frame, before any monster is updated.) States $20 and up are a candle fire: ignored.
;       Dodongo centre = (ObjX + $10, ObjY + 8) facing left or right, (ObjX + 8, ObjY + 9) facing up
;       or down. Bomb centre = (bomb X + 8, bomb Y + 8).
; ----------------------------------------------------------------------------
Dodongo_CheckBombHit:
    ; Only a walking Dodongo (state 0) reacts to a bomb.
    LDA ObjState, X
    BNE @Exit

    ; [00] = centre X: ObjX + 8 facing up or down (direction >= 4), else ObjX + $10.
    LDA ObjX, X
    CLC
    ADC #$08
    LDY ObjDir, X
    CPY #$04
    BCS :+
    ADC #$08
:
    STA $00

    ; [01] = centre Y: ObjY + 8, or ObjY + 9 facing up or down (the carry set by CPY is added in).
    ; None of the ADCs from here on has a CLC: each adds the carry of the sum before it, which is 0
    ; unless that sum passed $FF.
    LDA ObjY, X
    ADC #$08
    STA $01

    ; [02] = bomb centre X: X of slot $10 + 8.
    LDY #$10
    LDA a:ObjX, Y
    ADC #$08
    STA $02

    ; [03] = bomb centre Y: Y of slot $10 + 8.
    LDA a:ObjY, Y
    ADC #$08
    STA $03

    ; Slot $10 empty: return.
    LDA a:ObjState, Y
    BEQ @Exit

    ; Bomb with its fuse burning (state $12): go see whether the Dodongo eats it.
    CMP #$12
    BEQ Dodongo_TryEatBomb

    ; State $20 and up is a fire, not a bomb: return.
    CMP #$20
    BCS @Exit

    ; Blast or cloud (states $13, $14). If its centre is within range 0 of the Dodongo's centre,
    ; the Dodongo is stunned (state 2).
    LDY #$00
    JSR Dodongo_IsBombInRange
    BNE @Exit
    LDA #$02
    STA ObjState, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Makes the Dodongo swallow the ticking bomb in slot $10 if the bomb lies just ahead of its centre.
; IN:   X = object slot; [00]-[03] = Dodongo and bomb centres (from Dodongo_CheckBombHit).
; OUT:  If eaten: ObjState,X = 1 (bloated), Dodongo_ObjBloatedSubstate,X = 0, slot $10 emptied.
; NOTE: Entered only by the branch in Dodongo_CheckBombHit (Dodongo state 0, bomb state $12).
;       Two tests on d = Dodongo centre minus bomb centre: first -16..+16 pixels on both axes
;       (Dodongo_IsBombInRange, range 1), then the mouth box for the facing direction:
;         right: dx -16..-1, dy -4..+3        left: dx 0..+15, dy -4..+3
;         down:  dy -16..-1, dx -8..+7        up:   dy 0..+15, dx -8..+7
;       That is the half of the body ahead of its centre. The swallowed bomb never explodes; it is
;       counted on the next frame (Dodongo_State1Bloated_Wait).
; ----------------------------------------------------------------------------
Dodongo_TryEatBomb:
    ; Coarse test (range 1): the bomb must be within -16..+16 pixels on both axes. It also leaves
    ; the distances in [04] (horizontal) and [05] (vertical).
    LDY #$01
    JSR Dodongo_IsBombInRange
    BNE @Exit

    ; [00] counts the two passes: 1 = horizontal, 0 = vertical.
    LDA #$01
    STA $00

    ; Y = facing direction >> 1: 0 right, 1 left, 2 down, 4 up (3 does not occur).
    LDA ObjDir, X
    LSR
    TAY

    ; Start with the horizontal distance.
    LDA $04

@LoopLimit:
    ; Return unless lower limit <= distance < upper limit for this direction. The compares are
    ; signed by the N flag, which is safe because the distances are within -16..+16.
    CMP Dodongo_MouthNegativeLimitsX, Y
    BMI @Exit
    CMP Dodongo_MouthPositiveLimitsX, Y
    BPL @Exit

    ; Y + $0A: the same entry of the vertical tables (Dodongo_MouthNegativeLimitsY and
    ; Dodongo_MouthPositiveLimitsY lie $0A bytes after the horizontal ones).
    TYA
    CLC
    ADC #$0A
    TAY

    ; Second pass: the vertical distance.
    LDA $05
    DEC $00
    BPL @LoopLimit

    ; The bomb is in the mouth: state 0 -> 1 (bloated).
    INC ObjState, X

    ; Remove the bomb from slot $10 and start at bloated substate 0.
    LDY #$10
    LDA #$00
    STA a:ObjState, Y
    STA Dodongo_ObjBloatedSubstate, X

@Exit:
    RTS

; TABLE: upper limit (exclusive) of the centre distance for Dodongo_IsBombInRange; index = range 0-1.
; Value = signed pixels. Range 0 = blast or cloud (stun), range 1 = ticking bomb (coarse eat test).
Dodongo_BombPositiveLimits:
    .BYTE $0C, $11              ; 0: +12; 1: +17

; TABLE: lower limit (inclusive) of the centre distance for Dodongo_IsBombInRange; index = range 0-1.
; Value = signed pixels.
Dodongo_BombNegativeLimits:
    .BYTE $F4, $F0              ; 0: -12; 1: -16

; ----------------------------------------------------------------------------
; Tests whether the bomb's centre is within a square range of the Dodongo's centre.
; IN:   Y = range: 0 blast or cloud (-12..+11 pixels), 1 ticking bomb (-16..+16 pixels).
;       [00], [01] = Dodongo centre X, Y; [02], [03] = bomb centre X, Y.
; OUT:  A = 0 and Z set if in range on both axes; else A = 1 (one axis) or 3 (none), Z clear.
;       [04] = horizontal distance, [05] = vertical distance (Dodongo minus bomb), each written only
;       if that axis is in range. [06], [07] = upper, lower limit. [08] = A. Y = $FF.
; ----------------------------------------------------------------------------
Dodongo_IsBombInRange:
    ; [06] = upper limit (exclusive), [07] = lower limit (inclusive).
    LDA Dodongo_BombPositiveLimits, Y
    STA $06
    LDA Dodongo_BombNegativeLimits, Y
    STA $07

    ; [08] = 3. It is shifted right once for each axis that is in range, so 0 means both.
    LDA #$03
    STA $08

    ; Y = 1 for the vertical axis, then 0 for the horizontal one.
    LDY #$01

@LoopAxis:
    ; A = Dodongo centre minus bomb centre on this axis: [00+Y] - [02+Y].
    LDA $0000, Y
    SEC
    SBC $0002, Y

    ; Out of range if distance >= upper limit or < lower limit (signed, by the N flag).
    CMP $06
    BPL @NextAxis
    CMP $07
    BMI @NextAxis

    ; In range on this axis: keep the distance in [04+Y] and shift [08].
    STA $0004, Y
    LSR $08

@NextAxis:
    DEY
    BPL @LoopAxis

    ; A = 0 if in range on both axes.
    LDA $08
    RTS

; TABLE: frame image of the half drawn at ObjX, walking or stunned; index = (ObjDir >> 1) + 5 * frame.
; Two sets of 5, one per animation frame (0, 1); value = frame image number. The half drawn at
; ObjX + $10 (facing left or right) uses the value XOR 1. Entry 3 of each set is unused ($FF).
Dodongo_FrameImages:
    .BYTE $00, $01, $06, $FF, $08, $02, $03, $06    ; frame 0: right 0, left 1, down 6, -, up 8;
    .BYTE $FF, $08                                  ; frame 1: right 2, left 3, down 6, -, up 8

; TABLE: horizontal flip for each entry of Dodongo_FrameImages; same index; value = $40 flip, 0 none.
; Copied to [0F], where any nonzero value flips. Facing down or up the two frames differ only in the
; flip.
Dodongo_FrameHFlips:
    .BYTE $00, $40, $00, $FF, $00, $00, $40, $40    ; frame 0: left flipped;
    .BYTE $FF, $40                                  ; frame 1: left, down and up flipped

; TABLE: frame images of the bloated Dodongo; index = ObjDir >> 1; value = frame image number.
; Read as Dodongo_FrameImages + $14 (bloated substates 1 and 4). Only the first 5 entries are indexed.
Dodongo_BloatedFrameImages:
    .BYTE $04, $05, $07, $FF, $09, $04, $05, $07    ; right 4, left 5, down 7, -, up 9;
    .BYTE $FF, $09                                  ; then the same 5 again, never indexed

; TABLE: horizontal flip for each entry of Dodongo_BloatedFrameImages; same index; $40 flip, 0 none.
; Read as Dodongo_FrameHFlips + $14. Only the first 5 entries are indexed.
Dodongo_BloatedFrameHFlips:
    .BYTE $00, $40, $00, $FF, $00, $00, $40, $00    ; left flipped;
    .BYTE $FF, $00                                  ; second set never indexed

; ----------------------------------------------------------------------------
; Draws the Dodongo: one 16-pixel image facing up or down, two side by side facing left or right.
; IN:   X = object slot.
; NOTE: The animation frame is one bit of FrameCounter: bit 3 when walking and in bloated substate 0
;       (it changes every 8 frames), bit 5 when stunned (every $20 frames). Bloated substates 1 and 4
;       use the bloated images. Bloated substates 2 and 3 (second bomb) use the walking images of
;       frame 0 and are drawn only while FrameCounter bit 1 is set (2 frames in every 4).
;       All sprites use sprite attribute 3 (Anim_SetSpriteDescriptorLevelPaletteRow).
; ----------------------------------------------------------------------------
Dodongo_Draw:
    JSR Anim_SetSpriteDescriptorLevelPaletteRow

    ; [00] = facing direction >> 1: 0 right, 1 left, 2 down, 4 up. It indexes the frame tables.
    LDA ObjDir, X
    LSR
    STA $00

    ; State 0 (walking): go animate on FrameCounter bit 3.
    LDY ObjState, X
    BEQ @DrawWalkingFast

    ; State 2 (stunned): go animate on FrameCounter bit 5.
    LDA #$20
    DEY
    BNE @DrawWalking

    ; State 1 (bloated). Substate 0 looks like walking.
    LDY Dodongo_ObjBloatedSubstate, X
    BEQ @DrawWalkingFast

    ; Substates 2 and 3 (the second bomb's last wait, and dying): go draw flickering.
    CPY #$02
    BEQ @DrawFlickering
    CPY #$03
    BEQ @DrawFlickering

    ; Substates 1 and 4: add $14 to the index to reach the bloated images and flips.
    LDA $00
    CLC
    ADC #$14
    TAY
    JMP @PrepareToDraw

@DrawFlickering:
    ; Skip drawing while FrameCounter bit 1 is clear; else use the walking images of frame 0.
    LDA FrameCounter
    AND #$02
    BEQ Dodongo_Draw_Return
    LDY $00
    JMP @PrepareToDraw

@DrawWalkingFast:
    LDA #$08                    ; FrameCounter bit 3: the frame changes every 8 frames.

@DrawWalking:
    ; A = the FrameCounter bit that selects the animation frame. Bit clear: frame 0.
    LDY $00
    AND FrameCounter
    BEQ @PrepareToDraw

    ; Frame 1: add 5 to the index to reach the second set of images and flips.
    TYA
    CLC
    ADC #$05
    TAY

@PrepareToDraw:
    ; Y = index into the frame tables. [00], [01] = ObjX, ObjY; [0F] = 0.
    JSR Anim_FetchObjPosForSpriteDescriptor
    TYA                         ; Save the table index.
    PHA

    LDA Dodongo_FrameHFlips, Y
    STA $0F                     ; [0F] = horizontal flip (nonzero flips).
    LDA Dodongo_FrameImages, Y

    ; Frame images 7 and 9 (bloated, facing down and up) are drawn mirrored; all others are not.
    CMP #$07
    BEQ :+
    CMP #$09
    BNE @DrawNotMirrored
:
    JSR Anim_DrawObjectMirrored
    JMP @DrawSecondHalf

@DrawNotMirrored:
    JSR Anim_DrawObjectNotMirrored

@DrawSecondHalf:
    PLA                         ; Restore the table index.
    TAY

    ; Facing up or down: there is no second half. Return.
    LDA ObjDir, X
    AND #$03
    BEQ Dodongo_Draw_Return

    ; Facing left or right: draw the second half at ObjX + $10. [01] still holds ObjY.
    LDA ObjX, X
    CLC
    ADC #$10
    STA $00
    JSR Anim_SetSpriteDescriptorLevelPaletteRow

    ; The second half's image is the first half's XOR 1 (0<->1, 2<->3, 4<->5), drawn with the same
    ; flip in [0F]. Facing right the half at ObjX is image 0 (2 in frame 1) and the half at
    ; ObjX + $10 is image 1 (3). Facing left both are flipped and trade places: image 1 (3) at ObjX,
    ; image 0 (2) at ObjX + $10.
    LDA Dodongo_FrameImages, Y
    EOR #$01
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Sets both sprite attribute bytes of the sprite descriptor to 3 (sprite palette 3).
; OUT:  A = 3; [04] = [05] = 3 (left and right sprite attributes).
; NOTE: Sprite palette 3 is palette row 7, which Mode5Play_Init loads per room: for a room whose
;       slot 1 holds a Dodongo, the row shared by Digdogger and Dodongo. Also used by Digdogger.
; ----------------------------------------------------------------------------
Anim_SetSpriteDescriptorLevelPaletteRow:
    LDA #$03
    JSR Anim_SetSpriteDescriptorAttributes

; Shared exit of Dodongo_Draw; the routine above also returns through it.
Dodongo_Draw_Return:
    RTS

; Filler: 31 bytes of $FF before the next routine. No label; nothing refers to them.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF
