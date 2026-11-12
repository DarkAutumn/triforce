; ============================================================================
; FILE:     bank4/lanmola.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9671-10027
; SUBJECTS: Lanmola
; TOPIC:    Lanmola (types $3A red, $3B blue): per-segment update, death handling that shortens the
;           worm from the tail, the head's direction choice, and the pixel mover.
; SEE ALSO: bank4/zelda-lanmola-inits.asm (Lanmola_Init: 10 segments in two worms, speed, start spot);
;           bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink, Damage_ObjTypeDamageToLink);
;           bank7/object-update-table.asm (HP table, dead dummy turning into nothing);
;           bank4/digdogger.asm (a mover built the same way as Lanmola_Move).
; KEY FACTS:
;   - A room's Lanmola object makes two worms of 5 segments: slots $01-$05 and $06-$0A; slots $05
;     and $0A are the heads, $01 and $06 the tails (Lanmola_Update, Lanmola_UpdateHead).
;   - Speed is 1 pixel per frame for type $3A and 2 for type $3B (Lanmola_Speed, Lanmola_Move).
;   - Every segment can be hit. A killed segment is revived with HP $20 and the worm's lowest live
;     slot becomes a dead dummy ($5D) instead; only when the head is the last segment does a
;     Lanmola object really die (Lanmola_Update@FindTail).
;   - So each worm takes 5 kills. Every one of them adds to WorldKillCount and HelpDropCount, but
;     only the head's death advances WorldKillCycle, adds to RoomKillCount and can drop an item;
;     dead dummies do none of these (Lanmola_Update; Damage_KillMonster, Obj_EndMetastate).
;   - With the clock the segments stop but are still drawn and can be hit (Lanmola_Update).
;   - The head decides at every 16-pixel grid point: Random,X < $80 toward Link, else straight or
;     a perpendicular turn by Random+1,X; never reverses at random (Lanmola_UpdateHead).
;   - Directions pass one segment down the chain every 8 pixels, so segments trail 8 pixels apart
;     (Lanmola_UpdateHead@PropagateDirs).
;   - Segments do not test tiles when moving; only the head's chosen direction is tested against
;     the room bounds and tiles (Lanmola_UpdateHead@SetDirAndCheckTiles, Lanmola_Move).
; CONTENTS: (generated; L = line in this file)
;   L0043 Lanmola_Update             code  Updates one Lanmola segment: moves it, draws it, takes weapon hits and shortens the worm on a kill.
;   L0151 Lanmola_UpdateHead_Return  code  Shared RTS for the early exits of Lanmola_UpdateHead.
;   L0166 Lanmola_UpdateHead         code  Steers a Lanmola head: passes directions down the chain every 8 pixels, picks a new one every 16.
;   L0334 Lanmola_Move               code  Moves a Lanmola segment Lanmola_Speed pixels (1 or 2) in its direction.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates one Lanmola segment: moves it, draws it, takes weapon hits and shortens the worm on a kill.
; IN:   X = object slot ($01-$05 first Lanmola, $06-$0A second; $05 and $0A are the heads).
; OUT:  on a kill the lowest live slot of that Lanmola becomes a dead dummy ($5D) at the hit
;       segment's position and the hit segment is revived with HP $20.
; NOTE: With the clock (InvClock) segments do not move but are still drawn and can still be hit.
; NOTE: GuardFire_Update_Return is an RTS in the chunk before this one.
; ----------------------------------------------------------------------------
Lanmola_Update:
    ; A segment with direction 0 has not been set in motion yet (see Lanmola_UpdateHead): it is
    ; neither drawn nor checked for collisions.
    LDA ObjDir, X
    BEQ GuardFire_Update_Return

    ; With the clock, skip movement.
    LDA InvClock
    BNE @DrawAndCheckCollisions
    JSR Lanmola_Move

    ; Only the heads (slots $05 and $0A) choose directions.
    CPX #$05
    BEQ :+
    CPX #$0A
    BNE @DrawAndCheckCollisions
:
    JSR Lanmola_UpdateHead

@DrawAndCheckCollisions:
    LDA ObjX, X
    PHA                         ; Save the X coordinate.

    ; A segment is one 8-pixel-wide sprite. Center it in the 16-pixel cell by drawing it at X + 4.
    CLC
    ADC #$04
    STA ObjX, X

    ; Sprite attributes [03] = speed EOR 3: speed 1 (type $3A) -> palette 2, speed 2 (type $3B)
    ; -> palette 1.
    LDA Lanmola_Speed
    EOR #$03
    STA $03

    ; Tile $9E for a head, $A0 for any other segment.
    LDA #$9E
    CPX #$05
    BEQ @Draw
    CPX #$0A
    BEQ @Draw
    LDA #$A0

@Draw:
    JSR Anim_WriteSprite
    PLA                         ; Restore the X coordinate.
    STA ObjX, X

    ; Collision_CheckMonsterVsWeaponsAndLink may change the facing direction; the chain depends on it, so keep it.
    LDA ObjDir, X
    PHA
    JSR Collision_CheckMonsterVsWeaponsAndLink
    PLA
    STA ObjDir, X

    ; Metastate 0 = still alive: done.
    LDA ObjMetastate, X
    BEQ GuardFire_Update_Return

    ; The segment was killed this frame. Instead of letting it die, revive it and remove the
    ; segment at the tail end of its Lanmola.
    JSR Shove_Reset

    ; Give the hit segment its full HP ($20) again.
    LDA #$20
    STA ObjHP, X

    ; Find the tail: the lowest slot of this Lanmola that still has the Lanmola object type.
    ; Y + 1 is the slot tested: start at slot $01 for the first Lanmola (X < 6), $06 for the second.
    LDY #$FF
    CPX #$06
    BCC @FindTail
    LDY #$04

@FindTail:
    INY
    LDA ObjType+1, Y
    CMP Lanmola_Type
    BNE @FindTail

    ; Timer $11 for the tail slot. It has no effect: DeadDummy_StartDeathSparkle does not read the timer, and
    ; the first sparkle step (metastate $10) advances at once and reloads the timer with 6
    ; (Obj_AnimateMetastate). When the tail is the head, the sparkle is already at $10.
    LDA #$11
    STA a:ObjTimer+1, Y

    ; Move the tail slot to where the hit segment is (the sparkle appears there), with the hit
    ; segment's invincibility timer.
    LDA ObjInvincibilityTimer, X
    STA ObjInvincibilityTimer+1, Y
    LDA ObjX, X
    STA a:ObjX+1, Y
    LDA ObjY, X
    STA a:ObjY+1, Y

    ; If the tail found is a head (slot $05 or $0A), the head itself was the last segment: leave
    ; it dying, so that it dies as an ordinary monster.
    CPY #$04
    BEQ GuardFire_Update_Return
    CPY #$09
    BEQ GuardFire_Update_Return

    ; Turn the tail slot into a dead dummy ($5D), which plays the death sparkle, and bring the
    ; hit segment back to life.
    LDA #$5D
    STA ObjType+1, Y
    JMP Obj_ResetMetastate

; Shared RTS for the early exits of Lanmola_UpdateHead.
Lanmola_UpdateHead_Return:
    RTS

; ----------------------------------------------------------------------------
; Steers a Lanmola head: passes directions down the chain every 8 pixels, picks a new one every 16.
; IN:   X = head's object slot ($05 or $0A).
; OUT:  ObjDir of the head and of the four segments behind it; LanmolaViableDirMask.
; NOTE: At each 16-pixel grid point: Random,X < $80 -> turn toward Link; else Random+1,X >= $80 ->
;       straight on; else a perpendicular turn ($40-$7F: up if moving horizontally, left if
;       vertically; < $40: down or right). The head never picks its opposite direction at random,
;       but the turn toward Link may be vertical even when that reverses it.
; NOTE: A direction blocked by the room bounds or an unwalkable tile is replaced by the next
;       allowed one, rotating 8 -> 4 -> 2 -> 1 -> 8. The search has no exit if all three allowed
;       directions are blocked.
; ----------------------------------------------------------------------------
Lanmola_UpdateHead:
    ; Act only when X is a multiple of 8 ...
    LDA ObjX, X
    AND #$07
    BNE Lanmola_UpdateHead_Return

    ; ... and (Y + 3) is a multiple of 8 (object Y is offset 3 pixels from the tile grid).
    LDA ObjY, X
    CLC
    ADC #$03
    AND #$07
    BNE Lanmola_UpdateHead_Return

    ; Pass directions one segment toward the tail, 4 copies counted in [00].
    LDA #$04
    STA $00

    ; Y = tail slot - 1: 0 for the head in slot $05, 5 for the head in slot $0A.
    LDY #$00
    CPX #$05
    BEQ @PropagateDirs
    LDY #$05

@PropagateDirs:
    ; From the tail up: each segment takes the direction of the segment ahead of it. A segment
    ; whose direction becomes nonzero starts moving, so segments follow 8 pixels apart.
    LDA a:ObjDir+2, Y
    STA a:ObjDir+1, Y
    INY
    DEC $00
    BNE @PropagateDirs

    ; Choose a new direction only when X is a multiple of $10 ...
    LDA ObjX, X
    AND #$0F
    BNE @Exit

    ; ... and (Y + 3) is a multiple of $10.
    LDA ObjY, X
    CLC
    ADC #$03
    AND #$0F
    BNE @Exit

    ; Build the opposite of the facing direction: swap bits 1<->2 and 4<->8.
    LDA ObjDir, X
    LSR
    AND #$05
    STA $00
    LDA ObjDir, X
    ASL
    AND #$0A
    ORA $00

    ; The viable directions are all but the opposite one.
    EOR #$0F
    STA LanmolaViableDirMask

    ; Random,X < $80: go turn toward Link.
    LDA Random, X
    CMP #$80
    BCC @TurnTowardLink

    ; Otherwise go straight or turn perpendicular, by Random+1,X (in Y).
    LDA ObjDir, X
    LDY Random+1, X

    ; Random+1,X >= $80: keep the facing direction.
    CPY #$80
    BCS @SetDirAndCheckTiles

@LoopFindPerpendicular:
    ; Rotate the direction bit right (8 -> 4 -> 2 -> 1 -> 8) ...
    LSR
    BCC :+
    LDA #$08
:
    ; ... skipping the opposite direction.
    BIT LanmolaViableDirMask
    BEQ @LoopFindPerpendicular

    ; Y >= $40: take this first perpendicular direction (up if moving horizontally, else left).
    CPY #$40
    BCS @SetDirAndCheckTiles

    ; Y < $40: rotate once more to the second perpendicular direction (down if moving
    ; horizontally, else right). Y = $40 ends the loop on the next pass.
    LDY #$40
    BCC @LoopFindPerpendicular  ; Always taken (C is clear from the CPY above).

@SetDirAndCheckTiles:
    STA ObjDir, X

    ; [0F] = moving direction for the room bound and tile tests. Room_BoundDir zeroes it if blocked.
    STA $0F
    JSR Room_BoundDir
    LDA $0F
    BNE @CheckTileCollision

@FindNonOpposite:
    ; Blocked: try the next viable direction after the one just tried.
    LDA ObjDir, X

@LoopFindNonOpposite:
    ; Rotate the direction bit right (8 -> 4 -> 2 -> 1 -> 8).
    LSR
    BCC :+
    LDA #$08
:
    ; Viable: go test it. Opposite of the original facing: rotate again.
    BIT LanmolaViableDirMask
    BNE @SetDirAndCheckTiles
    BEQ @LoopFindNonOpposite

@CheckTileCollision:
    ; Not blocked by the room bounds. A tile at or above RoomFirstUnwalkableTile ahead blocks too.
    JSR Tile_GetAheadOfObj
    CMP RoomFirstUnwalkableTile
    BCS @FindNonOpposite

@Exit:
    ; If all three viable directions were blocked, the search above would never end. Whether
    ; any room with Lanmolas allows that was not checked (unverified).
    RTS

@TurnTowardLink:
    ; [02] = horizontal direction toward Link: 1 (right) if Link's X >= the head's X, else 2.
    LDA #$01
    STA $02
    LDA ObjX
    SEC
    SBC ObjX, X
    BCS :+
    ASL $02
:
    ; [03] = vertical direction toward Link: 4 (down) if Link's Y >= the head's Y, else 8.
    LDA #$04
    STA $03
    LDA ObjY
    SEC
    SBC ObjY, X
    BCS :+
    ASL $03
:
    ; Use the horizontal direction only if it is viable (not the opposite of the facing) ...
    LDA $02
    BIT LanmolaViableDirMask
    BEQ @SetVertical

    ; ... and it equals the direction Link is facing (ObjDir of slot 0).
    BIT ObjDir
    BNE @TestDir

@SetVertical:
    ; Otherwise use the vertical direction toward Link. It is not tested against the mask, so
    ; it may be the opposite of the facing direction.
    LDA $03
@TestDir:
    ; Set it and test it against the room bounds and tiles.
    JMP @SetDirAndCheckTiles

; ----------------------------------------------------------------------------
; Moves a Lanmola segment Lanmola_Speed pixels (1 or 2) in its direction.
; IN:   X = object slot; ObjDir,X = direction bits.
; OUT:  ObjX,X / ObjY,X.
; NOTE: No position fraction, no tile or bounds test here; only the head's choices keep the worm
;       in the room.
; ----------------------------------------------------------------------------
Lanmola_Move:
    ; [02] starts as mask $A1 and is shifted left before each later test: bit 0 walks through the
    ; direction bits 1, 2, 4, 8, while the bit shifted out (1, 0, 1) sets the carry needed by the
    ; following SBC, ADC, SBC.
    LDA #$A1
    STA $02

    ; Right (1): X += speed.
    LDA ObjDir, X
    BIT $02
    BEQ @Left
    LDA ObjX, X
    CLC
    ADC Lanmola_Speed
    STA ObjX, X

@Left:
    ; Left (2): X -= speed.
    LDA ObjDir, X
    ASL $02                     ; Mask $42; carry set for SBC.
    BIT $02
    BEQ @Down
    LDA ObjX, X
    SBC Lanmola_Speed
    STA ObjX, X

@Down:
    ; Down (4): Y += speed.
    LDA ObjDir, X
    ASL $02                     ; Mask $84; carry clear for ADC.
    BIT $02
    BEQ @Up
    LDA ObjY, X
    ADC Lanmola_Speed
    STA ObjY, X

@Up:
    ; Up (8): Y -= speed.
    LDA ObjDir, X
    ASL $02                     ; Mask $08; carry set for SBC.
    BIT $02
    BEQ @Exit
    LDA ObjY, X
    SBC Lanmola_Speed
    STA ObjY, X

@Exit:
    RTS
