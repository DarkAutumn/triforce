; ============================================================================
; FILE:     bank4/wizzrobe.asm           BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 7027-7636
; SUBJECTS: Wizzrobe
; TOPIC:    Blue Wizzrobe ($23: walks, teleports through blocks, shoots magic shot $58) and Red Wizzrobe
;           ($24: appears near Link, shoots magic shot $59, vanishes); their shared tile test, shooting,
;           weapon collisions and drawing. Ganon reuses the Blue Wizzrobe's movement.
; SEE ALSO: bank4/shoot-helpers-candle.asm (Shooter_ShootLimited: free slot and the 4-shot limit);
;           bank4/monster-shots-fireballs.asm (MonsterShot_Update moves shots $58/$59);
;           bank1/sword-bomb-arrow-collision.asm (the weapon tests called here, damage types);
;           bank1/link-collision-harm.asm (Link_CheckMonsterTouch, Damage_ObjTypeDamageToLink use, shield rule);
;           bank7/tile-collision.asm (Tile_GetUnderObj); bank4/ganon.asm (Ganon_MoveAndShoot calls
;           Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile); bank7/object-init-table.asm (both types are
;           initialized by Obj_ResetMetastateAndTimer: no spawn cloud).
; KEY FACTS:
;   - Weapons: Wizzrobe_DrawAndCheckCollisions sets ObjInvincibilityMask to $F6 and tests only object
;     slots $0E, $10, $11 and $0D. Only sword-type damage (sword, sword shot) and bombs hurt. Fire and
;     Link's magic shot are tested but masked (the magic shot still ends on the overlap). The
;     boomerang ($0F) and the arrow or rod ($12) are never tested: they pass through, no stun.
;   - While InvClock is set both types skip all movement and shooting and are drawn and hittable every
;     frame, even a hidden Red Wizzrobe (Wizzrobe_Blue_Update, Wizzrobe_Red_Update).
;   - Blue walk: 1 pixel every second frame while ObjTimer >= $10, then 14 frames standing, then at
;     timer 1 it picks a diagonal teleport target (Wizzrobe_Blue_WalkOrTeleport, Wizzrobe_Blue_WalkEveryOtherFrame).
;   - Blue teleport: $20 frames at 1 pixel per frame per axis; on odd ObjRemDistance it is neither drawn
;     nor collision-tested (cannot be hit, cannot hurt). Afterwards ObjTimer = Random,X OR $70
;     (Wizzrobe_Blue_BeginTeleporting, Wizzrobe_DrawAndCheckCollisionsIntermittently,
;     Wizzrobe_Blue_AlignWithNearestSquareAndRandomizeTimer).
;   - Blue walking into a block tile ($B0-$B3) or a tile >= $F4 starts a $20-pixel teleport straight
;     through it; any other unwalkable tile reverses it (Wizzrobe_Blue_MoveAndCheckTile).
;   - Blue turns toward Link every $40 walking steps (128 frames), alternating between the horizontal
;     and the vertical axis (Wizzrobe_Blue_TurnTowardLink).
;   - Blue shoots only when not teleporting, on frames where FrameCounter AND $1F = 0, when it faces
;     Link along a shared 16-pixel row, or along its column with Link's X exactly on that column
;     (Wizzrobe_Blue_TryShooting, Wizzrobe_Blue_TryShooting_CheckColumn).
;   - Red cycle is 208 frames, driven by ObjState counting down: 1 frame choosing a place, 63 fading in,
;     64 solid, 16 fading out, 64 hidden (Wizzrobe_Red_Update and its four handlers).
;   - Red appears 2, 3, 4 or 5 squares from Link in a cardinal direction, facing Link, on a walkable
;     tile with Y in $5D-$C3; a rejected spot is retried every frame (Wizzrobe_Red_State3Appear).
;   - Red shoots magic shot $59 once per cycle, at state $B0: the 16th solid frame (Wizzrobe_Red_State2Solid).
;   - Every shot needs a free monster slot and fewer than 4 counted shots (Wizzrobe_ShootMagicShot -> Shooter_ShootLimited).
;   - While fading (either type), on skipped frames there is no weapon test and no test against Link
;     (Wizzrobe_DrawAndCheckCollisionsIntermittently). A hidden Red Wizzrobe is never tested.
; CONTENTS: (generated; L = line in this file)
;   L0095 Wizzrobe_Blue_Update                                   code  Updates a Blue Wizzrobe (type $23): walk or teleport, try to shoot, then draw and check collisions.
;   L0103 Wizzrobe_DrawAndCheckCollisionsIntermittently          code  Draws and checks collisions only on frames where ObjRemDistance,X is even (flicker while fading).
;   L0112 Wizzrobe_JumpToDrawAndCheckCollisions                  code  Jumps to Wizzrobe_DrawAndCheckCollisions (out of branch range from above); also the clock path.
;   L0123 Wizzrobe_Blue_WalkOrTeleport                           code  Runs one frame of the Blue Wizzrobe's walk, pause or teleport, chosen by ObjTimer and ObjRemDistance.
;   L0150 Wizzrobe_Blue_WalkOrTeleport_Return                    code  Shared exit of the routines above.
;   L0154 Wizzrobe_Blue_WalkEveryOtherFrame                      code  Walking step: on even frames count a step, turn if due and move; on odd frames only test the turn.
;   L0164 Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile         code  Counts a step, turns toward Link if due, moves 1 pixel and handles the tile ahead.
;   L0174 Wizzrobe_Blue_MoveAndCheckTile                         code  Moves 1 pixel in ObjDir, then reverses at a wall or starts a teleport through a block or water tile.
;   L0226 Wizzrobe_Blue_AdvanceCounterAndTurnTowardLinkIfNeeded  code  Adds 1 to the Blue Wizzrobe's turn counter, then turns toward Link if the counter's low 6 bits are 0.
;   L0230 Wizzrobe_Blue_TurnTowardLinkIfNeeded                   code  Turns toward Link when the turn counter is a multiple of $40; otherwise returns.
;   L0241 Wizzrobe_Blue_TurnTowardLink                           code  Faces the Blue Wizzrobe toward Link on one axis and, if the direction changed, aligns it to a square.
;   L0271 Wizzrobe_Blue_DirStepsX                                data  TABLE: X step of Wizzrobe_Blue_Move; index = direction value $00-$0A (8-way); value = signed pixels.
;   L0276 Wizzrobe_Blue_DirStepsY                                data  TABLE: Y step of Wizzrobe_Blue_Move; index = direction value $00-$0A (8-way); value = signed pixels.
;   L0286 Wizzrobe_Blue_Move                                     code  Moves the object 1 pixel in its 8-way direction ObjDir (no fraction, no collision test).
;   L0298 Wizzrobe_Blue_Move_Return                              code  Shared exit of Wizzrobe_Blue_Move and the turn routines above.
;   L0302 Wizzrobe_Blue_TeleportTargetOffsetsX                   data  TABLE: X offset of the teleport target; index = random 0-3; value = signed pixels (-$20 or +$20).
;   L0306 Wizzrobe_Blue_TeleportTargetOffsetsY                   data  TABLE: Y offset of the teleport target; index = random 0-3; value = signed pixels (-$20 or +$20).
;   L0310 Wizzrobe_Blue_TeleportDirs                             data  TABLE: diagonal direction of the teleport; index = random 0-3; value = direction bits.
;   L0320 Wizzrobe_Blue_ChooseTeleportTarget                     code  Picks a random diagonal spot 2 squares away; teleports there if its tile is walkable, else walks on.
;   L0363 Wizzrobe_Blue_BeginTeleporting                         code  Starts a Blue Wizzrobe teleport of $20 pixels in the current ObjDir and aligns it to a square.
;   L0375 Wizzrobe_Blue_AlignWithNearestSquareAndRandomizeTimer  code  Starts a walking phase: ObjTimer = Random,X OR $70, then aligns to the nearest square.
;   L0386 Wizzrobe_Blue_AlignWithNearestSquare                   code  Snaps the object to the nearest 16-pixel square: X to a multiple of $10, Y to a multiple of $10 minus 3.
;   L0397 Wizzrobe_SetYAlignedToSquare                           code  Second entry: A = a Y coordinate; stores (A AND $F0) - 3 in ObjY,X and returns it in A.
;   L0405 Wizzrobe_TileTestOffsetsX                              data  TABLE: X offset of the Wizzrobe tile test point; index = direction value - 1 ($00-$09); pixels.
;   L0411 Wizzrobe_TileTestOffsetsY                              data  TABLE: Y offset of the Wizzrobe tile test point; index = direction value - 1 ($00-$09); signed pixels.
;   L0420 Wizzrobe_GetCollidableTile                             code  Tests the tile in front of a Wizzrobe (or Ganon) for its facing direction ObjDir.
;   L0428 Wizzrobe_GetCollidableTileForDir                       code  Tests the tile at the object's position plus the test offset of an 8-way direction.
;   L0455 Wizzrobe_GetBaseCollidableTile                         code  Reads the tile at (ObjX, ObjY + $0B) and compares it with the room's first unwalkable tile.
;   L0469 Wizzrobe_Blue_TryShooting                              code  Lets a Blue Wizzrobe shoot magic shot $58 when it faces Link along its row or column.
;   L0497 Wizzrobe_Blue_TryShooting_Return                       code  Shared exit of the shooting routines.
;   L0503 Wizzrobe_Blue_TryShooting_CheckColumn                  code  Column half of Wizzrobe_Blue_TryShooting: shoots if Link is in the monster's column and it faces him.
;   L0523 Wizzrobe_ShootMagicShot58                              code  Shoots the Blue Wizzrobe's magic shot (object type $58).
;   L0534 Wizzrobe_ShootMagicShot                                code  Shoots a magic shot of the given object type from the monster, with the magic sound.
;   L0550 Wizzrobe_Red_Update                                    code  Updates a Red Wizzrobe (type $24): counts ObjState down and runs the handler for its top two bits.
;   L0569 Wizzrobe_Red_Update_JumpTable                          data  TABLE: Red Wizzrobe handlers; index = ObjState >> 6.
;   L0577 Wizzrobe_Red_AppearOffsetsX                            data  TABLE: X offset from Link where a Red Wizzrobe appears; index = Random,X AND $0F; signed pixels.
;   L0582 Wizzrobe_Red_AppearOffsetsY                            data  TABLE: Y offset from Link where a Red Wizzrobe appears; index = Random,X AND $0F; signed pixels.
;   L0588 Wizzrobe_Red_AppearDirs                                data  TABLE: facing direction of an appearing Red Wizzrobe; index = Random,X AND 3; direction bits.
;   L0597 Wizzrobe_Red_State1FadeOut                             code  Red Wizzrobe state group 1 ($7F-$40), fading out: flickers for 16 frames.
;   L0615 Wizzrobe_Red_State3Appear                              code  Red Wizzrobe state group 3 ($FF-$C0): at $FF picks where to appear; $FE-$C0 fades in (63 frames).
;   L0659 Wizzrobe_Red_State0Hidden                              code  Red Wizzrobe state group 0 ($3F-$00), hidden: does nothing (not drawn, no collisions) for 64 frames.
;   L0667 Wizzrobe_Red_State2Solid                               code  Red Wizzrobe state group 2 ($BF-$80), solid for 64 frames: shoots magic shot $59 at state $B0.
;   L0683 Wizzrobe_DrawAndCheckCollisions                        code  Tests a Wizzrobe against Link's weapons and against Link, then draws it.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Blue Wizzrobe (type $23): walk or teleport, try to shoot, then draw and check collisions.
; IN:   X = object slot.
; NOTE: Two phases. Walking: ObjTimer,X nonzero. Teleporting: ObjTimer,X = 0 and ObjRemDistance,X = pixels
;       left. With the clock (InvClock nonzero) it only draws and checks collisions, every frame.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_Update:
    LDA InvClock
    BNE Wizzrobe_JumpToDrawAndCheckCollisions
    JSR Wizzrobe_Blue_WalkOrTeleport
    JSR Wizzrobe_Blue_TryShooting

; Draws and checks collisions only on frames where ObjRemDistance,X is even (flicker while fading).
; The Red Wizzrobe enters here with its fade counter, which is the same byte (Wizzrobe_ObjRedFadeCounter).
Wizzrobe_DrawAndCheckCollisionsIntermittently:
    ; Blue: 0 = walking, so it is drawn every frame; while teleporting the distance left counts down by
    ; 1 per frame, so every other frame is skipped. A skipped frame has no drawing, no weapon test and
    ; no test against Link.
    LDA ObjRemDistance, X
    LSR
    BCS Wizzrobe_Blue_WalkOrTeleport_Return             ; Odd: skip this frame.

; Jumps to Wizzrobe_DrawAndCheckCollisions (out of branch range from above); also the clock path.
Wizzrobe_JumpToDrawAndCheckCollisions:
    JMP Wizzrobe_DrawAndCheckCollisions

; ----------------------------------------------------------------------------
; Runs one frame of the Blue Wizzrobe's walk, pause or teleport, chosen by ObjTimer and ObjRemDistance.
; IN:   X = object slot.
; NOTE: ObjTimer >= $10: walk (1 pixel every second frame). $0F-$02: stand still (14 frames). 1: choose a
;       teleport target. 0 with distance left: move 1 pixel per frame. 0 with no distance left: the
;       teleport is over (also the very first update): align to a square, set ObjTimer to Random,X OR
;       $70 ($70-$7F or $F0-$FF) and turn toward Link.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_WalkOrTeleport:
    LDA ObjTimer, X
    BNE @Walk

    ; Timer = 0: teleporting. While distance remains, count it down and move.
    LDA ObjRemDistance, X
    BEQ @WalkAgain
    DEC ObjRemDistance, X
    JMP Wizzrobe_Blue_MoveAndCheckTile

@WalkAgain:
    ; Timer = 0 and distance = 0: start a walking phase.
    JSR Wizzrobe_Blue_AlignWithNearestSquareAndRandomizeTimer
    JMP Wizzrobe_Blue_TurnTowardLink

@Walk:
    ; A = timer (nonzero).
    ;   >= $10: turn if needed and move, every other frame
    ;   $0F-2:  do nothing
    ;   1:      choose a place to teleport to
    CMP #$10
    BCS Wizzrobe_Blue_WalkEveryOtherFrame
    CMP #$01
    BNE Wizzrobe_Blue_WalkOrTeleport_Return
    JSR Wizzrobe_Blue_ChooseTeleportTarget

; Shared exit of the routines above.
Wizzrobe_Blue_WalkOrTeleport_Return:
    RTS

; Walking step: on even frames count a step, turn if due and move; on odd frames only test the turn.
Wizzrobe_Blue_WalkEveryOtherFrame:
    LDA FrameCounter
    LSR
    BCS Wizzrobe_Blue_TurnTowardLinkIfNeeded   ; Odd frame: no step.

; ----------------------------------------------------------------------------
; Counts a step, turns toward Link if due, moves 1 pixel and handles the tile ahead.
; IN:   X = object slot.
; NOTE: Also called every frame by Ganon, with ObjRemDistance,X forced to 1 (so blocks never stop him).
; ----------------------------------------------------------------------------
Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile:
    JSR Wizzrobe_Blue_AdvanceCounterAndTurnTowardLinkIfNeeded

; ----------------------------------------------------------------------------
; Moves 1 pixel in ObjDir, then reverses at a wall or starts a teleport through a block or water tile.
; IN:   X = object slot; ObjRemDistance,X nonzero = already teleporting.
; NOTE: "Block or water" = tile $B0-$B3 or tile >= $F4 (primary squares $B0 block and $F4, the dungeon
;       water tile the ladder crosses). A wall is any other tile >= RoomFirstUnwalkableTile: the direction
;       is reversed on both axes and the step is taken back in the same frame.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_MoveAndCheckTile:
    JSR Wizzrobe_Blue_Move
    JSR Wizzrobe_GetCollidableTile
    BCC @Exit                   ; Walkable: done.

    ; Unwalkable. Sort the tile into block/water or wall.
    LDA Wizzrobe_ObjLastTile, X
    AND #$FC
    CMP #$B0
    BEQ @HitBlockOrWater        ; Tiles $B0-$B3.
    CMP #$F4
    BCC @HitWall                ; Below $F4: wall.

@HitBlockOrWater:
    ; Already teleporting: keep going through it next frame.
    LDA ObjRemDistance, X
    BNE Wizzrobe_Blue_WalkOrTeleport_Return

    ; Walking: start a teleport through the obstacle, in the current direction.
    JMP Wizzrobe_Blue_BeginTeleporting

@HitWall:
    ; Flip the vertical component (8 <-> 4) of the facing direction, if it has one.
    LDY ObjDir, X
    TYA
    AND #$0C
    BEQ @CheckHorizontal
    TYA
    EOR #$0C
    STA ObjDir, X
    TAY

@CheckHorizontal:
    ; Flip the horizontal component (1 <-> 2), if it has one.
    TYA
    AND #$03
    BEQ @StepBack
    TYA
    EOR #$03
    STA ObjDir, X

@StepBack:
    ; Step back out of the wall.
    JMP Wizzrobe_Blue_Move

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Adds 1 to the Blue Wizzrobe's turn counter, then turns toward Link if the counter's low 6 bits are 0.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_AdvanceCounterAndTurnTowardLinkIfNeeded:
    INC Wizzrobe_ObjBlueTurnCounter, X

; Turns toward Link when the turn counter is a multiple of $40; otherwise returns.
Wizzrobe_Blue_TurnTowardLinkIfNeeded:
    LDA Wizzrobe_ObjBlueTurnCounter, X
    AND #$3F
    BNE Wizzrobe_Blue_Move_Return

; ----------------------------------------------------------------------------
; Faces the Blue Wizzrobe toward Link on one axis and, if the direction changed, aligns it to a square.
; IN:   X = object slot; bit 6 of Wizzrobe_ObjBlueTurnCounter,X: 0 = horizontal axis, 1 = vertical.
; OUT:  ObjDir,X; ObjX,X and ObjY,X snapped to a square only when the direction changed.
; NOTE: Bit 6 alternates as the counter passes each multiple of $40, and Wizzrobe_Blue_BeginTeleporting flips it.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_TurnTowardLink:
    LDA Wizzrobe_ObjBlueTurnCounter, X
    AND #$40
    BNE @TurnVertically

    ; Monster's X >= Link's X: left (2). Else right (1).
    LDA #$02
    LDY ObjX, X
    CPY ObjX
    BCS :+
    LSR
:
    JMP @SetDir

@TurnVertically:
    ; Monster's Y >= Link's Y: up (8). Else down (4).
    LDA #$08
    LDY ObjY, X
    CPY ObjY
    BCS @SetDir
    LSR

@SetDir:
    ; Already facing that way: return without aligning.
    CMP ObjDir, X
    BEQ Wizzrobe_Blue_Move_Return
    STA ObjDir, X
    JMP Wizzrobe_Blue_AlignWithNearestSquare

; TABLE: X step of Wizzrobe_Blue_Move; index = direction value $00-$0A (8-way); value = signed pixels.
Wizzrobe_Blue_DirStepsX:
    .BYTE $00, $01, $FF, $00, $00, $01, $FF, $00    ; -, right +1, left -1, -, down 0, 5 +1, 6 -1, -
    .BYTE $00, $01, $FF                             ; up 0, 9 (up-right) +1, $A (up-left) -1

; TABLE: Y step of Wizzrobe_Blue_Move; index = direction value $00-$0A (8-way); value = signed pixels.
Wizzrobe_Blue_DirStepsY:
    .BYTE $00, $00, $00, $00, $01, $01, $01, $00    ; -, right 0, left 0, -, down +1, 5 +1, 6 +1, -
    .BYTE $FF, $FF, $FF                             ; up -1, 9 (up-right) -1, $A (up-left) -1

; ----------------------------------------------------------------------------
; Moves the object 1 pixel in its 8-way direction ObjDir (no fraction, no collision test).
; IN:   X = object slot; ObjDir,X = direction bits (diagonals allowed).
; OUT:  ObjX,X and ObjY,X changed by -1, 0 or +1 each.
; NOTE: Used by the Blue Wizzrobe (walking and teleporting) and by Ganon.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_Move:
    LDY ObjDir, X               ; The direction value itself is the table index.
    LDA ObjX, X
    CLC
    ADC Wizzrobe_Blue_DirStepsX, Y
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC Wizzrobe_Blue_DirStepsY, Y
    STA ObjY, X

; Shared exit of Wizzrobe_Blue_Move and the turn routines above.
Wizzrobe_Blue_Move_Return:
    RTS

; TABLE: X offset of the teleport target; index = random 0-3; value = signed pixels (-$20 or +$20).
Wizzrobe_Blue_TeleportTargetOffsetsX:
    .BYTE $E0, $20, $E0, $20

; TABLE: Y offset of the teleport target; index = random 0-3; value = signed pixels (-$20 or +$20).
Wizzrobe_Blue_TeleportTargetOffsetsY:
    .BYTE $E0, $E0, $20, $20

; TABLE: diagonal direction of the teleport; index = random 0-3; value = direction bits.
Wizzrobe_Blue_TeleportDirs:
    .BYTE $0A, $09, $06, $05    ; up-left, up-right, down-left, down-right

; ----------------------------------------------------------------------------
; Picks a random diagonal spot 2 squares away; teleports there if its tile is walkable, else walks on.
; IN:   X = object slot. Called when the walking timer reaches 1.
; OUT:  Success: ObjDir,X = the diagonal, then as Wizzrobe_Blue_BeginTeleporting. Failure: aligned to a square and
;       ObjTimer,X = Random,X OR $70 (another walking phase; no retry this frame).
; NOTE: RNG: Random,X AND 3 selects the diagonal. Only the target tile is tested, not the path.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_ChooseTeleportTarget:
    LDA Random, X
    AND #$03
    TAY

    ; Move the object to the target temporarily, saving its real position.
    LDA ObjX, X
    PHA
    CLC
    ADC Wizzrobe_Blue_TeleportTargetOffsetsX, Y
    STA ObjX, X
    LDA ObjY, X
    PHA
    CLC
    ADC Wizzrobe_Blue_TeleportTargetOffsetsY, Y
    STA ObjY, X
    TYA                         ; Save the random index.
    PHA

    ; Test the tile at the target, with the test offset of the diagonal direction.
    LDA Wizzrobe_Blue_TeleportDirs, Y
    TAY
    JSR Wizzrobe_GetCollidableTileForDir
    PLA                         ; Restore the random index.
    TAY
    PLA                         ; Restore the original Y.
    STA ObjY, X
    PLA                         ; Restore the original X.
    STA ObjX, X

    ; C set = not walkable: give up and start another walking phase.
    BCS Wizzrobe_Blue_AlignWithNearestSquareAndRandomizeTimer

    ; Face along the chosen diagonal.
    LDA Wizzrobe_Blue_TeleportDirs, Y
    STA ObjDir, X

; ----------------------------------------------------------------------------
; Starts a Blue Wizzrobe teleport of $20 pixels in the current ObjDir and aligns it to a square.
; IN:   X = object slot; ObjDir,X = diagonal (chosen target) or cardinal (walked into a block or water).
; OUT:  ObjRemDistance,X = $20; ObjTimer,X = 0 (teleporting); bit 6 of the turn counter flipped.
; NOTE: $20 frames; a diagonal covers $20 pixels on each axis. No shooting until it ends.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_BeginTeleporting:
    LDA #$20
    STA ObjRemDistance, X

    ; Flip bit 6 of the turn counter: the next turn toward Link uses the other axis.
    LDA Wizzrobe_ObjBlueTurnCounter, X
    EOR #$40
    STA Wizzrobe_ObjBlueTurnCounter, X
    LDA #$00                    ; Timer 0 = teleporting.
    BEQ :+

; Starts a walking phase: ObjTimer = Random,X OR $70, then aligns to the nearest square.
Wizzrobe_Blue_AlignWithNearestSquareAndRandomizeTimer:
    LDA Random, X
    ORA #$70                    ; $70-$7F or $F0-$FF frames.
:
    STA ObjTimer, X

; ----------------------------------------------------------------------------
; Snaps the object to the nearest 16-pixel square: X to a multiple of $10, Y to a multiple of $10 minus 3.
; IN:   X = object slot.
; OUT:  ObjX,X = (X + 8) AND $F0; ObjY,X = ((Y + 8) AND $F0) - 3; A = new Y.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_AlignWithNearestSquare:
    LDA ObjX, X
    CLC
    ADC #$08
    AND #$F0
    STA ObjX, X
    LDA ObjY, X
    CLC
    ADC #$08

; Second entry: A = a Y coordinate; stores (A AND $F0) - 3 in ObjY,X and returns it in A.
Wizzrobe_SetYAlignedToSquare:
    AND #$F0
    SEC
    SBC #$03
    STA ObjY, X
    RTS

; TABLE: X offset of the Wizzrobe tile test point; index = direction value - 1 ($00-$09); pixels.
Wizzrobe_TileTestOffsetsX:
    .BYTE $0F, $00, $00, $04, $08, $00, $00, $04    ; right, left, -, down, down-right, down-left, -, up
    .BYTE $08, $00                                  ; up-right, up-left

; TABLE: Y offset of the Wizzrobe tile test point; index = direction value - 1 ($00-$09); signed pixels.
;   Tile_GetUnderObj adds another $0B to Y.
Wizzrobe_TileTestOffsetsY:
    .BYTE $04, $04, $00, $08, $08, $08, $00, $F8    ; right, left, -, down, down-right, down-left, -, up
    .BYTE $00, $00                                  ; up-right, up-left

; ----------------------------------------------------------------------------
; Tests the tile in front of a Wizzrobe (or Ganon) for its facing direction ObjDir.
; IN:   X = object slot.
; OUT:  C clear = walkable, set = unwalkable; A and Wizzrobe_ObjLastTile,X = the tile.
; ----------------------------------------------------------------------------
Wizzrobe_GetCollidableTile:
    LDY ObjDir, X

; ----------------------------------------------------------------------------
; Tests the tile at the object's position plus the test offset of an 8-way direction.
; IN:   X = object slot; Y = direction value ($01-$0A).
; OUT:  C clear = walkable, set = unwalkable; A and Wizzrobe_ObjLastTile,X = the tile. Position unchanged.
; ----------------------------------------------------------------------------
Wizzrobe_GetCollidableTileForDir:
    DEY                         ; Direction - 1 indexes the offset tables.

    ; Shift the object to the test point, saving its real position.
    LDA ObjX, X
    PHA
    CLC
    ADC Wizzrobe_TileTestOffsetsX, Y
    STA ObjX, X
    LDA ObjY, X
    PHA
    CLC
    ADC Wizzrobe_TileTestOffsetsY, Y
    STA ObjY, X
    JSR Wizzrobe_GetBaseCollidableTile
    PLA                         ; Restore the original Y.
    STA ObjY, X
    PLA                         ; Restore the original X.
    STA ObjX, X
    RTS

; ----------------------------------------------------------------------------
; Reads the tile at (ObjX, ObjY + $0B) and compares it with the room's first unwalkable tile.
; IN:   X = object slot.
; OUT:  C clear = walkable (tile < RoomFirstUnwalkableTile), set = unwalkable; A and
;       Wizzrobe_ObjLastTile,X = the tile (also left in ObjCollidedTile,X).
; ----------------------------------------------------------------------------
Wizzrobe_GetBaseCollidableTile:
    JSR Tile_GetUnderObj
    LDA ObjCollidedTile, X
    CMP RoomFirstUnwalkableTile ; Sets C for the caller; STA below does not change it.
    STA Wizzrobe_ObjLastTile, X
    RTS

; ----------------------------------------------------------------------------
; Lets a Blue Wizzrobe shoot magic shot $58 when it faces Link along its row or column.
; IN:   X = object slot.
; NOTE: Tried only when not teleporting and FrameCounter AND $1F = 0 (once per 32 frames; also during
;       the 14-frame pause). Row test: same high nibble of Y. Column test: see
;       Wizzrobe_Blue_TryShooting_CheckColumn. The direction toward Link must equal ObjDir,X.
; ----------------------------------------------------------------------------
Wizzrobe_Blue_TryShooting:
    LDA ObjRemDistance, X
    BNE Wizzrobe_Blue_TryShooting_Return             ; Teleporting.
    LDA FrameCounter
    AND #$1F
    BNE Wizzrobe_Blue_TryShooting_Return

    ; Same 16-pixel row as Link (Y AND $F0 equal)? If not, try the column.
    LDA ObjY, X
    AND #$F0
    STA $00
    LDA ObjY
    AND #$F0
    CMP $00
    BNE Wizzrobe_Blue_TryShooting_CheckColumn

    ; Monster's X >= Link's X: left (2). Else right (1).
    LDA #$02
    LDY ObjX, X
    CPY ObjX
    BCS :+
    LSR
:
    ; Shoot only if already facing that way.
    CMP ObjDir, X
    BEQ Wizzrobe_ShootMagicShot58

; Shared exit of the shooting routines.
Wizzrobe_Blue_TryShooting_Return:
    RTS

; Column half of Wizzrobe_Blue_TryShooting: shoots if Link is in the monster's column and it faces him.
; NOTE: Link's X is compared unmasked with (monster X AND $F0), so Link must stand exactly on the
;       monster's 16-pixel column (X a multiple of $10), unlike the row test.
Wizzrobe_Blue_TryShooting_CheckColumn:
    LDA ObjX, X
    AND #$F0
    STA $00
    LDA ObjX
    CMP $00
    BNE Wizzrobe_Blue_TryShooting_Return

    ; Monster's Y >= Link's Y: up (8). Else down (4).
    LDA #$08
    LDY ObjY, X
    CPY ObjY
    BCS :+
    LSR
:
    ; Shoot only if already facing that way.
    CMP ObjDir, X
    BNE Wizzrobe_Blue_TryShooting_Return

; Shoots the Blue Wizzrobe's magic shot (object type $58).
Wizzrobe_ShootMagicShot58:
    LDA #$58

; ----------------------------------------------------------------------------
; Shoots a magic shot of the given object type from the monster, with the magic sound.
; IN:   A = shot object type ($58 Blue Wizzrobe, $59 Red Wizzrobe); X = object slot of the shooter.
; OUT:  As Shooter_ShootLimited: C set = shot created, Y = its slot. [00] = shot type. With the clock it
;       returns at once and C is left as it was.
; NOTE: Nothing happens with the clock. The sound (Tune0Request 4) plays even when Shooter_ShootLimited
;       refuses the shot (no free slot, or 4 counted shots already in flight).
; ----------------------------------------------------------------------------
Wizzrobe_ShootMagicShot:
    STA $00
    LDA InvClock
    BNE Wizzrobe_Blue_TryShooting_Return
    LDA #$04                    ; Magic shot sound.
    STA Tune0Request
    JMP Shooter_ShootLimited

; ----------------------------------------------------------------------------
; Updates a Red Wizzrobe (type $24): counts ObjState down and runs the handler for its top two bits.
; IN:   X = object slot.
; NOTE: ObjState,X is decremented first, then (state >> 6) selects: 3 ($FF-$C0) appear and fade in,
;       2 ($BF-$80) solid, 1 ($7F-$40) fade out, 0 ($3F-$00) hidden. A full cycle is 208 frames (group
;       1 is cut to 16 frames). With the clock the state is frozen and the monster is drawn and
;       collision-tested every frame, whatever the state.
; ----------------------------------------------------------------------------
Wizzrobe_Red_Update:
    LDA InvClock
    BEQ :+
    JMP Wizzrobe_DrawAndCheckCollisions
:
    INC Wizzrobe_ObjRedAnimCounter, X
    DEC ObjState, X

    ; State >> 6 = handler index 0-3.
    LDA ObjState, X
    LSR
    LSR
    LSR
    LSR
    LSR
    LSR
    JSR Util_TableJump

; TABLE: Red Wizzrobe handlers; index = ObjState >> 6.
Wizzrobe_Red_Update_JumpTable:
    .ADDR Wizzrobe_Red_State0Hidden   ; $00-$3F hidden
    .ADDR Wizzrobe_Red_State1FadeOut   ; $40-$7F fading out
    .ADDR Wizzrobe_Red_State2Solid   ; $80-$BF solid, shoots at $B0
    .ADDR Wizzrobe_Red_State3Appear   ; $C0-$FF choose a place ($FF), then fade in

; TABLE: X offset from Link where a Red Wizzrobe appears; index = Random,X AND $0F; signed pixels.
;   Index AND 3: 0 above Link, 1 below, 2 left, 3 right (matches Wizzrobe_Red_AppearDirs).
Wizzrobe_Red_AppearOffsetsX:
    .BYTE $00, $00, $E0, $20, $00, $00, $C0, $40    ; distance $20 (indexes 0-3), $40 (4-7)
    .BYTE $00, $00, $D0, $30, $00, $00, $B0, $50    ; distance $30 (8-$B), $50 ($C-$F)

; TABLE: Y offset from Link where a Red Wizzrobe appears; index = Random,X AND $0F; signed pixels.
Wizzrobe_Red_AppearOffsetsY:
    .BYTE $E0, $20, $00, $00, $C0, $40, $00, $00    ; distance $20 (indexes 0-3), $40 (4-7)
    .BYTE $D0, $30, $00, $00, $B0, $50, $00, $00    ; distance $30 (8-$B), $50 ($C-$F)

; TABLE: facing direction of an appearing Red Wizzrobe; index = Random,X AND 3; direction bits.
;   Each entry points back at Link from the side the same index selects in the offset tables.
Wizzrobe_Red_AppearDirs:
    .BYTE $04, $08, $01, $02    ; above Link: down; below: up; left of Link: right; right: left

; ----------------------------------------------------------------------------
; Red Wizzrobe state group 1 ($7F-$40), fading out: flickers for 16 frames.
; IN:   X = object slot.
; NOTE: State $7F jumps to $4F, so the group lasts 16 frames instead of 64. Also the fade-in body of
;       group 3. Drawn and collision-tested only on frames where the fade counter is even.
; ----------------------------------------------------------------------------
Wizzrobe_Red_State1FadeOut:
    LDA ObjState, X
    CMP #$7F
    BNE :+
    LDA #$4F                    ; Skip $30 states.
    STA ObjState, X
:
    INC Wizzrobe_ObjRedFadeCounter, X
    JMP Wizzrobe_DrawAndCheckCollisionsIntermittently

; ----------------------------------------------------------------------------
; Red Wizzrobe state group 3 ($FF-$C0): at $FF picks where to appear; $FE-$C0 fades in (63 frames).
; IN:   X = object slot.
; NOTE: RNG: one byte Random,X. Bits 0-1 pick the side of Link (and the facing, toward Link); bits 0-3
;       pick the offset: 2, 3, 4 or 5 squares. The spot is square-aligned (Y = multiple of $10 minus 3)
;       and accepted only if $5D <= Y < $C4 and its tile is walkable; otherwise the state goes back to
;       $00 and the choice is repeated next frame. X is not range-checked, only the tile is.
; ----------------------------------------------------------------------------
Wizzrobe_Red_State3Appear:
    LDY ObjState, X
    INY
    BNE Wizzrobe_Red_State1FadeOut     ; Not $FF: fade in, same as fading out.

    ; State $FF. Face toward Link from the side chosen by the low 2 random bits.
    LDA Random, X
    PHA                         ; Save the random value.
    AND #$03
    TAY
    LDA Wizzrobe_Red_AppearDirs, Y
    STA ObjDir, X
    PLA                         ; Restore the random value.

    ; X = (Link's X + offset) AND $F0. C is clear here (left so by the ASL in Util_TableJump).
    AND #$0F
    TAY
    LDA ObjX
    ADC Wizzrobe_Red_AppearOffsetsX, Y
    AND #$F0
    STA ObjX, X

    ; Y = ((Link's Y + 3 + offset) AND $F0) - 3. Rows are 16 pixels apart at Y = multiple of $10 minus
    ; 3. A Link on a row keeps that row; a Link 8 pixels below a row (halfway to the next) gets the
    ; row above him. Then the offset moves whole rows.
    LDA ObjY
    CLC
    ADC #$03
    ADC Wizzrobe_Red_AppearOffsetsY, Y
    JSR Wizzrobe_SetYAlignedToSquare

    ; A = new Y. Outside $5D-$C3 the spot is rejected without a tile test.
    CMP #$5D
    BCC @Retry
    CMP #$C4
    BCS @Retry
    JSR Wizzrobe_GetCollidableTile

    ; Walkable: keep the spot (state stays $FF; the next frame's decrement starts the fade-in).
    BCC Wizzrobe_Red_State0Hidden
@Retry:
    INC ObjState, X             ; $FF -> $00; the next frame's decrement gives $FF again.

; Red Wizzrobe state group 0 ($3F-$00), hidden: does nothing (not drawn, no collisions) for 64 frames.
Wizzrobe_Red_State0Hidden:
    RTS

; ----------------------------------------------------------------------------
; Red Wizzrobe state group 2 ($BF-$80), solid for 64 frames: shoots magic shot $59 at state $B0.
; IN:   X = object slot.
; NOTE: $B0 is the 16th solid frame. Drawn and collision-tested every frame.
; ----------------------------------------------------------------------------
Wizzrobe_Red_State2Solid:
    LDA ObjState, X
    CMP #$B0
    BNE Wizzrobe_DrawAndCheckCollisions
    LDA #$59
    JSR Wizzrobe_ShootMagicShot

; ----------------------------------------------------------------------------
; Tests a Wizzrobe against Link's weapons and against Link, then draws it.
; IN:   X = object slot.
; NOTE: ObjInvincibilityMask,X = $F6: only damage types 1 (sword, sword shot) and 8 (bomb) hurt.
;       Slots tested: $0E sword shot or magic shot, $10 and $11 bomb or fire, $0D sword. Fire and Link's
;       magic shot are masked. Slots $0F (boomerang, food) and $12 (arrow, rod) are not tested at all:
;       no boomerang stun, no arrow or rod damage. Weapon tests are skipped while
;       ObjInvincibilityTimer,X is nonzero; the test against Link is not.
; ----------------------------------------------------------------------------
Wizzrobe_DrawAndCheckCollisions:
    LDA #$F6
    STA ObjInvincibilityMask, X
    JSR Obj_GetMiddle
    LDA ObjInvincibilityTimer, X
    BNE @CheckLink
    LDY #$0E
    JSR SwordShot_CheckMonsterCollision
    LDY #$10
    JSR Bomb_CheckBombOrFireVsMonster
    LDY #$11
    JSR Bomb_CheckBombOrFireVsMonster
    LDY #$0D
    JSR Sword_CheckMonsterCollision
@CheckLink:
    ; Touching Link harms him even while the monster itself is invincible.
    JSR Link_CheckMonsterTouch

    ; Frame image 0 or 1 = bit 2 of the counter at $412 (Wizzrobe_ObjRedAnimCounter and
    ; Wizzrobe_ObjBlueTurnCounter are the same byte): Red changes image every 4 frames, Blue every 4
    ; walking steps (8 frames) and not at all while teleporting or paused.
    LDA Wizzrobe_ObjRedAnimCounter, X
    LSR
    LSR
    AND #$01
    PHA                         ; Save the frame image number.
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Facing up (also up-left and up-right): images 2 and 3, drawn mirrored.
    LDA ObjDir, X
    AND #$08
    BNE @FacingUp

    ; Otherwise [0F] = bit 1 of the direction: flip horizontally when facing left.
    LDA ObjDir, X
    LSR
    AND #$01
    STA $0F
    PLA                         ; Restore the frame image number.
    JMP Anim_DrawObjectNotMirrored
@FacingUp:
    PLA
    CLC
    ADC #$02
    JMP Anim_DrawObjectMirrored
