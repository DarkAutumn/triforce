; ============================================================================
; FILE:     bank7/walker-collision-screen-edge.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2829-3161
; SUBJECTS: Link, Mode5Play, Obj, Unused, Walker
; TOPIC:    What a grid walker does on a grid point: the test of the tile ahead
;           (Walker_CheckTileCollision), a blocked monster's search for another direction, Link's
;           screen edge test (Link_CheckScreenEdge) and the start of a room change (Mode5Play_GoToNextMode).
; SEE ALSO: bank7/walker-move.asm (Walker_Move calls this before Obj_MoveByQSpeed; speed and grid offset);
;           bank7/tile-collision.asm (Tile_GetAheadOfObj: which tile is "ahead");
;           bank5/link-handle-input.asm (Link_ModifyDirOnGridLine: a turn back between grid points
;           can leave a grid offset whose sign is opposite to the heading);
;           bank5/border-input-mask.asm (Link_FilterInput near the screen border);
;           bank1/room-bounds.asm (Room_BoundDir); bank5/cave-cellar-exits.asm (Warp_CheckCaveEdgeExit calls
;           Link_CheckScreenEdge); bank5/doorway.asm (Doorway_CheckLink sets [0E] and DoorwayDir);
;           bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate sets the grid offset back to 0).
; KEY FACTS:
;   - Tiles and the screen edge are tested only on a frame that starts with ObjGridOffset = 0, once,
;     before the move. Walker_CheckTileCollision@CheckGridOffset, Walker_AcceptMovingDir
;   - A tile is walkable when its number < RoomFirstUnwalkableTile ($89 OW, $78 UW). Walker_TestTileAhead
;   - Screen edge: Link's ObjY must EQUAL $3D (input up) or $DD (down), or his ObjX EQUAL $00 (left)
;     or $F0 (right); the direction is the lowest set bit of ObjInputDir. There is no "at or
;     beyond" test. Link_ScreenEdgeBounds, Link_CheckScreenEdge
;   - The edge test also needs game mode 5, no ladder laid down and, outside a doorway, [0F] <> 0
;     on arrival (Link not using an item, not stopped by an earlier check). In the OW it runs even
;     when the tile ahead is unwalkable. Walker_AcceptMovingDir, Link_StopAtUnwalkableTile
;   - Missing the edge: if Link's grid offset is counting toward 0 at the edge grid point (its sign
;     opposite to his heading) and a 2-pixel frame takes him from 1 pixel short of the bound to 1
;     pixel past it, no frame starts on the bound. In the OW nothing else stops him, and his 8-bit
;     coordinate wraps to the far side of the same room. Link_CheckScreenEdge, Obj_MoveByQSpeed
;   - In a dungeon doorway (DoorwayDir <> 0) Link's tile test is skipped. Walker_CheckTileCollision
;   - Link walking into an unwalkable tile on a grid point gets [0F] = 0 and ButtonsPressed = 0.
;     Link_StopAtUnwalkableTile
;   - A blocked monster tries: a random direction perpendicular to ObjDir (bit 7 of Random,X), the
;     opposite of that, the reverse of ObjDir, then does not move this frame.
;     Walker_GetNextAltDir_JumpTable
;   - A room change sets GameMode 6 and clears Link's ObjState, knockback and invincibility timer.
;     Mode5Play_GoToNextMode
;   - Object attribute $10 ("turn around when blocked") is in no Obj_TypeAttributes entry for
;     types $01-$5E, and every other store to ObjAttr writes $81 or copies a table value: the
;     Reverse path is dead code. Reverse
; CONTENTS: (generated; L = line in this file)
;   L0070 Link_ScreenEdgeBounds                       data  TABLE: Link's screen edge coordinates; index = 0 up, 1 down, 2 left, 3 right; value = ObjY or ObjX.
;   L0089 Walker_CheckTileCollision                   code  On a grid point, tests the tile ahead: Link stops or gets the screen edge test; a monster turns.
;   L0130 Unused_WalkerReverseWhenBlocked             code  Dead code: a blocked monster with object attribute $10 would reverse ObjDir here.
;   L0138 Walker_TestTileAhead                        code  Tests the tile ahead in direction [0F]; loop head of the monsters' alternate direction search.
;   L0150 Walker_RejectDirAndTryNext                  code  Monster's direction rejected (tile or room bound): turn around (attribute $10) or try the next.
;   L0157 Walker_TryNextAltDir                        code  Takes the next alternate direction into [0F] and tests it, until the search ends.
;   L0166 Link_StopAtUnwalkableTile                   code  Link on a grid point facing an unwalkable tile: stop; in the OW still run the screen edge test.
;   L0186 Walker_CheckTileCollision_Return            code  Shared RTS of Walker_CheckTileCollision and Walker_AcceptMovingDir.
;   L0193 Obj_ResetMovingDir                          code  Cancels the move: sets the moving direction [0F] to 0.
;   L0199 Walker_AcceptMovingDir                      code  Accepts direction [0F]: a monster goes to the room bound test, Link to the screen edge test.
;   L0242 Link_CheckScreenEdge                        code  Starts a room change when Link stands exactly on the screen edge his input direction points at.
;   L0280 Mode5Play_GoToNextMode                      code  Ends play mode: adds 1 to GameMode (5 -> 6, leave room) and clears Link's movement state.
;   L0292 Link_CheckScreenEdge_ReturnX0               code  Shared exit of Walker_AcceptMovingDir, Link_CheckScreenEdge and Mode5Play_GoToNextMode: returns X = 0 (Link).
;   L0297 Walker_CheckRoomBound                       code  Monster with a walkable tile ahead: accepts [0F] as ObjDir unless it leads across a room bound.
;   L0315 Walker_GetNextAltDir                        code  Gives the next direction to try for a blocked walker and advances the search step [0E].
;   L0321 Walker_GetNextAltDir_JumpTable              data  TABLE: alternate direction search steps; index = step in [0E] (0-3); value = routine address.
;   L0334 Walker_AltDir_GetRandomObjPerpendicularDir  code  Search step 0: returns a random direction perpendicular to the object's facing.
;   L0355 Walker_AltDir_GetMovingOppositeDir          code  Search step 1: returns the opposite of the direction in [0F] (the one step 0 tried).
;   L0376 Obj_ReverseDirAndMovingDir                  code  Reverses an object's facing and makes it the moving direction; also search step 2.
;   L0386 Walker_AltDir_EndLoop                       code  Search step 3: ends the search; returns A = 0 and [0E] = 0.
;   L0402 Walker_FaceUnblockedDir                     code  Turns a monster on a grid point to face an open direction, if the search below finds one.
;   L0427 Walker_FaceUnblockedDir_Return              code  Shared RTS of Walker_FaceUnblockedDir; also the early exit of Link_EndMoveAndAnimate.
; ============================================================================

; TABLE: Link's screen edge coordinates; index = 0 up, 1 down, 2 left, 3 right; value = ObjY or ObjX.
; Up $3D, down $DD (ObjY); left $00, right $F0 (ObjX). Each is a grid point: X a multiple of 8, Y a
; multiple of 8 plus 5. Link_CheckScreenEdge compares for equality only (CMP / BNE): a Link who is past
; a bound, or who passes over it between two frames, does not change rooms.
Link_ScreenEdgeBounds:
    .BYTE $3D, $DD, $00, $F0

; ----------------------------------------------------------------------------
; On a grid point, tests the tile ahead: Link stops or gets the screen edge test; a monster turns.
; IN:   X = object slot; [0F] = moving direction; [0E] = $FF if a door blocks Link (Doorway_CheckLink).
; OUT:  [0F] = direction to move in, or 0; ObjDir,X = [0F] for a monster that found a way; [0E] =
;       alternate direction step (scratch). For Link: X = 0, and GameMode may have gone to 6.
; NOTE: Link, in this order: (1) in a doorway (DoorwayDir <> 0): no tile test, on to
;       Walker_AcceptMovingDir; (2) a door blocks him ([0E] = $FF): return; (3) ObjGridOffset <> 0: return,
;       so between grid points neither the tile nor the screen edge is tested; (4) [0F] = 0:
;       return; (5) tile ahead walkable: Walker_AcceptMovingDir; (6) unwalkable: [0F] = 0 and
;       ButtonsPressed = 0, then in the OW Walker_AcceptMovingDir all the same, in a dungeon return.
; NOTE: A monster on a grid point whose way is closed by a tile or a room bound tries, in order:
;       a random direction perpendicular to ObjDir, the opposite of the direction just tried, the
;       reverse of ObjDir, then gives up with [0F] = 0 (Walker_GetNextAltDir). Between grid points
;       it keeps [0F] untested.
; NOTE: Walkable = tile number < RoomFirstUnwalkableTile ($89 OW, $78 UW).
; ----------------------------------------------------------------------------
Walker_CheckTileCollision:
    ; Link in a dungeon doorway: skip the tile test, whatever [0E] and [0F] hold.
    CPX #$00
    BNE @CheckGridOffset
    LDA DoorwayDir
    BEQ :+
    JMP Walker_AcceptMovingDir
:
    ; Link, not in a doorway: return if Doorway_CheckLink reported a blocking door ([0E] = $FF).
    LDA $0E
    BMI Walker_CheckTileCollision_Return

@CheckGridOffset:
    ; Between grid points (grid offset <> 0): return. Nothing is tested and [0F] stands.
    LDA ObjGridOffset, X
    BNE Walker_CheckTileCollision_Return

    ; Grid offset = 0: the object is on a grid point.
    ; Reset [0E] (A = 0), from here on the alternate direction search step.
    STA $0E

    ; If there is a moving direction, go test the tile ahead.
    LDA $0F
    BNE Walker_TestTileAhead

    ; [0F] = 0. Link: return. (No screen edge test either: standing still, using the sword or an
    ; item, or stopped by a tile object, person or room bound, Link does not change rooms.)
    CPX #$00
    BEQ Walker_CheckTileCollision_Return

    ; A monster with [0F] = 0: it has no input direction, or Room_BoundDir zeroed [0F] in Walker_Move.
    ; Without attribute $10: take ObjInputDir,X as the moving direction and start the alternate
    ; direction search (the first step replaces it with a direction perpendicular to ObjDir).
    LDA ObjAttr, X
    AND #$10                    ; Object attribute $10: turn around when blocked (never set).
    BNE Unused_WalkerReverseWhenBlocked
    LDA ObjInputDir, X
    STA $0F
    JMP Walker_TryNextAltDir

; Dead code: a blocked monster with object attribute $10 would reverse ObjDir here.
Unused_WalkerReverseWhenBlocked:
    ; No entry of Obj_TypeAttributes for types $01-$5E has bit $10 (only the $FF of type $00).
    ; Every other store to ObjAttr writes $81 (types $5F up, cave persons, dropped items) or copies
    ; a slot's table value, so the two branches to here are never taken.
    JSR Obj_ReverseDirAndMovingDir
    JMP Walker_CheckRoomBound

; Tests the tile ahead in direction [0F]; loop head of the monsters' alternate direction search.
Walker_TestTileAhead:
    ; Link and monsters. Walkable (tile < RoomFirstUnwalkableTile): go to Walker_AcceptMovingDir.
    ; For Link this one test is all; for a monster it repeats for each alternate direction.
    JSR Tile_GetAheadOfObj
    CMP RoomFirstUnwalkableTile
    BCC Walker_AcceptMovingDir

    ; The tile ahead is unwalkable. Link: go stop.
    CPX #$00
    BEQ Link_StopAtUnwalkableTile

; Monster's direction rejected (tile or room bound): turn around (attribute $10) or try the next.
Walker_RejectDirAndTryNext:
    ; Attribute $10 is never set (see Reverse), so this always falls into Walker_TryNextAltDir.
    LDA ObjAttr, X
    AND #$10
    BNE Unused_WalkerReverseWhenBlocked

; Takes the next alternate direction into [0F] and tests it, until the search ends.
Walker_TryNextAltDir:
    ; [0E] = 0 after the last step: return with [0F] = 0, the monster does not move this frame.
    JSR Walker_GetNextAltDir
    STA $0F                     ; New moving direction, or 0 at the end of the search.
    LDA $0E
    BNE Walker_TestTileAhead
    RTS

; Link on a grid point facing an unwalkable tile: stop; in the OW still run the screen edge test.
Link_StopAtUnwalkableTile:
    ; OW only: Spawn_CheckTouchedArmosOrGravestone (bank 1) may spawn an Armos or a Ghini when the tile Link
    ; walked into is an armos statue or gravestone tile (ObjCollidedTile $BC-$C3).
    LDA CurLevel
    BNE @StopMoving
    LDA #$01
    JSR Mmc1_SwitchBank
    JSR Spawn_CheckTouchedArmosOrGravestone

@StopMoving:
    ; [0F] = 0 (no movement) and ButtonsPressed = 0 (A from Obj_ResetMovingDir): buttons newly pressed
    ; on controller 1 are dropped for the rest of this frame.
    ; Dungeon: return. OW: go on to Walker_AcceptMovingDir, so Link standing on a screen edge bound can
    ; leave the room even though the tile ahead of him is unwalkable.
    JSR Obj_ResetMovingDir
    STA ButtonsPressed
    LDA CurLevel
    BEQ Walker_AcceptMovingDir

; Shared RTS of Walker_CheckTileCollision and Walker_AcceptMovingDir.
Walker_CheckTileCollision_Return:
    RTS

; ----------------------------------------------------------------------------
; Cancels the move: sets the moving direction [0F] to 0.
; OUT:  A = 0; [0F] = 0; Z set.
; ----------------------------------------------------------------------------
Obj_ResetMovingDir:
    LDA #$00
    STA $0F
    RTS

; Accepts direction [0F]: a monster goes to the room bound test, Link to the screen edge test.
Walker_AcceptMovingDir:
    ; Monster: the tile ahead is walkable; test the room bound next.
    CPX #$00
    BNE Walker_CheckRoomBound

    ; Link gets here from a doorway (no tile test), from a walkable tile ahead, and in the OW
    ; from an unwalkable tile ahead (Link_StopAtUnwalkableTile).
    ;
    ; The screen edge is tested only in mode 5 (play) and while the ladder is not laid down.
    LDA GameMode
    CMP #$05
    BNE Link_CheckScreenEdge_ReturnX0
    LDA LadderSlot
    BNE Walker_CheckTileCollision_Return

    ; ... and only on a grid point. (Only the doorway path can arrive here with grid offset <> 0.)
    LDA ObjGridOffset
    BNE Link_CheckScreenEdge_ReturnX0

; ----------------------------------------------------------------------------
; Starts a room change when Link stands exactly on the screen edge his input direction points at.
; IN:   Link's ObjInputDir, ObjX, ObjY. Reached by falling in from Walker_AcceptMovingDir (mode 5, no
;       ladder, grid offset 0) and called directly by Warp_CheckCaveEdgeExit in caves, where those three
;       conditions are not tested: every frame in mode $B; in mode $C only on a grid point with
;       ObjY <> $9D (Warp_CheckSubroomExit).
; OUT:  X = 0. On a match: ObjDir = the single input direction, then Mode5Play_GoToNextMode
;       (IsUpdatingMode = 0 tells a caller that it happened). Otherwise nothing changes.
; NOTE: The direction is the lowest set bit of ObjInputDir (order right, left, down, up), not
;       [0F] and not ObjDir. The coordinate is ObjY for up and down, ObjX for left and right, and
;       it must EQUAL Link_ScreenEdgeBounds for that direction: $3D up, $DD down, $00 left, $F0
;       right. A coordinate beyond the bound does not count.
; NOTE: How the test is missed. It runs once a frame, before the move, and (from Walker_AcceptMovingDir)
;       only on a frame that starts with grid offset 0. Normally Link's offset counts away from 0
;       and Obj_MoveByQSpeed stops him at +-8, exactly on the next grid point; the offset is set to 0
;       there and the next frame starts on the bound. If instead the offset's sign is opposite to
;       his heading (after the turn back in Link_ModifyDirOnGridLine followed by turning again),
;       it counts toward 0 at the grid point, where Obj_MoveByQSpeed has no stop: a frame that moves 2
;       pixels can take Link from 1 pixel short of the bound (offset +-1) to 1 pixel past it
;       (offset -+1). No frame then starts on the bound, the equality never holds, and Link walks
;       on while the tiles allow: in the OW Room_BoundDir is not applied to him. His coordinate is 8
;       bits and wraps ($00 - 1 = $FF, $FF + 1 = $00), so he comes in from the far side of the same
;       room. In a dungeon the edges are reached only through doorways.
; ----------------------------------------------------------------------------
Link_CheckScreenEdge:
    LDX ObjY                    ; X = Link's Y for now (replaced below for a horizontal direction).

    ; No direction held: return.
    LDA ObjInputDir
    BEQ Link_CheckScreenEdge_ReturnX0

    ; Y = reverse index of the lowest set bit of ObjInputDir: 0 up, 1 down, 2 left, 3 right.
    ; Vertical direction: test ObjY. Horizontal: test ObjX.
    JSR Dir_GetOpposite
    LDA Dir_ReverseIndexToDir, Y    ; A = that single direction.
    AND #$0C
    BNE :+
    LDX ObjX
:
    STX $00                     ; [00] = the coordinate on the direction's axis.

    ; The coordinate must equal the bound for that direction; any other value returns. Nothing
    ; here asks whether Link is already beyond the bound.
    LDA $00
    CMP Link_ScreenEdgeBounds, Y
    BNE Link_CheckScreenEdge_ReturnX0

    ; Link stands on the bound: the room change starts. Face the single input direction.
    LDA Dir_ReverseIndexToDir, Y
    STA ObjDir

; ----------------------------------------------------------------------------
; Ends play mode: adds 1 to GameMode (5 -> 6, leave room) and clears Link's movement state.
; OUT:  X = 0; A = 0; GameSubmode = 0; IsUpdatingMode = 0 (the new mode's init runs next frame);
;       [0F] = 0 (no move this frame); Link's ObjState, ObjShoveDir, ObjShoveDistance and
;       ObjInvincibilityTimer = 0.
; NOTE: Entered from Link_CheckScreenEdge (falls in), from Doorway_CheckLink when Link passes a false wall
;       or a bombed wall (door types 2, 3, 4), from the whirlwind that carries Link off and from
;       the raft. In a cave Warp_CheckCaveEdgeExit then replaces the incremented mode with $A.
; NOTE: Leaving a room this way drops Link's remaining invincibility frames and any knockback,
;       and ends a sword or item state (ObjState = 0).
; ----------------------------------------------------------------------------
Mode5Play_GoToNextMode:
    INC GameMode
    LDA #$00
    STA GameSubmode
    STA IsUpdatingMode
    STA $0F
    STA ObjState
    STA ObjShoveDir
    STA ObjShoveDistance
    STA ObjInvincibilityTimer

; Shared exit of Walker_AcceptMovingDir, Link_CheckScreenEdge and Mode5Play_GoToNextMode: returns X = 0 (Link).
Link_CheckScreenEdge_ReturnX0:
    LDX #$00
    RTS

; Monster with a walkable tile ahead: accepts [0F] as ObjDir unless it leads across a room bound.
Walker_CheckRoomBound:
    ; Room_BoundDir returns A = [0F], zeroed when the monster is at or beyond a room bound and [0F]
    ; points across it. Zero: go try another direction. Else ObjDir,X = [0F]; return.
    ; (When the bound rejects the first, perpendicular direction, [0F] is 0 as the second step
    ; runs, so that step yields 0 and the search goes on to the reverse of ObjDir.)
    JSR Room_BoundDir
    BEQ Walker_RejectDirAndTryNext
    STA ObjDir, X
    RTS

; ----------------------------------------------------------------------------
; Gives the next direction to try for a blocked walker and advances the search step [0E].
; IN:   X = object slot; [0E] = step (0-3); [0F] = direction tried last (used by step 1).
; OUT:  A = direction for this step, 0 at the end; Z set when A = 0; [0E] = next step, or 0 after
;       step 3; step 2 rewrites ObjDir,X and [0F].
; NOTE: Steps: 0 a random direction perpendicular to ObjDir,X; 1 the opposite of [0F]; 2 the
;       reverse of ObjDir,X; 3 end of the search.
; ----------------------------------------------------------------------------
Walker_GetNextAltDir:
    LDA $0E
    INC $0E
    JSR Util_TableJump

; TABLE: alternate direction search steps; index = step in [0E] (0-3); value = routine address.
Walker_GetNextAltDir_JumpTable:
    .ADDR Walker_AltDir_GetRandomObjPerpendicularDir    ; 0: random perpendicular of ObjDir
    .ADDR Walker_AltDir_GetMovingOppositeDir            ; 1: opposite of the direction just tried
    .ADDR Obj_ReverseDirAndMovingDir                                 ; 2: reverse of ObjDir
    .ADDR Walker_AltDir_EndLoop                         ; 3: end, returns 0

; ----------------------------------------------------------------------------
; Search step 0: returns a random direction perpendicular to the object's facing.
; IN:   X = object slot.
; OUT:  A = direction: ObjDir,X vertical -> left (2) if bit 7 of Random,X is set, else right (1);
;       ObjDir,X horizontal or 0 -> up (8) if the bit is set, else down (4).
; NOTE: RNG use: bit 7 of Random,X (the byte indexed by the object slot), 50% each way.
; ----------------------------------------------------------------------------
Walker_AltDir_GetRandomObjPerpendicularDir:
    LDY #$00
    LDA Random, X
    ASL                         ; C = bit 7 of the random byte.
    BCS :+
    INY                         ; Bit clear: the second direction of the pair.
:
    LDA ObjDir, X
    AND #$0C
    BEQ :+
    INY                         ; Facing up or down: use the horizontal pair (indexes 2, 3).
    INY
:
    LDA Dir_ReverseIndexToDir, Y    ; 0 -> up (8), 1 -> down (4), 2 -> left (2), 3 -> right (1).
    RTS

; ----------------------------------------------------------------------------
; Search step 1: returns the opposite of the direction in [0F] (the one step 0 tried).
; OUT:  A = [0F] shifted right when it is left or up (2 -> 1, 8 -> 4), else shifted left (1 -> 2,
;       4 -> 8); 0 when [0F] = 0.
; ----------------------------------------------------------------------------
Walker_AltDir_GetMovingOppositeDir:
    LDA $0F
    PHA
    AND #$0A
    BEQ @IncreasingDir
    PLA
    LSR
    RTS

@IncreasingDir:
    PLA
    ASL
    RTS

; ----------------------------------------------------------------------------
; Reverses an object's facing and makes it the moving direction; also search step 2.
; IN:   X = object slot.
; OUT:  A = ObjDir,X = [0F] = the opposite of the old ObjDir,X; Y = reverse index of the old one.
; NOTE: As a search step it rewrites ObjDir,X before the new direction is tested; the monster
;       stays turned around even if that direction is closed too.
; ----------------------------------------------------------------------------
Obj_ReverseDirAndMovingDir:
    LDA ObjDir, X
    JSR Dir_GetOpposite
    STA ObjDir, X
    STA $0F
    RTS

; ----------------------------------------------------------------------------
; Search step 3: ends the search; returns A = 0 and [0E] = 0.
; ----------------------------------------------------------------------------
Walker_AltDir_EndLoop:
    LDA #$00
    STA $0E
    RTS

; ----------------------------------------------------------------------------
; Turns a monster on a grid point to face an open direction, if the search below finds one.
; IN:   X = object slot.
; OUT:  ObjDir,X = the direction found; [0F] = last direction tried, or 0; [0E] = search step.
; NOTE: Does nothing unless ObjGridOffset,X = 0. Tries the steps of Walker_GetNextAltDir in order
;       and takes the first direction whose tile ahead is walkable and that Room_BoundDir allows.
;       The search ends as soon as a step returns 0: when the room bound (not a tile) rejects the
;       step 0 direction, Room_BoundDir leaves [0F] = 0, step 1 returns 0, and ObjDir,X is unchanged.
;       Step 2 reverses ObjDir,X even if the reverse direction is then found closed.
; NOTE: Used by the Rope when it picks a new heading (Rope_Update).
; ----------------------------------------------------------------------------
Walker_FaceUnblockedDir:
    LDA ObjGridOffset, X
    BNE Walker_FaceUnblockedDir_Return

    ; On a grid point: [0E] = 0 (A) starts the search at step 0.
    STA $0E

@LoopSearchStep:
    JSR Walker_GetNextAltDir

    ; [0F] = the direction to test. A = 0 (Z still set from the step routine): give up.
    STA $0F
    BEQ Walker_FaceUnblockedDir_Return

    ; Closed by an unwalkable tile ahead or by the room bound: try the next step.
    JSR Tile_GetAheadOfObj
    CMP RoomFirstUnwalkableTile
    BCS @LoopSearchStep
    JSR Room_BoundDir
    BEQ @LoopSearchStep

    ; Open: face it.
    STA ObjDir, X

; Shared RTS of Walker_FaceUnblockedDir; also the early exit of Link_EndMoveAndAnimate.
Walker_FaceUnblockedDir_Return:
    RTS
