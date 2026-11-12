; ============================================================================
; FILE:     bank7/walker-move.asm        BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2600-2827
; SUBJECTS: Walker
; TOPIC:    The per-frame move of a grid walker (Link and walking monsters): choose the moving
;           direction, apply every blocker (knockback, item use, tile objects, persons, room
;           bounds, doorways, tiles, ladder), then move by the q-speed (MoveObject).
; SEE ALSO: bank7/walker-collision-screen-edge.asm (Walker_CheckTileCollision: tile test, monsters'
;           alternate directions, Link's screen edge test); bank1/qspeed-direction-math.asm
;           (AddQSpeedToPositionFraction / SubQSpeedFromPositionFraction: fraction, pixel carry, grid
;           offset limit); bank5/link-handle-input.asm (Link_HandleInput sets Link's ObjDir and
;           ObjInputDir just before); bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate sets the
;           grid offset back to 0 just after); bank7/mode5-play-update.asm (UpdatePlayer: the order);
;           bank7/shove.asm (Obj_Shove); bank1/room-bounds.asm (BoundByRoom); bank5/doorway.asm
;           (CheckDoorway); bank5/ladder.asm (CheckLadder); bank5/cave-cellar-exits.asm (CheckSubroom).
; KEY FACTS:
;   - Every test that can stop a walker (tile objects, persons, room bounds, doorways, tiles, the
;     screen edge) runs before the pixels are added, on the position the previous frame left;
;     Walker_Move tests nothing after the move. Walker_Move
;   - MoveObject adds the q-speed to the position fraction 4 times a frame and moves 1 pixel per
;     carry: pixels per frame = q-speed / 64, at most 4. Link's $60 = 1.5: he moves 1, 2, 1, 2 ...
;     pixels per frame. MoveObject
;   - One axis per call. If [0F] holds several direction bits the order is right, left, down, up.
;     MoveObject
;   - ObjGridOffset: +1 per pixel right or down, -1 per pixel left or up. A sub-step moves nothing
;     while the offset equals +limit or -limit (8 for Link, $10 for other slots): the object stops
;     exactly one cell from the grid point it left. There is no such stop at 0: an offset whose sign
;     is opposite to the heading counts through 0, and a 2-pixel frame can take it from +1 to -1 (or
;     -1 to +1) so that no frame starts with offset 0 at that grid point. MoveObject
;   - Between grid points (offset <> 0) Link moves in ObjDir whichever directions are held and stops
;     when none is held; on a grid point he and the monsters move in the lowest set bit of
;     ObjInputDir (right, left, down, up). Walker_Move@ChooseObjDirOrInputDir
;   - Link's moving direction is zeroed while (ObjState AND $F0) is $10 or $20 (sword or item in
;     use). CheckDoorway (a door lets him pass) and CheckLadder (on the laid ladder) run later and
;     can store a direction again, so he can still move there. Walker_Move@SetMovingDir
;   - A monster does not move while InvClock or its ObjStunTimer is nonzero. Walker_Move@CheckStunned
;   - A knocked-back object (ObjShoveDir <> 0) is moved by Obj_Shove only. Walker_Move
;   - Room bounds (BoundByRoom) hold every monster, but Link only in a dungeon room outside doorways
;     (not in the OW, caves or cellars): in the OW only tiles and the screen edge test limit Link.
;     Walker_Move@CheckBoundary
;   - Link cannot move up while his Y < $8E when slot 1 holds a Grumble Goriya ($36) or a dungeon
;     person ($4B-$52), and cannot move at all within $10 pixels (both axes) of a tile object in
;     state 1. Walker_Move
; CONTENTS: (generated; L = line in this file)
;   L0064 Walker_Move  code  Moves Link or a walking monster for one frame: picks the direction, applies every blocker, moves.
;   L0244 MoveObject   code  Moves an object along one axis by its q-speed: 4 sub-steps per call, 1 pixel per fraction carry.
; ============================================================================

; ----------------------------------------------------------------------------
; Moves Link or a walking monster for one frame: picks the direction, applies every blocker, moves.
; IN:   X = object slot ($00 = Link); ObjInputDir,X = wanted direction bits; ObjDir,X = facing.
; OUT:  ObjX,X or ObjY,X, ObjPosFrac,X and ObjGridOffset,X updated; ObjDir,X may change (a monster
;       turning, Link at a doorway or screen edge); X preserved, with the exception below.
;       Scratch: [0F] = moving direction, [0E] = doorway result, then the alternate direction step.
; NOTE: Every test (tile objects, persons, room bounds, doorways, tiles, screen edge) runs before
;       the move, on the position the previous frame left. Nothing is tested after the pixels are
;       added, and the tile test runs only when ObjGridOffset,X = 0 (Walker_CheckTileCollision).
; NOTE: For Link the switchable bank is left at 5 (routines of banks 1 and 5 are called), except
;       when he is knocked back (Obj_Shove path).
; NOTE: On the frame CheckLadder puts the ladder away it returns X = the ladder's former slot, not
;       0. MoveObject then runs on that freed slot (with Link's [0F] and the $10 limits): Link
;       does not move that frame, and Walker_Move returns with X <> 0.
; ----------------------------------------------------------------------------
Walker_Move:
    ; A knocked-back object is moved by Obj_Shove instead; nothing below runs.
    LDA ObjShoveDir, X
    BEQ @ChooseObjDirOrInputDir
    JMP Obj_Shove

@ChooseObjDirOrInputDir:
    ; Choose the moving direction (A, then [0F]):
    ; - Link between grid points (ObjGridOffset <> 0): 0 when no direction is held, else ObjDir,
    ;   whichever directions are held. He can stop between grid points but moves only along the
    ;   line he is on; Link_HandleInput has already reversed ObjDir where a turn back is allowed.
    ; - Link on a grid point, and monsters: the lowest set bit of ObjInputDir,X (order right, left,
    ;   down, up), or 0 when ObjInputDir,X = 0.
    ; - A monster does not move at all while InvClock or its ObjStunTimer is nonzero.
    CPX #$00
    BNE @CheckStunned
    LDA ObjGridOffset
    BEQ @CheckStunned
    LDA ObjInputDir
    BEQ @ZeroMovingDir
    LDA ObjDir
    BNE @SetMovingDir           ; Always taken unless ObjDir = 0.

@CheckStunned:
    CPX #$00
    BEQ @FilterInput
    LDA InvClock
    ORA ObjStunTimer, X
    BNE L1EFCF_Exit             ; Clock held or monster stunned: return (an RTS in bank7/flute.asm).

@FilterInput:
    LDA ObjInputDir, X
    BEQ @ZeroMovingDir
    JSR GetOppositeDir          ; Y = reverse index of the lowest set direction bit.
    LDA ReverseDirections, Y    ; A = that single direction.
    BNE @SetMovingDir

@ZeroMovingDir:
    LDA #$00

@SetMovingDir:
    AND #$0F                    ; Keep the four direction bits.
    STA $0F                     ; [0F] = moving direction. The blockers below may zero it.

    ; [0E] = 0. For Link in a dungeon CheckDoorway sets it below: the reverse index of the doorway
    ; he is in or at, or $FF when its door blocks him.
    LDA #$00
    STA $0E

    ; Link's moving direction is zeroed while (ObjState AND $F0) is $10 or $20: the sword or an
    ; item is in use (states $10/$11, $20/$21). State $31, the last step of a use, does not block.
    ; Two later calls can store a direction in [0F] again: CheckDoorway when Link's input points
    ; through a doorway whose door lets him pass, and CheckLadder while he is on the laid ladder.
    CPX #$00
    BNE @OnlyLink
    LDA ObjState, X
    AND #$F0
    CMP #$10
    BEQ :+
    CMP #$20
    BNE @OnlyLink
:
    STX $0F                     ; X = 0: [0F] = 0, no movement.

@OnlyLink:
    ; The checks from here to @CheckBoundary are Link's only.
    CPX #$00
    BNE @CheckBoundary

    ; A push block ($68), rock ($62), gravestone ($65) or armos tile object ($66) in state 1 within
    ; $10 pixels of Link on both axes zeroes [0F] (CheckTileObjectsBlocking, bank 1).
    LDA #$01
    JSR SwitchBank
    JSR CheckTileObjectsBlocking

    ; If object slot 1 holds the Grumble Goriya ($36) or a dungeon person ($4B-$52), Link cannot
    ; move up while his Y < $8E (CheckPersonBlocking zeroes [0F]).
    LDA ObjType+1
    CMP #$36                    ; Grumble Goriya
    BEQ @CheckBlocked
    CMP #$4B                    ; First dungeon person type.
    BCC @CheckSubroomOrDoorways
    CMP #$53                    ; One past the last dungeon person type ($52).
    BCS @CheckSubroomOrDoorways

@CheckBlocked:
    JSR CheckPersonBlocking

@CheckSubroomOrDoorways:
    ; In a doorway (dungeon only): skip the subroom and room bound checks.
    LDA DoorwayDir
    BNE @CheckDoorways

    ; Modes 9 (cellar), $B (cave) and $C (shortcut cave): CheckSubroom (bank 5) handles the cellar
    ; exits, the person blocking the upper half of a cave, the shortcut stairs and the cave exit.
    LDA GameMode
    CMP #$09
    BEQ @InSubroom
    CMP #$0B
    BEQ @InSubroom
    CMP #$0C
    BNE @SkipSubroom

@InSubroom:
    LDA #$05
    JSR SwitchBank
    JSR CheckSubroom

    ; Link is held inside the room bounds only in a dungeon room, outside doorways: BoundByRoom is
    ; skipped in a cellar (mode 9), in the OW and its caves (CurLevel = 0) and in a doorway. So in
    ; the OW only tiles and the screen edge test limit Link.
    LDA GameMode
    CMP #$09
    BEQ @CheckDoorways

@SkipSubroom:
    LDA CurLevel
    BEQ @CheckDoorways
    LDA DoorwayDir
    BNE @CheckDoorways

@CheckBoundary:
    ; All monsters, and Link in a dungeon room: zero [0F] when the object is at or beyond a room
    ; bound and [0F] points across it (RoomBoundLeft/Right/Up/Down). This also stops Link in front
    ; of a doorway; CheckDoorway below sets [0F] again when the door lets him through.
    JSR BoundByRoom

@CheckDoorways:
    ; Link in a dungeon, except in a cellar (mode 9): CheckDoorway (bank 5) finds the doorway he is
    ; in or at, touches its door, and sets DoorwayDir, [0E], and [0F]/ObjDir when he may pass.
    CPX #$00
    BNE @CheckTiles
    LDA CurLevel
    BEQ @CheckTiles
    LDA GameMode
    CMP #$09
    BEQ @CheckTiles
    LDA #$05
    JSR SwitchBank
    JSR CheckDoorway
    LDX #$00                    ; Restore Link's object slot.

@CheckTiles:
    ; All objects: on a grid point, test the tile ahead. For a monster this picks another
    ; direction when blocked; for Link it holds the screen edge test, which may start a room change
    ; (then [0F] = 0 and Link does not move this frame).
    JSR Walker_CheckTileCollision

    ; Link only: CheckLadder (bank 5) may change [0F] while the ladder is laid down. It returns
    ; X = 0 except on the frame it puts the ladder away, when X is still the ladder's slot and
    ; the MoveObject below moves that freed slot instead of Link.
    CPX #$00
    BNE MoveObject
    LDA #$05
    JSR SwitchBank
    JSR CheckLadder

; ----------------------------------------------------------------------------
; Moves an object along one axis by its q-speed: 4 sub-steps per call, 1 pixel per fraction carry.
; IN:   X = object slot; [0F] = direction bits (0 = do not move); ObjQSpeedFrac,X = q-speed.
; OUT:  ObjX,X or ObjY,X, ObjPosFrac,X and ObjGridOffset,X updated; PositiveGridCellSize /
;       NegativeGridCellSize = 8 / $F8 for Link, $10 / $F0 for any other slot (also when [0F] = 0).
; NOTE: Speed: each sub-step adds (right, down) or subtracts (left, up) the q-speed to the 8-bit
;       ObjPosFrac,X and moves 1 pixel on carry or borrow, so pixels per frame = q-speed / 64, at
;       most 4. Link's q-speed $60 gives 1.5: ObjPosFrac alternates $00/$80 and he moves 1, 2, 1,
;       2 ... pixels per frame.
; NOTE: Only one axis moves: the first set bit of [0F] in the order right, left, down wins; if
;       none of those three is set the object moves up.
; NOTE: Grid offset: each pixel moved right or down adds 1 to ObjGridOffset,X, each pixel left or
;       up subtracts 1. A sub-step moves nothing (the fraction still changes) while the offset
;       equals the positive or the negative limit, whichever way the object is heading. That stops
;       an object exactly one cell from the grid point it left, and keeps it there until its own
;       code sets the offset back to 0 (Link: Link_EndMoveAndAnimate, after the move, same frame).
; NOTE: There is no stop at offset 0. If the offset's sign is opposite to the heading (Link after
;       turning back between grid points: Link_ModifyDirOnGridLine), it counts toward 0 and on
;       through it. A frame that moves 2 pixels from offset +1 ends on -1 (or from -1 on +1), so no
;       frame starts with offset 0 at that grid point and the tests that need offset 0 (tile ahead,
;       screen edge) are not made there. The coordinate is 8 bits and wraps ($00 - 1 = $FF,
;       $FF + 1 = $00).
; ----------------------------------------------------------------------------
MoveObject:
    LDA #$08                    ; Link's grid cell is 8 pixels: limits +8 and -8 ($F8).
    LDY #$F8
    CPX #$00
    BEQ :+
    LDA #$10                    ; Every other slot: $10 pixels, limits +$10 and -$10 ($F0).
    LDY #$F0
:
    STA PositiveGridCellSize
    STY NegativeGridCellSize
    LDA $0F
    BEQ @Exit                   ; No direction: no movement; fraction and grid offset untouched.

    ; ObjQSpeedFrac is a quarter of the per-frame speed: apply it 4 times (3 calls, then fall in).
    JSR @ApplyQSpeedToPosition
    JSR @ApplyQSpeedToPosition
    JSR @ApplyQSpeedToPosition

@ApplyQSpeedToPosition:
    ; One sub-step. Shift the direction bits out in turn: 1 right, 2 left, 4 down, otherwise up.
    LDA $0F
    LSR
    BCS @Right
    LSR
    BCS @Left
    LSR
    BCS @Down

    ; Up. C = 0 (borrow) when a pixel is to be moved; SBC #$00 then subtracts 1 from Y.
    JSR SubQSpeedFromPositionFraction
    LDA ObjY, X
    SBC #$00
    STA ObjY, X

@Exit:
    RTS

@Down:
    ; C = 1 when a pixel is to be moved; ADC #$00 then adds 1 to Y.
    JSR AddQSpeedToPositionFraction
    LDA ObjY, X
    ADC #$00
    STA ObjY, X
    RTS

@Right:
    ; As @Down, on X.
    JSR AddQSpeedToPositionFraction
    LDA ObjX, X
    ADC #$00
    STA ObjX, X
    RTS

@Left:
    ; As up, on X.
    JSR SubQSpeedFromPositionFraction
    LDA ObjX, X
    SBC #$00
    STA ObjX, X
    RTS
