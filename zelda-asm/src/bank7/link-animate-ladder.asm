; ============================================================================
; FILE:     bank7/link-animate-ladder.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 3163-3448
; SUBJECTS: Link
; TOPIC:    Link's end-of-frame step after moving: reset a finished grid step, lay the ladder down
;           at water, check warps (stairs, caves, level entrances), animate and draw Link and his
;           shield tile. Also the tables for the ladder and the shield, and bank trampolines.
; SEE ALSO: bank7/walker-move.asm (Walker_Move and MoveObject run just before; grid offset limits);
;           bank7/mode5-play-update.asm (UpdatePlayer: Link_HandleInput, Walker_Move, then this);
;           bank5/ladder.asm (CheckLadder: movement on the laid ladder and putting it away);
;           bank5/warps-stairs-caves.asm (CheckWarps); bank7/tile-collision.asm
;           (GetCollidingTileMoving, WalkableTiles); bank7/object-animation.asm (AnimateLinkBase,
;           AnimateLinkObjState); bank1/object-animation-draw.asm (DrawObjectWithAnim, Link's tiles).
; KEY FACTS:
;   - A grid offset that is a nonzero multiple of 8 (Link's limits +-8) is set to 0 here, after the
;     move, in modes 5 and 7 up; no other value is corrected. Link_EndMoveAndAnimate@TruncGridOffset
;   - Finishing a step in mode 5 clears UndergroundExitType: after coming out of a cave, cellar or
;     level Link must complete one step before warps work again.
;     Link_EndMoveAndAnimate@TruncGridOffset
;   - The ladder is laid only in mode 5 on a grid point, with InvLadder, outside a doorway, Link not
;     halted, no ladder out, a free monster slot, a water tile ahead in ObjDir, and ObjInputDir
;     (as Link_HandleInput left it) equal to ObjDir. Link_EndMoveAndAnimate@CheckLadder,
;     Link_EndMoveAndAnimate@SetUpLadder
;   - OW rooms where the ladder works: $17, $18, $19, $27, $4F, $5F. Every dungeon room allows it.
;     LadderRoomsOW
;   - Ladder tiles: dungeon $F4; OW $8D-$98 as returned by the tile lookup, which has already
;     turned the walkable $8D and $91 into $26. Link_EndMoveAndAnimate@CheckWaterOW
;   - The ladder is object type $5F, state 1, in the highest free slot of $0B..$01, one cell ahead
;     of Link. LinkToLadderOffsetsX, LinkToLadderOffsetsY, Link_EndMoveAndAnimate@SetUpLadder
;   - CheckWarps runs once a frame in mode 5 only, after the move. Link_EndMoveAndAnimate@CheckWarps
;   - While a whirlwind carries Link nothing here runs: no warps, no drawing. Link_EndMoveAndAnimate
;   - Link is drawn 2 pixels below ObjY where CurLevel = 0 (OW and its caves) and in cellars (mode
;     9); at ObjY in dungeon rooms. Link_EndMoveAndAnimate@Animate
;   - Link_EndMoveAndDraw holds ObjAnimCounter at 6, so Link's walk frame and his sword or item
;     state do not advance while it is used (person textboxes, ending). Link_EndMoveAndDraw
;   - Between rooms (leave-room init, room scroll, room entry) Link is drawn only in the OW.
;     Link_EndMoveAndAnimateBetweenRooms
;   - Shield sprite: without the magic shield only the facing-down walk tiles get a shield ($08 ->
;     $58, $0A -> $5A); with it $02 -> $80, $06 -> $54, $08 and $0A -> $60. Facing up and the
;     item-use frames get none. LinkHeadTiles, LinkHeadMagicShieldTiles
; CONTENTS: (generated; L = line in this file)
;   L0058 LinkToLadderOffsetsX                data  TABLE: X offset from Link to the ladder he lays down; index = 0 up, 1 down, 2 left, 3 right; pixels.
;   L0062 LinkToLadderOffsetsY                data  TABLE: Y offset from Link to the ladder he lays down; index = 0 up, 1 down, 2 left, 3 right; pixels.
;   L0068 LinkHeadTiles                       data  TABLE: tiles of Link's sprite that get the magic shield; index = 0-3; value = sprite tile number.
;   L0072 LinkHeadMagicShieldTiles            data  TABLE: magic shield replacements for LinkHeadTiles; same index; value = sprite tile number.
;   L0076 LadderRoomsOW                       data  TABLE: OW rooms in which Link can lay the ladder; index = 0-5 (searched); value = room id.
;   L0082 Link_EndMoveAndAnimate_Bank4        code  Link_EndMoveAndAnimate for callers in bank 4: returns with bank 4 switched in.
;   L0090 Link_EndMoveAndDraw_Bank1           code  Link_EndMoveAndDraw for callers in bank 1: returns with bank 1 switched in.
;   L0099 Link_EndMoveAndAnimate_Bank1        code  Link_EndMoveAndAnimate for callers in bank 1: returns with bank 1 switched in.
;   L0106 Link_EndMoveAndDraw_Bank4           code  Link_EndMoveAndDraw for callers in bank 4: returns with bank 4 switched in.
;   L0116 Link_EndMoveAndDraw                 code  Link_EndMoveAndAnimate with the animation frozen: Link is drawn in his current walk frame.
;   L0126 Link_EndMoveAndAnimateBetweenRooms  code  Link_EndMoveAndAnimate for the frames between rooms: OW only; in a dungeon Link is not drawn.
;   L0150 Link_EndMoveAndAnimate              code  Link's end-of-frame step: resets a finished grid step, lays the ladder, checks warps, draws him.
;   L0391 L1F36A_Exit                         code  Shared RTS of Link_EndMoveAndAnimate; also used by the sword shot code (UpdateSwordShotOrMagicShot).
; ============================================================================

; TABLE: X offset from Link to the ladder he lays down; index = 0 up, 1 down, 2 left, 3 right; pixels.
LinkToLadderOffsetsX:
    .BYTE $00, $00, $F0, $10    ; up 0, down 0, left -$10, right +$10

; TABLE: Y offset from Link to the ladder he lays down; index = 0 up, 1 down, 2 left, 3 right; pixels.
LinkToLadderOffsetsY:
    .BYTE $FB, $13, $03, $03    ; up -5, down +$13, left +3, right +3

; TABLE: tiles of Link's sprite that get the magic shield; index = 0-3; value = sprite tile number.
; $02 and $06 are the right halves of the two sideways walk frames, $08 and $0A the two halves of
; the facing-down frame (ObjAnimFrameHeap entries 0-2 of Link's animation).
LinkHeadTiles:
    .BYTE $02, $06, $08, $0A

; TABLE: magic shield replacements for LinkHeadTiles; same index; value = sprite tile number.
LinkHeadMagicShieldTiles:
    .BYTE $80, $54, $60, $60    ; $02 -> $80, $06 -> $54, $08 -> $60, $0A -> $60

; TABLE: OW rooms in which Link can lay the ladder; index = 0-5 (searched); value = room id.
LadderRoomsOW:
    .BYTE $17, $18, $19, $27, $4F, $5F

; ----------------------------------------------------------------------------
; Link_EndMoveAndAnimate for callers in bank 4: returns with bank 4 switched in.
; ----------------------------------------------------------------------------
Link_EndMoveAndAnimate_Bank4:
    JSR Link_EndMoveAndAnimate
    LDA #$04
    JMP SwitchBank

; ----------------------------------------------------------------------------
; Link_EndMoveAndDraw for callers in bank 1: returns with bank 1 switched in.
; ----------------------------------------------------------------------------
Link_EndMoveAndDraw_Bank1:
    JSR Link_EndMoveAndDraw
:
    LDA #$01
    JMP SwitchBank

; ----------------------------------------------------------------------------
; Link_EndMoveAndAnimate for callers in bank 1: returns with bank 1 switched in.
; ----------------------------------------------------------------------------
Link_EndMoveAndAnimate_Bank1:
    JSR Link_EndMoveAndAnimate
    JMP :-

; ----------------------------------------------------------------------------
; Link_EndMoveAndDraw for callers in bank 4: returns with bank 4 switched in.
; ----------------------------------------------------------------------------
Link_EndMoveAndDraw_Bank4:
    JSR Link_EndMoveAndDraw
    LDA #$04
    JMP SwitchBank

; ----------------------------------------------------------------------------
; Link_EndMoveAndAnimate with the animation frozen: Link is drawn in his current walk frame.
; NOTE: Sets ObjAnimCounter to 6 on every call, so the counter never reaches 0: the walk frame does
;       not switch and AnimateLinkObjState does not advance a sword or item state ($10, $11, ...).
; ----------------------------------------------------------------------------
Link_EndMoveAndDraw:
    LDA #$06
    STA ObjAnimCounter
    BNE Link_EndMoveAndAnimate  ; Always taken.

; ----------------------------------------------------------------------------
; Link_EndMoveAndAnimate for the frames between rooms: OW only; in a dungeon Link is not drawn.
; NOTE: Callers: the leave-room init (mode 6), the room scroll (mode 7 init and update) and the
;       between-rooms draw of the room entry code.
; ----------------------------------------------------------------------------
Link_EndMoveAndAnimateBetweenRooms:
    LDA CurLevel
    BNE L1F1FC_Exit             ; Dungeon: return.

; ----------------------------------------------------------------------------
; Link's end-of-frame step: resets a finished grid step, lays the ladder, checks warps, draws him.
; IN:   Link's slot 0 fields; GameMode.
; OUT:  Link's ObjGridOffset may be set to 0; a ladder object may be created (LadderSlot);
;       GameMode may change (CheckWarps); sprite records $12 and $13 written. X is not preserved
;       (0, 1 or 5); [0F] and Link's ObjCollidedTile may be overwritten by the ladder's tile probe.
; NOTE: Steps: (1) return at once while a whirlwind carries Link; (2) in modes 5 and 7 up, a grid
;       offset that is a nonzero multiple of 8 is set to 0, and in mode 5 UndergroundExitType is
;       cleared with it; (3) mode 5 on a grid point: lay the ladder if every condition holds;
;       (4) mode 5: CheckWarps; (5) animate and draw Link; (6) patch the shield tile.
; NOTE: Step 2 is the only correction of Link's grid offset after a move: it turns the limit
;       values +8 and -8 (where MoveObject stopped him) into 0. An offset that passed through 0
;       during the move is left as it is.
; NOTE: Ladder conditions, all required: mode 5; grid offset 0; a dungeon room, or OW room $17,
;       $18, $19, $27, $4F or $5F; not in a doorway; InvLadder; Link not halted; no ladder out;
;       the tile ahead in ObjDir is $F4 (dungeon) or $8D-$98 (OW); a free monster slot;
;       ObjInputDir (as Link_HandleInput left it) nonzero and equal to ObjDir. This is tested
;       every frame Link stands on a grid point.
; NOTE: Bank 5 is left switched in when the routine ran in mode 5.
; ----------------------------------------------------------------------------
Link_EndMoveAndAnimate:
    ; A whirlwind is carrying Link (WhirlwindTeleportingState <> 0): return. No warp check, and
    ; Link is not drawn.
    LDA WhirlwindTeleportingState
    BNE L1F1FC_Exit

    ; Modes 0-4 and 6: the grid offset is left alone (modes 4 and 6 walk Link until it reaches 0
    ; or a limit, see UpdateMode4and6EnterLeave); go straight to the drawing (@CheckWarps does
    ; nothing outside mode 5).
    TAX                         ; X = 0 (A is 0 here).
    LDA GameMode
    CMP #$06
    BEQ @CheckWarpsAndAnimate
    CMP #$05
    BCC @CheckWarpsAndAnimate

    ; Modes 5 and 7 up. Grid offset 0: Link is on a grid point, go see about the ladder.
    ; A nonzero multiple of 8 (low 3 bits clear): a finished step, go set it to 0.
    ; Anything else: Link is between grid points.
    LDA ObjGridOffset
    BEQ @CheckLadderRoom
    AND #$07
    BEQ @TruncGridOffset

@CheckWarpsAndAnimate:
    JMP @CheckWarps

@TruncGridOffset:
    ; Link has finished a step: he is on the next grid point, so the offset becomes 0.
    ; Outside mode 5 that is all.
    LDA #$00
    STA ObjGridOffset
    LDY GameMode
    CPY #$05
    BNE @CheckWarpsAndAnimate

    ; Mode 5: a whole step was walked, so clear UndergroundExitType (A = 0). While it is nonzero
    ; CheckWarps ignores the entrance Link has just come out of.
    STA UndergroundExitType

@CheckLadderRoom:
    ; On a grid point. The ladder is considered only in mode 5.
    LDA GameMode
    CMP #$05
    BNE @CheckWarpsAndAnimate

    ; Dungeon: every room qualifies. OW: only the 6 rooms of LadderRoomsOW.
    LDA CurLevel
    BNE @CheckLadder
    LDA RoomId
    LDY #$05

@LoopLadderRoom:
    CMP LadderRoomsOW, Y
    BEQ @CheckLadder
    DEY
    BPL @LoopLadderRoom
    BMI @CheckWarps

@CheckLadder:
    ; Not in a dungeon doorway, and Link owns the ladder.
    LDA DoorwayDir
    BNE @CheckWarps
    LDA InvLadder
    BEQ @CheckWarps

    ; Link is not halted ((ObjState AND $C0) = $40) and the ladder is not already laid down.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ @CheckWarps
    LDA LadderSlot
    BNE @CheckWarps

    ; Probe the tile ahead of Link in his facing direction. This overwrites [0F] and Link's
    ; ObjCollidedTile.
    LDX #$00
    LDA ObjDir
    STA $0F
    JSR GetCollidingTileMoving

    ; Dungeon: the tile must be $F4 (water).
    ; OW: the tile must be in $8D-$98. In the OW the lookup returns $26 for the walkable tiles
    ; of WalkableTiles, among them $8D and $91, so those two never match.
    LDY CurLevel
    BEQ @CheckWaterOW
    CMP #$F4
    BEQ @SetUpLadder
    BNE @CheckWarps

@CheckWaterOW:
    CMP #$8D
    BCC @CheckWarps
    CMP #$99
    BCS @CheckWarps

@SetUpLadder:
    ; The ladder needs a free monster slot (the highest free one of $0B..$01); Z set = none.
    ; ObjInputDir must be nonzero and equal to ObjDir.
    JSR FindEmptyMonsterSlot
    BEQ @CheckWarps
    LDA ObjInputDir
    BEQ @CheckWarps
    LDX EmptyMonsterSlot
    CMP ObjDir
    BNE @CheckWarps

    ; X = the ladder's object slot. Remember it in LadderSlot; the ladder faces as Link does.
    STX LadderSlot
    STA ObjDir, X

    ; Place the ladder one cell ahead of Link: Link's position plus the offsets for the direction.
    JSR GetOppositeDir          ; Y = reverse direction index: 0 up, 1 down, 2 left, 3 right.
    LDA ObjX
    CLC
    ADC LinkToLadderOffsetsX, Y
    STA ObjX, X
    LDA ObjY
    CLC
    ADC LinkToLadderOffsetsY, Y
    STA ObjY, X

    ; Object type $5F (ladder); no knockback, no invincibility timer.
    LDA #$5F
    STA ObjType, X
    JSR ResetShoveInfo          ; Returns A = 0.
    STA ObjInvincibilityTimer, X

    ; Ladder state 1: laid down, Link not on it yet (CheckLadder sets 2 once he is).
    LDA #$01
    STA ObjState, X

@CheckWarps:
    ; Mode 5 only: CheckWarps (bank 5) looks for stairs, a cave or a level entrance under Link.
    ; Link's ObjCollidedTile is saved around the call (CheckWarps does its own tile lookup).
    LDX #$00                    ; Link's object slot.
    LDA GameMode
    CMP #$05
    BNE @Animate
    LDA ObjCollidedTile
    PHA
    LDA #$05
    JSR SwitchBank
    JSR CheckWarps
    LDX #$00                    ; Link's object slot.
    PLA
    STA ObjCollidedTile

@Animate:
    ; AnimateLinkBase returns Y = animation frame (0, 1 sideways; 2 down; 3 up), [00]/[01] = the
    ; sprite position, [0F] = horizontal flip.
    JSR AnimateLinkBase
    LDA GameMode
    CMP #$09
    BEQ @ShiftDown2Pixels       ; In a cellar (mode 9): draw Link 2 pixels lower.
    LDA CurLevel                ; CurLevel = 0 (OW, caves): the same. Dungeon room: draw at ObjY.
    BNE @ChangeLookForState

@ShiftDown2Pixels:
    INC $01
    INC $01

@ChangeLookForState:
    ; While (ObjState AND $30) is $10 or $20 (sword or item in use) add 4 to the frame: frames
    ; 4-7 are the item-use poses. State $31 (AND $30 = $30) uses the walk frames again.
    LDA ObjState
    AND #$30
    CMP #$10
    BEQ @WieldingObject
    CMP #$20
    BNE @Draw

@WieldingObject:
    TYA
    CLC
    ADC #$04
    TAY

@Draw:
    TYA                         ; A = frame.
    LDY #$00                    ; Animation index 0 = Link.
    JSR DrawObjectWithAnim      ; Writes sprite records $12 (left half) and $13 (right half).

    ; Patch the shield into the sprite tiles just written. Sprites+72 is record $12; with X = 1
    ; Sprites+72,X is the left half's tile and Sprites+73,X its attributes; with X = 5 the same
    ; two bytes of record $13, the right half.
    LDA InvMagicShield
    BNE @HandleMagicShield

    ; No magic shield: only when Link faces down (ObjDir = 4), and only on the walk tiles.
    LDA ObjDir
    CMP #$04
    BNE L1F36A_Exit
    LDX #$01                    ; Left half.
    LDA Sprites+72, X
    CMP #$0B                    ; The facing-down walk frame puts tile $08 or $0A here.
    BCS L1F36A_Exit             ; Any higher tile (the item-use frame): no shield, return.
    PHA                         ; Save the original tile for @FixHFlip.
    CLC
    ADC #$50                    ; $08 -> $58, $0A -> $5A: the tiles used without the magic shield.
    JMP @ApplyShieldSprite

@HandleMagicShield:
    ; Magic shield: patch the half that carries it. Facing right (bit 0 of ObjDir) that is the
    ; right half, otherwise the left half.
    LDX #$01
    LDA ObjDir
    LSR
    BCC :+
    LDX #$05
:
    LDY #$04
    LDA Sprites+72, X
    PHA                         ; Save the original tile for @FixHFlip.

@LoopHeadTile:
    ; Look the tile up in LinkHeadTiles (index 3 down to 0). No match (facing up, item-use
    ; frames): leave the tile alone.
    DEY
    BMI @FixHFlip
    CMP LinkHeadTiles, Y
    BNE @LoopHeadTile
    LDA LinkHeadMagicShieldTiles, Y    ; The magic shield tile for it.

@ApplyShieldSprite:
    ; Write the shield tile.
    STA Sprites+72, X

@FixHFlip:
    PLA                         ; A = the original tile.
    CMP #$0A
    BNE L1F36A_Exit

    ; The original tile was $0A: the facing-down frame drawn flipped (its right half on the left).
    ; The replacement tile is drawn unflipped: clear the upper attribute nibble (flip and priority
    ; bits) of that sprite.
    LDA Sprites+73, X
    AND #$0F
    STA Sprites+73, X

; Shared RTS of Link_EndMoveAndAnimate; also used by the sword shot code (UpdateSwordShotOrMagicShot).
L1F36A_Exit:
    RTS
