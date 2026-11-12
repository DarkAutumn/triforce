; ============================================================================
; FILE:     bank5/lay-out-doors.asm      BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4721-5036
; SUBJECTS: Door
; TOPIC:    Drawing the four doors of a dungeon room into the play area tile map by door type and
;           open/closed state (Door_LayOut), and the per-wall tables used for that and for the door
;           animation.
; SEE ALSO: bank5/room-layouts-uw.asm (Door_FaceTilesE/W/S/N: the faces and their layout);
;           bank5/fill-walls.asm (Door_FindTypeByDirBit; RoomLayout_FillWalls draws the walls first);
;           bank5/update-doors.asm (Door_Update opens and closes a door, then calls Door_LayOut);
;           bank5/door-flags-dark-room.asm (Door_ShiftOpenDoorwayMask, Door_AddRoomFlagsToOpenedDoors);
;           bank5/shutters-door-flags.asm (RoomFlags_SetDoorOpened); bank5/mode12-lay-out-room.asm (RoomLayout_LayOutRoom
;           calls Door_LayOut); bank5/mode7-scroll-init.asm (Door_LayOutWithPrevOpened;
;           Door_SetEnteringDoorwayOpened); bank5/boss-sound-door-face.asm
;           (Door_WriteFaceTileHorizontally reads Door_PassageTileIndexes); bank5/touch-door.asm
;           (what each door type does to Link).
; KEY FACTS:
;   - Direction index used by all door tables: 0 = right (E), 1 = left (W), 2 = down (S), 3 = up (N);
;     direction bit = 1 << index. The saved door flags of a room use the same bit order. Door_DirBits.
;   - Face drawn per door type: 0 open -> open doorway; 1 wall, 2 and 3 false wall -> nothing (the
;     plain wall stays); 4 bombable -> plain-wall tiles, or the hole when open; 5 and 6 key door ->
;     locked door, or open doorway when open; 7 shutter -> shutter, or open doorway when open.
;     "Open" means the side's bit is set in CurOpenedDoors. Door_LayOut.
;   - False walls and intact bombable walls are drawn with exactly the tiles of a plain wall.
;     Door_LayOut (Door_FaceTilesE face 4).
;   - Door_LayOut clears from CurOpenedDoors the bits of sides that are not true doors (types 0-3).
;   - Door_LayOut sets the saved door flag of the current room for every key door and bombable wall
;     that is in CurOpenedDoors; never for a shutter. RoomLayout_LayOutRoom first ORs the room's saved door
;     flags into CurOpenedDoors (Door_AddRoomFlagsToOpenedDoors), so a saved door is drawn open.
;   - The scroll code puts the side Link enters by into CurOpenedDoors before the new room is laid
;     out, so a key door or bombable wall on that side of the new room is drawn open and its flag is
;     set, however the other room's side was passed. Door_LayOut
;     (Door_SetEnteringDoorwayOpened).
;   - Door_LayOut also rebuilds OpenDoorwayMask: bit set for an open side (type 0) and for a type 4-7
;     side whose saved door flag is set. Nothing but the submenu map reads it, after recomputing it.
;   - Door positions in the play area tile map (column, row of the top-left tile): E (28, 9), W (1, 9),
;     S (14, 18), N (14, 1). E and W doors are 3 x 4 tiles, S and N doors 4 x 3.
;     Door_PlayAreaAddrsLo.
;   - Opening a key door or bombable wall also sets the opposite door's flag in the room behind it:
;     room id +1 (E), -1 (W), +$10 (S), -$10 (N). Door_NextRoomIdOffsets (used by Door_Update).
; CONTENTS: (generated; L = line in this file)
;   L0064 Door_DirBits                         data  TABLE: direction bit of a door; index = direction index (0 right/E, 1 left/W, 2 down/S, 3 up/N).
;   L0068 Door_FaceTilesAddrsLo                data  TABLE: low bytes of the door face tile tables; index = direction index (E, W, S, N).
;   L0075 Door_FaceTilesAddrsHi                data  TABLE: high bytes of the door face tile tables; index = direction index (E, W, S, N).
;   L0083 Door_PlayAreaAddrsLo                 data  TABLE: low bytes of the tile map address of a door's top-left tile; index = direction index.
;   L0090 Door_PlayAreaAddrsHi                 data  TABLE: high bytes of the tile map address of a door's top-left tile; index = direction index.
;   L0099 Door_NextTileOffsets                 data  TABLE: step of the tile map pointer after copying a door tile; index = rows left in the column.
;   L0104 Door_SecondHalfOffsets               data  TABLE: tile map offset from a door's first half to its second half; index = direction index.
;   L0108 Door_HalfColumnCounts                data  TABLE: columns in one half of a door face; index = direction index (3 for E and W, 2 for S and N).
;   L0112 Door_HalfRowCountsMinusOne           data  TABLE: rows in one half of a door face, minus 1; index = direction index (2 rows E, W; 3 S, N).
;   L0131 Door_LayOut                          code  Draws the four doors of the dungeon room RoomId into the play area tile map, open or closed.
;   L0135 Door_LayOut_LoopDoors                code  Door_LayOut loop over the four sides: X = direction index, 3 (up) down to 0 (right).
;   L0142 Door_LayOut_LoopHalves               code  Door_LayOut loop over the two halves of one door: X = direction index, [06] = 1 then 0.
;   L0319 Door_FetchFaceTilesAndPlayAreaAddrs  code  Points [02:03] at one door face's 12 tiles and [00:01] at the door's place in the tile map.
;   L0337 Door_Return                          code  Shared RTS of Door_LayOut, Door_FetchFaceTilesAndPlayAreaAddrs and Door_Update.
;   L0345 Door_PassageTileIndexes              data  TABLE: face tile indexes of a door's 2x2 passage block; index = base for the wall + 0..3.
;   L0350 Door_PassageTileIndexBases           data  TABLE: offset into Door_PassageTileIndexes of a wall's four entries; index = direction index.
;   L0356 Door_PassageVramAddrsHi              data  TABLE: high bytes of the name table address of a door's 2x2 passage block; index = direction index.
;   L0361 Door_PassageVramAddrsLo              data  TABLE: low bytes of the name table address of a door's 2x2 passage block; index = direction index.
;   L0365 Door_NextRoomIdOffsets               data  TABLE: room id offset to the room behind a door; index = direction index.
; ============================================================================

; TABLE: direction bit of a door; index = direction index (0 right/E, 1 left/W, 2 down/S, 3 up/N).
Door_DirBits:
    .BYTE $01, $02, $04, $08

; TABLE: low bytes of the door face tile tables; index = direction index (E, W, S, N).
Door_FaceTilesAddrsLo:
    .LOBYTES Door_FaceTilesE
    .LOBYTES Door_FaceTilesW
    .LOBYTES Door_FaceTilesS
    .LOBYTES Door_FaceTilesN

; TABLE: high bytes of the door face tile tables; index = direction index (E, W, S, N).
Door_FaceTilesAddrsHi:
    .HIBYTES Door_FaceTilesE
    .HIBYTES Door_FaceTilesW
    .HIBYTES Door_FaceTilesS
    .HIBYTES Door_FaceTilesN

; TABLE: low bytes of the tile map address of a door's top-left tile; index = direction index.
;   The tile map starts at $6530 and has 22 bytes per column.
Door_PlayAreaAddrsLo:
    .LOBYTES $67A1              ; E door: column 28, row 9
    .LOBYTES $654F              ; W door: column 1, row 9
    .LOBYTES $6676              ; S door: column 14, row 18
    .LOBYTES $6665              ; N door: column 14, row 1

; TABLE: high bytes of the tile map address of a door's top-left tile; index = direction index.
Door_PlayAreaAddrsHi:
    .HIBYTES $67A1
    .HIBYTES $654F
    .HIBYTES $6676
    .HIBYTES $6665

; TABLE: step of the tile map pointer after copying a door tile; index = rows left in the column.
;   Index 0 (the column's last tile): $14, which from row 2 of a 3-row column reaches row 0 of the
;   next column (22 - 2). A 2-row column (E and W doors) gets 1 more from the code (22 - 1 = $15).
Door_NextTileOffsets:
    .BYTE $14, $01, $01

; TABLE: tile map offset from a door's first half to its second half; index = direction index.
;   E and W: 2 (two rows down). S and N: $2C (two columns right).
Door_SecondHalfOffsets:
    .BYTE $02, $02, $2C, $2C

; TABLE: columns in one half of a door face; index = direction index (3 for E and W, 2 for S and N).
Door_HalfColumnCounts:
    .BYTE $03, $03, $02, $02

; TABLE: rows in one half of a door face, minus 1; index = direction index (2 rows E, W; 3 S, N).
Door_HalfRowCountsMinusOne:
    .BYTE $01, $01, $02, $02

; ----------------------------------------------------------------------------
; Draws the four doors of the dungeon room RoomId into the play area tile map, open or closed.
; IN:   RoomId; CurOpenedDoors = direction bits of the doors to draw open.
; OUT:  PlayAreaTiles: a door face on each side whose type is not a wall (1-3).
;       CurOpenedDoors: bits of sides of type 0-3 cleared.
;       Room flags of RoomId: door bit set for each key door (5, 6) and bombable wall (4) that is in
;       CurOpenedDoors.
;       OpenDoorwayMask (first byte) = direction bits of sides that are open (type 0) or whose saved
;       door flag is set (types 4-7).
;       Uses [00]-[06], [0B]; X = $FF on return.
; NOTE: Writes the tile map only; nothing is sent to the screen here. Called when a dungeon room is
;       laid out (RoomLayout_LayOutRoom), by Door_LayOutWithPrevOpened (with PrevOpenedDoors as the open doors: when a
;       scroll goes up, and in the death sequence) and at the end of every door command
;       (Door_Update). Each door is handled in two passes (halves), and the type-to-face decision,
;       with its RoomFlags_SetDoorOpened call, is made again in each pass.
; ----------------------------------------------------------------------------
Door_LayOut:
    LDX #$03

; Door_LayOut loop over the four sides: X = direction index, 3 (up) down to 0 (right).
Door_LayOut_LoopDoors:
    ; [06] = half of the door to draw: 1 = first half (top half of an E or W door, left half of an
    ; S or N door), then 0 = second half.
    LDA #$01
    STA $06

; Door_LayOut loop over the two halves of one door: X = direction index, [06] = 1 then 0.
Door_LayOut_LoopHalves:
    TXA
    PHA                         ; Keep the direction index on the stack until @NextLoopDoorHalf.
    STA $0B                     ; [0B] = direction index.
    LDA Door_DirBits, X
    STA $02                     ; [02] = direction bit of this side.
    JSR Door_FindTypeByDirBit

    ; Turn the door type into a "provisional face" (PF) for the closed state:
    ; types 5-7 keep their number, type 4 becomes 8, type 0 becomes 4, types 1-3 keep their number.
    CMP #$05
    BCS @HaveClosedFace           ; Types 5, 6 (key door) and 7 (shutter): PF = type.
    CMP #$04
    BNE @ClearFromOpenedMask    ; Types 0-3.
    LDA #$08                    ; Type 4 (bombable wall): PF = 8.
    BNE @HaveClosedFace           ; Always taken.

@UseHoleFace:
    ; An opened bombable wall: PF = 9, the hole.
    LDA #$09
    BNE @HaveFace             ; Always taken.

@ClearFromOpenedMask:
    ; Types 0-3 (open, wall, false walls) are not doors that open or close: remove this side from
    ; CurOpenedDoors.
    PHA
    LDA $02
    EOR #$FF
    AND CurOpenedDoors
    STA CurOpenedDoors
    PLA

    CMP #$01
    BCS @HaveClosedFace           ; Types 1-3: PF = type (below 4 = draw nothing).
    LDA #$04                    ; Type 0 (open): PF = 4.

@HaveClosedFace:
    ; A = PF for the closed state:
    ;
    ;   type  PF  meaning
    ;   0     4   open
    ;   1     1   wall
    ;   2     2   false wall
    ;   3     3   false wall
    ;   4     8   bombable wall
    ;   5     5   key door
    ;   6     6   key door
    ;   7     7   shutter
    PHA

    ; Is this side in CurOpenedDoors? (Only types 4-7 can be: the others were just cleared.)
    LDA CurOpenedDoors
    AND $02
    TAX
    PLA                         ; A = PF again.
    CPX $02
    BNE @HaveFace             ; Not open: keep the closed PF.

    ; The door is open. X = direction index (copied from the top of the stack), A = PF.
    TAY
    PLA
    PHA
    TAX
    TYA

    ; A shutter (PF 7) is drawn open but is not remembered. A key door or bombable wall gets its
    ; saved door flag set in the room flags of RoomId.
    CMP #$07
    BEQ @UseOpenFace
    PHA
    JSR RoomFlags_SetDoorOpened             ; X = direction index. Changes A, Y and [00:01].
    PLA

    CMP #$08
    BEQ @UseHoleFace                 ; Bombable wall: PF = 9 (hole).
@UseOpenFace:
    LDA #$04                    ; Key door or shutter: PF = 4 (open doorway).

@HaveFace:
    ; A = final PF:
    ;
    ;   type  closed  open  meaning
    ;   0     4       -     open
    ;   1-3   1-3     -     wall, false walls
    ;   4     8       9     bombable wall
    ;   5, 6  5, 6    4     key door
    ;   7     7       4     shutter
    ;
    ; In the first pass ([06] = 1) only, shift this side's bit into OpenDoorwayMask. The sides come
    ; in the order up, down, left, right, so after the fourth side the mask holds direction bits.
    LDX $06
    BEQ :+
    LDX $0B                     ; X = direction index.
    PHA
    JSR Door_FindTypeByDirBit   ; A = door type again ([02] still holds the direction bit).
    JSR Door_ShiftOpenDoorwayMask     ; Type 0: 1. Types 1-3: 0. Types 4-7: the saved door flag.
    PLA
:
    ; PF 1-3 (wall, false walls): draw nothing; the wall written by RoomLayout_FillWalls stays.
    CMP #$04
    BCC @NextLoopDoorHalf

    ; Face index = PF - 3, less 1 more if that is 3 or above:
    ;   PF 4 -> 1 open doorway, 5 and 6 -> 2 locked door, 7 -> 3 shutter,
    ;   8 -> 4 bombable wall (plain-wall tiles), 9 -> 5 bombed hole.
    SEC
    SBC #$03
    TAY
    CPY #$03
    BCC :+
    DEY
:
    PLA                         ; A = direction index (left on the stack).
    PHA
    JSR Door_FetchFaceTilesAndPlayAreaAddrs    ; [00:01] = tile map, [02:03] = face tiles, X = index.

    ; Second pass ([06] = 0): move both pointers to the second half of the door.
    LDA $06
    BNE :+
    LDA Door_SecondHalfOffsets, X    ; Tile map: 2 rows down (E, W) or 2 columns right (S, N).
    JSR Math_AddToInt16At0
    LDA #$06                    ; Face tiles: each half is 6 tiles.
    JSR Math_AddToInt16At2
:
    ; Copy the half: 3 columns of 2 tiles (E, W) or 2 columns of 3 tiles (S, N), each column top to
    ; bottom. Y stays 0; the pointers move.
    LDY #$00
    LDA Door_HalfColumnCounts, X
    STA $05                     ; [05] = columns left.

@LoopColumn:
    PLA                         ; X = direction index (left on the stack).
    PHA
    TAX
    LDA Door_HalfRowCountsMinusOne, X
    TAX                         ; X = rows left in this column after the current tile.

@LoopRowTile:
    LDA ($02), Y                ; Copy one tile.
    STA ($00), Y
    JSR Math_Add1ToInt16At2
    LDA Door_NextTileOffsets, X  ; 1 = next row; $14 after the column's last tile.
    JSR Math_AddToInt16At0

    ; After the last tile of a column of an E or W door (2 rows), $14 is one short of the top of the
    ; next column: add 1 more.
    CPX #$00
    BNE :+
    PLA                         ; A = direction index (left on the stack).
    PHA
    CMP #$02
    BCS :+                      ; S or N door: 3 rows, $14 was right.
    JSR Math_Add1ToInt16At0
:
    DEX
    BPL @LoopRowTile
    DEC $05
    BNE @LoopColumn

@NextLoopDoorHalf:
    PLA
    TAX                         ; X = direction index.
    DEC $06                     ; Next half; done after the second one.
    BMI :+
    JMP Door_LayOut_LoopHalves
:
    DEX                         ; Next side; return after index 0.
    BMI Door_Return
    JMP Door_LayOut_LoopDoors

; ----------------------------------------------------------------------------
; Points [02:03] at one door face's 12 tiles and [00:01] at the door's place in the tile map.
; IN:   A = direction index (0 right/E, 1 left/W, 2 down/S, 3 up/N); Y = face index (1-5).
; OUT:  X = direction index; [00:01] = play area tile map address of the door's top-left tile;
;       [02:03] = address of the face's tiles (table for the wall + $0C * (face - 1)); Y = 0.
; NOTE: Y = 0 on entry is not handled: the loop would add $0C 255 times.
; ----------------------------------------------------------------------------
Door_FetchFaceTilesAndPlayAreaAddrs:
    TAX
    LDA Door_FaceTilesAddrsLo, X
    STA $02
    LDA Door_FaceTilesAddrsHi, X
    STA $03
    LDA Door_PlayAreaAddrsLo, X
    STA $00
    LDA Door_PlayAreaAddrsHi, X
    STA $01
:
    DEY                         ; Add $0C (12 tiles per face) for each face before the wanted one.
    BEQ Door_Return
    LDA #$0C
    JSR Math_AddToInt16At2
    JMP :-

; Shared RTS of Door_LayOut, Door_FetchFaceTilesAndPlayAreaAddrs and Door_Update.
Door_Return:
    RTS

; TABLE: face tile indexes of a door's 2x2 passage block; index = base for the wall + 0..3.
;   The four entries of a wall are in screen order: top-left, top-right, bottom-left, bottom-right.
;   Door_Update sends these four tiles to the screen (Door_WriteFaceTileHorizontally). Faces 1-3 of
;   a wall differ from each other only in them, and so do faces 4 and 5. The entries of the S and N
;   walls overlap by two bytes.
Door_PassageTileIndexes:
    .BYTE $01, $03, $06, $08, $03, $05, $08, $0A    ; E: 1 3 / 6 8.  W: 3 5 / 8 10.
    .BYTE $03, $06, $04, $07, $05, $08              ; S: 3 6 / 4 7.  N: 4 7 / 5 8 (starts at the $04).

; TABLE: offset into Door_PassageTileIndexes of a wall's four entries; index = direction index.
Door_PassageTileIndexBases:
    .BYTE $00, $04, $08, $0A

; TABLE: high bytes of the name table address of a door's 2x2 passage block; index = direction index.
;   Addresses: E $225C, W $2242, S $234F, N $214F. The play area's first tile row is at $2100, so
;   these are play area (column, row) E (28, 10), W (2, 10), S (15, 18), N (15, 2).
Door_PassageVramAddrsHi:
    .BYTE $22, $22, $23, $21

; TABLE: low bytes of the name table address of a door's 2x2 passage block; index = direction index.
;   Bit 5 is clear in all four, so the second tile row is reached by setting it (Door_Update).
Door_PassageVramAddrsLo:
    .BYTE $5C, $42, $4F, $4F

; TABLE: room id offset to the room behind a door; index = direction index.
Door_NextRoomIdOffsets:
    .BYTE $01, $FF, $10, $F0    ; E +1, W -1, S +$10, N -$10
