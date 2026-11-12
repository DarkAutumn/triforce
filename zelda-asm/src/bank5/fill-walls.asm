; ============================================================================
; FILE:     bank5/fill-walls.asm         BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4537-4719
; SUBJECTS: (none)
; TOPIC:    The door type lookup for one side of a dungeon room (FindDoorTypeByDoorBit), and FillWalls,
;           which writes the wall outline shared by all dungeon rooms into the play area tile map.
; SEE ALSO: bank5/room-layouts-uw.asm (WallTileList); bank5/lay-out-doors.asm (LayOutDoors draws the
;           doors over the walls); bank5/mode12-lay-out-room.asm (LayOutRoom, the only caller of
;           FillWalls); bank5/touch-door.asm (TouchDoor_JumpTable: what each door type does when Link
;           walks into it); bank5/doorway.asm, bank5/shutters-door-flags.asm, bank5/submenu-map.asm,
;           bank5/mode3-mode6-init.asm (GetPassedDoorType), bank7/next-room-room-flags.asm
;           (CalculateNextRoom), bank7/bomb.asm (other callers of FindDoorTypeByDoorBit);
;           bank1/map-marker-curtain.asm (the 16-bit pointer helpers).
; KEY FACTS:
;   - A dungeon room has one door type (3 bits) per side: LevelBlockAttrsA bits 7-5 = N, bits 4-2 = S;
;     LevelBlockAttrsB bits 7-5 = W, bits 4-2 = E. FindDoorTypeByDoorBit.
;   - Door types: 0 open, 1 wall, 2 and 3 false wall, 4 bombable wall, 5 and 6 key door, 7 shutter.
;     FindDoorTypeByDoorBit returns 8 when its direction argument has none of bits 0-3 set.
;   - FindDoorTypeByDoorBit reads the level data of RoomId only. Whether a door is open now is in
;     CurOpenedDoors, and whether it was opened before is in the room flags; neither is looked at here.
;   - FindDoorTypeByDoorBit tests the sides in the order right, left, down, up and stops at the first
;     bit of [02] that matches. It also returns that side's reverse direction index in [03]
;     (3 right, 2 left, 1 down, 0 up).
;   - Every dungeon room gets the same wall outline: 78 bytes (WallTileList) describe the left wall and
;     the left half of the top wall; FillWalls mirrors them top to bottom, then rotates the left half
;     of the room by 180 degrees into the right half.
;   - Wall tile numbering that the mirroring relies on: below $DC a shape has four tiles T, T+1
;     (flipped top to bottom), T+2 (T rotated 180 degrees), T+3; $DC/$DD are the horizontal edge line
;     of the top/bottom wall, $DE/$DF the vertical edge line of the left/right wall; $E0 and above are
;     copied unchanged by the rotation. FillWalls, LoopWallTile.
;   - FillWalls runs before LayOutDoors and LayoutUWFloor, on a tile map already filled with brick
;     tile $F6; it leaves the floor area and the outer ring of tiles as $F6. FillWalls.
; CONTENTS: (generated; L = line in this file)
;   L0055 FindDoorTypeByDoorBit  code  Looks up the door type (0-7) of the side of the current dungeon room selected by a direction bit.
;   L0100 ReachedTopWallBottom   code  FillWalls, part: a $00 in WallTileList ends a 3-tile top-wall column; steps both pointers on.
;   L0110 DecBottomOffset        code  FillWalls, part: moves the bottom pointer up one row after a tile pair.
;   L0125 FillWalls              code  Writes the wall outline shared by all dungeon rooms into the play area tile map.
;   L0148 LoopWallTile           code  FillWalls pass 1 loop: writes one tile of WallTileList at the top pointer and at the bottom pointer.
;   L0178 MoveWallPtrs           code  FillWalls, part: adds A to the top pointer; X = 1 moves the bottom pointer up a row, else adds X.
;   L0186 NextLoopWallTile       code  FillWalls, part: advances the list pointer; after the last list byte, falls into pass 2 (rotation).
; ============================================================================

; ----------------------------------------------------------------------------
; Looks up the door type (0-7) of the side of the current dungeon room selected by a direction bit.
; IN:   [02] = direction bit (1 right/E, 2 left/W, 4 down/S, 8 up/N); RoomId.
; OUT:  A = door type: 0 open, 1 wall, 2 and 3 false wall, 4 bombable wall, 5 and 6 key door,
;       7 shutter; 8 if [02] has none of bits 0-3 set.
;       [01] = the direction bit that matched ($10 if none).
;       [03] = reverse direction index of that side: 3 right, 2 left, 1 down, 0 up ($FF if none).
;       [00] = scratch. Clobbers Y. Keeps X and [02].
; NOTE: The sides are tested in the order right, left, down, up; if [02] has several bits set, the
;       first match wins. The type comes from the level data alone (LevelBlockAttrsA bits 7-5 = N,
;       4-2 = S; LevelBlockAttrsB bits 7-5 = W, 4-2 = E): it does not say whether the door is open.
; ----------------------------------------------------------------------------
FindDoorTypeByDoorBit:
    LDA #$01                    ; [01] = direction bit under test, starting with 1 (right).
    STA $01
    LDA #$03                    ; [03] = reverse direction index, 3 (right) down to 0 (up).
    STA $03

@LoopDoorBit:
    LDY RoomId
    LDA LevelBlockAttrsA, Y     ; Attribute A holds the N and S door types.
    LDY $03
    CPY #$02
    BCC :+                      ; Index 3 or 2 (right, left): attribute B holds the W and E types.
    LDY RoomId
    LDA LevelBlockAttrsB, Y
:
    STA $00
    LDA $03
    AND #$01
    BNE :+                      ; Odd index (right, down): the type is in bits 4-2.
    LSR $00                     ; Even index (left, up): the type is in bits 7-5; 3 more shifts.
    LSR $00
    LSR $00
:
    LSR $00                     ; Shift the type down to bits 2-0 (bits 3 and up may hold the
    LSR $00                     ; other side's type; they are masked off at @Found).

    ; Is this the side the caller asked for?
    LDA $02
    BIT $01
    BNE @Found
    ASL $01                     ; Next direction bit.
    DEC $03
    BPL @LoopDoorBit

    ; [02] had none of bits 0-3 set: return 8, which is not a door type. (CalculateNextRoom's jump
    ; table has a ninth entry for it, "no next room".)
    LDA #$08
    RTS

@Found:
    LDA $00
    AND #$07
    RTS

; FillWalls, part: a $00 in WallTileList ends a 3-tile top-wall column; steps both pointers on.
ReachedTopWallBottom:
    ; The top pointer is at row 4 and the bottom pointer at row 17. One column is 22 ($16) bytes, so
    ; row 1 of the next column is $13 ahead of the top pointer and row 20 of the next column is $19
    ; ahead of the bottom pointer.
    LDA #$13
    STA $06                     ; Also refill the tile counter, so it cannot run out in a 3-tile column.
    LDX #$19
    BNE MoveWallPtrs            ; Always taken.

; FillWalls, part: moves the bottom pointer up one row after a tile pair.
DecBottomOffset:
    JSR Sub1FromInt16At4
    JMP NextLoopWallTile

; ----------------------------------------------------------------------------
; Writes the wall outline shared by all dungeon rooms into the play area tile map.
; IN:   PlayAreaTiles already filled with brick tile $F6 (LayOutRoom does that first).
; OUT:  PlayAreaTiles columns 1-3 and 28-30 (rows 1-20) and rows 1-3 and 18-20 (columns 4-27) hold
;       the four walls, without doors. Uses [00]-[06]; Y = 0 throughout.
; NOTE: Pass 1 walks WallTileList with a top pointer [02:03] going down from row 1 and a bottom
;       pointer [04:05] going up from row 20, writing each tile at both (the bottom copy flipped).
;       That covers columns 1-15. Pass 2 copies columns 0-15 to columns 31-16 rotated by 180 degrees:
;       the tile at (column c, row r) goes to (31 - c, 21 - r), changed to its rotated form.
;       The play area tile map is stored by column, 22 ($16) bytes per column, from $6530.
; ----------------------------------------------------------------------------
FillWalls:
    ; [00:01] = read pointer into WallTileList.
    LDA #<WallTileList
    STA $00
    LDA #>WallTileList
    STA $01

    ; [02:03] = top pointer = $6547 = PlayAreaTiles + $17: column 1, row 1.
    LDA #$47
    STA $02
    LDA #$65
    STA $03

    ; [04:05] = bottom pointer = $655A = PlayAreaTiles + $2A: column 1, row 20.
    LDA #$5A
    STA $04
    LDA #$65
    STA $05
    LDA #$0A                    ; [06] = tiles left in a left-wall column: 10 (rows 1-10 and 20-11).
    STA $06
    LDY #$00

; FillWalls pass 1 loop: writes one tile of WallTileList at the top pointer and at the bottom pointer.
LoopWallTile:
    LDA ($00), Y
    BEQ ReachedTopWallBottom    ; $00 = end of a top-wall column.
    STA ($02), Y
    STA ($04), Y

    ; The bottom copy must be flipped top to bottom. Tile $DE (vertical edge line) and tiles $E2 and
    ; above look the same flipped; every other tile's flipped form is the next tile number
    ; ($E0 -> $E1, $DC -> $DD, $C4 -> $C5).
    CMP #$DE
    BEQ :+
    CMP #$E2
    BCS :+
    ADC #$01                    ; Carry is clear here.
    STA ($04), Y
:
    LDA #$01                    ; Top pointer: down one row.
    LDX #$01                    ; X = 1: bottom pointer up one row (see MoveWallPtrs).
    DEC $06
    BNE MoveWallPtrs

    ; Tenth tile of a left-wall column: the top pointer is at row 10, the bottom pointer at row 11.
    ; Row 1 of the next column is $0D ahead of the top pointer; row 20 of the next column is $1F ahead
    ; of the bottom pointer.
    LDA #$0A
    STA $06
    LDA #$0D
    LDX #$1F

; FillWalls, part: adds A to the top pointer; X = 1 moves the bottom pointer up a row, else adds X.
MoveWallPtrs:
    JSR AddToInt16At2
    TXA
    DEX
    BEQ DecBottomOffset         ; X was 1: subtract 1 from the bottom pointer.
    JSR AddToInt16At4           ; Else add X ($19 or $1F) to move it to row 20 of the next column.

; FillWalls, part: advances the list pointer; after the last list byte, falls into pass 2 (rotation).
NextLoopWallTile:
    JSR Add1ToInt16At0          ; Returns the pointer's new low byte in A.
    CMP #<(WallTileList + $4E)  ; End of the 78-byte list? (Only the low byte is compared.)
    BNE LoopWallTile

    ; Pass 2. Columns 1-15 now hold the left wall and the left halves of the top and bottom walls.
    ; Copy the left half of the tile map to the right half, rotated by 180 degrees.
    ; Source [02:03] runs forward from $6530 (column 0, row 0);
    ; destination [04:05] runs backward from $67EF (column 31, row 21).
    LDA #$30
    STA $02
    LDA #$65
    STA $03
    LDA #$EF
    STA $04
    LDA #$67
    STA $05

@LoopRotate:
    LDA ($02), Y                ; Copy the tile, then replace the copy with its rotated form:
    STA ($04), Y
    CMP #$DD
    BEQ @SwapWithVertical       ; $DD (bottom wall's edge line) -> $DC (top wall's edge line).
    CMP #$E0
    BCS @NextLoopRotate         ; $E0 and above (corners, wall interior $F5, brick $F6): unchanged.
    CMP #$DC
    BCS :+                      ; $DC -> $DD, $DE -> $DF: add 1.
    ADC #$01                    ; Below $DC: add 2 in all (carry is clear; the store here is
    STA ($04), Y                ; overwritten below).
:
    CLC
    ADC #$01

@SetRotatedTile:
    STA ($04), Y

@NextLoopRotate:
    JSR Sub1FromInt16At4
    JSR Add1ToInt16At2          ; Returns the source pointer's new low byte in A.
    CMP #$90                    ; Stop when the source reaches $6690 (column 16, row 0): 16 columns
    BNE @LoopRotate             ; of 22 tiles copied.
    LDA $03
    CMP #$66
    BNE @LoopRotate
    RTS

@SwapWithVertical:
    LDA #$DC
    BNE @SetRotatedTile         ; Always taken.
