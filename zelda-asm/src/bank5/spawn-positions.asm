; ============================================================================
; FILE:     bank5/spawn-positions.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1870-2037
; SUBJECTS: (none)
; TOPIC:    Where the objects made on room entry start: the nine in-room spawn spots for monsters, the
;           four cellar Keese, the cave person; and the test that keeps a spawn off blocked tiles and
;           away from Link (also used for monsters entering from the screen edge).
; SEE ALSO: bank5/enter-room.asm (decodes the room's monster list into ObjType, then calls
;           AssignObjSpawnPositions); bank5/mode-inits-obj-lists.asm (SpawnPosList0-3, the spot lists);
;           bank7/object-init-table.asm (edge spawns call IsDistanceSafeToSpawn);
;           bank7/tile-collision.asm (GetCollidableTileStill); bank5/edge-spawn-cell.asm (edge cells).
; KEY FACTS:
;   - In-room spawn spots come from one of four 9-entry lists chosen by the direction Link faces on
;     entry (bit index of ObjDir: 0 right, 1 left, 2 down, 3 up). AssignObjSpawnPositions.
;   - A spot byte is row in the high nibble, column in the low nibble: X = column * $10,
;     Y = row * $10 + $0D. AssignObjSpawnPositions@LoopSpawnSpot.
;   - The walk through the list starts at SpawnCycle and SpawnCycle is stored back afterwards, so it
;     carries over from room to room; the same room can hand out different spots on each visit.
;     AssignObjSpawnPositions.
;   - Object slots $01-$09 all get a position, whatever the room's monster count is. A spot is skipped
;     when its tile blocks movement, or when it is within $21 pixels of Link on both axes.
;     IsSafeToSpawn, IsDistanceSafeToSpawn.
;   - No in-room positions are assigned when the list id is 0 or Zelda ($37), when the monster count is
;     0, or in an overworld room whose monsters enter from the screen edges (RoomAttrsF bit 3).
;     AssignObjSpawnPositions.
;   - A cellar (mode 9) always gets four blue Keese (type $1B) in slots $01-$04 at X $20, $60, $90,
;     $D0. All four get Y $9D: the Y table is indexed with Y, which is 0 here, instead of X.
;     AssignObjSpawnPositions@AssignSpecialPositions, CellarKeeseYs.
;   - In a cave (modes $0B, $0C) slots $01-$08 are emptied and slot $01 becomes the cave person:
;     type $6A + (cave index - $10), cave index = bits 7-2 of LevelBlockAttrsB.
;     AssignObjSpawnPositions@InCave.
;   - A monster entering from the screen edge is refused (and retried later) only by the distance
;     test; its tile is not tested. IsDistanceSafeToSpawn.
; CONTENTS: (generated; L = line in this file)
;   L0044 CellarKeeseXs            data  TABLE: X of the four cellar Keese; index = 0-3 (object slot - 1); value = X in pixels.
;   L0049 CellarKeeseYs            data  TABLE: Y of the cellar Keese, in pixels. Only entry 0 ($9D) is ever read: the reader indexes it
;   L0064 AssignObjSpawnPositions  code  Gives the objects made on room entry their start positions (spawn spots, cellar Keese, cave person).
;   L0189 IsSafeToSpawn            code  Tests whether an object may spawn where it is: walkable tile, and not close to Link.
;   L0208 IsDistanceSafeToSpawn    code  Tests whether an object is far enough from Link to spawn.
;   L0229 ReturnUnsafeToSpawn      code  Shared "not safe" exit of the two tests above: returns C set.
; ============================================================================

; TABLE: X of the four cellar Keese; index = 0-3 (object slot - 1); value = X in pixels.
CellarKeeseXs:
    .BYTE $20, $60, $90, $D0

; TABLE: Y of the cellar Keese, in pixels. Only entry 0 ($9D) is ever read: the reader indexes it
; with Y = 0 (see AssignObjSpawnPositions@AssignSpecialPositions), so $5D, $7D, $9D are unused.
CellarKeeseYs:
    .BYTE $9D, $5D, $7D, $9D

; ----------------------------------------------------------------------------
; Gives the objects made on room entry their start positions (spawn spots, cellar Keese, cave person).
; IN:   [02] = the room's monster list id (0 = none); ObjType+1.. already filled by the caller;
;       RoomObjCount = number of monsters; ObjDir = direction Link faces; SpawnCycle.
; OUT:  ObjX/ObjY of slots $01-$09 (in-room spawns); SpawnCycle advanced; in a cellar, types and
;       positions of slots $01-$04; in a cave, types of slots $01-$08.
; NOTE: The spot list is walked cyclically from SpawnCycle; an unsafe spot (IsSafeToSpawn) is skipped
;       without advancing the slot, so with fewer than nine safe spots several slots share a spot. The
;       loop only ends once slot $09 is placed (it would not end if no spot were safe).
; NOTE: Nothing here is random: the positions depend only on Link's entry direction, SpawnCycle and
;       which spots are safe.
; ----------------------------------------------------------------------------
AssignObjSpawnPositions:
    LDY RoomObjCount            ; Y = monster count. It is 0 whenever the cellar code below is reached.

    ; If the monster list id is 0, or it is Zelda ($37; she is placed by her own init), then
    ; skip the in-room spawn spots.
    LDA $02
    BEQ @AssignSpecialPositions
    CMP #$37                    ; Zelda
    BEQ @AssignSpecialPositions

    ; In the OW, skip the spawn spots if this room's monsters enter from the screen edges.
    LDA CurLevel
    BNE :+
    LDA RoomAttrsF
    AND #$08                    ; OW: monsters enter from the screen edges
    BNE @AssignSpecialPositions
:
    LDA RoomObjCount
    BEQ @AssignSpecialPositions ; No monsters: skip the spawn spots.

    ; Turn Link's direction bit into an index: 0 right, 1 left, 2 down, 3 up.
    LDA ObjDir
    LDY #$FF
:
    INY
    LSR
    BCC :-

    ; Put the address of the spawn spot list for that direction in [06:07].
    LDA SpawnPosListAddrsLo, Y
    STA $06
    LDA SpawnPosListAddrsHi, Y
    STA $07

    ; Assign coordinates to the 9 object slots $01-$09, continuing in the list where the
    ; previous room stopped.
    LDY SpawnCycle
    LDX #$01                    ; First slot to place.

@LoopSpawnSpot:
    LDA ($06), Y                ; Spot byte: high nibble = row, low nibble = column.
    PHA
    ASL                         ; X = column * $10.
    ASL
    ASL
    ASL
    STA ObjX, X
    PLA                         ; Y = row * $10 + $0D.
    AND #$F0
    ORA #$0D
    STA ObjY, X
    JSR IsSafeToSpawn
    BCS :+                      ; Unsafe: keep this slot and try the next spot for it.
    INX                         ; Safe: go on to the next slot.
:
    INY                         ; Next spot; the list has 9 entries and wraps around.
    CPY #$09
    BCC :+
    LDY #$00
:
    CPX #$0A                    ; Until slot $09 has been placed.
    BCC @LoopSpawnSpot
    STY SpawnCycle

@AssignSpecialPositions:
    ; In mode 9 (entering a cellar), make 4 blue Keese in slots $01-$04.
    LDA GameMode
    CMP #$09
    BNE @CheckCaves
    LDX #$03
:
    LDA #$1B                    ; Blue Keese
    STA ObjType+1, X
    LDA CellarKeeseXs, X
    STA ObjX+1, X
    ; The Y table is indexed by Y instead of X. In mode 9 the caller zeroes [02] and RoomObjCount,
    ; so this point is reached by the first branch above with Y = RoomObjCount = 0: every Keese
    ; gets CellarKeeseYs[0] = $9D.
    LDA CellarKeeseYs, Y
    STA ObjY+1, X
    DEX
    BPL :-
    RTS

@CheckCaves:
    ; A still holds GameMode. Modes $0B (cave) and $0C (shortcut stairs cave) go on; others return.
    CMP #$0B
    BEQ @InCave
    CMP #$0C
    BNE @Exit

@InCave:
    ; In a cave: empty slots $01-$08.
    LDX #$07
    LDA #$00
:
    STA ObjType+1, X
    DEX
    BPL :-

    ; LevelBlockAttrsB bits 7-2 hold the cave index ($10 and up for caves with a person).
    ; A = cave index - $10.
    LDY RoomId
    LDA LevelBlockAttrsB, Y
    AND #$FC                    ; Cave index << 2
    SEC
    SBC #$40
    LSR
    LSR

    ; Cave person object type = $6A + (cave index - $10), put in slot $01.
    CLC
    ADC #$6A
    STA ObjType+1

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Tests whether an object may spawn where it is: walkable tile, and not close to Link.
; IN:   X = object slot (ObjX, ObjY already set).
; OUT:  C set = not safe; C clear = safe. Y preserved; ObjCollidedTile,X = the tile found.
; NOTE: The tile is the one GetCollidableTileStill finds for the slot (no direction); it blocks when
;       it is >= RoomFirstUnwalkableTile. Falls through to the distance test.
; ----------------------------------------------------------------------------
IsSafeToSpawn:
    TYA                         ; Save Y (the caller's index into the spot list).
    PHA
    JSR GetCollidableTileStill
    PLA                         ; Restore Y.
    TAY
    LDA ObjCollidedTile, X
    CMP RoomFirstUnwalkableTile
    BCS ReturnUnsafeToSpawn     ; The tile blocks movement: not safe.

; ----------------------------------------------------------------------------
; Tests whether an object is far enough from Link to spawn.
; IN:   X = object slot.
; OUT:  C set = too close; C clear = safe.
; NOTE: Too close means |Link X - X| < $22 and |Link Y - Y| < $22 (a $43-pixel square around Link's
;       position). $22 or more on either axis is safe. Entry point for screen-edge spawns.
; NOTE: Each difference is an 8-bit subtraction made positive by Abs, so it wraps: a real distance
;       of $DF or more reads as $21 or less (Link at X $00, object at X $F0 = "distance" $10).
; ----------------------------------------------------------------------------
IsDistanceSafeToSpawn:
    ; Horizontal distance to Link.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$22
    BCS :+                      ; $22 or more: safe.

    ; Vertical distance to Link.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$22
    BCC ReturnUnsafeToSpawn     ; Both distances below $22: not safe.
:
    CLC                         ; Safe
    RTS

; Shared "not safe" exit of the two tests above: returns C set.
ReturnUnsafeToSpawn:
    SEC                         ; Not safe
    RTS
