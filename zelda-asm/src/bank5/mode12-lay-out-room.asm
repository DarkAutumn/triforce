; ============================================================================
; FILE:     bank5/mode12-lay-out-room.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5537-5749
; SUBJECTS: (none)
; TOPIC:    Mode $12 (end of level, after a Triforce piece is taken): init and the five update submodes
;           (wait, flash, refill hearts, close the curtain, return to the overworld). LayOutRoom, which
;           builds the play area tile map of any room, and the routines that queue one tile column or
;           row of that map for transfer to the screen.
; SEE ALSO: bank5/next-room-calc.asm (EndGameMode12: leaves the level); bank1/map-marker-curtain.asm
;           (UpdateWorldCurtainEffect); bank7/status-bar-lift-item.asm (Link holding up the item);
;           bank5/layout-room-ow.asm (LayoutRoomOW, PatchColumnDirectoryForCellar);
;           bank5/uw-floor-push-block.asm (LayoutUWFloor); bank5/fill-walls.asm, bank5/lay-out-doors.asm
;           (dungeon walls and doors); bank5/ow-shortcut-map-square.asm (CheckShortcut and the callers
;           of CopyRowToTileBuf); bank5/mode7-scroll-update.asm (row and column copies while scrolling).
; KEY FACTS:
;   - Mode $12 sequence: ObjTimer[0] = $30 wait (UpdateMode12EndLevel_Sub0); $30 frames of palette flash,
;     4 frames white / 4 frames level palette (UpdateMode12EndLevel_Sub1); hearts refill to full
;     (UpdateMode12EndLevel_Sub2); $80 frame wait, then the curtain closes from both screen edges, one
;     tile column per side every 5 frames, 16 steps (UpdateMode12EndLevel_Sub3); $80 frame wait, then
;     EndGameMode12 (UpdateMode12EndLevel_Sub4).
;   - The end of a level always refills Link's hearts completely: World_IsFillingHearts = 2
;     (StartFillingHearts), and the sequence does not go on until the refill clears it.
;   - During all of mode $12 Link is drawn holding up item $1B (Triforce piece) and all other object
;     sprites are hidden (InitMode12, UpdateMode12EndLevel_Full).
;   - LayOutRoom: overworld (CurLevel = 0) = LayoutRoomOW then CheckShortcut; dungeon = fill the map with
;     tile $F6, then walls, doors, floor. It only writes PlayAreaTiles; the screen is updated by the
;     row/column copies.
;   - The play area tile map is stored by column: $16 (22) tiles per column, $20 (32) columns, from $6530
;     (CopyColumnToTileBuf, CopyRowToTileBuf).
;   - CopyColumnToTileBuf appends a 22-tile vertical record at PPU $2100 + column to DynTileBuf;
;     CopyRowToTileBuf overwrites DynTileBuf from its start with one 32-tile row record at
;     $2100 + CurRow * $20.
; CONTENTS: (generated; L = line in this file)
;   L0056 InitMode12                           code  Init of mode $12 (end of level): starts the song, blanks the tile map, makes Link lift the Triforce.
;   L0078 UpdateMode12EndLevel_Full            code  Update of mode $12: hides object sprites, draws Link lifting the Triforce, runs the submode.
;   L0085 UpdateMode12EndLevel_Full_JumpTable  data  TABLE: mode $12 update submode handlers; index = GameSubmode.
;   L0093 UpdateMode12EndLevel_Sub0            code  Mode $12 submode 0: waits for ObjTimer[0] ($30 from InitMode12), then arms $30 frames of flashing.
;   L0101 UpdateMode12EndLevel_Sub1            code  Mode $12 submode 1: flashes the background palette for $30 frames, then starts the heart refill.
;   L0114 L16887_Exit                          code  Shared return of mode $12 submodes 0 and 1.
;   L0118 StartFillingHearts                   code  End of mode $12 submode 1: starts refilling Link's hearts and advances the submode.
;   L0124 L1688C_IncSubmode                    code  Advances GameSubmode and returns (shared by mode $12 submodes 0 and 1).
;   L0129 UpdateMode12EndLevel_Sub2            code  Mode $12 submode 2: refills hearts each frame; when full, arms a $80 frame wait and advances.
;   L0138 UpdateMode12EndLevel_Sub3            code  Mode $12 submode 3: after the wait, closes the curtain; once closed, arms a $80 frame wait.
;   L0156 UpdateMode12EndLevel_Sub4            code  Mode $12 submode 4: after the wait, hides all sprites and leaves the level (EndGameMode12).
;   L0178 LayOutRoom                           code  Builds the play area tile map of the room RoomId (overworld or dungeon); nothing is drawn yet.
;   L0201 CopyColumnToTileBuf                  code  Queues one tile column (22 tiles) of the play area tile map for transfer to the screen.
;   L0249 CopyRowToTileBuf                     code  Queues one tile row (32 tiles) of the play area tile map for transfer to the screen.
; ============================================================================

; ----------------------------------------------------------------------------
; Init of mode $12 (end of level): starts the song, blanks the tile map, makes Link lift the Triforce.
; OUT:  SongRequest = $04; ObjX[$0C] = $20 and ObjX[$0D] = 1 (curtain columns); ObjTimer[0] = $30;
;       PlayAreaTiles all $24 (blank; the screen itself is not redrawn yet); IsUpdatingMode = 1;
;       ItemTypeToLift = $1B.
; ----------------------------------------------------------------------------
InitMode12:
    LDA #$04                    ; Request the end-of-level song.
    STA SongRequest
    LDA #$20                    ; Curtain column that moves left (UpdateWorldCurtainEffect).
    STA ObjX+12
    LDA #$01                    ; Curtain column that moves right.
    STA ObjX+13
    LDA #$30                    ; Submode 0 waits until this timer runs out.
    STA ObjTimer
    LDA #$24                    ; Fill the tile map with blank tiles: the curtain copies its columns.
    STA $0A
    JSR FillTileMap
    INC IsUpdatingMode
    JSR HideObjectSprites
    LDA #$1B                    ; Item id $1B: Triforce piece.
    STA ItemTypeToLift
    JMP SetUpAndDrawLinkLiftingItem

; ----------------------------------------------------------------------------
; Update of mode $12: hides object sprites, draws Link lifting the Triforce, runs the submode.
; IN:   GameSubmode 0-4.
; ----------------------------------------------------------------------------
UpdateMode12EndLevel_Full:
    JSR HideObjectSprites
    JSR DrawLinkLiftingItem
    LDA GameSubmode
    JSR TableJump

; TABLE: mode $12 update submode handlers; index = GameSubmode.
UpdateMode12EndLevel_Full_JumpTable:
    .ADDR UpdateMode12EndLevel_Sub0
    .ADDR UpdateMode12EndLevel_Sub1
    .ADDR UpdateMode12EndLevel_Sub2
    .ADDR UpdateMode12EndLevel_Sub3
    .ADDR UpdateMode12EndLevel_Sub4

; Mode $12 submode 0: waits for ObjTimer[0] ($30 from InitMode12), then arms $30 frames of flashing.
UpdateMode12EndLevel_Sub0:
    LDA ObjTimer
    BNE L16887_Exit             ; Wait until the timer runs out.
    LDA #$30                    ; Submode 1 flashes while this timer runs.
    STA ObjTimer
    BNE L1688C_IncSubmode       ; Always taken.

; Mode $12 submode 1: flashes the background palette for $30 frames, then starts the heart refill.
UpdateMode12EndLevel_Sub1:
    ; Each frame select a palette transfer: timer low 3 bits 4-7 = white, 0-3 = the level's palette.
    LDY #$18                    ; Transfer buffer selector $18: LevelInfo_PalettesTransferBuf.
    LDA ObjTimer
    BEQ StartFillingHearts      ; Timer ran out (no transfer is selected on this frame).
    AND #$07
    CMP #$04
    BCC :+
    LDY #$78                    ; Selector $78: WhitePaletteBottomHalfTransferBuf.
:
    STY TileBufSelector

; Shared return of mode $12 submodes 0 and 1.
L16887_Exit:
    RTS

; End of mode $12 submode 1: starts refilling Link's hearts and advances the submode.
StartFillingHearts:
    ; Any nonzero value starts the refill; the end of a level uses 2.
    LDA #$02
    STA World_IsFillingHearts

; Advances GameSubmode and returns (shared by mode $12 submodes 0 and 1).
L1688C_IncSubmode:
    INC GameSubmode
    RTS

; Mode $12 submode 2: refills hearts each frame; when full, arms a $80 frame wait and advances.
UpdateMode12EndLevel_Sub2:
    JSR UpdateHeartsAndRupees
    LDA World_IsFillingHearts
    BEQ :+                      ; Refill finished: go set the timer and advance (in submode 3's code).
    RTS

; Mode $12 submode 3: after the wait, closes the curtain; once closed, arms a $80 frame wait.
; NOTE: UpdateWorldCurtainEffect copies two (blank) tile columns to the screen every 5 frames:
;       ObjX[$0C] goes $20 down to $10 and ObjX[$0D] goes 1 up to $11, so 16 steps cover all 32 columns.
UpdateMode12EndLevel_Sub3:
    LDA ObjTimer
    BNE :++                     ; Still waiting: return.
    JSR UpdateWorldCurtainEffect
    LDA ObjX+12
    CMP #$11
    BCS :++                     ; The left-moving column has not passed the middle ($11 or more): return.
:
    ; Also the tail of submode 2.
    LDA #$80                    ; The next submode starts with a wait of this many frames.
    STA ObjTimer
    INC GameSubmode
:
    RTS                         ; Also the return of submode 4.

; Mode $12 submode 4: after the wait, hides all sprites and leaves the level (EndGameMode12).
; NOTE: EndGameMode12 sets CurLevel = 0, GameMode = 2 (load level) and UndergroundExitType = 2, so Link
;       comes out of the level's entrance on the overworld.
UpdateMode12EndLevel_Sub4:
    LDA ObjTimer
    BNE :-                      ; Still waiting: return.
    JSR HideAllSprites
    LDA CurPpuControl
    AND #$FB                    ; Clear bit 2: PPU address increment of 1.
    STA CurPpuControl
    STA PpuControl
    JMP EndGameMode12
:
    ; Overworld branch of LayOutRoom (reached from its BEQ below).
    JSR LayoutRoomOW
    JMP CheckShortcut           ; Draw the shortcut stairs if this room shows them.

; ----------------------------------------------------------------------------
; Builds the play area tile map of the room RoomId (overworld or dungeon); nothing is drawn yet.
; IN:   RoomId; CurLevel (0 = overworld); for a dungeon, the door state used by LayOutDoors.
; OUT:  PlayAreaTiles written; ColumnDirectoryOW entry 0 patched for the current world. Overworld:
;       RoomTileObjType/X/Y set when the room has a tile object, shortcut stairs added.
; NOTE: the overworld part is the unnamed block directly above this label. Callers copy the map to the
;       screen afterwards, by rows (room entry) or by rows/columns (scrolling, curtain).
; ----------------------------------------------------------------------------
LayOutRoom:
    JSR PatchColumnDirectoryForCellar
    LDA CurLevel
    BEQ :-                      ; Overworld: go to the block above.

    ; Dungeon: fill the whole map with tile $F6, the brick seen at the margins; then walls, doors
    ; and floor go on top.
    LDA #$F6
    STA $0A
    JSR FillTileMap
    JSR AddDoorFlagsToCurOpenedDoors
    JSR FillWalls
    JSR LayOutDoors
    JMP LayoutUWFloor

; ----------------------------------------------------------------------------
; Queues one tile column (22 tiles) of the play area tile map for transfer to the screen.
; IN:   CurColumn = tile column + 1 (1-$20); DynTileBufLen = where to append.
; OUT:  a record appended to DynTileBuf: PPU address $2100 + column, control $96 (vertical, $16
;       tiles), 22 tiles, then the $FF end marker; DynTileBufLen += $19. Uses [00:01], [06].
; NOTE: [00:01] starts at $651A, $16 below PlayAreaTiles ($6530), because the loop adds $16 once more
;       than the column number.
; ----------------------------------------------------------------------------
CopyColumnToTileBuf:
    LDA #$1A
    STA $00
    LDA #$65
    STA $01
    LDX CurColumn
    DEX                         ; X = tile column (0-$1F).
    TXA
    LDY DynTileBufLen           ; Append after what is already in the dynamic transfer buffer.
    STA DynTileBuf+1, Y         ; PPU address $21xx: the top play area row of nametable 0, at this column.
    LDA #$21
    STA DynTileBuf, Y
:
    ; Add $16 (the height of a column) X+1 times: [00:01] = $6530 + column * $16.
    LDA #$16
    JSR AddToInt16At0
    DEX
    BPL :-
    LDA #$96                    ; Control byte: bit 7 = write downward (step 32), count $16.
    STA DynTileBuf+2, Y
    TXA                         ; X is $FF here: the end marker, placed after the 22 tiles.
    STA DynTileBuf+25, Y
    TYA                         ; X = offset of the record in the buffer.
    TAX
    LDY #$00                    ; [06] counts the tiles copied.
    STY $06
:
    LDA ($00), Y                ; Tile from the tile map,
    STA DynTileBuf+3, X         ; into the record.
    JSR Add1ToInt16At0
    INX
    INC $06
    LDA $06
    CMP #$16
    BCC :-                      ; Until $16 tiles are copied.
    ; X advanced by $16; 3 more for the record header gives the offset of the end marker.
    INX
    INX
    INX
    STX DynTileBufLen
    RTS

; ----------------------------------------------------------------------------
; Queues one tile row (32 tiles) of the play area tile map for transfer to the screen.
; IN:   CurRow = tile row 0-$15.
; OUT:  DynTileBuf rewritten from its start: PPU address $2100 + CurRow * $20, count $20, 32 tiles,
;       $FF end marker; DynTileBufLen = $23. Uses [00:01]. Anything queued before is overwritten.
; ----------------------------------------------------------------------------
CopyRowToTileBuf:
    ; [00:01] = $6530 + CurRow: the row's tile in the first column.
    LDA #$65
    STA $01
    LDA CurRow
    TAX
    CLC
    ADC #$30
    STA $00
    BCC :+
    INC $01                     ; Only for CurRow >= $D0; never happens with rows 0-$15.
:
    ; PPU address = $20E0 + $20 * (CurRow + 1) = $2100 + CurRow * $20.
    LDA #$20
    STA DynTileBuf
    LDA #$E0
    STA DynTileBuf+1

@Add20H:
    LDA DynTileBuf+1
    CLC
    ADC #$20
    STA DynTileBuf+1
    BCC :+
    INC DynTileBuf
:
    DEX
    BPL @Add20H
    LDA #$20                    ; Control byte: $20 tiles, written left to right.
    STA DynTileBuf+2
    STX DynTileBuf+35           ; X is $FF: the end marker after the 32 tiles.

    ; Copy the row: the same row of each of the 32 columns, $16 bytes apart in the tile map.
    LDX #$00
    LDY #$00
:
    LDA ($00), Y                ; Copy a tile.
    STA DynTileBuf+3, X
    LDA #$16
    JSR AddToInt16At0           ; Advance to the tile in the same row, but next column of play area.
    INX                         ; Advance to the next tile in the row in transfer buf.
    CPX #$20
    BCC :-                      ; If we haven't written $20 tiles, then go write another.
    LDA #$23                    ; The transfer buf is $23 bytes (3 for header, $20 for payload).
    STA DynTileBufLen
    RTS
