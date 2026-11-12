; ============================================================================
; FILE:     bank7/tile-obj-tiles.asm     BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1130-1255
; SUBJECTS: Room, Tile
; TOPIC:    Helpers for a room's layout id and for changing background tiles during play: replace the
;           2x2 tiles under a tile object (push block, rock, armos, revealed stairs), fill the tile map.
; SEE ALSO: bank5/ow-shortcut-map-square.asm (Tile_ChangePlayAreaSquare, Tile_FetchPlayAreaTilesAddr: the play area tile
;           map); bank1/mazes-lost-woods-hills.asm (Ppu_MapScreenPosToAddr); bank4/push-block.asm,
;           bank4/rock-wall-tree-secrets.asm, bank4/ghini-armos.asm, bank5/underworld-secrets.asm
;           (callers of Tile_ChangeTileObjTiles); bank5/mode12-lay-out-room.asm (Mode12EndLevel_Init, RoomLayout_LayOutRoom) and
;           bank4/zelda-guard-fire.asm (Zelda_State1WaitForEnding): the callers of Tile_FillTileMap;
;           bank5/layout-room-ow.asm (RoomLayout_LayOutRoomOW takes 7 bits of the same byte as its layout id).
; KEY FACTS:
;   - The unique room id is bits 5-0 of LevelBlockAttrsD[RoomId]. Dungeons use it as the room layout
;     id; cellars are ids $3E (tunnel) and $3F (item cellar). The overworld layout code reads bits
;     6-0 of the same byte itself. Room_GetUniqueRoomId
;   - A tile object's square is redrawn as two vertical 2-tile records appended to DynTileBuf (10 bytes
;     plus the end marker), and the same square is written into the play area tile map, so both the
;     picture and tile collision change. Tile_ChangeTileObjTiles
;   - On screen, tiles $46-$F2 are drawn as four consecutive tiles (t, t+1 left column; t+2, t+3
;     right column) and any other tile four times. The tile map write (Tile_ChangePlayAreaSquare) has its
;     own rule: $27-$F2 consecutive, others looked up in RoomLayout_OWSquareTiles. Tile_ChangeTileObjTiles
;   - Tile_ChangeTileObjTiles returns with bank 5 switched in unless the caller set ReturnToBank4 first (the
;     bank 4 callers do); the flag is cleared on return. Tile_ChangeTileObjTiles
;   - Tile_FillTileMap overwrites the whole play area tile map $6530-$67EF (704 bytes = 32 columns of 22
;     rows) with one tile: $F6 (brick) before every dungeon room layout, blank $24 at the end of a
;     level (mode $12 init) and when Zelda's room hands over to mode $13. Tile_FillTileMap
; CONTENTS: (generated; L = line in this file)
;   L0040 Room_GetUniqueRoomId     code  Gets the current room's unique room id (bits 5-0 of its attribute byte D).
;   L0057 Tile_ChangeTileObjTiles  code  Replaces the 2x2 background tiles of the square under an object, on screen and in the tile map.
;   L0153 Tile_FillTileMap         code  Fills the whole play area tile map ($6530-$67EF) with one tile.
; ============================================================================

; ----------------------------------------------------------------------------
; Gets the current room's unique room id (bits 5-0 of its attribute byte D).
; OUT:  A = LevelBlockAttrsD[RoomId] AND $3F; Y = RoomId.
; NOTE: In a dungeon this is the room layout id; cellars have ids $3E (tunnel) and $3F (item cellar)
;       and callers test for those. The overworld layout code does not call this: it takes bits 6-0.
; ----------------------------------------------------------------------------
Room_GetUniqueRoomId:
    LDY RoomId
    LDA LevelBlockAttrsD, Y
    AND #$3F                    ; Bits 5-0 = unique room id.
    RTS

; ----------------------------------------------------------------------------
; Replaces the 2x2 background tiles of the square under an object, on screen and in the tile map.
; IN:   A = first tile; X = object slot (ObjX,X / ObjY,X give the square);
;       ReturnToBank4 nonzero = switch to bank 4 before returning.
; OUT:  X preserved; DynTileBufLen advanced by $0A; ReturnToBank4 = 0; bank 5 or 4 switched in.
;       Uses [00]-[03], [05] and whatever Tile_ChangePlayAreaSquare uses.
; NOTE: On screen, tiles $46-$F2 expand to t, t+1 (left column) and t+2, t+3 (right column); other
;       tiles are repeated four times. The tile map write (Tile_ChangePlayAreaSquare, used in dungeons
;       too despite its name) expands $27-$F2 and looks other tiles up in RoomLayout_OWSquareTiles. The
;       tiles callers pass ($24, $26, $70, $74, $B0, $BC, $C0, $C8) come out the same under both rules.
; ----------------------------------------------------------------------------
Tile_ChangeTileObjTiles:
    ; First, cue the transfer of the 4 tiles to nametable 0.
    ;
    ; [05] holds the first tile.
    STA $05
    TXA                         ; Save object slot.
    PHA
    LDA ObjX, X
    STA $03                     ; [03] holds the X coordinate.
    LDA ObjY, X
    STA $02                     ; [02] holds the Y coordinate.
    JSR Ppu_MapScreenPosToAddr   ; [00:01] = PPU address (hi, lo) of the tile at (X, Y).
    LDX DynTileBufLen           ; Append at the buffer's current end marker.

    ; Two transfer records are written to the buffer, then an end marker.
    ;
    ; Each record is 5 bytes (address hi, address lo, control, 2 tiles) and
    ; transfers 2 tiles arranged vertically. In total, 11 bytes are written.
    ;
    ; First, write the high byte of the PPU address in each record.
    LDA $00
    STA DynTileBuf, X
    STA DynTileBuf+5, X

    ; Write the low address in the first record (left column). In the second
    ; record, write (low address + 1) (right column).
    LDA $01
    STA DynTileBuf+1, X
    STA DynTileBuf+6, X
    INC DynTileBuf+6, X

    ; Write the first tile twice in each record. Patch them below.
    LDA $05
    STA DynTileBuf+3, X
    STA DynTileBuf+4, X
    STA DynTileBuf+8, X
    STA DynTileBuf+9, X

    ; If tile >= $46 and < $F3, add 2 to the tile bytes in the second record.
    ; Add 1 more to the second byte in each record.
    ; Example result for tile $70 (stairs):
    ;   record 1 (left column):  $70, $71
    ;   record 2 (right column): $72, $73
    CMP #$46
    BCC @SetCountBytes          ; Tile < $46: keep four copies of the same tile.
    CMP #$F3
    BCS @SetCountBytes          ; Tile >= $F3: keep four copies of the same tile.
    CLC
    ADC #$02
    STA DynTileBuf+8, X
    STA DynTileBuf+9, X
    INC DynTileBuf+4, X
    INC DynTileBuf+9, X

@SetCountBytes:
    ; Set the control byte in each record to $82:
    ; bit 7 = step 32 (downward), count 2.
    LDA #$82
    STA DynTileBuf+2, X
    STA DynTileBuf+7, X

    ; Write the end marker.
    LDA #$FF
    STA DynTileBuf+10, X

    ; Advance the buffer length by the 10 record bytes; it now points at the end marker.
    TXA
    CLC
    ADC #$0A
    STA DynTileBufLen
    PLA                         ; Restore object slot.
    TAX
    LDA #$05
    JSR Mmc1_SwitchBank

    ; Change the same square in the play area tile map (reads the tile from [05]).
    JSR Tile_ChangePlayAreaSquare

    ; Switch to bank 4, if requested; otherwise bank 5 stays switched in.
    LDA ReturnToBank4
    BEQ :+
    LDA #$04
    JSR Mmc1_SwitchBank
:
    LDA #$00
    STA ReturnToBank4
    RTS

; ----------------------------------------------------------------------------
; Fills the whole play area tile map ($6530-$67EF) with one tile.
; IN:   [0A] = tile.
; OUT:  [00:01] = $67F0; Y = 0; bank 5 switched in.
; NOTE: Only the tile map in RAM changes; nothing is sent to the PPU here. Three callers: RoomLayout_LayOutRoom
;       fills with $F6 before laying out any dungeon room; Mode12EndLevel_Init (end of level) and
;       Zelda_State1WaitForEnding (hand-over to mode $13) fill with blank tile $24 for the curtain.
; ----------------------------------------------------------------------------
Tile_FillTileMap:
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Tile_FetchPlayAreaTilesAddr        ; [00:01] = $6530, the start of the tile map.
    LDY #$00

@Loop:
    LDA $0A
    STA ($00), Y
    JSR Math_Add1ToInt16At0
    LDA $00                     ; Stop when [00:01] reaches $67F0, one past the last byte.
    CMP #$F0
    BNE @Loop
    LDA $01
    CMP #$67
    BNE @Loop
    RTS

; DATA: filler ($FF), 2 bytes.
    .BYTE $FF, $FF
