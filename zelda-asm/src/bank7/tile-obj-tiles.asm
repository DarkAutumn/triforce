; ============================================================================
; FILE:     bank7/tile-obj-tiles.asm     BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 1130-1255
; SUBJECTS: (none)
; TOPIC:    GetUniqueRoomId, tile object tiles, FillTileMap
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 GetUniqueRoomId     code  Returns:
;   L0029 ChangeTileObjTiles  code  Params:
;   L0120 FillTileMap         code  Params:
; ============================================================================

; Returns:
; A: unique room ID
; Y: room ID
;
GetUniqueRoomId:
    LDY RoomId
    LDA LevelBlockAttrsD, Y
    AND #$3F                    ; Unique room ID in both OW and UW.
    RTS

; Params:
; A: first tile
; X: object index
; [F7]: not 0 to switch to bank 4 before returning
;
ChangeTileObjTiles:
    ; First, cue the transfer of the 4 tiles to nametable 0.
    ;
    ; [05] holds the first tile.
    STA $05
    TXA                         ; Save object index.
    PHA
    LDA ObjX, X
    STA $03                     ; [03] holds the X coordinate.
    LDA ObjY, X
    STA $02                     ; [02] holds the Y coordinate.
    JSR MapScreenPosToPpuAddr
    LDX DynTileBufLen           ; Start writing from where we left off last time.

    ; Two transfer records will be written to the buffer and an end marker.
    ;
    ; Each record is 5 bytes and will transfer 2 tiles arranged vertically.
    ; In total, 11 bytes will be written.
    ;
    ; First, write the high byte of PPU address in each record.
    LDA $00
    STA DynTileBuf, X
    STA DynTileBuf+5, X

    ; Write the low address in the first record. In the second record,
    ; write (low address + 1).
    LDA $01
    STA DynTileBuf+1, X
    STA DynTileBuf+6, X
    INC DynTileBuf+6, X

    ; Write the first tile twice in each record. Patch them later.
    LDA $05
    STA DynTileBuf+3, X
    STA DynTileBuf+4, X
    STA DynTileBuf+8, X
    STA DynTileBuf+9, X

    ; If tile >= $46 and < $F3, add 2 to the tile bytes in the second record.
    ; Add 1 more to the second byte in each record.
    ; Here is an example result of this manipulation:
    ;   record 1: $70, $71
    ;   record 2: $72, $73
    CMP #$46
    BCC @SetCountBytes
    CMP #$F3
    BCS @SetCountBytes
    CLC
    ADC #$02
    STA DynTileBuf+8, X
    STA DynTileBuf+9, X
    INC DynTileBuf+4, X
    INC DynTileBuf+9, X

@SetCountBytes:
    ; Set the count byte in each record to $82:
    ; 2 tiles arranged vertically
    LDA #$82
    STA DynTileBuf+2, X
    STA DynTileBuf+7, X

    ; Write the end marker.
    LDA #$FF
    STA DynTileBuf+10, X

    ; Update dynamic buffer length with 10 new bytes.
    TXA
    CLC
    ADC #$0A
    STA DynTileBufLen
    PLA                         ; Restore object index.
    TAX
    LDA #$05
    JSR SwitchBank

    ; Change the tiles in the play area map.
    JSR ChangePlayMapSquareOW

    ; Switch to bank 4, if requested.
    LDA ReturnToBank4
    BEQ :+
    LDA #$04
    JSR SwitchBank
:
    LDA #$00
    STA ReturnToBank4
    RTS

; Params:
; [$0A]: tile
;
FillTileMap:
    LDA #$05
    JSR SwitchBank
    JSR FetchTileMapAddr
    LDY #$00

@Loop:
    LDA $0A
    STA ($00), Y
    JSR Add1ToInt16At0
    LDA $00
    CMP #$F0
    BNE @Loop
    LDA $01
    CMP #$67
    BNE @Loop
    RTS

; Unknown block
    .BYTE $FF, $FF
