; ============================================================================
; FILE:     bank5/ow-shortcut-map-square.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6079-6371
; SUBJECTS: (none)
; TOPIC:    Shortcut stairs, map square change; padding
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 CheckShortcut                                      code
;   L0076 ChangePlayMapSquareOW                              code
;   L0153 FetchTileMapAddr                                   code  Returns:
;   L0163 CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone  code  Returns:
;   L0171 CopyNextRowToTransferBuf                           code  Returns:
;   L0178 LayoutRoom_SubmodeTask                             code
; ============================================================================

CheckShortcut:
    JSR GetRoomFlags
    ASL
    BCS @Exit                   ; If the player found the secret, then return.
    LDA ($00), Y
    AND #$20
    BEQ @Exit                   ; If the shortcut wasn't seen in this room, then return.
    JSR FetchTileMapAddr
    JSR GetShortcutOrItemXY

    ; Divide X coordinate by 4 to get offset into column address table.
    ; Think of it this way. Divide X by 8 to get tile column number.
    ; Then multiply by 2 to turn it into an offset for an address.
    LSR
    LSR
    TAX
    LDA PlayAreaColumnAddrs, X  ; Put address of column that has X coordinate into [$00:01].
    STA $00
    LDA PlayAreaColumnAddrs+1, X
    STA $01
    TYA                         ; Subtract $40 from X coordinate to get rid of status bar.
    SEC
    SBC #$40
    LSR                         ; Divide new Y coordinate by 8 to get a tile row.
    LSR
    LSR
    TAY                         ; Keep tile row (offset) in Y register.
    LDA ($00), Y                ; Get the tile that's where the shortcut should be.
    CMP #$C4
    BEQ @WriteStairs            ; If it's a tree, go prepare a stairs square.
    CMP #$BC
    BEQ @Exit                   ; If it's a gravestone, then return.
    CMP #$D8
    BNE @WriteStairs            ; If it's not a rock wall, then go prepare a stairs square.
    LDA RoomTileObjType
    CMP #$62
    BEQ @WriteStairs            ; If the room's tile object is a rock, go prepare a stairs square.

    ; Else make the tile object nothing, and write a black tile.
    ; But this branch seems to be unused.
    ; Begin unverified code 16BAF
    LDA #$00
    STA RoomTileObjType
    LDA #$0C
    STA $0D
    ; End unverified code

@Write:
    JSR WriteSquareOW

@Exit:
    RTS

@WriteStairs:
    LDA #$10                    ; The first type 1 square index.
    STA $0D
    LDA #$70
    BNE @Write                  ; Go write a stairs square.

ChangePlayMapSquareOW:
    TXA                         ; Save object index.
    PHA

    ; We're dealing with squares.
    ; So, align the object's X coordinate with 16 pixels.
    LDA ObjX, X
    AND #$F0

    ; Divide it by 4 to get the offset of the address of the column
    ; in play area map.
    ;
    ; The calculation is: address offset = (X / 8) * 2
    ; 8 for the width of the column; 2 for the width of the address
    LSR
    LSR
    TAX

    ; Store the address of the column in [00:01].
    LDA PlayAreaColumnAddrs, X
    STA $00
    LDA PlayAreaColumnAddrs+1, X
    STA $01
    PLA                         ; Get object index.
    PHA
    TAX

    ; Align the object's Y coordinate with 16 pixels.
    LDA ObjY, X
    AND #$F0

    ; Subtract $40 for the status bar; and divide by 8 for
    ; the height of each row.
    SEC
    SBC #$40
    LSR
    LSR
    LSR

    ; Add this row offset to the column address.
    JSR AddToInt16At0

    ; Assume that we'll write a type 1 square with primary square
    ; taken from [05]. If so, the square index doesn't matter
    ; as long as >= $10. See WriteSquareOW.
    LDY #$00
    LDX #$10
    LDA $05

    ; If tile < $27 or >= $F3, we'll write a type 3 square.
    ; So, we have to look up the square index corresponding
    ; to primary square/tile in [05].
    ;
    ; Look in primary square table from $E to 1. If a match is found,
    ; then the X register will have the square index value.
    CMP #$27
    BCC :+
    CMP #$F3
    BCC @Write
:
    LDX #$0E
:
    CMP PrimarySquaresOW, X
    BEQ @Write
    DEX
    BNE :-

@Write:
    ; Write the square.
    STX $0D
    JSR WriteSquareOW
    PLA                         ; Restore object index.
    TAX
    RTS

; Returns:
; [$00:01]: address of room tile map
FetchTileMapAddr:
    LDA #$30
    STA $00
    LDA #$65
    STA $01
    RTS

; Returns:
; C: 1 if copied the last row
;
CopyNextRowToTransferBufAndAdvanceSubmodeWhenDone:
    JSR CopyNextRowToTransferBuf
    BCS :+                      ; If done, then go to the next submode.
    RTS

; Returns:
; C: 1 if copied the last row
;
CopyNextRowToTransferBuf:
    JSR CopyRowToTileBuf
    INC CurRow
    LDA CurRow
    CMP #$16
    RTS

LayoutRoom_SubmodeTask:
    JSR LayOutRoom
    LDA #$00
    STA CurRow
:
    INC GameSubmode
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF
