; ============================================================================
; FILE:     bank5/boss-sound-door-face.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4129-4226
; SUBJECTS: (none)
; TOPIC:    Boss roar; padding; door face tile helper
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 BossSoundEffects               data
;   L0017 CheckBossSoundEffectUW         code
;   L0081 WriteDoorFaceTileHorizontally  code  Params:
; ============================================================================

BossSoundEffects:
    .BYTE $00, $10, $20, $40

CheckBossSoundEffectUW:
    JSR GetRoomFlags
    LDY LevelInfo_BossRoomId
    LDA ($00), Y                ; Get room flags for boss room.
    AND #$C0
    CMP #$C0
    BEQ :+                      ; If the boss was defeated, go turn off ambient sound effects.
    LDY RoomId
    LDA LevelBlockAttrsE, Y
    AND #$60                    ; Sound effect index
    ASL                         ; Shift the sound effect index to the low end of the byte.
    ROL
    ROL
    ROL
    TAX
    LDA BossSoundEffects, X
    BEQ :+                      ; If the room has no boss sound effect, go turn off any that might be playing.
    ORA #$80                    ; Add the flag to repeat
    STA SampleRequest
    RTS
:
    LDA #$80                    ; Silence sample and tune1.
    JMP PlayEffect

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
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF

; Params:
; [02:03]: address of door tiles for direction and face
; [04]: count of tiles remaining
; [07]: current index (0 to 3) of the tile to copy
;       to the dynamic transfer buf
; [09]: door direction index
;
; Returns:
; Z: 1 if wrote the last tile / [04] became 0
; [04]: original value - 1
; [07]: original value + 1
;
; The door face tile list at [02:03] has pairs of tiles arranged
; vertically. The point of this routine is to access them
; horizontally in order to transfer them to a nametable.
;
WriteDoorFaceTileHorizontally:
    ; For the direction index in [09], look up the base offset of the
    ; set of 4 indexes in HorizontalDoorFaceIndexes that point to
    ; the door face tiles at [02:03].
    LDY $09
    LDA HorizontalDoorFaceIndexesBaseOffsets, Y

    ; Add [07] to the base offset we got above. This yields one
    ; of four consecutive indexes into HorizontalDoorFaceIndexes.
    CLC
    ADC $07                     ; Current index of a tile, abstractly (0 to 3)
    TAY

    ; With that index, look up the index to use with the door face
    ; list of tiles at [02:03].
    LDA HorizontalDoorFaceIndexes, Y
    TAY

    ; Now we can read one of four tiles inside a door face tile map,
    ; and copy it to the dynamic transfer buf.
    LDA ($02), Y
    STA DynTileBuf, X

    ; Prepare for the next call:
    ; - increment dynamic transfer buf pointer
    ; - increment index of tile to copy [07]
    ; - decrement count of tiles remaining
    INX
    INC $07
    DEC $04
    RTS
