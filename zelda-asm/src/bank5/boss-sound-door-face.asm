; ============================================================================
; FILE:     bank5/boss-sound-door-face.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4129-4226
; SUBJECTS: (none)
; TOPIC:    The looping boss roar heard in dungeon rooms near a living boss; 174 bytes of padding; the
;           helper that copies one door face tile into the dynamic transfer buffer.
; SEE ALSO: bank7/init-mode5-play.asm (RunCrossRoomTasksAndBeginUpdateMode calls
;           CheckBossSoundEffectUW when mode 5 starts in a dungeon room); bank5/room-history-uw.asm
;           (SaveKillCountUW sets the "all dead" flag tested here); bank0/sound-effects.asm
;           (DriveEffect: EffectRequest bit 7 silences the sample); bank5/update-doors.asm (UpdateDoors
;           calls WriteDoorFaceTileHorizontally); bank5/lay-out-doors.asm (HorizontalDoorFaceIndexes).
; KEY FACTS:
;   - A dungeon room plays a roar when bits 6-5 of its LevelBlockAttrsE byte are not 0 and the
;     level's boss room (LevelInfo_BossRoomId) is not saved as all dead (room flags bits 6-7 = 3).
;     CheckBossSoundEffectUW
;   - Roar index 1, 2, 3 requests sample bit $10, $20, $40, each with bit $80 added; the value is
;     stored into SampleRequest, replacing any other sample request of the frame. BossSoundEffects,
;     CheckBossSoundEffectUW
;   - With no roar to play, EffectRequest bit 7 is set, which stops the sample channel and tune
;     channel 1 on entry to every such dungeon room. CheckBossSoundEffectUW
;   - The test runs once per room, when mode 5 begins in a dungeon. CheckBossSoundEffectUW
; CONTENTS: (generated; L = line in this file)
;   L0030 BossSoundEffects               data  TABLE: boss roar sample requests; index = roar index (LevelBlockAttrsE bits 6-5); value =
;   L0042 CheckBossSoundEffectUW         code  Starts a dungeon room's looping boss roar, or silences it when the boss is dead or it has none.
;   L0100 WriteDoorFaceTileHorizontally  code  Copies one tile of a door face to the dynamic transfer buffer, reading the face row by row.
; ============================================================================

; TABLE: boss roar sample requests; index = roar index (LevelBlockAttrsE bits 6-5); value =
; SampleRequest bit (0 = no roar in this room).
BossSoundEffects:
    .BYTE $00, $10, $20, $40

; ----------------------------------------------------------------------------
; Starts a dungeon room's looping boss roar, or silences it when the boss is dead or it has none.
; IN:   RoomId; LevelInfo_BossRoomId; LevelBlockAttrsE
; OUT:  SampleRequest = roar bit | $80, or EffectRequest |= $80 (silence sample and tune channel 1).
;       Clobbers A, X, Y, [00:01].
; NOTE: The boss counts as dead when the boss room's flags have bits 6-7 = 3 ("all dead", written
;       by SaveKillCountUW when that room is left). Called once per room at the start of mode 5, in
;       dungeons only.
; ----------------------------------------------------------------------------
CheckBossSoundEffectUW:
    JSR GetRoomFlags            ; Only for [00:01] = address of the room flags.
    LDY LevelInfo_BossRoomId
    LDA ($00), Y                ; Flags of the boss room.
    AND #$C0
    CMP #$C0
    BEQ :+                      ; Boss room all dead: silence.
    LDY RoomId
    LDA LevelBlockAttrsE, Y
    AND #$60                    ; Roar index in bits 6-5.
    ASL                         ; Rotate it down to bits 1-0.
    ROL
    ROL
    ROL
    TAX
    LDA BossSoundEffects, X
    BEQ :+                      ; Index 0: no roar in this room: silence.
    ORA #$80                    ; Bit 7 with the sample bit (DriveSample loops samples $10-$40).
    STA SampleRequest
    RTS
:
    LDA #$80                    ; EffectRequest bit 7: silence the sample and tune channel 1.
    JMP PlayEffect

; DATA: filler ($FF), 174 bytes, no label. Nothing refers to it.
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

; ----------------------------------------------------------------------------
; Copies one tile of a door face to the dynamic transfer buffer, reading the face row by row.
; IN:   X = write offset in DynTileBuf; [02:03] = address of the door face's tile list (stored as
;       vertical pairs); [04] = tiles left to write; [07] = running tile index of this door, 0-3;
;       [09] = door direction index
; OUT:  X + 1; [07] + 1; [04] - 1; Z set when [04] reached 0 (the record is complete)
; NOTE: The tile list holds the face column by column. HorizontalDoorFaceIndexes, entered at
;       (base offset for direction [09]) + [07], gives the list offset of each tile in row order,
;       so that a nametable row can be written left to right.
; ----------------------------------------------------------------------------
WriteDoorFaceTileHorizontally:
    ; Base offset of this direction's entries in HorizontalDoorFaceIndexes.
    LDY $09
    LDA HorizontalDoorFaceIndexesBaseOffsets, Y

    ; Add the running tile index.
    CLC
    ADC $07
    TAY

    ; Look up the offset of the tile in the door face tile list.
    LDA HorizontalDoorFaceIndexes, Y
    TAY

    ; Copy the tile to the dynamic transfer buffer.
    LDA ($02), Y
    STA DynTileBuf, X

    ; Advance the buffer offset and the tile index; count the tile.
    INX
    INC $07
    DEC $04
    RTS
