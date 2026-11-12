; ============================================================================
; FILE:     bank5/update-doors.asm       BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 5038-5309
; SUBJECTS: (none)
; TOPIC:    Carrying out a door command in a dungeon room: the two-step opening or closing of the
;           triggered door, the tiles sent to the screen, the saved door flags in this room and the
;           room behind the door, and CurOpenedDoors.
; SEE ALSO: bank5/shutters-door-flags.asm (CheckShutters and TriggerOpenDoor start an open command;
;           SetDoorFlag, ResetDoorFlag); bank5/touch-door.asm (TouchDoorKey spends the key and starts
;           the open command); bank7/bomb.asm (a bomb near a bombable wall starts the open command);
;           bank5/enter-room.asm (InitMode_EnterRoom starts the close command);
;           bank5/lay-out-doors.asm (LayOutDoors, the door tables, the shared RTS L165D4_Exit);
;           bank5/fill-walls.asm (FindDoorTypeByDoorBit); bank5/boss-sound-door-face.asm
;           (WriteDoorFaceTileHorizontally); bank5/room-layouts-uw.asm (the door faces);
;           bank7/mode5-play-update.asm (UpdateMode5Play calls UpdateDoors when CurLevel is not 0).
; KEY FACTS:
;   - A door command is TriggeredDoorCmd (6 = open, 2 = close) plus TriggeredDoorDir (the side's
;     direction bit). It takes two steps 8 frames apart (DoorTimer): 6 -> 7 -> done, 2 -> 3 -> done;
;     then TriggeredDoorCmd is 0 again. Only one command runs at a time. UpdateDoors.
;   - UpdateDoors is called on every frame on which the play update runs in a dungeon, after Link,
;     his weapons and the monsters have been updated and right after CheckShutters, so a command
;     set on a frame gets its first step on that frame. It does nothing while GameMode is $12 (end
;     of level) or DoorTimer is not 0. UpdateDoors (UpdateMode5Play).
;   - Opening: the first step sends the closed tiles again (nothing changes on screen); the second
;     step, 8 frames later, sends the open tiles and on that frame adds the side to CurOpenedDoors
;     and redraws the doors in the tile map. The side counts as closed until then. UpdateDoors.
;   - Opening a key door (types 5, 6) or bombable wall (type 4) sets its saved door flag in this room
;     and the flag of the opposite side in the room behind it (room flags bits 0-3), so it is open
;     from both sides from then on. The neighbouring room id is not range-checked and the door type
;     of the neighbour's side is not looked at. UpdateDoors.
;   - A shutter (type 7) that opens only enters CurOpenedDoors; no flag is saved. UpdateDoors.
;   - The close command is issued when Link enters a room through a side that is in CurOpenedDoors
;     (after LayOutDoors that can only be a side of type 4-7). Its first step sets Link's ObjTimer
;     to $30 frames whatever the type; then only a shutter closes (closed tiles at once; 8 frames
;     later its saved door flag is cleared and its bit removed from CurOpenedDoors). For a key door
;     or bombable wall the command is dropped and the door stays open. UpdateDoors.
;   - Both steps request sample bit $04 (the door sound) for key doors and shutters; a bombable wall
;     requests none here. PrepareWriteHorizontalDoorTransferRecords.
;   - Only the 2x2 passage tiles of the door are sent to the screen, as two 2-tile transfer records
;     in DynTileBuf. UpdateDoors, PrepareWriteHorizontalDoorTransferRecords.
; CONTENTS: (generated; L = line in this file)
;   L0060 UpdateDoors                                   code  Runs one step of the pending door command: opens or closes the triggered dungeon door.
;   L0168 L_SetOpenedDoorMaskAndResetCmdAndLayOutDoors  code  UpdateDoors, end of a command: stores A as CurOpenedDoors, then falls into the reset below.
;   L0172 L_ResetDoorCmdAndLayOutDoors                  code  UpdateDoors, end of a command: clears TriggeredDoorCmd and redraws the doors in the tile map.
;   L0217 PrepareWriteHorizontalDoorTransferRecords     code  Picks the face to show for the triggered door and sets up its two 2-tile transfer records.
; ============================================================================

; ----------------------------------------------------------------------------
; Runs one step of the pending door command: opens or closes the triggered dungeon door.
; IN:   TriggeredDoorCmd (2 or 3 = close, 6 or 7 = open; 0 = none); TriggeredDoorDir = direction bit;
;       DoorTimer; GameMode; CurOpenedDoors; RoomId.
; OUT:  DynTileBuf/DynTileBufLen: two 2-tile records for the door; TriggeredDoorCmd + 1 and
;       DoorTimer = 8 after the first step; after the second step TriggeredDoorCmd = 0,
;       CurOpenedDoors and the room flags updated and the doors redrawn in the tile map
;       (LayOutDoors). Uses [00]-[09], [0B].
; NOTE: Does nothing in GameMode $12 or while DoorTimer is not 0. The first step of a close command
;       sets Link's ObjTimer to $30. A close command for anything but a shutter ends at once without
;       closing the door. A door finishes opening 8 frames after its first step.
; ----------------------------------------------------------------------------
UpdateDoors:
    LDA GameMode
    CMP #$12                    ; End-of-level mode: leave the doors alone.
    BEQ L165D4_Exit
    LDA DoorTimer
    BNE L165D4_Exit             ; Still waiting between the two steps.
    LDA TriggeredDoorCmd
    BEQ L165D4_Exit             ; No command.

    ; From the command, work out [08]: 2 = show the open face, 0 = show the closed face.
    ;
    ;   command              2 (close)  3 (close, 2nd)  6 (open)  7 (open, 2nd)
    ;   after the LSR below  2          1               6         3
    ;   [08]                 0          0               0         2
    ;
    ; So only the second step of opening shows the open face. Commands other than 2, 3, 6 and 7 are
    ; never set.
    AND #$07
    LDY #$01
    STY $02
    BIT $02
    BEQ :+
    LSR                         ; Odd command (second step): halve it.
:
    CMP #$02
    BNE :+
    LDY #$30                    ; Command 2 (first step of closing): Link's timer = $30 frames.
    STY ObjTimer
:
    AND #$03
    SEC
    SBC #$01
    AND #$02
    STA $08

    ; A close command (2 or 3) only acts on a shutter. For a key door or bombable wall it ends here:
    ; the door stays open.
    LDA TriggeredDoorCmd
    CMP #$05
    BCS @WriteTiles             ; Open command.
    LDA TriggeredDoorDir
    STA $02
    JSR FindDoorTypeByDoorBit
    CMP #$07
    BEQ @WriteTiles             ; Closing a shutter.
    JMP L_ResetDoorCmdAndLayOutDoors

@WriteTiles:
    ; Queue the 2x2 passage tiles of the face chosen by [08] for the screen: two records of two
    ; tiles. Sets X = DynTileBufLen, [00] = address high, [01] = address low, [05] = [06] = 2,
    ; [07] = 0, [09] = direction index.
    JSR PrepareWriteHorizontalDoorTransferRecords

@LoopTransferRec:
    LDA $06
    STA $04                     ; [04] = tiles left in this record (2).

    ; Record header: name table address high, low, then the tile count.
    LDA $00
    STA DynTileBuf, X
    INX
    LDA $01
    STA DynTileBuf, X
    INX
    LDA $06
    STA DynTileBuf, X
    INX
:
    JSR WriteDoorFaceTileHorizontally    ; Append one tile; Z set after the record's last tile.
    BNE :-

    ; Second record: one tile row lower (+$20; bit 5 is clear in all four low address bytes).
    LDA #$20
    ORA $01
    STA $01
    DEC $05                     ; [05] = records left.
    BNE @LoopTransferRec

    ; End marker; DynTileBufLen = its offset.
    LDA #$FF
    STA DynTileBuf, X
    TXA
    STA DynTileBufLen

    ; Step the command. 2 -> 3 and 6 -> 7: wait 8 frames for the second step.
    ; 3 -> 4 and 7 -> 8 (low two bits 0): the command is complete.
    INC TriggeredDoorCmd
    LDA TriggeredDoorCmd
    AND #$03
    BEQ :+
    LDA #$08
    STA DoorTimer
    RTS
:
    ; Command 4: a shutter has closed. Clear the side's saved door flag and remove the side from
    ; CurOpenedDoors. (No SetDoorFlag call sets the flag for a shutter, but the neighbour-room write
    ; below sets a flag without looking at that side's door type.) Command 8 branches to the
    ; unnamed label after L_ResetDoorCmdAndLayOutDoors.
    LDA TriggeredDoorCmd
    CMP #$04
    BNE :+
    LDX $09                     ; X = direction index.
    JSR ResetDoorFlag
    LDA TriggeredDoorDir
    EOR #$0F
    AND CurOpenedDoors

; UpdateDoors, end of a command: stores A as CurOpenedDoors, then falls into the reset below.
L_SetOpenedDoorMaskAndResetCmdAndLayOutDoors:
    STA CurOpenedDoors

; UpdateDoors, end of a command: clears TriggeredDoorCmd and redraws the doors in the tile map.
L_ResetDoorCmdAndLayOutDoors:
    LDA #$00
    STA TriggeredDoorCmd
    JMP LayOutDoors
:
    ; Command 8: a door has opened. A shutter is only added to CurOpenedDoors.
    LDA TriggeredDoorDir
    STA $02
    JSR FindDoorTypeByDoorBit
    CMP #$07
    BEQ @AddToOpenedMask

    ; A key door or bombable wall: set its saved door flag in this room...
    LDX $09                     ; X = direction index.
    JSR SetDoorFlag             ; Returns Y = RoomId and [00:01] = address of the room flags.

    ; ...and the flag of the opposite side in the room behind the door, so that the door is open
    ; from there too. The neighbouring room id is RoomId +1, -1, +$10 or -$10; it is not range-checked.
    TYA
    CLC
    ADC DoorNextRoomIdOffsets, X
    TAY
    TXA
    EOR #$01                    ; Opposite direction index: right <-> left, down <-> up.
    TAX
    LDA ($00), Y
    ORA LevelMasks, X
    STA ($00), Y

@AddToOpenedMask:
    LDA TriggeredDoorDir
    ORA CurOpenedDoors
    JMP L_SetOpenedDoorMaskAndResetCmdAndLayOutDoors

; ----------------------------------------------------------------------------
; Picks the face to show for the triggered door and sets up its two 2-tile transfer records.
; IN:   TriggeredDoorDir = direction bit; [08] = 0 to show the closed face, nonzero the open face.
; OUT:  X = DynTileBufLen (where the records go); [00] = name table address high byte and
;       [01] = low byte of the door's 2x2 passage block; [02:03] = address of the face's tiles;
;       [05] = 2 (records); [06] = 2 (tiles per record); [07] = 0 (first tile); [09] = direction
;       index. Clobbers A, Y, [03].
; NOTE: Requests sample bit $04 (the door sound) when the door is a key door or shutter, on every
;       call (both steps of a command). Only meaningful for door types 4-7: for types 0, 2 and 3
;       the face index comes out as nonsense, and UpdateDoors never calls it for them.
; ----------------------------------------------------------------------------
PrepareWriteHorizontalDoorTransferRecords:
    LDA TriggeredDoorDir
    STA $02
    JSR FindDoorTypeByDoorBit   ; A = door type; [03] = reverse direction index.
    CMP #$05
    BCC :+
    PHA
    LDA #$04                    ; Types 5-7: request the door sound (sample request bit $04).
    JSR PlaySample
    PLA
:
    ; Face index, as in LayOutDoors. First a provisional value: type 4 (bombable wall) -> 8;
    ; types 5-7 keep their number.
    CMP #$04
    BNE :+
    LDA #$08
:
    ; Type 1 (wall) -> 4. No trigger in the code starts a command for a wall (the triggers test for
    ; types 4, 5, 6 and 7), so this is not expected to run.
    CMP #$01
    BNE :+
    LDA #$04
:
    SEC
    SBC #$03
    TAY                         ; Y = 2, 3, 4 for types 5, 6, 7; 5 for type 4.

    ; Open face wanted ([08] not 0): 5 (bombed hole) for a bombable wall, else 1 (open doorway).
    ; Closed face wanted: subtract 1 from values of 3 and above, giving 2 (locked door) for types 5
    ; and 6, 3 (shutter) for type 7, 4 (plain-wall tiles) for type 4.
    LDA $08
    BEQ :+
    CPY #$05
    BEQ @MakeForwardIndex
    LDY #$01
:
    CPY #$03
    BCC @MakeForwardIndex
    DEY

@MakeForwardIndex:
    ; FindDoorTypeByDoorBit left the reverse direction index in [03]; direction index = 3 - [03].
    LDA #$03
    SEC
    SBC $03
    JSR FetchDoorAddrsFaceTilesSrcAndPlayAreaDst    ; [02:03] = face tiles; X = direction index.

    ; [00] = high byte and [01] = low byte of the name table address of the passage block. (This
    ; overwrites the tile map address that the call above left in [00:01].)
    LDA DoorVramAddrsHi, X
    STA $00
    LDA DoorVramAddrsLo, X
    STA $01

    STX $09                     ; [09] = direction index.
    LDX DynTileBufLen           ; X = where the next transfer record goes.
    LDA #$00
    STA $07                     ; [07] = index of the first passage tile (0-3).
    LDA #$02
    STA $06                     ; [06] = tiles per record.
    STA $05                     ; [05] = records.
    RTS
