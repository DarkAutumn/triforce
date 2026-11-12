; ============================================================================
; FILE:     bank5/next-room-calc.asm     BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7482-7565
; SUBJECTS: (none)
; TOPIC:    Works out NextRoomId from the direction Link leaves a room in; leaving the map (as through
;           a dungeon's entrance) loads the overworld instead. Also the end of the end-level sequence
;           and a small PPU mask helper.
; SEE ALSO: bank7/next-room-room-flags.asm (CalculateNextRoom and its jump table, which dispatch here by
;           door type); bank5/mode7-scroll-init.asm (the caller of CalculateNextRoom);
;           bank1/mazes-lost-woods-hills.asm (CheckMazes); bank5/mode12-lay-out-room.asm (jumps to
;           EndGameMode12); bank7/init-mode-dispatch.asm (InitMode3_Sub1 places Link in the OW);
;           bank5/enter-room.asm (uses CalcNextRoomByDir to find the room Link came from).
; KEY FACTS:
;   - Next room = RoomId + offset: up -$10, down +$10, left -1, right +1, in 8-bit arithmetic with no
;     column test, so left from column 0 gives the last room of the row above and right from column
;     $F the first room of the row below (NextRoomIdOffsets, CalculateNextRoomForDoor).
;   - A result with bit 7 set (up from row 0, down from row 7, left from $00, right from $7F) is "no
;     such room": the overworld is loaded through mode 2 with UndergroundExitType = 2. This is how
;     walking out of a dungeon's entrance works (CalculateNextRoomForDoor, EndGameMode12).
;   - In the overworld CheckMazes may replace NextRoomId (Lost Woods, Lost Hills) before the validity
;     test (CalculateNextRoomForDoor).
;   - Dungeon door type 1 (wall) computes no next room at all (CalculateNoNextRoom).
;   - Finishing a level (mode $12) ends in the same code: back to the overworld through mode 2
;     (EndGameMode12).
; CONTENTS: (generated; L = line in this file)
;   L0036 NextRoomIdOffsets         data  TABLE: value added to RoomId to get the neighbouring room; index = reverse direction index (0 up,
;   L0046 CalculateNoNextRoom       code  Door types with no room behind them: clears the direction in [E7] and computes nothing.
;   L0069 CalculateNextRoomForDoor  code  Sets NextRoomId to the room beyond the side Link is leaving by; loads the OW when there is none.
;   L0100 EndGameMode12             code  Leaves the current level for the overworld: mode 2 (load level) with CurLevel = 0.
;   L0114 MaskCurPpuMaskGrayscale   code  Clears bit 0 (grayscale) of the PPU mask copy CurPpuMask.
;   L0127 CalcNextRoomByDir         code  Computes the id of the room next to the current one in a direction, with no checks.
; ============================================================================

; TABLE: value added to RoomId to get the neighbouring room; index = reverse direction index (0 up,
;        1 down, 2 left, 3 right); value = $F0 (-$10), $10, $FF (-1), $01.
NextRoomIdOffsets:
    .BYTE $F0, $10, $FF, $01

; ----------------------------------------------------------------------------
; Door types with no room behind them: clears the direction in [E7] and computes nothing.
; NOTE: Entries 1 and 8 of CalculateNextRoom_JumpTable: door type 1 (wall), and the value 8 that
;       FindDoorTypeByDoorBit returns when Link's direction has none of the four direction bits.
;       NextRoomId keeps its old value. [E7] is the direction bit stored by
;       CalculateNextRoom_TableJump. Whether either entry is reached in play is unverified.
; ----------------------------------------------------------------------------
CalculateNoNextRoom:
    LDA #$00
    STA $E7
    RTS
:
    ; Part of the loop of CalculateNextRoomForDoor below: try the next direction bit.
    ASL $00
    DEX
    JMP :+

; ----------------------------------------------------------------------------
; Sets NextRoomId to the room beyond the side Link is leaving by; loads the OW when there is none.
; IN:   [E7] = direction bit (1 right, 2 left, 4 down, 8 up), set by CalculateNextRoom
; OUT:  NextRoomId; $04E4 = the current room's unique room id (not read anywhere by that address);
;       bit 0 of CurPpuMask cleared; when no room exists: GameMode = 2, CurLevel = 0,
;       UndergroundExitType = 2
; NOTE: Used for every door type except 1 (in the OW every side counts as type 0, open).
; NOTE: The search for the bit has no end test: [E7] without any of bits 0-3 would loop forever.
; NOTE: In the OW, CheckMazes then gets the chance to change NextRoomId (rooms $61 and $1B).
; NOTE: The validity test is only "bit 7 clear". Rooms are $00-$7F, 16 per row, so only leaving
;       the top row upward, the bottom row downward, room $00 leftward or room $7F rightward is
;       caught; other left/right moves across a row end give a room on the neighbouring row.
; ----------------------------------------------------------------------------
CalculateNextRoomForDoor:
    ; Find X, the reverse direction index of the direction bit in [E7]:
    ; 1 (right) -> 3, 2 (left) -> 2, 4 (down) -> 1, 8 (up) -> 0.
    LDA #$01
    STA $00                     ; [00] = single-bit mask, shifted left each try.
    LDX #$03
:
    LDA $E7
    BIT $00
    BEQ :--                     ; No match: shift the mask, decrement X (code above the header).
    JSR GetUniqueRoomId
    STA $04E4                   ; Written here only; no other reference to this address was found.
    LDA NextRoomIdOffsets, X
    CLC
    ADC RoomId
    STA NextRoomId
    LDA CurLevel
    BNE :+
    JSR CheckMazes              ; OW only: the Lost Woods and Lost Hills may keep Link in the room.
:
    LDA NextRoomId
    BPL MaskCurPpuMaskGrayscale ; A valid room ($00-$7F): finish. Else fall through and load the OW.

; ----------------------------------------------------------------------------
; Leaves the current level for the overworld: mode 2 (load level) with CurLevel = 0.
; OUT:  GameMode = 2, submode 0, not updating; CurLevel = 0; UndergroundExitType = 2 (came out of a
;       dungeon); [E7] = 0; Tune0Request = $80 (stop the song); bit 0 of CurPpuMask cleared
; NOTE: Reached by fall-through when Link walks off the map (out of a dungeon's entrance), and by
;       jump from the last step of the end-level sequence (mode $12), hence the upstream name.
; NOTE: The OW room Link appears in is LevelEntranceRoomId (chosen by mode 3), not set here.
; ----------------------------------------------------------------------------
EndGameMode12:
    JSR EndGameMode             ; Returns A = 0.
    STA $E7
    STA CurLevel                ; Level 0 = the OW.
    LDA #$02
    STA GameMode
    STA UndergroundExitType     ; 2 = leaving a dungeon.
    LDA #$80                    ; Stop the song.
    STA Tune0Request

; ----------------------------------------------------------------------------
; Clears bit 0 (grayscale) of the PPU mask copy CurPpuMask.
; NOTE: Only the copy is changed here; the register is written elsewhere.
; ----------------------------------------------------------------------------
MaskCurPpuMaskGrayscale:
    LDA CurPpuMask
    AND #$FE
    STA CurPpuMask
    RTS

; ----------------------------------------------------------------------------
; Computes the id of the room next to the current one in a direction, with no checks.
; IN:   A = direction bit (1 right, 2 left, 4 down, 8 up)
; OUT:  A = RoomId + offset for that direction (may be outside $00-$7F); X = reverse direction
;       index; [00] = the single-bit mask that matched
; NOTE: With A = 0 the loop never ends (no guard).
; ----------------------------------------------------------------------------
CalcNextRoomByDir:
    LDX #$01
    STX $00                     ; [00] = single-bit mask.
    LDX #$03                    ; X = reverse direction index, starting with right.
:
    BIT $00                     ; Does the direction in A have this bit?
    BNE :+
    ASL $00
    DEX
    JMP :-
:
    LDA NextRoomIdOffsets, X
    CLC
    ADC RoomId
    RTS
