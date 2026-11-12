; ============================================================================
; FILE:     bank5/warps-stairs-caves.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7255-7407
; SUBJECTS: (none)
; TOPIC:    Stairs/cave/dungeon entrance detection, kill count save
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 CheckWarps     code
;   L0086 SetTargetMode  code
;   L0104 SaveKillCount  code
;   L0111 L1746E_Exit    code
;   L0114 HandleWarpOW   code
; ============================================================================

CheckWarps:
    ; If just came out of a cave, dungeon, or cellar; or if grid offset <> 0;
    ; then return.
    LDA UndergroundExitType
    ORA ObjGridOffset
    BNE L1746E_Exit

    ; If in OW room $22 and Link's X is not a multiple of 8, then return.
    ; This is a special case, because Level 6's entrance is wide.
    LDA CurLevel
    BNE @EnsureSquareX
    LDA RoomId
    CMP #$22
    BNE @EnsureSquareX
    LDA ObjX
    AND #$07
    BNE L1746E_Exit
    BEQ @EnsureSquareY

@EnsureSquareX:
    ; In other OW rooms and in UW, make sure X is a multiple of $10.
    LDA ObjX
    AND #$0F
    BNE L1746E_Exit

@EnsureSquareY:
    ; If Link's Y is not at ((multiple of $10) + $D), then return.
    LDA ObjY
    AND #$0F
    CMP #$0D
    BNE L1746E_Exit

    ; Check tile collision standing still.
    JSR GetCollidableTileStill

    ; If in OW, go handle the tile separately.
    LDA ObjCollidedTile
    LDY CurLevel
    BEQ HandleWarpOW

    ; In UW.
    ;
    ; If tile is not part of stairs square (tiles $70 to $74), return.
    CMP #$70
    BCC L1746E_Exit
    CMP #$74
    BCS L1746E_Exit

    ; Prepare to leave this room.
    JSR SaveKillCount
    LDA RoomId
    STA CellarSourceRoomId      ; Remember what room we were in.

    ; Look for a room in cellar array that has the current room as
    ; destination room A or B.
    LDX #$FF
:
    INX
    LDA LevelInfo_CellarRoomIdArray, X
    TAY
    LDA RoomId
    CMP LevelBlockAttrsA, Y
    BEQ :+
    CMP LevelBlockAttrsB, Y
    BNE :-
:
    ; Make the cellar found the current room, and the target mode 9.
    STY RoomId
    LDA #$09

SetTargetMode:
    STA ModeAfterStairs

    ; If the target mode is not 9 (as in it's a cave), then silence all sound.
    CMP #$09
    BEQ :+
    JSR SilenceAllSound
    STA Tune1Request

    ; Reset the flute timer.
    STA FluteTimer
:
    ; Go to mode $10.
    LDA #$10
    STA GameMode
    JSR MaskCurPpuMaskGrayscale
    JMP EndPrepareMode

SaveKillCount:
    LDA CurLevel
    BNE :+
    JMP SaveKillCountOW
:
    JSR SaveKillCountUW

L1746E_Exit:
    RTS

HandleWarpOW:
    ; Save the tile that Link is standing on; so that we know how
    ; to go underground.
    STA UndergroundEntranceTile

    ; If (tile < $70 or >= $74) except for $24 and $88; then return.
    CMP #$24
    BEQ :+
    CMP #$88
    BEQ :+
    CMP #$70
    BCC L1746E_Exit
    CMP #$74
    BCS L1746E_Exit

    ; If Link touched a stairs tile ($70 to $74), then
    ; use $70 to represent them all.
    LDA #$70
    STA ObjCollidedTile
:
    JSR SaveKillCount

    ; Get the cave index attribute.
    LDY RoomId
    LDA LevelBlockAttrsB, Y
    AND #$FC                    ; Cave index

    ; If < $40, go deal with a level.
    ; $40 means cave index $10. Levels have cave indexes 1 to 9.
    CMP #$40
    BCC @LoadLevel

    ; If attribute <> $50, go to mode $B for a regular cave.
    LDY #$0B
    CMP #$50
    BNE :+

    ; Attribute = $50 meaning cave index $14 (shortcuts).
    ; Go to mode $C for a shortcut cave.
    INY
:
    TYA
    JMP SetTargetMode

@LoadLevel:
    ; Load a level.
    ;
    ; Shift right by two to get the level number.
    LSR
    LSR
    STA CurLevel
    LDA RoomId
    STA LevelEntranceRoomId        ; Remember where we came from.
    LDA #$02                    ; Target mode is 2 to load a level.
    BNE SetTargetMode
