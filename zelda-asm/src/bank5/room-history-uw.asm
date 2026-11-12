; ============================================================================
; FILE:     bank5/room-history-uw.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4019-4127
; SUBJECTS: (none)
; TOPIC:    Dungeon room history, kill counts, non-recurring monsters
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 ModifyObjCountByHistoryUW  code
;   L0065 SaveKillCountUW            code
; ============================================================================

ModifyObjCountByHistoryUW:
    ; Look for the room in the history.
    LDY #$05
    LDA RoomId
:
    CMP RoomHistory, Y
    BEQ @CalcObjCount           ; If found, go subtract kill count from object count.
    DEY
    BPL :-

    ; It wasn't found. Check the kill count in the room flags.
    JSR GetRoomFlags
    AND #$C0
    CMP #$C0
    BNE @CalcObjCount           ; If not all defeated, go subtract kill count from object count.

    ; The room is not in history, and all foes were defeated.
    ; Does the object list ID indicate a non-recurring object?
    LDA $02
    CMP #$32
    BCC :+
    CMP #$3A
    BEQ :+
    CMP #$3B
    BEQ :+
    CMP #$49
    BCC @ResetObjCount          ; If it's non-recurring, go reset the object count.
:
    ; The object list ID is for recurring objects.
    ; So, reset the kill count in room flags and level block.
    LDA ($00), Y
    AND #$3F
    STA ($00), Y
    LDA #$00
    STA LevelKillCounts, Y
    RTS

@CalcObjCount:
    ; Subtract the level block's kill count from the object count.
    LDY RoomId
    LDA $03
    SEC
    SBC LevelKillCounts, Y
    BPL :+                      ; If the result >= 0, go set the object count to this.

@ResetObjCount:
    LDA #$00                    ; Else, reset object count and object list ID.
    STA $02
:
    STA $03
    RTS

SaveKillCountUW:
    JSR GetRoomFlags
    AND #$3F                    ; Reset kill count part of the room's flags.
    STA ($00), Y
    LDA RoomObjCount
    BEQ @StoreMax               ; If no monsters were made in this room, go store the max kill count.
    LDA RoomKillCount
    BEQ :+                      ; If RoomKillCount = 0, go compare it to RoomFoeCount.

    ; If the object template type is ...
    ; >= $32 and
    ; <> $3A and
    ; <> $3B and
    ; <  $49,
    ; then it refers to a non-recurring foe that shouldn't be made
    ; again. So, go store the max kill count for the room.
    ; 
    LDY RoomObjTemplateType
    CPY #$32
    BCC :+
    CPY #$3A
    BEQ :+
    CPY #$3B
    BEQ :+
    CPY #$49
    BCC @StoreMax
:
    ; Compare RoomKillCount and RoomFoeCount.
    CMP RoomObjCount
    BCS @StoreMax               ; If RoomKillCount >= RoomFoeCount, go store the max kill count for the room.

    ; RoomKillCount < _RoomObjCount.
    ; Add RoomKillCount to level kill count for this room.
    LDY RoomId
    CLC
    ADC LevelKillCounts, Y
    STA LevelKillCounts, Y
    CMP #$03                    ; Cap the kill count to 2.
    BCC :+
    LDA #$02
:
    CLC                         ; Shift the adjusted kill count (up to 2) into the top 2 bits.
    ROR
    ROR
    ROR
    JMP @CombineWithFlags       ; Go combine this mask with the room's flags.

@StoreMax:
    LDY RoomId                  ; For this room in the level block, set kill count to max ($F).
    LDA #$0F
    STA LevelKillCounts, Y
    LDA #$C0                    ; For this room in world flags, set kill count to max (3).

@CombineWithFlags:
    ORA ($00), Y                ; Combine the mask with the room's flags.
    STA ($00), Y
    RTS
