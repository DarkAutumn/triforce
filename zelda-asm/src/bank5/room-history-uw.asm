; ============================================================================
; FILE:     bank5/room-history-uw.asm    BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 4019-4127
; SUBJECTS: RoomHistory
; TOPIC:    The dungeon side of the room history: how many monsters a dungeon room makes on entry
;           (kills banked in LevelKillCounts, non-recurring monsters) and how its kills are saved on
;           leaving.
; SEE ALSO: bank5/room-history-ow.asm (the overworld versions); bank5/enter-room.asm
;           (Room_EnterCommon builds list id [02] and count [03], calls RoomHistory_ModifyObjCountUW,
;           and stores RoomObjTemplateType); bank5/warps-stairs-caves.asm (RoomHistory_SaveKillCount);
;           bank7/init-mode5-play.asm (Room_RunEntryTasks adds the room to
;           RoomHistory); bank7/init-mode-dispatch.asm (RoomHistory_Clear; LevelKillCounts zeroed when
;           a level loads); bank5/boss-sound-door-face.asm (the boss room's "all dead" flag stops
;           the roar).
; KEY FACTS:
;   - Two stores: LevelKillCounts[RoomId] (RAM, per loaded level; $0F = all dead) is what reduces the
;     monster count; room flags bits 6-7 (saved) hold min(count, 2), or 3 for all dead.
;     RoomHistory_SaveKillCountUW, RoomHistory_ModifyObjCountUW
;   - On entry the room makes [03] - LevelKillCounts[RoomId] monsters (none if negative), unless the
;     room is outside the 6-room history and its flags say all dead. RoomHistory_ModifyObjCountUW
;   - All dead and out of the history: a room whose monster list id is $32-$48 except $3A and $3B
;     (non-recurring: bosses and the like) makes nothing, permanently; any other room has its flags
;     bits 6-7 and LevelKillCounts cleared and refills. RoomHistory_ModifyObjCountUW
;   - Partial flags (1 or 2 in bits 6-7) never reduce the count by themselves; after a level reload
;     (LevelKillCounts zeroed) a partly cleared room is full again. RoomHistory_ModifyObjCountUW
;   - A room is saved as all dead when it made no monsters, when RoomKillCount >= RoomObjCount, or
;     when at least one kill was made and the room's first monster type (RoomObjTemplateType) is
;     non-recurring. RoomHistory_SaveKillCountUW
;   - Lanmola ($3A, $3B) and Dodongo type $31 are outside the non-recurring range. RoomHistory_SaveKillCountUW
; CONTENTS: (generated; L = line in this file)
;   L0050 RoomHistory_ModifyObjCountUW  code  Reduces the monster count of the dungeon room being entered by the kills banked for it.
;   L0110 RoomHistory_SaveKillCountUW   code  Banks the kills made in the dungeon room being left: LevelKillCounts and room flags bits 6-7.
; ============================================================================

; ----------------------------------------------------------------------------
; Reduces the monster count of the dungeon room being entered by the kills banked for it.
; IN:   [02] = monster list id; [03] = monster count from the room attributes; RoomId
; OUT:  [03] = monsters to make; [02] = 0 as well when the subtraction went negative or the room is
;       a cleared non-recurring one; may clear room flags bits 6-7 and LevelKillCounts[RoomId]
; NOTE: Cases, in order:
;         - room in RoomHistory (6 entries), or flags bits 6-7 <> 3: [03] - LevelKillCounts[RoomId];
;         - else (all dead, not in the history) list id in $32-$48 except $3A, $3B: nothing, ever;
;         - else: flags bits 6-7 and LevelKillCounts[RoomId] are cleared, the full set is made.
;       LevelKillCounts = $0F (all dead) goes negative for any count below $0F, so a cleared room in
;       the history is empty. The non-recurring ids are object types: $32 Dodongo, $33/$34 Gohma,
;       $35 rupee stash, $36 Grumble, $37 Zelda, $38/$39 Digdogger, $3C Manhandla, $3D Aquamentus,
;       $3E Ganon, $3F guard fire, $40 standing fire, $41 Moldorm, $42-$45 Gleeok, $46 Gleeok head, $47/$48
;       Patra. $3A/$3B (Lanmola) are excluded; $31 (Dodongo) is below the range.
; ----------------------------------------------------------------------------
RoomHistory_ModifyObjCountUW:
    ; Look for the room in the history (6 entries).
    LDY #$05
    LDA RoomId
:
    CMP RoomHistory, Y
    BEQ @CalcObjCount           ; Found: subtract the banked kills.
    DEY
    BPL :-

    ; Not in the history. Bits 6-7 of the room flags = 3 means all dead.
    JSR RoomFlags_Get
    AND #$C0
    CMP #$C0
    BNE @CalcObjCount           ; Not all dead: subtract the banked kills.

    ; Not in the history and all dead. Is the monster list id a non-recurring type
    ; ($32-$48 except $3A and $3B)?
    LDA $02
    CMP #$32
    BCC :+
    CMP #$3A
    BEQ :+
    CMP #$3B
    BEQ :+
    CMP #$49
    BCC @ResetObjCount          ; Non-recurring: make nothing.
:
    ; Recurring monsters: forget the kills (flags bits 6-7 and the banked count); [03] stays full.
    LDA ($00), Y
    AND #$3F
    STA ($00), Y
    LDA #$00
    STA LevelKillCounts, Y
    RTS

@CalcObjCount:
    ; Subtract the kills banked for this room from the monster count.
    LDY RoomId
    LDA $03
    SEC
    SBC LevelKillCounts, Y
    BPL :+                      ; Result >= 0: that many monsters are made.

@ResetObjCount:
    LDA #$00                    ; No monsters: clear the list id and the count.
    STA $02
:
    STA $03
    RTS

; ----------------------------------------------------------------------------
; Banks the kills made in the dungeon room being left: LevelKillCounts and room flags bits 6-7.
; IN:   RoomKillCount; RoomObjCount; RoomObjTemplateType; RoomId
; OUT:  all dead: LevelKillCounts[RoomId] = $0F and flags bits 6-7 = 3; otherwise
;       LevelKillCounts[RoomId] += RoomKillCount and flags bits 6-7 = min(that sum, 2)
; NOTE: All dead when (a) RoomObjCount = 0 (the room made no monsters), (b) RoomKillCount <> 0 and
;       the first monster's type is $32-$48 except $3A and $3B (one kill clears a boss room), or
;       (c) RoomKillCount >= RoomObjCount.
; ----------------------------------------------------------------------------
RoomHistory_SaveKillCountUW:
    JSR RoomFlags_Get
    AND #$3F                    ; Clear the kill count bits (6-7) of the room's flags.
    STA ($00), Y
    LDA RoomObjCount
    BEQ @StoreMax               ; The room made no monsters: all dead.
    LDA RoomKillCount
    BEQ @CompareCounts          ; No kills: skip the non-recurring test.

    ; At least one kill. If the first monster's type is non-recurring ($32-$48 except $3A and
    ; $3B), the room is all dead.
    LDY RoomObjTemplateType
    CPY #$32
    BCC @CompareCounts
    CPY #$3A
    BEQ @CompareCounts
    CPY #$3B
    BEQ @CompareCounts
    CPY #$49
    BCC @StoreMax

@CompareCounts:
    ; A = RoomKillCount.
    CMP RoomObjCount
    BCS @StoreMax               ; Killed everything the room made: all dead.

    ; Some monsters are left: add the kills of this visit to the banked count.
    LDY RoomId
    CLC
    ADC LevelKillCounts, Y
    STA LevelKillCounts, Y
    CMP #$03                    ; The flags hold at most 2 for a partial count.
    BCC :+
    LDA #$02
:
    CLC                         ; Move the value (0-2) from bits 0-1 to bits 6-7.
    ROR
    ROR
    ROR
    JMP @CombineWithFlags

@StoreMax:
    LDY RoomId                  ; Banked count = $0F: all dead.
    LDA #$0F
    STA LevelKillCounts, Y
    LDA #$C0                    ; Flags bits 6-7 = 3: all dead.

@CombineWithFlags:
    ORA ($00), Y                ; Put the kill count bits into the room's flags.
    STA ($00), Y
    RTS
