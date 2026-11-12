; ============================================================================
; FILE:     bank7/flute.asm              BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 2496-2598
; SUBJECTS: (none)
; TOPIC:    Using the recorder (flute): tune, $98-frame freeze, and its effect by place: in the UW a
;           "flute used" flag (Digdogger reads it); in the OW a flute secret in the listed rooms,
;           the whirlwind elsewhere.
; SEE ALSO: bank5/wield-items.asm (WieldItem_JumpTable calls WieldFlute for selected slot 5);
;           bank7/mode5-play-update.asm (UpdateMode5Play: the freeze while FluteTimer runs);
;           bank1/whirlwind.asm (SummonWhirlwind: conditions and destinations);
;           bank7/flute-secret-pond.asm (object $5E: the pond drains and stairs appear);
;           bank4/digdogger.asm (FluteUsedState splits Digdogger); bank5/warps-stairs-caves.asm
;           (SetTargetMode clears FluteTimer).
; KEY FACTS:
;   - Every use requests the flute tune (Tune1Request = $10) and sets FluteTimer = $98 (152 frames),
;     wherever Link is (WieldFlute). While it runs, play is not updated at all (UpdateMode5Play).
;   - UW (CurLevel <> 0): FluteUsedState goes 0 -> 1; a use while it is nonzero changes nothing
;     (WieldFlute@FlagFluteUsed). Digdogger's update reads it.
;   - OW, GameMode <> 5 (the play update also runs mode $B, a cave, and mode $C, the shortcut
;     stairs room): tune and freeze only (WieldFlute).
;   - OW rooms with a flute secret (FluteRoomSecretsOW): $42 in Q1 only; $06, $29, $2B, $30, $3A,
;     $3C, $58, $60, $6E, $72 in Q2 only. There the flute makes a flute secret object (type $5E)
;     and does not summon the whirlwind (WieldFlute@RevealSecret).
;   - The secret is made only when SecretColorCycle = 0 (not already started in this room) and one of
;     object slots $01-$09 is empty; the highest empty slot is used (WieldFlute@FindEmptySlot).
;   - Everywhere else in the OW, including $42 in Q2 and the ten Q2 rooms in Q1, the flute calls
;     SummonWhirlwind (WieldFlute@SummonWhirlwind).
; CONTENTS: (generated; L = line in this file)
;   L0036 FluteRoomSecretsOW  data  TABLE: overworld rooms where the flute reveals a secret; index = 0-10; value = room id.
;   L0049 WieldFlute          code  Uses the flute: plays the tune, freezes play for $98 frames, then acts by level, room and quest.
;   L0134 L1EFCF_Exit         code  Shared return; also the early exit of Walker_Move (bank7/walker-move.asm).
; ============================================================================

; TABLE: overworld rooms where the flute reveals a secret; index = 0-10; value = room id.
; Entry 0 ($42) is the Q1 room; entries 1-10 are the Q2 rooms (WieldFlute tests the quest).
FluteRoomSecretsOW:
    .BYTE $42, $06, $29, $2B, $30, $3A, $3C, $58
    .BYTE $60, $6E, $72

; ----------------------------------------------------------------------------
; Uses the flute: plays the tune, freezes play for $98 frames, then acts by level, room and quest.
; OUT:  FluteTimer = $98; Tune1Request = $10; bank 5 switched in after the whirlwind path.
; NOTE: UW: sets FluteUsedState to 1 if it was 0. Nothing else happens here.
; NOTE: OW, mode 5 only: in a FluteRoomSecretsOW room of the current quest, creates object $5E (flute
;       secret) in the highest empty slot of $09-$01 unless SecretColorCycle <> 0; otherwise calls
;       SummonWhirlwind (which has its own conditions).
; NOTE: The object and the whirlwind are created in this frame but first update when the freeze ends.
; ----------------------------------------------------------------------------
WieldFlute:
    LDA #$10                    ; Flute tune
    STA Tune1Request

    ; Freeze play for $98 (152) frames.
    LDA #$98
    STA FluteTimer

    ; In the UW, only flag that the flute was used.
    LDA CurLevel
    BNE @FlagFluteUsed

    ; In the OW outside mode 5 (a cave, mode $B, or the shortcut stairs room, mode $C), nothing more.
    LDA GameMode
    CMP #$05
    BNE @Exit

    ; Push the quest number: 0 = Q1, 1 = Q2.
    LDY CurSaveSlot
    LDA QuestNumbers, Y
    PHA

    ; Look for the current room among the 11 entries (index $0A down to 0) of the secret rooms.
    LDA RoomId
    LDY #$0A

@FindSecretRoom:
    CMP FluteRoomSecretsOW, Y
    BEQ @Found
    DEY
    BPL @FindSecretRoom
    BMI @FluteSecretNotFound    ; Always taken.

@Found:
    ; Entry 0 (room $42) has its secret in Q1 only: in Q2 summon the whirlwind instead.
    CPY #$00
    BNE @Normal
    PLA
    BNE @SummonWhirlwind        ; Q2
    BEQ @RevealSecret           ; Q1

@Normal:
    ; Entries 1-10 have their secret in Q2 only: in Q1 summon the whirlwind instead.
    PLA
    BEQ @SummonWhirlwind        ; Q1

@RevealSecret:
    ; If the secret's color cycle has started or finished in this room, return.
    LDA SecretColorCycle
    BNE @Exit

    ; Look for an empty object slot from $09 down to $01 (Y = 8 down to 0, indexing from slot 1).
    LDY #$09

@FindEmptySlot:
    DEY
    BMI @Exit                   ; No empty slot: nothing happens (the tune and freeze still do).
    LDA ObjType+1, Y
    BNE @FindEmptySlot

    ; Make a flute secret object (type $5E) there. It runs the rest of the secret.
    LDA #$5E
    STA ObjType+1, Y

@Exit:
    RTS

@FluteSecretNotFound:
    ; The room is not in the list. Discard the quest number.
    PLA

@SummonWhirlwind:
    LDA #$01
    JSR SwitchBank
    JSR SummonWhirlwind
    LDA #$05                    ; Switch back to bank 5, where the caller (WieldItem) lives.
    JMP SwitchBank

@FlagFluteUsed:
    ; UW: FluteUsedState 0 -> 1; left alone if already nonzero.
    LDA FluteUsedState
    BNE L1EFCF_Exit
    INC FluteUsedState

; Shared return; also the early exit of Walker_Move (bank7/walker-move.asm).
L1EFCF_Exit:
    RTS
