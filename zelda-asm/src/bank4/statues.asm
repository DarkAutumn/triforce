; ============================================================================
; FILE:     bank4/statues.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1571-1755
; SUBJECTS: Statue
; TOPIC:    Dungeon statues that shoot fireballs at Link: which rooms have them, where the fireballs start,
;           the per-statue timers, and the same code reused for the two fires beside an attacked person.
; SEE ALSO: bank4/monster-shots-fireballs.asm (Fireball_Shoot, Fireball_Update: how the fireball aims and
;           flies); bank7/mode5-play-update.asm (the dungeon-only call, once per frame after the object
;           loop); bank1/underworld-person-update.asm and bank1/monster-collisions.asm (set
;           IsPersonFireballsEnabled); bank7/tile-obj-tiles.asm (Room_GetUniqueRoomId = room layout id).
; KEY FACTS:
;   - Statues are not objects. Statue_Update runs once per frame from the play update whenever
;     CurLevel <> 0 (no game mode test: modes 5, 9, $A, $B, $C) and creates fireballs (type $55) at
;     fixed coordinates. No weapon test exists for a statue. Statue_Update
;   - Only two room layouts have shooting statues: layout $24 (4 statues, pattern 0) and layout $23
;     (2 statues, pattern 1). Statue_RoomLayoutIds, Statue_FireballCountsMinus1
;   - Pattern 2 (2 fireballs from ($48,$82) and ($A8,$86)) is used in any dungeon room once
;     IsPersonFireballsEnabled is set (the room's person was attacked); it takes priority over the
;     layout test. Statue_Update
;   - Each statue has its own timer in Statue_FireballTimers (0 on room entry, so every statue may fire
;     on the first frame). When a timer runs out: with Random,X >= $F0 (X = statue number) nothing is
;     shot and the statue waits $100 frames; else the timer is reloaded with $50, $80, $F0 or $60
;     frames (Random,X AND 3), so the next chance comes 81, 129, 241 or 97 frames later.
;     Statue_Update@NextStatue, Statue_FireballStartTimes
;   - A statue whose timer ran out does not shoot when Link is within $18 pixels of the fireball's
;     start point on both axes (LinkX - X and LinkY - Y each in -$18..+$17); the timer is reloaded
;     anyway. Statue_Update@LoopAxis
;   - Nothing happens at all (timers do not even count) on a frame where the highest empty object slot
;     of $01-$0B is below $06, or no slot is empty. Statue_Update@Shoot
;   - Statue fireballs are not counted in MonsterShotCount and have no limit but a free monster slot.
;     Fireball_Shoot (bank4/monster-shots-fireballs.asm)
;   - When several statues shoot in one frame, the frame's first fireball ends up at the start point
;     of the last statue that shot (its slot doubles as the "shooter"). Statue_Update@Shoot
; CONTENTS: (generated; L = line in this file)
;   L0045 Statue_RoomLayoutIds               data  TABLE: room layout ids (Room_GetUniqueRoomId) whose statues shoot; index = fireball pattern; value = layout...
;   L0049 Statue_FireballCountsMinus1        data  TABLE: number of fireball sources minus 1; index = fireball pattern (0, 1, 2); value = last statue number.
;   L0054 Statue_FireballStartTimes          data  TABLE: reload values for Statue_FireballTimers; index = Random,X AND 3; value = frames.
;   L0058 Statue_PatternBasePositionIndexes  data  TABLE: where each pattern's coordinates start in Statue_FireballXs/StatueYs; index = fireball pattern; valu...
;   L0064 Statue_FireballXs                  data  TABLE: X where a fireball starts; index = pattern base + statue number; value = pixels.
;   L0069 Statue_FireballYs                  data  TABLE: Y where a fireball starts; index = pattern base + statue number; value = pixels.
;   L0086 Statue_Update                      code  Counts down each statue's timer and shoots a fireball (type $55) at Link from statues that are due.
; ============================================================================

; TABLE: room layout ids (Room_GetUniqueRoomId) whose statues shoot; index = fireball pattern; value = layout id.
Statue_RoomLayoutIds:
    .BYTE $24, $23              ; pattern 0: layout $24 (4 statues); pattern 1: layout $23 (2 statues)

; TABLE: number of fireball sources minus 1; index = fireball pattern (0, 1, 2); value = last statue number.
Statue_FireballCountsMinus1:
    .BYTE $03, $01, $01         ; pattern 0: 4 sources; pattern 1: 2; pattern 2 (person's fires): 2

; TABLE: reload values for Statue_FireballTimers; index = Random,X AND 3; value = frames.
; The next chance to shoot comes (value + 1) frames after the reload: 81, 129, 241 or 97 frames.
Statue_FireballStartTimes:
    .BYTE $50, $80, $F0, $60

; TABLE: where each pattern's coordinates start in Statue_FireballXs/StatueYs; index = fireball pattern; value = index.
Statue_PatternBasePositionIndexes:
    .BYTE $00, $04, $06

; TABLE: X where a fireball starts; index = pattern base + statue number; value = pixels.
; Entries 0-3: pattern 0, statues 0-3. Entries 4-5: pattern 1, statues 0-1. Entries 6-7: pattern 2
; (the person's fires), sources 0-1. Fireball_Shoot adds 4 to this X for the fireball it creates.
Statue_FireballXs:
    .BYTE $24, $C8, $24, $C8, $64, $88, $48, $A8

; TABLE: Y where a fireball starts; index = pattern base + statue number; value = pixels.
; Entries grouped as in Statue_FireballXs.
Statue_FireballYs:
    .BYTE $C0, $BC, $64, $5C, $94, $8C, $82, $86

; ----------------------------------------------------------------------------
; Counts down each statue's timer and shoots a fireball (type $55) at Link from statues that are due.
; IN:   nothing (called once per frame from the play update, only when CurLevel <> 0; the caller
;       does not test the game mode, so cellars count too).
; OUT:  may create fireballs in empty monster slots; writes Statue_FireballTimers; uses [00]-[04], [0A],
;       [0B]; X, Y, A not preserved.
; NOTE: Does nothing (timers frozen) unless the highest empty slot of $01-$0B is $06 or above.
; NOTE: RNG: Random,X with X = statue number 0-3 (the bytes Random to Random+3). >= $F0: no shot and a
;       $100-frame wait; else bits 0-1 pick the reload time from Statue_FireballStartTimes.
; NOTE: No shot when Link is within $18 pixels of the start point on both axes.
; NOTE: When several statues shoot in the same frame, the frame's first fireball sits in the slot
;       used as the "shooter" and is moved to each later statue's start point (without the +4 in
;       X); it ends up at the last one. The later fireballs start at their own statues.
; ----------------------------------------------------------------------------
Statue_Update:
    ; If the room's person was attacked (IsPersonFireballsEnabled), use pattern 2: two fireball
    ; sources at ($48,$82) and ($A8,$86), left and right of the person.
    LDY #$02
    LDA IsPersonFireballsEnabled
    BNE @Shoot

    ; Else look for the room's layout id in Statue_RoomLayoutIds:
    ; - layout $23: pattern 1: 2 statues
    ; - layout $24: pattern 0: 4 statues
    ; Any other layout: return.
    JSR Room_GetUniqueRoomId         ; A := layout id (LevelBlockAttrsD AND $3F)
    LDY #$01
:
    CMP Statue_RoomLayoutIds, Y
    BEQ @Shoot
    DEY
    BPL :-
    RTS

@Shoot:
    ; Y = fireball pattern.
    ;
    ; Find the highest empty slot of $0B down to $01. If there is none, return.
    ;
    ; Fireball_Shoot needs a "shooter" object: it creates the fireball at the shooter's coordinates
    ; (X + 4). The empty slot found here plays the shooter: before each shot its ObjX/ObjY are set to
    ; the statue's coordinates. Fireball_Shoot then searches for an empty slot itself, and because this
    ; one is still empty and the highest, the first fireball of the frame is created in this very
    ; slot (it is its own shooter; ObjX/ObjY survive the slot's reset). A second fireball in the same
    ; frame overwrites the coordinates of that first fireball before being created in another slot.
    TYA                         ; Save the fireball pattern.
    PHA
    JSR Obj_FindEmptyMonsterSlot    ; found: A = 0, Y = slot. None: A = type in slot 1 (nonzero).
    CMP #$01                    ; C set when no slot was found (A >= 1).
    PLA                         ; Restore the fireball pattern (C is kept).
    BCS @Exit

    ; If the slot found is below $06, return. As the search goes downward, this means slots $06-$0B
    ; are all in use.
    CPY #$06
    BCC @Exit

    ; [0A] := the shooter slot; [0B] := the fireball pattern.
    STY $0A
    STA $0B

    ; X := number of fireball sources minus 1 for this pattern.
    TAY
    LDX Statue_FireballCountsMinus1, Y

@NextStatue:
    ; For each statue number X (3 or 1 down to 0):
    ;
    ; Decrement the statue's timer (it wraps from 0 to $FF).
    LDY Statue_FireballTimers, X
    DEY
    TYA
    STA Statue_FireballTimers, X

    ; If the timer was not 0 before the decrement, this statue keeps waiting.
    INY
    BNE @SkipStatue

    ; The timer ran out (it now holds $FF). If Random,X >= $F0, do not shoot: the statue waits
    ; another $100 frames.
    LDA Random, X
    CMP #$F0
    BCS @SkipStatue

    ; Else reload the timer with one of four times chosen by bits 0-1 of the same random byte.
    AND #$03
    TAY
    LDA Statue_FireballStartTimes, Y
    STA Statue_FireballTimers, X

    ; Y := index into Statue_FireballXs/StatueYs = the pattern's base index + statue number.
    LDY $0B                     ; Pattern
    TXA
    CLC
    ADC Statue_PatternBasePositionIndexes, Y
    TAY
    TXA                         ; Save the statue number.
    PHA

    ; [00], [01] := Link's X, Y.
    LDX #$00
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Switch to the shooter slot.
    LDX $0A

    ; Put the shooter slot at the statue's start point; keep the point in [02], [03].
    LDA Statue_FireballXs, Y
    STA $02
    STA ObjX, X
    LDA Statue_FireballYs, Y
    STA $03
    STA ObjY, X

    ; [04] := 3, a mask that is shifted right once for each axis on which Link is close.
    LDA #$03
    STA $04

    ; For the Y axis (index 1), then the X axis (index 0): A := Link's coordinate - start point.
    ; If A is in $00-$17 or $E8-$FF (distance from -$18 to +$17 pixels), shift the mask right.
    LDY #$01

@LoopAxis:
    LDA $0000, Y
    SEC
    SBC $0002, Y
    CMP #$18
    BPL :+                      ; Skip unless (A - $18) has bit 7 set: A in $00-$17 or $98-$FF.
    CMP #$E8
    BMI :+                      ; Skip unless (A - $E8) has bit 7 clear: A in $E8-$FF or $00-$67.
    LSR $04
:
    DEY
    BPL @LoopAxis

    ; [04] is 3 or 1 when Link is $18 pixels or more away on at least one axis: shoot.
    ; [04] is 0 when Link is close on both axes: no fireball (the timer stays reloaded).
    LDA $04
    BEQ :+
    LDA #$55                    ; Fireball object type
    JSR Fireball_Shoot           ; Fails silently when no monster slot is empty.
:
    PLA                         ; Restore the statue number.
    TAX

@SkipStatue:
    DEX
    BPL @NextStatue

@Exit:
    RTS

; DATA: filler ($FF), 3 bytes; not referenced.
    .BYTE $FF, $FF, $FF
