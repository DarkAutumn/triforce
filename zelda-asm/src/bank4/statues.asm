; ============================================================================
; FILE:     bank4/statues.asm            BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1571-1755
; SUBJECTS: (none)
; TOPIC:    Dungeon statue fireball emitters
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 StatueRoomLayouts                 data
;   L0021 StatueFireballCounts              data
;   L0024 StatueFireballStartTimes          data
;   L0027 StatuePatternToBasePositionIndex  data
;   L0032 StatueXs                          data  A list of coordinates divided into sets for each pattern.
;   L0037 StatueYs                          data  A list of coordinates divided into sets for each pattern.
;   L0040 UpdateStatues                     code
; ============================================================================

StatueRoomLayouts:
    .BYTE $24, $23

StatueFireballCounts:
    .BYTE $03, $01, $01

StatueFireballStartTimes:
    .BYTE $50, $80, $F0, $60

StatuePatternToBasePositionIndex:
    .BYTE $00, $04, $06

; A list of coordinates divided into sets for each pattern.
;
StatueXs:
    .BYTE $24, $C8, $24, $C8, $64, $88, $48, $A8

; A list of coordinates divided into sets for each pattern.
;
StatueYs:
    .BYTE $C0, $BC, $64, $5C, $94, $8C, $82, $86

UpdateStatues:
    ; If person fireballs are enabled (from the two fires), then
    ; go make fireballs with pattern 2.
    LDY #$02
    LDA PersonFireballsEnabled
    BNE @Shoot

    ; Else if the room layout has two or four statues that shoot fireballs,
    ; then go shoot fireballs with the corresponding pattern in Y register:
    ; - layout $23: pattern 1: 2 statues
    ; - layout $24: pattern 0: 4 statues
    JSR GetUniqueRoomId
    LDY #$01
:
    CMP StatueRoomLayouts, Y
    BEQ @Shoot
    DEY
    BPL :-
    RTS

@Shoot:
    ; Look for an empty monster slot.
    ; If none is found, return.
    ;
    ; Later, for each fireball, we'll call a routine to make the fireball
    ; object. This routine takes a shooter object, and instantiates
    ; a fireball object at the shooter's location.
    ;
    ; The object slot we set up here will behave as the shooter.
    ; For each fireball index, this object slot will have its coordinates
    ; changed.
    ;
    ; Note that this means that:
    ; 1. The first fireball made will be its own shooter.
    ;    Practically, there's nothing special about this.
    ; 2. If there is more than one fireball made in a frame,
    ;    then two of them will come from the same statue.
    TYA                         ; Save the fireball pattern.
    PHA
    JSR FindEmptyMonsterSlot
    CMP #$01
    PLA                         ; Restore the fireball pattern.
    BCS @Exit

    ; If we found a slot, but it < 6, return.
    CPY #$06
    BCC @Exit

    ; Store the object slot in [0A].
    ; Store fireball pattern in [0B].
    STY $0A
    STA $0B

    ; Look for the number of fireballs to make (minus 1) by pattern number.
    TAY
    LDX StatueFireballCounts, Y

@LoopFireball:
    ; For each fireball to make, indexed by X, (1 or 3 down to 0):
    ;
    ; Decrement the fireball timer at this index.
    LDY Statue_ObjFireballTimer, X
    DEY
    TYA
    STA Statue_ObjFireballTimer, X

    ; If it was not 0, then this fireball index has to keep waiting
    ; to make a fireball. Go loop again.
    INY
    BNE @NextLoopFireball

    ; If the random value for this fireball index >= $F0, go loop again.
    ; This fireball index will wait another $100 frames.
    LDA Random, X
    CMP #$F0
    BCS @NextLoopFireball

    ; Else choose one of four start times for the fireball based on
    ; the random value.
    AND #$03
    TAY
    LDA StatueFireballStartTimes, Y
    STA Statue_ObjFireballTimer, X

    ; The coordinate lists are divided into sets for each pattern.
    ;
    ; Look up the index of the base of the set for the pattern in use.
    ; Add it to the fireball index in X register to get the index of
    ; the coordinate we want within each coordinate list.
    LDY $0B                     ; Pattern
    TXA
    CLC
    ADC StatuePatternToBasePositionIndex, Y
    TAY
    TXA                         ; Save the fireball index.
    PHA

    ; Get Link's coordinates into [00] and [01].
    LDX #$00
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Switch to the new fireball's object slot.
    LDX $0A

    ; Look up and set the X coordinate for the fireball object.
    ; Also store it in [02].
    LDA StatueXs, Y
    STA $02
    STA ObjX, X

    ; Look up and set the Y coordinate for the fireball object.
    ; Also store it in [03].
    LDA StatueYs, Y
    STA $03
    STA ObjY, X

    ; Set mask value 3 in [04].
    LDA #$03
    STA $04

    ; Loop to compare each of Link's coordinates to the fireball's.
    ; Each time the distance < $18, the mask in [04] is shifted right.
    LDY #$01

@LoopAxis:
    LDA $0000, Y
    SEC
    SBC $0002, Y
    CMP #$18
    BPL :+
    CMP #$E8
    BMI :+
    LSR $04
:
    DEY
    BPL @LoopAxis

    ; If [04] <> 0, then its value is 1 or 3, meaning that it was shifted
    ; 0 or 1 time, instead of 2.
    ;
    ; This, in turn, means that the distance between Link and the
    ; fireball was >= $18 in at least one axis.
    ;
    ; So, we're OK to finish making the fireball.
    LDA $04
    BEQ :+
    LDA #$55
    JSR ShootFireball
:
    PLA                         ; Restore the fireball index.
    TAX

@NextLoopFireball:
    ; Bottom of the loop.
    ; Loop while fireball index >= 0.
    DEX
    BPL @LoopFireball

@Exit:
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF
