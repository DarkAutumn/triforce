; ============================================================================
; FILE:     bank4/moldorm-init.asm       BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4771-4840
; SUBJECTS: (none)
; TOPIC:    InitMoldorm: ten segments in two worms
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0012 InitMoldorm  code
; ============================================================================

InitMoldorm:
    ; Loop to make 10 moldorm segments, from 9 to 0, indexed by Y register.
    ; Object slots $A to 1 will be used.
    LDY #$09

@LoopSegment:
    ; Start at position ($80, $70).
    LDA #$80
    STA a:ObjX+1, Y
    LDA #$70
    STA a:ObjY+1, Y

    ; Start with no direction, no deferred bounce direction,
    ; and ready to update.
    LDA #$00
    STA a:ObjDir+1, Y
    STA Moldorm_ObjBounceDir+1, Y
    STA ObjMetastate+1, Y
    STA ObjUninitialized+1, Y

    ; Copy object attributes from the first monster slot.
    LDA ObjAttr+1
    STA ObjAttr+1, Y

    ; Copy HP from the first monster slot.
    LDA ObjHP+1
    STA ObjHP+1, Y

    ; Start with flying speed $80, and in flying state 2.
    LDA #$80
    STA Flyer_ObjSpeed+1, Y
    LDA #$02
    STA Flyer_ObjFlyingState+1, Y
    LDA #$41                    ; All segments are considered Moldorm.
    STA ObjType+1, Y

    ; Bottom of the loop.
    DEY
    BPL @LoopSegment

    ; Choose a random 8-way direction for the head segments
    ; in slots 4 and 9.
    LDA Random+5
    AND #$07
    TAY
    LDA Directions8, Y
    STA ObjDir+5
    STA Moldorm_ObjOldDir+5     ; Also store the current direction as the old direction.
    LDA Random+10
    AND #$07
    TAY
    LDA Directions8, Y
    STA ObjDir+10
    STA Moldorm_ObjOldDir+10    ; Also store the current direction as the old direction.

    ; Set minimum number of turns 1 for flying states.
    ;
    ; By setting this, the first time that a flight control routine
    ; (Flyer_Chase, Flyer_Wander) is called and returns;
    ; that flying state will immediately end, and object timer
    ; will be left 0: ready for moldorm to decide a flying state
    ; randomly and set timer to $10.
    LDA #$01
    STA Flyer_ObjTurns+5
    STA Flyer_ObjTurns+10
    LDA #$80                    ; The maximum speed for Moldorm is $80.
    STA FlyerMaxSpeedFrac
    LDA #$08                    ; 8 instead of 10, because 8 is the max kill count to save.
    STA RoomObjCount
    RTS
