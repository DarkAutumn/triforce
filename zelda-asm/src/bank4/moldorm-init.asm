; ============================================================================
; FILE:     bank4/moldorm-init.asm       BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4771-4840
; SUBJECTS: Moldorm
; TOPIC:    Moldorm (object type $41) init: fills object slots $01-$0A with ten segments, two worms
;           of five, all starting at ($80, $70); the heads are slots $05 and $0A.
; SEE ALSO: bank4/moldorm.asm (Moldorm_Update: movement, direction propagation, segment death);
;           bank4/fairy-flyer-common.asm (Flyer_State2Chase, Flyer_State3Wander, Flyer_Move, Dir_EightWayDirs);
;           bank7/object-init-table.asm (Obj_Init loads ObjAttr and ObjHP before calling this).
; KEY FACTS:
;   - One Moldorm object becomes ten: slots $01-$0A all get type $41, position ($80, $70), direction
;     0, and the attribute byte and HP of slot $01. Moldorm_Init@LoopSegment
;   - Worm A is slots $01-$05 with its head in $05; worm B is slots $06-$0A with its head in $0A.
;     Only the heads get a direction here (random, 8-way); a segment with direction 0 is neither
;     moved, drawn nor hittable (Moldorm_Update). Moldorm_Init
;   - Every segment has flyer speed $80 (1 pixel every 2 frames in Flyer_Move) and flying state 2
;     (chase); the heads start with 1 turn left, so a head's first update with ObjTimer = 0 ends
;     the state and picks a new one. Moldorm_Init
;   - RoomObjCount is set to 8 although ten segments are made (reason unverified). Moldorm_Init
;   - RNG: head directions come from Random+5 and Random+10 (AND 7 into Dir_EightWayDirs). Moldorm_Init
; CONTENTS: (generated; L = line in this file)
;   L0034 Moldorm_Init  code  Creates the ten Moldorm segments (two worms of five) in object slots $01-$0A.
; ============================================================================

; ----------------------------------------------------------------------------
; Creates the ten Moldorm segments (two worms of five) in object slots $01-$0A.
; IN:   ObjAttr+1 and ObjHP+1 already loaded for type $41 by Obj_Init. X is not used.
; OUT:  slots $01-$0A initialized; FlyerMaxSpeedFrac = $80; RoomObjCount = 8.
; NOTE: It clears ObjUninitialized for all ten slots, so it runs once per room. Copying from slot
;       $01 is right only if the room's Moldorm object is in slot $01 when this runs (unverified).
;       HP is $20 per segment and the attribute byte $C1 (type $41 in Obj_TypeHpPairs and
;       Obj_TypeAttributes).
; ----------------------------------------------------------------------------
Moldorm_Init:
    ; Loop over the 10 segments, Y = 9 down to 0. The arrays are addressed at +1,
    ; so object slots $0A down to $01 are filled.
    LDY #$09

@LoopSegment:
    ; Start at position ($80, $70).
    LDA #$80
    STA a:ObjX+1, Y
    LDA #$70
    STA a:ObjY+1, Y

    ; No direction (the segment stays idle and invisible until it gets one), no deferred bounce
    ; direction, metastate 0 (no spawn cloud), and flagged initialized.
    LDA #$00
    STA a:ObjDir+1, Y
    STA Moldorm_ObjBounceDir+1, Y
    STA ObjMetastate+1, Y
    STA ObjUninitialized+1, Y

    ; Copy the attribute byte from slot $01.
    LDA ObjAttr+1
    STA ObjAttr+1, Y

    ; Copy HP from slot $01.
    LDA ObjHP+1
    STA ObjHP+1, Y

    ; Flyer speed $80 (Flyer_Move moves 1 pixel every 2 frames), flying state 2 (Moldorm_Chase).
    LDA #$80
    STA Flyer_ObjSpeed+1, Y
    LDA #$02
    STA Flyer_ObjFlyingState+1, Y
    LDA #$41                    ; Every segment is object type $41 (Moldorm).
    STA ObjType+1, Y

    DEY
    BPL @LoopSegment

    ; Choose a random 8-way direction for each head: slot $05 (worm of slots $01-$05)
    ; and slot $0A (worm of slots $06-$0A). Also store it as the head's old direction,
    ; which the next segment takes over at the first propagation.
    LDA Random+5
    AND #$07
    TAY
    LDA Dir_EightWayDirs, Y
    STA ObjDir+5
    STA Moldorm_ObjOldDir+5
    LDA Random+10
    AND #$07
    TAY
    LDA Dir_EightWayDirs, Y
    STA ObjDir+10
    STA Moldorm_ObjOldDir+10

    ; Give each head 1 turn left in its flying state. Flyer_State2Chase decrements the count before
    ; turning, so on a head's first update with ObjTimer = 0 the count reaches 0, the state ends
    ; without a turn, and Moldorm_ChangeFlyingState picks a state at random and sets ObjTimer
    ; to $10.
    LDA #$01
    STA Flyer_ObjTurns+5
    STA Flyer_ObjTurns+10
    LDA #$80                    ; Flyer top speed $80. Moldorm's flight table has no speed-up
    STA FlyerMaxSpeedFrac       ; state, so its speed stays at the $80 set above.
    LDA #$08                    ; Monster count 8, not 10 (reason unverified).
    STA RoomObjCount
    RTS
