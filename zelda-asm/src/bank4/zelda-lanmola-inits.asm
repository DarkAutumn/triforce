; ============================================================================
; FILE:     bank4/zelda-lanmola-inits.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9456-9524
; SUBJECTS: Lanmola, Zelda
; TOPIC:    Init routines: Zelda (type $37) with her four guard fires (type $3F) and their start
;           positions; Lanmola (types $3A, $3B), which fills slots $01-$0A with two 5-segment worms.
; SEE ALSO: bank4/zelda-guard-fire.asm (Zelda_Update, GuardFire_Update); bank4/lanmola.asm (Lanmola_Update:
;           movement, segment death); bank7/object-init-table.asm (Obj_Init, which calls these and
;           sets ObjAttr and ObjHP first); bank7/object-update-table.asm (Obj_TypeHpPairs).
; KEY FACTS:
;   - Zelda is object slot 1 at ($78, $88); the guard fires are slots 2-5 at ($60, $B5), ($70, $9D),
;     ($80, $9D), ($90, $B5) (Zelda_Init, Zelda_RoomObjXs, Zelda_RoomObjYs).
;   - A Lanmola room has 10 segments in slots $01-$0A, all starting at ($40, $8D): two worms, heads in
;     slots $05 and $0A, tails in slots $01 and $06 (Lanmola_Init).
;   - Only the heads start with a direction (8 = up); a segment with direction 0 has not started
;     moving, and directions are handed down the worm by Lanmola_Update (Lanmola_Init).
;   - Lanmola speed is (object type - $39) pixels per frame: 1 for type $3A, 2 for type $3B
;     (Lanmola_Init, Lanmola_Speed).
;   - Every segment gets the HP of slot 1, $20 for both types (Lanmola_Init, Obj_TypeHpPairs).
;   - Lanmola_Init sets RoomObjCount to 8, not 10. Only the two heads' deaths add to RoomKillCount, so
;     killing both worms never reaches the "killed everything" test of the room history code.
; CONTENTS: (generated; L = line in this file)
;   L0031 Zelda_RoomObjXs  data  TABLE: start X of the objects in Zelda's room; index = object slot - 1 (0 = Zelda, 1-4 = guard
;   L0036 Zelda_RoomObjYs  data  TABLE: start Y of the objects in Zelda's room; index = object slot - 1 (0 = Zelda, 1-4 = guard
;   L0045 Zelda_Init       code  Places Zelda in object slot 1 and four guard fires in slots 2-5.
;   L0075 Lanmola_Init     code  Makes the ten Lanmola segments (two worms of five) in object slots $01-$0A.
; ============================================================================

; TABLE: start X of the objects in Zelda's room; index = object slot - 1 (0 = Zelda, 1-4 = guard
;        fires); value = X in pixels.
Zelda_RoomObjXs:
    .BYTE $78, $60, $70, $80, $90

; TABLE: start Y of the objects in Zelda's room; index = object slot - 1 (0 = Zelda, 1-4 = guard
;        fires); value = Y in pixels.
Zelda_RoomObjYs:
    .BYTE $88, $B5, $9D, $9D, $B5

; ----------------------------------------------------------------------------
; Places Zelda in object slot 1 and four guard fires in slots 2-5.
; OUT:  ObjX, ObjY, ObjType of slots 1-5 written; X = 0.
; NOTE: Slots 2-5 keep the uninitialized flag they got on room entry, so each guard fire is
;       initialized as a type $3F object (attributes, HP) on its own first update.
; ----------------------------------------------------------------------------
Zelda_Init:
    ; For object slots 5 down to 1:
    LDX #$05
:
    ; Position from the tables (index = slot - 1), type $3F.
    LDA Zelda_RoomObjXs-1, X
    STA ObjX, X
    LDA Zelda_RoomObjYs-1, X
    STA ObjY, X
    LDA #$3F                    ; Guard fire object type
    STA ObjType, X
    DEX
    BNE :-

    ; Slot 1 is Zelda herself: type $37.
    LDA #$37
    STA ObjType+1
    RTS

; ----------------------------------------------------------------------------
; Makes the ten Lanmola segments (two worms of five) in object slots $01-$0A.
; IN:   slot 1 holds the room's Lanmola (type $3A or $3B) with ObjAttr and ObjHP already set.
; OUT:  slots $01-$0A positioned, typed and marked initialized; Lanmola_Type, Lanmola_Speed,
;       RoomObjCount written.
; NOTE: Speed is type - $39: 1 pixel per frame for type $3A, 2 for type $3B.
;       RoomObjCount is set to 8, not 10. Its only later reader is the kill banking on leaving the
;       room (RoomHistory_SaveKillCountUW: all dead when RoomKillCount >= RoomObjCount; Lanmola is
;       exempt from the one-kill rule for boss types). Only a head's death adds to RoomKillCount
;       (tails die as type $5D), so the count is at most 2 and 8 behaves exactly as 10 would.
; ----------------------------------------------------------------------------
Lanmola_Init:
    ; For Y = 9 down to 0, object slots $0A down to $01:
    LDY #$09
:
    ; Every segment starts at ($40, $8D) with direction 0 (not moving yet).
    LDA #$40
    STA a:ObjX+1, Y
    LDA #$8D
    STA a:ObjY+1, Y
    LDA #$00
    STA a:ObjDir+1, Y

    ; No spawn cloud (metastate 0) and no init call of its own (uninitialized flag 0).
    STA ObjMetastate+1, Y
    STA ObjUninitialized+1, Y

    ; Copy the attributes, HP and type of the Lanmola in slot 1 to the segment.
    LDA ObjAttr+1
    STA ObjAttr+1, Y
    LDA ObjHP+1
    STA ObjHP+1, Y
    LDA ObjType+1
    STA ObjType+1, Y
    DEY
    BPL :-

    ; The heads, slots $05 and $0A, start facing up (8). The same value goes to the slot-array
    ; entries at $0380 + slot ($0385, $038A; ObjMovingLimit); the Lanmola update code does not
    ; read them.
    LDA #$08
    STA ObjDir+5
    STA $0385
    STA ObjDir+10
    STA $038A

    ; Remember the object type, and set the speed to (object type - $39): 1 or 2 pixels per frame.
    LDA ObjType+1
    STA Lanmola_Type
    SEC
    SBC #$39
    STA Lanmola_Speed
    LDA #$08                    ; 8, although 10 segments were made (no effect: see the header).
    STA RoomObjCount
    RTS
