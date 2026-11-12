; ============================================================================
; FILE:     bank4/zelda-lanmola-inits.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9456-9524
; SUBJECTS: (none)
; TOPIC:    Guard fire tables, InitZelda, Lanmola init
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 GuardFireXs  data
;   L0018 GuardFireYs  data
;   L0021 InitZelda    code
;   L0040 InitLamnola  code
; ============================================================================

GuardFireXs:
    .BYTE $78, $60, $70, $80, $90

GuardFireYs:
    .BYTE $88, $B5, $9D, $9D, $B5

InitZelda:
    ; Make Zelda and 4 flames that guard her, in object slots 5 to 1.
    LDX #$05
:
    ; Look up and set the location of each object.
    LDA GuardFireXs-1, X
    STA ObjX, X
    LDA GuardFireYs-1, X
    STA ObjY, X
    LDA #$3F                    ; Guard fire object type
    STA ObjType, X
    DEX
    BNE :-

    ; Replace the type of the object in slot 1 with $37: Zelda.
    LDA #$37
    STA ObjType+1
    RTS

InitLamnola:
    ; Make $A segments.
    LDY #$09
:
    ; Starting location is ($40, $8D) and direction is 0.
    LDA #$40
    STA a:ObjX+1, Y
    LDA #$8D
    STA a:ObjY+1, Y
    LDA #$00
    STA a:ObjDir+1, Y

    ; Each segment is flagged initialized and ready to update.
    STA ObjMetastate+1, Y
    STA ObjUninitialized+1, Y

    ; This Lamnola that's making additional segments is in slot 1.
    ; Copy its object attributes, HP, and type to the others.
    LDA ObjAttr+1
    STA ObjAttr+1, Y
    LDA ObjHP+1
    STA ObjHP+1, Y
    LDA ObjType+1
    STA ObjType+1, Y
    DEY
    BPL :-

    ; Only the heads in slots 5 and $A have a direction and [0380][X] set.
    LDA #$08
    STA ObjDir+5
    STA $0385
    STA ObjDir+10
    STA $038A

    ; Remember the object type, and set speed to (object type - $39):
    ; 1 or 2.
    LDA ObjType+1
    STA Lanmola_Type
    SEC
    SBC #$39
    STA Lanmola_Speed
    LDA #$08                    ; 8 instead of 10, because 8 is the max kill count to save.
    STA RoomObjCount
    RTS
