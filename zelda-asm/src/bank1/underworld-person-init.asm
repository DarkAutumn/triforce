; ============================================================================
; FILE:     bank1/underworld-person-init.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1011-1168
; SUBJECTS: (none)
; TOPIC:    Dungeon old men, life-or-money and Grumble init, text selectors
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 InitUnderworldPerson_Full             code
;   L0031 InitUnderworldPerson_Full_JumpTable   data
;   L0043 UnderworldPersonTextSelectorsA        data
;   L0046 InitUnderworldPersonA                 code
;   L0069 InitUnderworldPersonLifeOrMoney_Full  code
;   L0083 UnderworldPerson_DestroyIfTaken       code
;   L0091 L_PlayCharacterSfx                    code
;   L0094 UnderworldPersonTextSelectorsB        data
;   L0097 InitUnderworldPersonB                 code
;   L0114 UnderworldPersonTextSelectorsC        data
;   L0117 InitUnderworldPersonC                 code
;   L0154 InitGrumble_Full                      code
;   L0178 PlayCharacterSfx                      code
; ============================================================================

InitUnderworldPerson_Full:
    ; Point to the front of the first line in VRAM.
    LDA TextboxLineAddrsLo+2
    STA PersonTextPtr
    LDA CurLevel
    JSR TableJump

InitUnderworldPerson_Full_JumpTable:
    .ADDR InitUnderworldPerson_DoNothing
    .ADDR InitUnderworldPersonA
    .ADDR InitUnderworldPersonA
    .ADDR InitUnderworldPersonB
    .ADDR InitUnderworldPersonB
    .ADDR InitUnderworldPersonA
    .ADDR InitUnderworldPersonB
    .ADDR InitUnderworldPersonA
    .ADDR InitUnderworldPersonB
    .ADDR InitUnderworldPersonC

UnderworldPersonTextSelectorsA:
    .BYTE $28, $26, $2E, $30, $32, $3E, $3E, $34

InitUnderworldPersonA:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Subtract $4B from the object type to get the old man's index.
    LDA ObjType, X
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA UnderworldPersonTextSelectorsA, Y
    STA PersonTextSelector

    ; If this is the man that offers more bomb capacity, then
    ; go see if the offer was flagged as already taken.
    LDA ObjType+1
    CMP #$4F
    BNE L_PlayCharacterSfx
    BEQ UnderworldPerson_DestroyIfTaken

InitUnderworldPersonLifeOrMoney_Full:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Set text selector for "life or money".
    LDA #$36
    STA PersonTextSelector

    ; Point to the front of the first line in VRAM.
    LDA TextboxLineAddrsLo+2
    STA PersonTextPtr

UnderworldPerson_DestroyIfTaken:
    ; Destroy this object if the secret/item was already taken.
    JSR GetRoomFlagUWItemState
    BEQ L_PlayCharacterSfx
    LDA #$00
    STA ObjState
    JMP DestroyMonster

L_PlayCharacterSfx:
    JMP PlayCharacterSfx

UnderworldPersonTextSelectorsB:
    .BYTE $2A, $38, $3A, $2C, $40, $42, $42, $3C

InitUnderworldPersonB:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Subtract $4B from the object type to get the old man's index.
    LDA ObjType, X
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA UnderworldPersonTextSelectorsB, Y
    STA PersonTextSelector
    JMP PlayCharacterSfx

UnderworldPersonTextSelectorsC:
    .BYTE $44, $46, $48, $4A

InitUnderworldPersonC:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects
    JSR PlayCharacterSfx
    LDA ObjType, X
    PHA                         ; Save the object type.

    ; Subtract $4B from the object type to get the old man's index.
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA UnderworldPersonTextSelectorsC, Y
    STA PersonTextSelector
    PLA                         ; Restore the object type.

    ; Return, if this is not the man at the entrance of Level 9,
    ; or Link has not gotten all the triforce pieces.
    CMP #$4B
    BNE @Exit
    LDA InvTriforce
    CMP #$FF
    BNE @Exit

    ; Otherwise, open the shutters, and destroy this object.
    LDA #$01
    STA ShutterTrigger
    LSR
    STA ObjState
    JSR DestroyMonster

@Exit:
    RTS

InitGrumble_Full:
    ; Grumble Goriya goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Store text selector $24.
    LDA #$24
    STA PersonTextSelector

    ; Set the low VRAM address of the first character to transfer
    ; to the front of the first line in NT0. It's at index 2.
    LDA TextboxLineAddrsLo+2
    STA PersonTextPtr

    ; If Grumble already got the food, then
    ; reset his type and state to get rid of him.
    JSR GetRoomFlagUWItemState
    BEQ PlayCharacterSfx
    LDA #$00
    STA ObjState
    STA ObjType+1
    RTS

PlayCharacterSfx:
    LDA #$08                    ; "item taken/character" sound effect
    STA Tune1Request
    RTS
