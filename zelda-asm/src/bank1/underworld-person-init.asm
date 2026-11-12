; ============================================================================
; FILE:     bank1/underworld-person-init.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1011-1168
; SUBJECTS: (none)
; TOPIC:    Init of the dungeon persons (object types $4B-$52) and of the Grumble Goriya ($36): position,
;           the two fires, the text selector by level and type, removal when the room's offer was taken.
; SEE ALSO: bank1/underworld-person-update.asm, bank1/life-or-money.asm, bank1/grumble-goriya.asm (the
;           updates); bank1/cave-init-person-text.asm (SetUpCommonCaveObjects, PersonTextAddrs);
;           bank1/person-textbox.asm (typing the text); bank1/take-item.asm (GetRoomFlagUWItemState);
;           bank7/object-stubs-destroy-monster.asm (fixed-bank entry points, DestroyMonster).
; KEY FACTS:
;   - Every person stands at ($78, $80) with fires (type $40) in slots 2 and 3 at X $48 and $A8, and
;     Link is halted (ObjState = $40) while the text is typed. InitUnderworldPersonA,
;     InitUnderworldPersonB, InitUnderworldPersonC, InitGrumble_Full
;   - The text depends on the level and on type - $4B: levels 1, 2, 5, 7 use
;     UnderworldPersonTextSelectorsA; levels 3, 4, 6, 8 use UnderworldPersonTextSelectorsB; level 9
;     uses UnderworldPersonTextSelectorsC. InitUnderworldPerson_Full_JumpTable
;   - In levels 1, 2, 5, 7 type $4F is the man who sells bomb capacity; he does not appear once the
;     room's item-taken flag ($10) is set. InitUnderworldPersonA, UnderworldPerson_DestroyIfTaken
;   - The life-or-money man (type $51) and the Grumble Goriya (type $36) likewise do not appear once
;     the room's item-taken flag is set. InitUnderworldPersonLifeOrMoney_Full, InitGrumble_Full
;   - Level 9, type $4B: when InvTriforce = $FF the man sets ShutterTrigger = 1 and is removed at
;     once; with any piece missing he stays. InitUnderworldPersonC
; CONTENTS: (generated; L = line in this file)
;   L0045 InitUnderworldPerson_Full             code  Inits a dungeon person (types $4B-$50, $52): dispatches by level to pick the text selector table.
;   L0053 InitUnderworldPerson_Full_JumpTable   data  TABLE: dungeon person init routines; index = CurLevel (0-9); value = routine address.
;   L0067 UnderworldPersonTextSelectorsA        data  TABLE: text selectors of dungeon persons in levels 1, 2, 5, 7; index = object type - $4B; value =
;   L0077 InitUnderworldPersonA                 code  Inits a dungeon person of levels 1, 2, 5, 7: places him and sets his text by type.
;   L0105 InitUnderworldPersonLifeOrMoney_Full  code  Inits the life-or-money man (type $51): places him, sets text $36, removes him if already paid.
;   L0121 UnderworldPerson_DestroyIfTaken       code  Removes the person in slot X and unhalts Link if the room's item-taken flag is set.
;   L0129 L_PlayCharacterSfx                    code  Branch island: jumps to PlayCharacterSfx.
;   L0134 UnderworldPersonTextSelectorsB        data  TABLE: text selectors of dungeon persons in levels 3, 4, 6, 8; index = object type - $4B; value =
;   L0143 InitUnderworldPersonB                 code  Inits a dungeon person of levels 3, 4, 6, 8: places him and sets his text. These only talk.
;   L0162 UnderworldPersonTextSelectorsC        data  TABLE: text selectors of dungeon persons in level 9; index = object type - $4B (only $4B-$4E have
;   L0172 InitUnderworldPersonC                 code  Inits a level 9 person; type $4B opens the shutters and vanishes if Link has all Triforce pieces.
;   L0216 InitGrumble_Full                      code  Inits the Grumble Goriya (type $36): places it, sets text $24, removes it if it was already fed.
;   L0242 PlayCharacterSfx                      code  Requests the sound played when a person appears (Tune1Request = $08).
; ============================================================================

; ----------------------------------------------------------------------------
; Inits a dungeon person (types $4B-$50, $52): dispatches by level to pick the text selector table.
; IN:   X = object slot (the person is in slot 1; parts of the code read ObjType+1 directly).
; OUT:  PersonTextVramLo = $A4 (front of textbox line 1); the rest is done by the level's routine.
; ----------------------------------------------------------------------------
InitUnderworldPerson_Full:
    ; Point to the front of the first line in VRAM.
    LDA TextboxLineAddrsLo+2
    STA PersonTextVramLo
    LDA CurLevel
    JSR TableJump

; TABLE: dungeon person init routines; index = CurLevel (0-9); value = routine address.
InitUnderworldPerson_Full_JumpTable:
    .ADDR InitUnderworldPerson_DoNothing    ; 0 (overworld): nothing
    .ADDR InitUnderworldPersonA             ; level 1
    .ADDR InitUnderworldPersonA             ; level 2
    .ADDR InitUnderworldPersonB             ; level 3
    .ADDR InitUnderworldPersonB             ; level 4
    .ADDR InitUnderworldPersonA             ; level 5
    .ADDR InitUnderworldPersonB             ; level 6
    .ADDR InitUnderworldPersonA             ; level 7
    .ADDR InitUnderworldPersonB             ; level 8
    .ADDR InitUnderworldPersonC             ; level 9

; TABLE: text selectors of dungeon persons in levels 1, 2, 5, 7; index = object type - $4B; value =
;        PersonTextSelector (offset into PersonTextAddrs).
UnderworldPersonTextSelectorsA:
    ; Types $4B, $4C, $4D, $4E, $4F (bomb capacity offer), $50, $51 (not used: type $51 has its own
    ; init and selector $36), $52.
    .BYTE $28, $26, $2E, $30, $32, $3E, $3E, $34

; ----------------------------------------------------------------------------
; Inits a dungeon person of levels 1, 2, 5, 7: places him and sets his text by type.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector; Link halted; type $4F is destroyed when the room's item-taken flag is set.
; ----------------------------------------------------------------------------
InitUnderworldPersonA:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Object type - $4B = index of the person.
    LDA ObjType, X
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA UnderworldPersonTextSelectorsA, Y
    STA PersonTextSelector

    ; Type $4F offers more bomb capacity. Go see if the room's item-taken flag
    ; says the offer was already taken. Every other type only plays the sound.
    LDA ObjType+1
    CMP #$4F
    BNE L_PlayCharacterSfx
    BEQ UnderworldPerson_DestroyIfTaken

; ----------------------------------------------------------------------------
; Inits the life-or-money man (type $51): places him, sets text $36, removes him if already paid.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector = $36; PersonTextVramLo = $A4; falls into UnderworldPerson_DestroyIfTaken.
; ----------------------------------------------------------------------------
InitUnderworldPersonLifeOrMoney_Full:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Text selector $36: the "life or money" text.
    LDA #$36
    STA PersonTextSelector

    ; Point to the front of the first line in VRAM.
    LDA TextboxLineAddrsLo+2
    STA PersonTextVramLo

; Removes the person in slot X and unhalts Link if the room's item-taken flag is set.
; Otherwise only plays the sound of a person appearing.
UnderworldPerson_DestroyIfTaken:
    JSR GetRoomFlagUWItemState  ; A = $10 if the room's flag is set, else 0.
    BEQ L_PlayCharacterSfx
    LDA #$00
    STA ObjState                ; Unhalt Link.
    JMP DestroyMonster          ; The two fires stay.

; Branch island: jumps to PlayCharacterSfx.
L_PlayCharacterSfx:
    JMP PlayCharacterSfx

; TABLE: text selectors of dungeon persons in levels 3, 4, 6, 8; index = object type - $4B; value =
;        PersonTextSelector (offset into PersonTextAddrs).
UnderworldPersonTextSelectorsB:
    ; Types $4B, $4C, $4D, $4E, $4F, $50, $51 (not used: type $51 has its own init), $52.
    .BYTE $2A, $38, $3A, $2C, $40, $42, $42, $3C

; ----------------------------------------------------------------------------
; Inits a dungeon person of levels 3, 4, 6, 8: places him and sets his text. These only talk.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector; Link halted.
; ----------------------------------------------------------------------------
InitUnderworldPersonB:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Object type - $4B = index of the person.
    LDA ObjType, X
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA UnderworldPersonTextSelectorsB, Y
    STA PersonTextSelector
    JMP PlayCharacterSfx

; TABLE: text selectors of dungeon persons in level 9; index = object type - $4B (only $4B-$4E have
;        an entry); value = PersonTextSelector (offset into PersonTextAddrs).
UnderworldPersonTextSelectorsC:
    .BYTE $44, $46, $48, $4A    ; $44 (type $4B) = the text about needing the Triforce.

; ----------------------------------------------------------------------------
; Inits a level 9 person; type $4B opens the shutters and vanishes if Link has all Triforce pieces.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector; for type $4B with InvTriforce = $FF: ShutterTrigger = 1, Link unhalted,
;       the person destroyed.
; NOTE: With a piece missing the man stays, and a person in slot 1 stops Link from walking up past him.
; ----------------------------------------------------------------------------
InitUnderworldPersonC:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects
    JSR PlayCharacterSfx
    LDA ObjType, X
    PHA                         ; Save the object type.

    ; Object type - $4B = index of the person.
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA UnderworldPersonTextSelectorsC, Y
    STA PersonTextSelector
    PLA                         ; Restore the object type.

    ; Return, if this is not type $4B (the man at the entrance of level 9),
    ; or Link does not have all eight Triforce pieces (InvTriforce <> $FF).
    CMP #$4B
    BNE @Exit
    LDA InvTriforce
    CMP #$FF
    BNE @Exit

    ; Otherwise, open the shutters, unhalt Link (A = 1 shifted right = 0),
    ; and destroy this object.
    LDA #$01
    STA ShutterTrigger
    LSR
    STA ObjState
    JSR DestroyMonster

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Inits the Grumble Goriya (type $36): places it, sets text $24, removes it if it was already fed.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector = $24; PersonTextVramLo = $A4; when the room's item-taken flag is set:
;       ObjType+1 = 0 and Link unhalted.
; ----------------------------------------------------------------------------
InitGrumble_Full:
    ; The Grumble Goriya goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR SetUpCommonCaveObjects

    ; Text selector $24: the "grumble, grumble" text.
    LDA #$24
    STA PersonTextSelector

    ; Point to the front of the first textbox line in VRAM (entry 2 of the table).
    LDA TextboxLineAddrsLo+2
    STA PersonTextVramLo

    ; If the room's item-taken flag is set (the Goriya already got the food), then
    ; unhalt Link and reset the object type to get rid of it.
    JSR GetRoomFlagUWItemState
    BEQ PlayCharacterSfx
    LDA #$00
    STA ObjState
    STA ObjType+1
    RTS

; ----------------------------------------------------------------------------
; Requests the sound played when a person appears (Tune1Request = $08).
; ----------------------------------------------------------------------------
PlayCharacterSfx:
    LDA #$08                    ; "item taken/character" sound effect
    STA Tune1Request
    RTS
