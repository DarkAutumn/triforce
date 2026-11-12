; ============================================================================
; FILE:     bank1/underworld-person-init.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1011-1168
; SUBJECTS: Grumble, Person
; TOPIC:    Init of the dungeon persons (object types $4B-$52) and of the Grumble Goriya ($36): position,
;           the two fires, the text selector by level and type, removal when the room's offer was taken.
; SEE ALSO: bank1/underworld-person-update.asm, bank1/life-or-money.asm, bank1/grumble-goriya.asm (the
;           updates); bank1/cave-init-person-text.asm (Person_SetUpWithFires, Person_TextAddrs);
;           bank1/person-textbox.asm (typing the text); bank1/take-item.asm (RoomFlags_GetItemTaken);
;           bank7/object-stubs-destroy-monster.asm (fixed-bank entry points, Obj_DestroyMonster).
; KEY FACTS:
;   - Every person stands at ($78, $80) with fires (type $40) in slots 2 and 3 at X $48 and $A8, and
;     Link is halted (ObjState = $40) while the text is typed. Person_InitUnderworldLevels1257,
;     Person_InitUnderworldLevels3468, Person_InitUnderworldLevel9, Grumble_Init_Full
;   - The text depends on the level and on type - $4B: levels 1, 2, 5, 7 use
;     Person_TextSelectorsLevels1257; levels 3, 4, 6, 8 use Person_TextSelectorsLevels3468; level 9
;     uses Person_TextSelectorsLevel9. Person_InitUnderworld_JumpTable
;   - In levels 1, 2, 5, 7 type $4F is the man who sells bomb capacity; he does not appear once the
;     room's item-taken flag ($10) is set. Person_InitUnderworldLevels1257, Person_DestroyIfTaken
;   - The life-or-money man (type $51) and the Grumble Goriya (type $36) likewise do not appear once
;     the room's item-taken flag is set. Person_InitLifeOrMoney_Full, Grumble_Init_Full
;   - Level 9, type $4B: when InvTriforce = $FF the man sets ShutterTrigger = 1 and is removed at
;     once; with any piece missing he stays. Person_InitUnderworldLevel9
; CONTENTS: (generated; L = line in this file)
;   L0045 Person_InitUnderworld_Full       code  Inits a dungeon person (types $4B-$50, $52): dispatches by level to pick the text selector table.
;   L0053 Person_InitUnderworld_JumpTable  data  TABLE: dungeon person init routines; index = CurLevel (0-9); value = routine address.
;   L0067 Person_TextSelectorsLevels1257   data  TABLE: text selectors of dungeon persons in levels 1, 2, 5, 7; index = object type - $4B; value =
;   L0077 Person_InitUnderworldLevels1257  code  Inits a dungeon person of levels 1, 2, 5, 7: places him and sets his text by type.
;   L0105 Person_InitLifeOrMoney_Full      code  Inits the life-or-money man (type $51): places him, sets text $36, removes him if already paid.
;   L0121 Person_DestroyIfTaken            code  Removes the person in slot X and unhalts Link if the room's item-taken flag is set.
;   L0129 Person_GoPlayCharacterSfx        code  Branch island: jumps to Person_PlayCharacterSfx.
;   L0134 Person_TextSelectorsLevels3468   data  TABLE: text selectors of dungeon persons in levels 3, 4, 6, 8; index = object type - $4B; value =
;   L0143 Person_InitUnderworldLevels3468  code  Inits a dungeon person of levels 3, 4, 6, 8: places him and sets his text. These only talk.
;   L0162 Person_TextSelectorsLevel9       data  TABLE: text selectors of dungeon persons in level 9; index = object type - $4B (only $4B-$4E have
;   L0172 Person_InitUnderworldLevel9      code  Inits a level 9 person; type $4B opens the shutters and vanishes if Link has all Triforce pieces.
;   L0216 Grumble_Init_Full                code  Inits the Grumble Goriya (type $36): places it, sets text $24, removes it if it was already fed.
;   L0242 Person_PlayCharacterSfx          code  Requests the sound played when a person appears (Tune1Request = $08).
; ============================================================================

; ----------------------------------------------------------------------------
; Inits a dungeon person (types $4B-$50, $52): dispatches by level to pick the text selector table.
; IN:   X = object slot (the person is in slot 1; parts of the code read ObjType+1 directly).
; OUT:  PersonTextVramLo = $A4 (front of textbox line 1); the rest is done by the level's routine.
; ----------------------------------------------------------------------------
Person_InitUnderworld_Full:
    ; Point to the front of the first line in VRAM.
    LDA Person_TextboxLineAddrsLo+2
    STA PersonTextVramLo
    LDA CurLevel
    JSR Util_TableJump

; TABLE: dungeon person init routines; index = CurLevel (0-9); value = routine address.
Person_InitUnderworld_JumpTable:
    .ADDR Person_InitUnderworld_Return    ; 0 (overworld): nothing
    .ADDR Person_InitUnderworldLevels1257             ; level 1
    .ADDR Person_InitUnderworldLevels1257             ; level 2
    .ADDR Person_InitUnderworldLevels3468             ; level 3
    .ADDR Person_InitUnderworldLevels3468             ; level 4
    .ADDR Person_InitUnderworldLevels1257             ; level 5
    .ADDR Person_InitUnderworldLevels3468             ; level 6
    .ADDR Person_InitUnderworldLevels1257             ; level 7
    .ADDR Person_InitUnderworldLevels3468             ; level 8
    .ADDR Person_InitUnderworldLevel9             ; level 9

; TABLE: text selectors of dungeon persons in levels 1, 2, 5, 7; index = object type - $4B; value =
;        PersonTextSelector (offset into Person_TextAddrs).
Person_TextSelectorsLevels1257:
    ; Types $4B, $4C, $4D, $4E, $4F (bomb capacity offer), $50, $51 (not used: type $51 has its own
    ; init and selector $36), $52.
    .BYTE $28, $26, $2E, $30, $32, $3E, $3E, $34

; ----------------------------------------------------------------------------
; Inits a dungeon person of levels 1, 2, 5, 7: places him and sets his text by type.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector; Link halted; type $4F is destroyed when the room's item-taken flag is set.
; ----------------------------------------------------------------------------
Person_InitUnderworldLevels1257:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR Person_SetUpWithFires

    ; Object type - $4B = index of the person.
    LDA ObjType, X
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA Person_TextSelectorsLevels1257, Y
    STA PersonTextSelector

    ; Type $4F offers more bomb capacity. Go see if the room's item-taken flag
    ; says the offer was already taken. Every other type only plays the sound.
    LDA ObjType+1
    CMP #$4F
    BNE Person_GoPlayCharacterSfx
    BEQ Person_DestroyIfTaken

; ----------------------------------------------------------------------------
; Inits the life-or-money man (type $51): places him, sets text $36, removes him if already paid.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector = $36; PersonTextVramLo = $A4; falls into Person_DestroyIfTaken.
; ----------------------------------------------------------------------------
Person_InitLifeOrMoney_Full:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR Person_SetUpWithFires

    ; Text selector $36: the "life or money" text.
    LDA #$36
    STA PersonTextSelector

    ; Point to the front of the first line in VRAM.
    LDA Person_TextboxLineAddrsLo+2
    STA PersonTextVramLo

; Removes the person in slot X and unhalts Link if the room's item-taken flag is set.
; Otherwise only plays the sound of a person appearing.
Person_DestroyIfTaken:
    JSR RoomFlags_GetItemTaken  ; A = $10 if the room's flag is set, else 0.
    BEQ Person_GoPlayCharacterSfx
    LDA #$00
    STA ObjState                ; Unhalt Link.
    JMP Obj_DestroyMonster          ; The two fires stay.

; Branch island: jumps to Person_PlayCharacterSfx.
Person_GoPlayCharacterSfx:
    JMP Person_PlayCharacterSfx

; TABLE: text selectors of dungeon persons in levels 3, 4, 6, 8; index = object type - $4B; value =
;        PersonTextSelector (offset into Person_TextAddrs).
Person_TextSelectorsLevels3468:
    ; Types $4B, $4C, $4D, $4E, $4F, $50, $51 (not used: type $51 has its own init), $52.
    .BYTE $2A, $38, $3A, $2C, $40, $42, $42, $3C

; ----------------------------------------------------------------------------
; Inits a dungeon person of levels 3, 4, 6, 8: places him and sets his text. These only talk.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector; Link halted.
; ----------------------------------------------------------------------------
Person_InitUnderworldLevels3468:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR Person_SetUpWithFires

    ; Object type - $4B = index of the person.
    LDA ObjType, X
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA Person_TextSelectorsLevels3468, Y
    STA PersonTextSelector
    JMP Person_PlayCharacterSfx

; TABLE: text selectors of dungeon persons in level 9; index = object type - $4B (only $4B-$4E have
;        an entry); value = PersonTextSelector (offset into Person_TextAddrs).
Person_TextSelectorsLevel9:
    .BYTE $44, $46, $48, $4A    ; $44 (type $4B) = the text about needing the Triforce.

; ----------------------------------------------------------------------------
; Inits a level 9 person; type $4B opens the shutters and vanishes if Link has all Triforce pieces.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector; for type $4B with InvTriforce = $FF: ShutterTrigger = 1, Link unhalted,
;       the person destroyed.
; NOTE: With a piece missing the man stays, and a person in slot 1 stops Link from walking up past him.
; ----------------------------------------------------------------------------
Person_InitUnderworldLevel9:
    ; The person goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR Person_SetUpWithFires
    JSR Person_PlayCharacterSfx
    LDA ObjType, X
    PHA                         ; Save the object type.

    ; Object type - $4B = index of the person.
    SEC
    SBC #$4B
    TAY

    ; Look up and store the text selector.
    LDA Person_TextSelectorsLevel9, Y
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
    JSR Obj_DestroyMonster

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Inits the Grumble Goriya (type $36): places it, sets text $24, removes it if it was already fed.
; IN:   X = object slot (1).
; OUT:  PersonTextSelector = $24; PersonTextVramLo = $A4; when the room's item-taken flag is set:
;       ObjType+1 = 0 and Link unhalted.
; ----------------------------------------------------------------------------
Grumble_Init_Full:
    ; The Grumble Goriya goes at location ($78, $80).
    LDA #$78
    LDY #$80
    JSR Person_SetUpWithFires

    ; Text selector $24: the "grumble, grumble" text.
    LDA #$24
    STA PersonTextSelector

    ; Point to the front of the first textbox line in VRAM (entry 2 of the table).
    LDA Person_TextboxLineAddrsLo+2
    STA PersonTextVramLo

    ; If the room's item-taken flag is set (the Goriya already got the food), then
    ; unhalt Link and reset the object type to get rid of it.
    JSR RoomFlags_GetItemTaken
    BEQ Person_PlayCharacterSfx
    LDA #$00
    STA ObjState
    STA ObjType+1
    RTS

; ----------------------------------------------------------------------------
; Requests the sound played when a person appears (Tune1Request = $08).
; ----------------------------------------------------------------------------
Person_PlayCharacterSfx:
    LDA #$08                    ; "item taken/character" sound effect
    STA Tune1Request
    RTS
