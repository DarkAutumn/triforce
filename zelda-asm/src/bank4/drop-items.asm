; ============================================================================
; FILE:     bank4/drop-items.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10995-11309
; SUBJECTS: (none)
; TOPIC:    HP nibble extract, drop tables, SetUpDroppedItem, UpdateItem, DestroyMonster
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0034 ExtractHitPointValue    code  Params:
;   L0055 PlayBossDeathCry        code
;   L0064 NoDropMonsterTypes      data  Types of objects that do not drop items.
;   L0067 DropItemMonsterTypes0   data
;   L0070 DropItemMonsterTypes1   data
;   L0074 DropItemMonsterTypes2   data
;   L0080 DropItemSetBaseOffsets  data  Multiples of $A to index the base of each row of items in DropItemTable.
;   L0086 DropItemRates           data  Top drop an item, a random value must be less than the
;   L0095 DropItemTable           data  ID's for items dropped by monsters.
;   L0102 SetUpDroppedItem        code
;   L0235 ItemTakerObjSlots       data
;   L0238 UpdateItem              code
;   L0327 DestroyMonster_Bank4    code  Params:
; ============================================================================

; Params:
; A: hit point doublet (byte)
; [00]: object type
;
; Returns:
; A: hit points for object type
;
; TRIFORCE: Extracts one enemy's HP from the packed byte in ObjectTypeToHpPairs.
; TRIFORCE: Even types: AND $F0 (keep high nibble). Odd types: low nibble << 4.
; TRIFORCE: Result always has low nibble=0. Python reads ObjHP >> 4 for HP count.
ExtractHitPointValue:
    PHA

    ; Even object types only have to mask off the bottom nibble.
    LDA $00
    LSR
    BCS @OddType
    PLA
    AND #$F0
    RTS

@OddType:
    PLA

    ; Odd object types have to multiply by 16.
    ASL
    ASL
    ASL
    ASL
    RTS

PlayBossDeathCry:
    LDA #$02
    STA SampleRequest
    LDA #$80
    STA EffectRequest
    RTS

; Types of objects that do not drop items.
;
NoDropMonsterTypes:
    .BYTE $5D, $14, $15, $1B, $1C, $1D, $17

DropItemMonsterTypes0:
    .BYTE $07, $08, $0E, $04, $0F, $23

DropItemMonsterTypes1:
    .BYTE $21, $22, $0D, $10, $13, $28, $2A, $27
    .BYTE $16

DropItemMonsterTypes2:
    .BYTE $09, $0A, $03, $01, $12, $06, $0B, $24
    .BYTE $30

; Multiples of $A to index the base of each row of items in DropItemTable.
;
DropItemSetBaseOffsets:
    .BYTE $00, $0A, $14, $1E

; Top drop an item, a random value must be less than the
; element for the monster group.
;
DropItemRates:
    .BYTE $50, $98, $68, $68

; ID's for items dropped by monsters.
; WorldKillCycle addresses the column.
;
; Monsters are arranged in groups that determine the row.
; There are four rows.
;
DropItemTable:
    .BYTE $22, $18, $22, $18, $23, $18, $22, $22
    .BYTE $18, $18, $0F, $18, $22, $18, $0F, $22
    .BYTE $21, $18, $18, $18, $22, $00, $18, $21
    .BYTE $18, $22, $00, $18, $00, $22, $22, $22
    .BYTE $23, $18, $22, $23, $22, $22, $22, $18

SetUpDroppedItem:
    ; Reset [01] which holds the row number of the drop item table.
    LDA #$00
    STA $01

    ; Compare dead object's type to array of 7 that don't drop items.
    ; If found, go destroy the monster.
    LDA Item_ObjMonsterType, X
    LDY #$06

@FindNoDropType:
    CMP NoDropMonsterTypes, Y
    BEQ @DestroyMonster         ; If found, go destroy the monster.
    DEY
    BPL @FindNoDropType

    ; Look for object type in array of 6. These use drop item table row 0.
    LDY #$05

@FindDrop0Type:
    CMP DropItemMonsterTypes0, Y
    BEQ @Found                  ; If found, use row 0.
    DEY
    BPL @FindDrop0Type

    ; Increment row number in [01] to 1.
    INC $01

    ; Look for object type in array of 9. These use drop item table row 1.
    LDY #$08

@FindDrop1Type:
    CMP DropItemMonsterTypes1, Y
    BEQ @Found                  ; If found, use row 1.
    DEY
    BPL @FindDrop1Type

    ; Increment row number in [01] to 2.
    INC $01

    ; Look for object type in array of 9. These use drop item table row 2.
    LDY #$08

@FindDrop2Type:
    CMP DropItemMonsterTypes2, Y
    BEQ @Found                  ; If found, use row 2.
    DEY
    BPL @FindDrop2Type

    ; Increment [01]. The rest use row 3.
    INC $01

@Found:
    ; If object slot = 1 and object is stalfos or gibdos, then
    ; go destroy the mosnter, because it might already be carrying
    ; a room item.
    CPX #$01
    BNE @LookUpItem
    CMP #$2A
    BEQ @DestroyMonster
    CMP #$30
    BEQ @DestroyMonster

@LookUpItem:
    ; Look up the base offset of the row in the drop item table.
    ; Add it and WorldKillCycle to look up the dropped item for
    ; this object type and store it in [00].
    LDY $01
    LDA DropItemSetBaseOffsets, Y
    CLC
    ADC WorldKillCycle
    TAY
    LDA DropItemTable, Y
    STA $00                     ; [00] Dropped item ID

    ; If the global kill count = $10, set the dropped item to a fairy.
    LDA #$23
    LDY WorldKillCount
    CPY #$10
    BEQ @SetHelpItem

    ; If help drop counter < $A, then use the dropped item ID as is.
    ; But there is a random chance that the item drop will be canceled.
    LDA HelpDropCount
    CMP #$0A
    BCC @RandomlyCancel

    ; Otherwise, the item will be guaranteed to be made.
    ; [51] HelpDropValue determines which kind:
    ; - 0:     5 rupees
    ; - other: bomb
    LDA #$0F
    LDY HelpDropValue
    BEQ @SetHelpItem
    LDA #$00                    ; Bomb

@SetHelpItem:
    STA $00                     ; [00] Item ID

    ; The player has been helped with a dropped item. Reset the
    ; help drop variables.
    LDA #$00
    STA HelpDropCount
    STA HelpDropValue
    BEQ @Commit

@RandomlyCancel:
    ; If the random value >= drop item rate for the monster's group,
    ; then go destroy the monster.
    LDY $01
    LDA Random, X
    CMP DropItemRates, Y
    BCS @DestroyMonster

@Commit:
    ; Set ObjItemLifetime to $FF.
    LDA #$FF
    STA Item_ObjItemLifetime, X

    ; Set ObjItemId.
    LDA $00
    STA Item_ObjItemId, X

    ; If the item is a fairy, we have to finish setting it up.
    CMP #$23
    BNE :+
    JMP SetUpFairyObject
:
    RTS

@DestroyMonster:
    JMP DestroyMonster_Bank4

ItemTakerObjSlots:
    .BYTE $00, $0F, $0D, $12

UpdateItem:
    ; Every other frame, decrement the item lifetime.
    LDA FrameCounter
    LSR
    BCC :+
    DEC Item_ObjItemLifetime, X
:
    ; If lifetime reached 0, go destroy the object.
    LDA Item_ObjItemLifetime, X
    BEQ DestroyMonster_Bank4

    ; Draw the item. Fairies are animated separately.
    LDA Item_ObjItemId, X
    CMP #$23
    BEQ @AnimateFairy
    JSR AnimateItemObject
    JMP :+

@AnimateFairy:
    JSR UpdateFairyObject
:
    ; If the player is halted, then return.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ @Exit

    ; In addition to Link, there are 3 objects that can pick up items: 
    ; * arrow
    ; * sword
    ; * boomerang
    ;
    ; For each weapon that's active, substitute it's location for
    ; Link's and try taking the item. Have Link himself check last.
    LDA #$04
    STA $0D                     ; [0D] holds index into array of item slots that can take an item.

@LoopItemTaker:
    ; Save Link's location.
    LDA ObjX
    PHA
    LDA ObjY
    PHA

    ; Set Link's location to this object's.
    LDY $0D
    LDX ItemTakerObjSlots-1, Y
    LDA ObjX, X
    STA ObjX
    LDA ObjY, X
    STA ObjY
    CPX #$00
    BEQ @TryToTake              ; If Link is checking directly, go try to take the item.

    ; If the object whose place Link is taking is not active, then
    ; don't try to take the item.
    LDA ObjState, X
    BEQ @SkipTaking
    BMI @SkipTaking

@TryToTake:
    ; Try to take the item.
    LDX CurObjIndex
    LDA Item_ObjItemId, X
    STA $04
    JSR TryTakeItem

@SkipTaking:
    ; Restore Link's location.
    PLA
    STA ObjY
    PLA
    STA ObjX
    LDX CurObjIndex             ; Reload object index in X.

    ; If item object's item type is invalid, then the item was taken.
    ; So, go destroy this object.
    LDA Item_ObjItemId, X
    CMP #$FF
    BEQ DestroyMonster_Bank4
    DEC $0D                     ; Decrease the index [0D].
    BNE @LoopItemTaker          ; If there are more object slots to check from, go try the next one.

@Exit:
    RTS

; Params:
; X: object index
;
DestroyMonster_Bank4:
    LDA #$00
    STA ObjType, X
    JSR SetShoveInfoWith0
    STA ObjTimer, X
    STA ObjState, X
    STA ObjInvincibilityTimer, X
    LDA #$FF
    STA ObjUninitialized, X
    LDA #$01
    STA ObjMetastate, X
    RTS
