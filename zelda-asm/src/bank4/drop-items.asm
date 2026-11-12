; ============================================================================
; FILE:     bank4/drop-items.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10995-11309
; SUBJECTS: Drop, Obj, Sample
; TOPIC:    What a dead monster leaves behind: the four drop groups, the drop table indexed by the kill
;           cycle, drop rates, the fairy at 16 kills and the forced help drops; the dropped item object
;           (type $60): lifetime, pickup by Link or his weapons. Also the HP nibble extractor, the boss
;           death cry and this bank's copy of Obj_DestroyMonster.
; SEE ALSO: bank7/object-update-table.asm (Obj_EndMetastate: advances WorldKillCycle, turns the dead
;           monster into type $60 and calls Drop_SetUpItem; Obj_TypeHpPairs);
;           bank1/weapon-damage-death.asm (Damage_KillMonster: WorldKillCount, HelpDropCount, IsHelpDropBomb);
;           bank1/link-collision-harm.asm (Link_BeHarmed zeroes the three counters);
;           bank1/take-item.asm (Item_TryTake: pickup range and effects); bank7/room-item-draw.asm
;           (Item_DrawObject: flashing); bank4/fairy-flyer-common.asm (Fairy_Init, Fairy_Update);
;           bank4/dodongo.asm (forces a bomb help drop); bank7/object-init-table.asm (Obj_Init sets ObjHP).
; KEY FACTS:
;   - Never drop: types $5D dead dummy, $14 child Gel, $15 Gel, $1B-$1D Keese, $17 Like-Like; and a
;     Stalfos ($2A) or Gibdo ($30) in object slot 1. Drop_NoDropTypes, Drop_SetUpItem@Found
;   - Group 0: $07/$08 red Octoroks, $0E Tektite, $04 Moblin, $0F Blue Leever, $23 Blue Wizzrobe.
;     Group 1: $21/$22 Ghini, $0D Tektite, $10 Red Leever, $13 Zol, $28 Rope, $2A Stalfos, $27 Wallmaster,
;     $16 Pols Voice. Group 2: $09/$0A blue Octoroks, $03 Moblin, $01 Lynel, $12 Vire, $06 Goriya,
;     $0B Darknut, $24 Red Wizzrobe, $30 Gibdo. Group 3: every other type. Drop_Group0Types/1/2
;   - The item is Drop_ItemTable[group * 10 + WorldKillCycle]; the cycle (0-9) was already advanced for
;     this kill by Obj_EndMetastate. Drop_ItemTable, Drop_SetUpItem@LookUpItem
;   - Without a forced drop the item appears only if Random,X (X = the monster's slot) < the group's
;     rate: $50 (31%), $98 (59%), $68 (41%), $68 (41%). Drop_GroupRates, Drop_SetUpItem@RandomlyCancel
;   - WorldKillCount = $10 when the death ends: the drop is a fairy, with no random test, whatever the
;     group. Drop_SetUpItem@LookUpItem
;   - Else HelpDropCount >= $0A: forced drop, a bomb if IsHelpDropBomb <> 0, otherwise 5 rupees. Both
;     forced kinds (and the fairy) zero HelpDropCount and IsHelpDropBomb. Drop_SetUpItem@SetHelpItem
;   - A dropped item's lifetime starts at $FF and drops by 1 on odd frames: gone after about 510
;     frames. While it is >= $F0 (about the first 32 frames) it flashes and cannot be taken.
;     Drop_UpdateItem, Drop_SetUpItem@Commit
;   - Items are taken by Link or by an active object in slot $12 (arrow/rod), $0D (sword) or $0F
;     (boomerang/food), tested in that order, Link last; not while Link is halted. Drop_UpdateItem,
;     Drop_ItemTakerObjSlots
;   - ObjHP = the type's nibble * $10 (even type: high nibble; odd type: low nibble).
;     Obj_ExtractStartHp
; CONTENTS: (generated; L = line in this file)
;   L0062 Obj_ExtractStartHp        code  Extracts one object type's starting HP from a packed byte of Obj_TypeHpPairs.
;   L0088 Sample_PlayBossDeathCry   code  Requests the boss death sounds: DMC sample 2 (the boss cry) and noise effect $80.
;   L0096 Drop_NoDropTypes          data  TABLE: object types that never drop an item; index = 0-6; value = object type.
;   L0100 Drop_Group0Types          data  TABLE: object types of drop group 0; index = 0-5; value = object type.
;   L0105 Drop_Group1Types          data  TABLE: object types of drop group 1; index = 0-8; value = object type.
;   L0111 Drop_Group2Types          data  TABLE: object types of drop group 2; index = 0-8; value = object type.
;   L0117 Drop_GroupRowOffsets      data  TABLE: offset of each group's row in Drop_ItemTable; index = drop group 0-3; value = group * 10.
;   L0122 Drop_GroupRates           data  TABLE: drop rate; index = drop group 0-3; value = threshold: an unforced drop happens when the
;   L0132 Drop_ItemTable            data  TABLE: dropped item ids; index = drop group * 10 + WorldKillCycle (0-9); value = item id
;   L0157 Drop_SetUpItem            code  Decides what a monster that finished dying drops, and makes the slot that item or frees it.
;   L0280 Drop_ItemTakerObjSlots    data  TABLE: object slots whose position can pick up a dropped item; index = 0-3 (tested 3 down to 0);
;   L0297 Drop_UpdateItem           code  Updates a dropped item (type $60): ages it, draws it, and lets Link or his weapons take it.
;   L0384 Obj_DestroyMonster_Bank4  code  Frees an object slot: type 0, no knockback, timer, state and invincibility timer cleared.
; ============================================================================

; ----------------------------------------------------------------------------
; Extracts one object type's starting HP from a packed byte of Obj_TypeHpPairs.
; IN:   A = byte of Obj_TypeHpPairs (index = type / 2); [00] = object type.
; OUT:  A = HP: the type's nibble * $10 (low nibble always 0).
; NOTE: Even types use the high nibble, odd types the low nibble. Weapon damage values are multiples
;       of $10 as well ($10 = the wooden sword's damage).
; ----------------------------------------------------------------------------
Obj_ExtractStartHp:
    PHA

    ; Bit 0 of the type into carry.
    LDA $00
    LSR
    BCS @OddType

    ; Even type: keep the high nibble.
    PLA
    AND #$F0
    RTS

@OddType:
    PLA

    ; Odd type: move the low nibble up.
    ASL
    ASL
    ASL
    ASL
    RTS

; ----------------------------------------------------------------------------
; Requests the boss death sounds: DMC sample 2 (the boss cry) and noise effect $80.
; ----------------------------------------------------------------------------
Sample_PlayBossDeathCry:
    LDA #$02
    STA SampleRequest
    LDA #$80
    STA EffectRequest
    RTS

; TABLE: object types that never drop an item; index = 0-6; value = object type.
Drop_NoDropTypes:
    .BYTE $5D, $14, $15, $1B, $1C, $1D, $17        ; dead dummy, child Gel, Gel, Keese x3, Like-Like

; TABLE: object types of drop group 0; index = 0-5; value = object type.
Drop_Group0Types:
    .BYTE $07, $08, $0E, $04, $0F, $23    ; red Octorok slow/fast, Tektite $0E, Moblin $04, Blue Leever,
                                          ; Blue Wizzrobe

; TABLE: object types of drop group 1; index = 0-8; value = object type.
Drop_Group1Types:
    .BYTE $21, $22, $0D, $10, $13, $28, $2A, $27    ; Ghini, flying Ghini, Tektite $0D, Red Leever, Zol,
                                                    ; Rope, Stalfos, Wallmaster
    .BYTE $16                                       ; Pols Voice

; TABLE: object types of drop group 2; index = 0-8; value = object type.
Drop_Group2Types:
    .BYTE $09, $0A, $03, $01, $12, $06, $0B, $24    ; blue Octorok slow/fast, Moblin $03, Lynel $01, Vire,
                                                    ; Goriya $06, Darknut $0B, Red Wizzrobe
    .BYTE $30                                       ; Gibdo

; TABLE: offset of each group's row in Drop_ItemTable; index = drop group 0-3; value = group * 10.
Drop_GroupRowOffsets:
    .BYTE $00, $0A, $14, $1E

; TABLE: drop rate; index = drop group 0-3; value = threshold: an unforced drop happens when the
;        random byte is below it ($50 = 80/256, $98 = 152/256, $68 = 104/256).
Drop_GroupRates:
    .BYTE $50, $98, $68, $68

; TABLE: dropped item ids; index = drop group * 10 + WorldKillCycle (0-9); value = item id
;        ($22 heart, $18 rupee, $0F 5 rupees, $23 fairy, $21 clock, $00 bomb).
;        The rows are 10 long, so they do not line up with the 8-byte lines below:
;        group 0: heart, rupee, heart, rupee, fairy, rupee, heart, heart, rupee, rupee
;        group 1: 5 rupees, rupee, heart, rupee, 5 rupees, heart, clock, rupee, rupee, rupee
;        group 2: heart, bomb, rupee, clock, rupee, heart, bomb, rupee, bomb, heart
;        group 3: heart, heart, fairy, rupee, heart, fairy, heart, heart, heart, rupee
Drop_ItemTable:
    .BYTE $22, $18, $22, $18, $23, $18, $22, $22    ; group 0, cycle 0-7
    .BYTE $18, $18, $0F, $18, $22, $18, $0F, $22    ; group 0, cycle 8-9; group 1, cycle 0-5
    .BYTE $21, $18, $18, $18, $22, $00, $18, $21    ; group 1, cycle 6-9; group 2, cycle 0-3
    .BYTE $18, $22, $00, $18, $00, $22, $22, $22    ; group 2, cycle 4-9; group 3, cycle 0-1
    .BYTE $23, $18, $22, $23, $22, $22, $22, $18    ; group 3, cycle 2-9

; ----------------------------------------------------------------------------
; Decides what a monster that finished dying drops, and makes the slot that item or frees it.
; IN:   X = object slot, already turned into type $60; Item_ObjMonsterType,X = the dead monster's type.
; OUT:  Either Item_ObjItemId,X = item id and Item_ObjItemLifetime,X = $FF, or the slot is freed.
;       [00] = item id, [01] = drop group (0-3).
; NOTE: Order of the decisions:
;       1. A type in Drop_NoDropTypes drops nothing.
;       2. A Stalfos or Gibdo in slot 1 drops nothing (slot 1 is the monster that carries the room item).
;       3. The table item is Drop_ItemTable[group * 10 + WorldKillCycle]. The caller advanced the cycle
;          for this kill first, so a kill made with the cycle at 0 uses column 1.
;       4. WorldKillCount = $10: a fairy instead, always.
;       5. Otherwise HelpDropCount >= $0A: a bomb if IsHelpDropBomb <> 0, else 5 rupees, always.
;       6. Otherwise the table item, only if Random,X < Drop_GroupRates[group].
;       Cases 4 and 5 zero HelpDropCount and IsHelpDropBomb. WorldKillCount is not changed here, so
;       every death that ends while it reads $10 gives a fairy. WorldKillCount and HelpDropCount are
;       incremented when the killing hit lands, this routine runs when the death sparkle ends: kills
;       made in between count. The random byte is the one indexed by the object slot.
; ----------------------------------------------------------------------------
Drop_SetUpItem:
    ; [01] = drop group, the row of the drop item table. Start with 0.
    LDA #$00
    STA $01

    ; The 7 types that never drop: go free the slot.
    LDA Item_ObjMonsterType, X
    LDY #$06

@FindNoDropType:
    CMP Drop_NoDropTypes, Y
    BEQ @FreeSlot
    DEY
    BPL @FindNoDropType

    ; The 6 types of group 0.
    LDY #$05

@FindDrop0Type:
    CMP Drop_Group0Types, Y
    BEQ @Found
    DEY
    BPL @FindDrop0Type

    INC $01

    ; The 9 types of group 1.
    LDY #$08

@FindDrop1Type:
    CMP Drop_Group1Types, Y
    BEQ @Found
    DEY
    BPL @FindDrop1Type

    INC $01

    ; The 9 types of group 2.
    LDY #$08

@FindDrop2Type:
    CMP Drop_Group2Types, Y
    BEQ @Found
    DEY
    BPL @FindDrop2Type

    ; Every other type is in group 3.
    INC $01

@Found:
    ; A Stalfos ($2A) or Gibdo ($30) in object slot 1 drops nothing: go free the slot.
    ; (The room item is drawn on a Stalfos, Gibdo or Like-Like in slot 1; see Item_MoveAndDrawRoomItem.)
    CPX #$01
    BNE @LookUpItem
    CMP #$2A
    BEQ @FreeSlot
    CMP #$30
    BEQ @FreeSlot

@LookUpItem:
    ; [00] = Drop_ItemTable[group * 10 + WorldKillCycle].
    LDY $01
    LDA Drop_GroupRowOffsets, Y
    CLC
    ADC WorldKillCycle
    TAY
    LDA Drop_ItemTable, Y
    STA $00                     ; [00] Dropped item ID

    ; If WorldKillCount = $10 (16 kills since Link was last harmed), drop a fairy ($23) instead.
    LDA #$23
    LDY WorldKillCount
    CPY #$10
    BEQ @SetHelpItem

    ; If the help drop counter < $0A, keep the table item, subject to the random test.
    LDA HelpDropCount
    CMP #$0A
    BCC @RandomlyCancel

    ; Help drop (10 kills since Link was last harmed): guaranteed.
    ; IsHelpDropBomb = 0: 5 rupees ($0F). Otherwise: a bomb ($00).
    LDA #$0F
    LDY IsHelpDropBomb
    BEQ @SetHelpItem
    LDA #$00                    ; Bomb

@SetHelpItem:
    STA $00                     ; [00] Item ID

    ; The forced drop (fairy, bomb or 5 rupees) was given. Reset the help drop variables.
    LDA #$00
    STA HelpDropCount
    STA IsHelpDropBomb
    BEQ @Commit                 ; Always taken.

@RandomlyCancel:
    ; If the random byte of this slot >= the group's drop rate, drop nothing: go free the slot.
    LDY $01
    LDA Random, X
    CMP Drop_GroupRates, Y
    BCS @FreeSlot

@Commit:
    ; The item starts with lifetime $FF.
    LDA #$FF
    STA Item_ObjItemLifetime, X

    LDA $00
    STA Item_ObjItemId, X

    ; A fairy flies: set up its flight.
    CMP #$23
    BNE :+
    JMP Fairy_Init
:
    RTS

@FreeSlot:
    JMP Obj_DestroyMonster_Bank4

; TABLE: object slots whose position can pick up a dropped item; index = 0-3 (tested 3 down to 0);
;        value = object slot: $00 Link, $0F boomerang or food, $0D sword, $12 arrow or rod.
Drop_ItemTakerObjSlots:
    .BYTE $00, $0F, $0D, $12

; ----------------------------------------------------------------------------
; Updates a dropped item (type $60): ages it, draws it, and lets Link or his weapons take it.
; IN:   X = object slot (= CurObjIndex).
; OUT:  Slot freed when the lifetime runs out or the item is taken.
; NOTE: The lifetime starts at $FF and drops by 1 on every frame with an odd FrameCounter: the item
;       is gone after about 510 frames. While the lifetime is >= $F0 (about the first 32 frames) it
;       flashes (Item_DrawObject) and cannot be taken (Item_TryTake).
;       Takers, in this order: slot $12 (arrow or rod), slot $0D (sword), slot $0F (boomerang or food),
;       then Link. A weapon slot counts while its state is nonzero with bit 7 clear; which object is
;       in the slot is not tested. The test is Item_TryTake with the taker's position in place of
;       Link's: both axis distances < 9 (the taker's Y + 3 against the item's Y).
;       Nothing is taken while Link is halted (ObjState & $C0 = $40), but the item keeps aging.
;       A fairy ($23) flies around instead of lying still.
; ----------------------------------------------------------------------------
Drop_UpdateItem:
    ; On odd frames, decrement the item lifetime.
    LDA FrameCounter
    LSR
    BCC :+
    DEC Item_ObjItemLifetime, X
:
    ; If the lifetime reached 0, free the slot.
    LDA Item_ObjItemLifetime, X
    BEQ Obj_DestroyMonster_Bank4

    ; Draw the item. A fairy moves and is drawn by its own routine.
    LDA Item_ObjItemId, X
    CMP #$23
    BEQ @AnimateFairy
    JSR Item_DrawObject
    JMP :+

@AnimateFairy:
    JSR Fairy_Update
:
    ; If Link is halted, then return.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ @Exit

    ; Besides Link, three weapon slots can pick up items: $12, $0D and $0F.
    ;
    ; For each one that's active, substitute its position for Link's and try taking the item.
    ; Link himself is tried last.
    LDA #$04
    STA $0D                     ; [0D] = 1-based index into Drop_ItemTakerObjSlots, 4 down to 1.

@LoopItemTaker:
    ; Save Link's position.
    LDA ObjX
    PHA
    LDA ObjY
    PHA

    ; Set Link's position to the taker's.
    LDY $0D
    LDX Drop_ItemTakerObjSlots-1, Y
    LDA ObjX, X
    STA ObjX
    LDA ObjY, X
    STA ObjY
    CPX #$00
    BEQ @TryToTake              ; Link himself needs no activity test.

    ; A weapon slot that is empty (state 0) or has bit 7 of its state set does not take items.
    LDA ObjState, X
    BEQ @SkipTaking
    BMI @SkipTaking

@TryToTake:
    ; Try to take the item ([04] = item id).
    LDX CurObjIndex
    LDA Item_ObjItemId, X
    STA $04
    JSR Item_TryTake

@SkipTaking:
    ; Restore Link's position.
    PLA
    STA ObjY
    PLA
    STA ObjX
    LDX CurObjIndex             ; Reload object index in X.

    ; Item_TryTake marks a taken item by setting its item id to $FF. If so, free the slot.
    LDA Item_ObjItemId, X
    CMP #$FF
    BEQ Obj_DestroyMonster_Bank4
    DEC $0D
    BNE @LoopItemTaker          ; More takers to try.

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Frees an object slot: type 0, no knockback, timer, state and invincibility timer cleared.
; IN:   X = object slot.
; OUT:  A = 1; ObjUninitialized,X = $FF; ObjMetastate,X = 1.
; NOTE: Bank 4's own copy of what Obj_DestroyMonster does in the fixed bank.
; ----------------------------------------------------------------------------
Obj_DestroyMonster_Bank4:
    LDA #$00
    STA ObjType, X
    JSR Shove_SetDirAndDistance
    STA ObjTimer, X
    STA ObjState, X
    STA ObjInvincibilityTimer, X
    LDA #$FF
    STA ObjUninitialized, X
    LDA #$01
    STA ObjMetastate, X
    RTS
