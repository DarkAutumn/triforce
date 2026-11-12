; ============================================================================
; FILE:     bank7/object-update-table.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5265-5518
; SUBJECTS: (none)
; TOPIC:    Type attributes, HP pairs, UpdateObject table, death to drop
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 ObjectTypeToAttributes  data
;   L0036 ObjectTypeToHpPairs     data  TRIFORCE: HP lookup table, 2 enemies packed per byte. Even types use high nibble (AND $F0),
;   L0046 UpdateObject            code  Params:
;   L0101 UpdateObject_JumpTable  data
;   L0209 UpdateMetaObject        code
;   L0218 DoNothing               code
;   L0221 UpdateMetaObjectEnd     code
; ============================================================================

ObjectTypeToAttributes:
    .BYTE $FF, $00, $00, $00, $00, $00, $00, $05
    .BYTE $05, $05, $05, $81, $81, $81, $81, $01
    .BYTE $01, $81, $01, $01, $43, $43, $81, $81
    .BYTE $81, $81, $01, $81, $81, $81, $01, $81
    .BYTE $81, $81, $81, $81, $81, $C3, $C3, $89
    .BYTE $89, $81, $81, $89, $89, $89, $89, $83
    .BYTE $81, $89, $89, $C9, $C9, $81, $81, $81
    .BYTE $A9, $A9, $41, $41, $89, $89, $81, $81
    .BYTE $81, $C1, $C1, $C1, $C1, $C1, $81, $81
    .BYTE $81, $A1, $A1, $81, $81, $81, $81, $81
    .BYTE $81, $81, $81, $E3, $E3, $E3, $E3, $E3
    .BYTE $E1, $E1, $E1, $E1, $E1, $81, $81

; TRIFORCE: HP lookup table, 2 enemies packed per byte. Even types use high nibble (AND $F0),
; TRIFORCE: odd types use low nibble (<<4). Result is always multiple of $10; Python reads >>4.
; TRIFORCE: 0-HP entries: Gel($14,$15), Keese($1B-$1D) — die in one hit.
; TRIFORCE: Verified empirically: goriya=$30(3HP), stalfos=$20(2HP), keese=$00.
ObjectTypeToHpPairs:
    .BYTE $06, $43, $25, $31, $12, $24, $81, $14
    .BYTE $22, $42, $00, $A9, $8F, $20, $00, $3F
    .BYTE $F9, $FA, $46, $62, $11, $2F, $FF, $FF
    .BYTE $7F, $F6, $2F, $FF, $FF, $22, $46, $F1
    .BYTE $F2, $AA, $AA, $FB, $BF, $F0

; Params:
; A: object type
;
UpdateObject:
    PHA
    LDA #$04
    JSR SwitchBank
    PLA

    ; If the object was initialized, go update it.
    LDY ObjUninitialized, X
    STY $0F                     ; [0F] holds the original value of "uninitialized" flag.
    BEQ @Update

    ; If the object type starts with a spawning cloud
    ; (type < $53, except armos and flying ghini),
    ; then set object timer to 7.
    ;
    ; But this will be overridden during initialization.
    LDA ObjType, X
    CMP #$1E                    ; Armos
    BEQ @Init
    CMP #$22                    ; Flying Ghini
    BEQ @Init
    CMP #$53                    ; Flying Rock
    BCS @Init
    LDA #$07
    STA ObjTimer, X

@Init:
    ; Flag the object initialized, and go initialize it.
    LDA #$00
    STA ObjUninitialized, X
    JMP InitObject

@Update:
    ; The object was already initialized.
    ;
    ; If meta-state <> 0, go update the meta-object
    ; (spawning cloud or dying sparkle).
    LDY ObjMetastate, X
    BEQ :+
    JMP UpdateMetaObject
:
    ; This is a normal update.
    ;
    ; If object type >= $6A, then switch to bank 1,
    ; and go update a cave.
    CMP #$6A
    BCC @NormalObject
    LDA #$01
    JSR SwitchBank
    JMP UpdateCavePerson

@NormalObject:
    ; Else call the object update routine.
    JSR TableJump

UpdateObject_JumpTable:
    .ADDR DoNothing
    .ADDR UpdateLynel
    .ADDR UpdateLynel
    .ADDR UpdateMoblin
    .ADDR UpdateMoblin
    .ADDR UpdateGoriya
    .ADDR UpdateGoriya
    .ADDR UpdateOctorock
    .ADDR UpdateOctorock
    .ADDR UpdateOctorock
    .ADDR UpdateOctorock
    .ADDR UpdateDarknut
    .ADDR UpdateDarknut
    .ADDR UpdateTektiteOrBoulder
    .ADDR UpdateTektiteOrBoulder
    .ADDR UpdateBlueLeever
    .ADDR UpdateRedLeever
    .ADDR UpdateZora
    .ADDR UpdateVire
    .ADDR UpdateZol
    .ADDR UpdateGel
    .ADDR UpdateGel
    .ADDR UpdatePolsVoice
    .ADDR UpdateLikeLike
    .ADDR UpdateDigdogger
    .ADDR DoNothing
    .ADDR UpdatePeahat
    .ADDR UpdateKeese
    .ADDR UpdateKeese
    .ADDR UpdateKeese
    .ADDR UpdateArmos
    .ADDR UpdateBoulderSet
    .ADDR UpdateTektiteOrBoulder
    .ADDR UpdateGhini
    .ADDR UpdateFlyingGhini
    .ADDR UpdateBlueWizzrobe
    .ADDR UpdateRedWizzrobe
    .ADDR UpdatePatraChild
    .ADDR UpdatePatraChild
    .ADDR UpdateWallmaster
    .ADDR UpdateRope
    .ADDR DoNothing
    .ADDR UpdateStalfos
    .ADDR UpdateBubble
    .ADDR UpdateBubble
    .ADDR UpdateBubble
    .ADDR UpdateWhirlwind
    .ADDR UpdatePondFairy
    .ADDR UpdateGibdo
    .ADDR UpdateDodongo
    .ADDR UpdateDodongo
    .ADDR UpdateGohma
    .ADDR UpdateGohma
    .ADDR UpdateRupeeStash
    .ADDR UpdateGrumble
    .ADDR UpdateZelda
    .ADDR UpdateDigdogger
    .ADDR UpdateDigdogger
    .ADDR UpdateLamnola
    .ADDR UpdateLamnola
    .ADDR UpdateManhandla
    .ADDR UpdateAquamentus
    .ADDR UpdateGanon
    .ADDR UpdateGuardFire
    .ADDR UpdateStandingFire
    .ADDR UpdateMoldorm
    .ADDR UpdateGleeok
    .ADDR UpdateGleeok
    .ADDR UpdateGleeok
    .ADDR UpdateGleeok
    .ADDR UpdateGleeokHead
    .ADDR UpdatePatra
    .ADDR UpdatePatra
    .ADDR UpdateTrap
    .ADDR UpdateTrap
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateUnderworldPersonLifeOrMoney
    .ADDR UpdateUnderworldPerson
    .ADDR UpdateMonsterShot
    .ADDR UpdateMonsterShot
    .ADDR UpdateFireball
    .ADDR UpdateFireball
    .ADDR UpdateMonsterShot
    .ADDR UpdateMonsterShot
    .ADDR UpdateMonsterShot
    .ADDR UpdateMonsterShot
    .ADDR UpdateMonsterArrow
    .ADDR UpdateArrowOrBoomerang
    .ADDR UpdateDeadDummy
    .ADDR UpdateFluteSecret
    .ADDR DoNothing
    .ADDR UpdateItem
    .ADDR UpdateDock
    .ADDR UpdateRockOrGravestone
    .ADDR UpdateRockWall
    .ADDR UpdateTree
    .ADDR UpdateRockOrGravestone
    .ADDR UpdateRockOrGravestone
    .ADDR UpdateRockWall
    .ADDR UpdateBlock
    .ADDR DoNothing

UpdateMetaObject:
    JSR AnimateAndDrawMetaObject

    ; Go handle end metastates (4 and $14) specially.
    LDA ObjMetastate, X
    AND #$0F
    CMP #$04
    BCS UpdateMetaObjectEnd

DoNothing:
    RTS

UpdateMetaObjectEnd:
    ; Metastate = 4 or $14.
    ;
    ; If it's 4, go reset the metastate. The object is ready to update
    ; on its own.
    LDA ObjMetastate, X
    AND #$10
    BEQ @Reset

    ; Metastate = $14
    ;
    ; Copy the object type, so that it can be used in setting up
    ; a dropped item.
    LDA ObjType, X
    STA Item_ObjMonsterType, X

    ; Certain objects don't advance the world kill cycle.
    CMP #$5D
    BEQ @DropItem
    CMP #$14                    ; Child Gel
    BEQ @DropItem
    CMP #$1C                    ; Red Keese
    BEQ @DropItem

    ; Advance the world kill cycle (0 to 9).
    LDA WorldKillCycle
    CLC
    ADC #$01
    CMP #$0A
    BNE :+
    LDA #$00
:
    STA WorldKillCycle

    ; If the object is not a zora, then increase room kill count.
    LDA ObjType, X
    CMP #$11                    ; Zora
    BEQ @DropItem
    INC RoomKillCount

@DropItem:
    ; Turn this object into a dropped item.
    LDA #$60
    STA ObjType, X
    STA ObjUninitialized, X     ; Flag it uninitialized.
    LDA #$81                    ; "Custom collision check and drawing" + "Reverse after hitting Link"
    STA ObjAttr, X
    JSR SetUpDroppedItem

@Reset:
    JMP ResetObjMetastate
