; ============================================================================
; FILE:     bank7/object-update-table.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5265-5518
; SUBJECTS: Obj
; TOPIC:    Per-object-type tables (behaviour attribute bits, packed HP) and the per-frame object
;           update dispatch: first-frame init, spawn cloud / death sparkle metastates, the update
;           routine of each type, and the end of a death (kill cycle, room kill count, dropped item).
; SEE ALSO: bank7/object-init-table.asm (Obj_Init reads both tables; Obj_Init_JumpTable);
;           bank7/mode5-play-update.asm (Mode5Play_Update calls Obj_Update for slots $0B..$01 and
;           tests attribute bits $01/$04); bank7/object-animation.asm (Obj_AnimateMetastate);
;           bank4/drop-items.asm (Obj_ExtractStartHp, Drop_SetUpItem, Drop_NoDropTypes);
;           bank1/monster-collisions.asm (Collision_CheckMonsterVsWeaponsAndLink: attribute bits $20, $40);
;           bank1/shove-knockback.asm (Shove_Begin: attribute bit $80);
;           bank7/walker-collision-screen-edge.asm (attribute bit $10);
;           bank1/object-animation-draw.asm (attribute bits $02, $08).
; KEY FACTS:
;   - Obj_TypeAttributes bits: $01 the type's own update draws it and checks its collisions (the
;     object loop does neither); $02 drawn half width; $04 the object loop does not draw it; $08 no
;     sprite attribute lookup when drawn; $10 reverse when blocked; $20 Link's weapons are not tested
;     against it; $40 half width for collisions; $80 it does not turn around after hitting Link.
;   - No type in Obj_TypeAttributes has bit $10 (only the unused entry 0, $FF), and no code found
;     sets it later, so the "reverse when blocked" branches of the walker code are not taken.
;   - Types $01-$06 (Lynel, Moblin, Goriya) have attribute $00: the object loop draws them as generic
;     walkers and checks their collisions. Octoroks ($07-$0A) have $05 (Obj_TypeAttributes).
;   - Weapons cannot touch (bit $20): Digdogger $38/$39, Trap $49/$4A, monster shots $53-$5C.
;   - HP is one nibble per type in Obj_TypeHpPairs (even type = high nibble, odd = low); ObjHP is
;     the nibble x $10. Nibble 0 (Gel, Keese): any weapon hit kills, even one of damage 0 (the
;     boomerang). The table ends at type $4B.
;   - Obj_Update order per slot: uninitialized -> Obj_Init (no update that frame); metastate <> 0
;     -> cloud or sparkle only; type >= $6A -> Cave_UpdatePerson (bank 1); else Obj_Update_JumpTable.
;   - Spawn cloud total: slot-number frames (first step, set by Obj_Init) + 6 + 6 frames; then the
;     metastate returns to 0 (Obj_EndMetastate). Armos ($1E), flying Ghini ($22) and types >= $53
;     get no timer for the first step.
;   - When a death sparkle ends (metastate $14): WorldKillCycle advances 0..9 except for types $5D
;     (dead dummy), $14 (child Gel), $1C (red Keese); RoomKillCount +1 except for those and Zora $11
;     (Obj_EndMetastate).
;   - The dead object then becomes type $60 (dropped item), uninitialized, attribute $81, with
;     Item_ObjMonsterType = the old type; Drop_SetUpItem decides what, if anything, drops.
; CONTENTS: (generated; L = line in this file)
;   L0068 Obj_TypeAttributes    data  TABLE: behaviour attribute bits per object type; index = object type $00-$5E; value = bits, copied
;   L0092 Obj_TypeHpPairs       data  TABLE: starting HP, two object types per byte; index = object type / 2; value = HP nibble of the
;   L0107 Obj_Update            code  Runs one object slot for this frame: its init, its spawn cloud or death sparkle, or its update.
;   L0159 Obj_Update_JumpTable  data  TABLE: update routine per object type; index = object type $00-$69; value = routine address.
;   L0272 Obj_UpdateMetastate   code  Draws and times an object's spawn cloud or death sparkle; hands the last step to the end routine.
;   L0282 Obj_DoNothing         code  Does nothing (RTS): the update or init routine of object types that have none of their own.
;   L0295 Obj_EndMetastate      code  Ends a spawn cloud (object starts acting) or a death (kill counters, slot becomes a dropped item).
; ============================================================================

; TABLE: behaviour attribute bits per object type; index = object type $00-$5E; value = bits, copied
; to ObjAttr by Obj_Init.
;   $01 the type's update routine draws the object and checks its collisions itself
;   $02 drawn half width                $04 the object loop does not draw it (tested only when $01 is 0)
;   $08 drawn without the sprite attribute lookup
;   $10 reverse when blocked (no type has it)
;   $20 Link's weapons are not tested against it (it can still hit Link)
;   $40 half width for collisions       $80 does not turn around after hitting Link
; Values by type (names are on the rows of Obj_Update_JumpTable):
;   $00  $01-$06                        $05  $07-$0A
;   $01  $0F $10 $12 $13 $1A $1E        $43  $14 $15
;   $C3  $25 $26                        $89  $27 $28 $2B-$2E $31 $32 $3C $3D
;   $83  $2F                            $C9  $33 $34
;   $A9  $38 $39                        $41  $3A $3B
;   $C1  $41-$45                        $A1  $49 $4A
;   $E3  $53-$57                        $E1  $58-$5C
;   $81  every other type from $0B to $5E;  $FF  type $00 (no object)
; NOTE: Types $5F and up index past the table; Obj_Init then overwrites their attribute with $81
;       (tile objects and items) or the cave init does (types >= $6A).
Obj_TypeAttributes:
    .BYTE $FF, $00, $00, $00, $00, $00, $00, $05    ; types $00-$07
    .BYTE $05, $05, $05, $81, $81, $81, $81, $01    ; $08-$0F
    .BYTE $01, $81, $01, $01, $43, $43, $81, $81    ; $10-$17
    .BYTE $81, $81, $01, $81, $81, $81, $01, $81    ; $18-$1F
    .BYTE $81, $81, $81, $81, $81, $C3, $C3, $89    ; $20-$27
    .BYTE $89, $81, $81, $89, $89, $89, $89, $83    ; $28-$2F
    .BYTE $81, $89, $89, $C9, $C9, $81, $81, $81    ; $30-$37
    .BYTE $A9, $A9, $41, $41, $89, $89, $81, $81    ; $38-$3F
    .BYTE $81, $C1, $C1, $C1, $C1, $C1, $81, $81    ; $40-$47
    .BYTE $81, $A1, $A1, $81, $81, $81, $81, $81    ; $48-$4F
    .BYTE $81, $81, $81, $E3, $E3, $E3, $E3, $E3    ; $50-$57
    .BYTE $E1, $E1, $E1, $E1, $E1, $81, $81         ; $58-$5E

; TABLE: starting HP, two object types per byte; index = object type / 2; value = HP nibble of the
; even type (high nibble) and of the odd type (low nibble). Obj_ExtractStartHp turns the nibble
; into nibble x $10, which Obj_Init stores in ObjHP (always a multiple of $10).
; Each type's nibble is on its row of Obj_Update_JumpTable. Nibble 0 (Gels $14/$15, Keese $1B-$1D):
; any weapon hit kills, even one of damage 0 (Damage_DealToMonster). Nibble $F is given to types that are not
; killed through ordinary HP loss (Boulder, Bubble, Dodongo, Digdogger, Ganon, Trap, ...; each
; type's own code decides, not checked here: (unverified)).
; NOTE: 38 bytes cover types $00-$4B. Obj_Init does not bound the index, so types $4C and up are
;       given a nibble of the bytes that follow the table (the code of Obj_Update: $48, $A9, $04,
;       $20 give types $4C-$53 the nibbles 4, 8, $A, 9, 0, 4, 2, 0).
Obj_TypeHpPairs:
    .BYTE $06, $43, $25, $31, $12, $24, $81, $14    ; types $00/$01 .. $0E/$0F
    .BYTE $22, $42, $00, $A9, $8F, $20, $00, $3F    ; $10/$11 .. $1E/$1F
    .BYTE $F9, $FA, $46, $62, $11, $2F, $FF, $FF    ; $20/$21 .. $2E/$2F
    .BYTE $7F, $F6, $2F, $FF, $FF, $22, $46, $F1    ; $30/$31 .. $3E/$3F
    .BYTE $F2, $AA, $AA, $FB, $BF, $F0              ; $40/$41 .. $4A/$4B

; ----------------------------------------------------------------------------
; Runs one object slot for this frame: its init, its spawn cloud or death sparkle, or its update.
; IN:   A = object type (ObjType,X, nonzero); X = object slot ($01-$0B, = CurObjIndex)
; OUT:  [0F] = ObjUninitialized,X as it was on entry; bank 4 switched in (bank 1 for cave persons)
; NOTE: An uninitialized slot only runs Obj_Init this frame. While ObjMetastate <> 0 the type's
;       update routine does not run (the object neither moves nor is hit). Types >= $6A go to
;       Cave_UpdatePerson; the rest through Obj_Update_JumpTable.
; ----------------------------------------------------------------------------
Obj_Update:
    PHA
    LDA #$04
    JSR Mmc1_SwitchBank
    PLA

    ; If the object was initialized, go update it.
    LDY ObjUninitialized, X
    STY $0F                     ; [0F] = the "uninitialized" flag as it was.
    BEQ @Update

    ; Types that start with a spawn cloud (type < $53, except Armos and flying Ghini) get object
    ; timer 7 here. Obj_Init, when it goes through with the init, replaces it with the slot
    ; number.
    LDA ObjType, X
    CMP #$1E                    ; Armos
    BEQ @Init
    CMP #$22                    ; Flying Ghini
    BEQ @Init
    CMP #$53                    ; Flying rock: first of the types without a cloud
    BCS @Init
    LDA #$07
    STA ObjTimer, X

@Init:
    ; Flag the object initialized, and go initialize it.
    LDA #$00
    STA ObjUninitialized, X
    JMP Obj_Init

@Update:
    ; The object was already initialized. A still holds the object type.
    ;
    ; Metastate <> 0: only animate the spawn cloud or the death sparkle.
    LDY ObjMetastate, X
    BEQ :+
    JMP Obj_UpdateMetastate
:
    ; Normal update. Types >= $6A are cave persons: their update is in bank 1.
    CMP #$6A
    BCC @NormalObject
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Cave_UpdatePerson

@NormalObject:
    ; Jump to the type's update routine (A = object type indexes the table below).
    JSR Util_TableJump

; TABLE: update routine per object type; index = object type $00-$69; value = routine address.
; Row comments: type, name, attribute byte (Obj_TypeAttributes), HP nibble (Obj_TypeHpPairs;
; "-" = past the end of that table).
Obj_Update_JumpTable:
    .ADDR Obj_DoNothing                             ; $00 no object                attr $FF  HP 0
    .ADDR Lynel_Update                           ; $01 Lynel (blue)             attr $00  HP 6
    .ADDR Lynel_Update                           ; $02 Lynel (red)              attr $00  HP 4
    .ADDR Moblin_Update                          ; $03 Moblin (blue)            attr $00  HP 3
    .ADDR Moblin_Update                          ; $04 Moblin (red)             attr $00  HP 2
    .ADDR Goriya_Update                          ; $05 Goriya (blue)            attr $00  HP 5
    .ADDR Goriya_Update                          ; $06 Goriya (red)             attr $00  HP 3
    .ADDR Octorok_Update                        ; $07 Octorok, slow            attr $05  HP 1
    .ADDR Octorok_Update                        ; $08 Octorok, fast            attr $05  HP 1
    .ADDR Octorok_Update                        ; $09 blue Octorok, slow       attr $05  HP 2
    .ADDR Octorok_Update                        ; $0A blue Octorok, fast       attr $05  HP 2
    .ADDR Darknut_Update                         ; $0B Darknut (red)            attr $81  HP 4
    .ADDR Darknut_Update                         ; $0C Darknut (blue)           attr $81  HP 8
    .ADDR Jumper_Update                ; $0D Tektite (blue)           attr $81  HP 1
    .ADDR Jumper_Update                ; $0E Tektite (red)            attr $81  HP 1
    .ADDR Leever_UpdateBlue                      ; $0F Blue Leever              attr $01  HP 4
    .ADDR Leever_UpdateRed                       ; $10 Red Leever               attr $01  HP 2
    .ADDR Zora_Update                            ; $11 Zora                     attr $81  HP 2
    .ADDR Vire_Update                            ; $12 Vire                     attr $01  HP 4
    .ADDR Zol_Update                             ; $13 Zol                      attr $01  HP 2
    .ADDR Gel_Update                             ; $14 child Gel (from a Zol)   attr $43  HP 0
    .ADDR Gel_Update                             ; $15 Gel                      attr $43  HP 0
    .ADDR PolsVoice_Update                       ; $16 Pols Voice               attr $81  HP A
    .ADDR LikeLike_Update                        ; $17 Like-Like                attr $81  HP 9
    .ADDR Digdogger_Update                       ; $18 small Digdogger          attr $81  HP 8
    .ADDR Obj_DoNothing                             ; $19 (no update routine)      attr $81  HP F
    .ADDR Peahat_Update                          ; $1A Peahat                   attr $01  HP 2
    .ADDR Keese_Update                           ; $1B Keese (blue)             attr $81  HP 0
    .ADDR Keese_Update                           ; $1C Keese (red)              attr $81  HP 0
    .ADDR Keese_Update                           ; $1D Keese (black)            attr $81  HP 0
    .ADDR Armos_Update                           ; $1E Armos                    attr $01  HP 3
    .ADDR Boulder_UpdateSpawner                      ; $1F boulder spawner          attr $81  HP F
    .ADDR Jumper_Update                ; $20 Boulder                  attr $81  HP F
    .ADDR Ghini_Update                           ; $21 Ghini                    attr $81  HP 9
    .ADDR Ghini_UpdateFlying                     ; $22 flying Ghini             attr $81  HP F
    .ADDR Wizzrobe_Blue_Update                    ; $23 Blue Wizzrobe            attr $81  HP A
    .ADDR Wizzrobe_Red_Update                     ; $24 Red Wizzrobe             attr $81  HP 4
    .ADDR Patra_UpdateChild                      ; $25 Patra child              attr $C3  HP 6
    .ADDR Patra_UpdateChild                      ; $26 Patra child              attr $C3  HP 6
    .ADDR Wallmaster_Update                      ; $27 Wallmaster               attr $89  HP 2
    .ADDR Rope_Update                            ; $28 Rope                     attr $89  HP 1
    .ADDR Obj_DoNothing                             ; $29 (no update routine)      attr $81  HP 1
    .ADDR Stalfos_Update                         ; $2A Stalfos                  attr $81  HP 2
    .ADDR Bubble_Update                          ; $2B Bubble                   attr $89  HP F
    .ADDR Bubble_Update                          ; $2C Bubble                   attr $89  HP F
    .ADDR Bubble_Update                          ; $2D Bubble                   attr $89  HP F
    .ADDR Whirlwind_Update                       ; $2E Whirlwind                attr $89  HP F
    .ADDR PondFairy_Update                       ; $2F Pond fairy               attr $83  HP F
    .ADDR Gibdo_Update                           ; $30 Gibdo                    attr $81  HP 7
    .ADDR Dodongo_Update                         ; $31 Dodongo                  attr $89  HP F
    .ADDR Dodongo_Update                         ; $32 Dodongo                  attr $89  HP F
    .ADDR Gohma_Update                           ; $33 Gohma                    attr $C9  HP 6
    .ADDR Gohma_Update                           ; $34 Gohma                    attr $C9  HP 2
    .ADDR RupeeStash_Update                      ; $35 Rupee stash              attr $81  HP F
    .ADDR Grumble_Update                         ; $36 Grumble Goriya           attr $81  HP F
    .ADDR Zelda_Update                           ; $37 Zelda                    attr $81  HP F
    .ADDR Digdogger_Update                       ; $38 Digdogger                attr $A9  HP F
    .ADDR Digdogger_Update                       ; $39 Digdogger                attr $A9  HP F
    .ADDR Lanmola_Update                         ; $3A Lanmola (red)            attr $41  HP 2
    .ADDR Lanmola_Update                         ; $3B Lanmola (blue)           attr $41  HP 2
    .ADDR Manhandla_Update                       ; $3C Manhandla                attr $89  HP 4
    .ADDR Aquamentus_Update                      ; $3D Aquamentus               attr $89  HP 6
    .ADDR Ganon_Update                           ; $3E Ganon                    attr $81  HP F
    .ADDR GuardFire_Update                       ; $3F Guard fire               attr $81  HP 1
    .ADDR StandingFire_Update                    ; $40 Standing fire            attr $81  HP F
    .ADDR Moldorm_Update                         ; $41 Moldorm                  attr $C1  HP 2
    .ADDR Gleeok_Update                          ; $42 Gleeok                   attr $C1  HP A
    .ADDR Gleeok_Update                          ; $43 Gleeok                   attr $C1  HP A
    .ADDR Gleeok_Update                          ; $44 Gleeok                   attr $C1  HP A
    .ADDR Gleeok_Update                          ; $45 Gleeok                   attr $C1  HP A
    .ADDR Gleeok_UpdateFlyingHead                      ; $46 flying Gleeok head       attr $81  HP F
    .ADDR Patra_Update                           ; $47 Patra                    attr $81  HP B
    .ADDR Patra_Update                           ; $48 Patra                    attr $81  HP B
    .ADDR Trap_Update                            ; $49 Trap                     attr $A1  HP F
    .ADDR Trap_Update                            ; $4A Trap                     attr $A1  HP F
    .ADDR Person_UpdateUnderworld                ; $4B dungeon person           attr $81  HP 0
    .ADDR Person_UpdateUnderworld                ; $4C dungeon person           attr $81  HP -
    .ADDR Person_UpdateUnderworld                ; $4D dungeon person           attr $81  HP -
    .ADDR Person_UpdateUnderworld                ; $4E dungeon person           attr $81  HP -
    .ADDR Person_UpdateUnderworld                ; $4F dungeon person           attr $81  HP -
    .ADDR Person_UpdateUnderworld                ; $50 dungeon person           attr $81  HP -
    .ADDR Person_UpdateLifeOrMoney     ; $51 life-or-money person     attr $81  HP -
    .ADDR Person_UpdateUnderworld                ; $52 dungeon person           attr $81  HP -
    .ADDR MonsterShot_Update                     ; $53 flying rock              attr $E3  HP -
    .ADDR MonsterShot_Update                     ; $54 monster shot             attr $E3  HP -
    .ADDR Fireball_Update                        ; $55 fireball                 attr $E3  HP -
    .ADDR Fireball_Update                        ; $56 fireball                 attr $E3  HP -
    .ADDR MonsterShot_Update                     ; $57 monster sword shot       attr $E3  HP -
    .ADDR MonsterShot_Update                     ; $58 monster magic shot       attr $E1  HP -
    .ADDR MonsterShot_Update                     ; $59 monster magic shot       attr $E1  HP -
    .ADDR MonsterShot_Update                     ; $5A monster shot             attr $E1  HP -
    .ADDR MonsterArrow_Update                    ; $5B monster arrow            attr $E1  HP -
    .ADDR Boomerang_UpdateOrArrow                ; $5C monster boomerang        attr $E1  HP -
    .ADDR DeadDummy_StartDeathSparkle                       ; $5D dead dummy               attr $81  HP -
    .ADDR FluteSecret_Update                     ; $5E flute secret             attr $81  HP -
    .ADDR Obj_DoNothing                             ; $5F ladder                   attr $81 (set at init)
    .ADDR Drop_UpdateItem                            ; $60 dropped item             attr $81 (set at init)
    .ADDR Dock_Update                            ; $61 dock                     attr $81 (set at init)
    .ADDR Rock_Update                ; $62 rock                     attr $81 (set at init)
    .ADDR RockWall_Update                        ; $63 rock wall                attr $81 (set at init)
    .ADDR Tree_Update                            ; $64 tree                     attr $81 (set at init)
    .ADDR Rock_Update                ; $65 gravestone               attr $81 (set at init)
    .ADDR Rock_Update                ; $66 armos tile object        attr $81 (set at init)
    .ADDR RockWall_Update                        ; $67 rock wall                attr $81 (set at init)
    .ADDR PushBlock_Update                           ; $68 push block               attr $81 (set at init)
    .ADDR Obj_DoNothing                             ; $69 (no update routine)      attr $81 (set at init)

; ----------------------------------------------------------------------------
; Draws and times an object's spawn cloud or death sparkle; hands the last step to the end routine.
; IN:   X = object slot with ObjMetastate <> 0
; NOTE: Returns after drawing unless the low nibble of the metastate has reached 4 (4 or $14).
; ----------------------------------------------------------------------------
Obj_UpdateMetastate:
    JSR Obj_AnimateMetastate

    ; Metastate 4 (cloud over) or $14 (sparkle over): go finish.
    LDA ObjMetastate, X
    AND #$0F
    CMP #$04
    BCS Obj_EndMetastate

; Does nothing (RTS): the update or init routine of object types that have none of their own.
Obj_DoNothing:
    RTS

; ----------------------------------------------------------------------------
; Ends a spawn cloud (object starts acting) or a death (kill counters, slot becomes a dropped item).
; IN:   X = object slot; ObjMetastate,X = 4 or $14
; OUT:  ObjMetastate,X = 0. After a death: ObjType,X = $60, ObjUninitialized,X <> 0, ObjAttr,X =
;       $81, Item_ObjMonsterType,X = the dead type, WorldKillCycle and RoomKillCount updated.
; NOTE: WorldKillCycle counts 0..9 and wraps; types $5D (dead dummy), $14 (child Gel) and $1C (red
;       Keese) do not advance it or RoomKillCount. A Zora ($11) advances the cycle but not
;       RoomKillCount. Every death goes through Drop_SetUpItem, which decides whether an item
;       appears.
; ----------------------------------------------------------------------------
Obj_EndMetastate:
    ; Metastate 4: the spawn cloud is over. Reset the metastate; the object's own update runs
    ; from the next frame.
    LDA ObjMetastate, X
    AND #$10
    BEQ @Reset

    ; Metastate $14: the death sparkle is over.
    ;
    ; Remember the object type; Drop_SetUpItem chooses the drop by it.
    LDA ObjType, X
    STA Item_ObjMonsterType, X

    ; These types advance neither the kill cycle nor the room kill count.
    CMP #$5D                    ; Dead dummy
    BEQ @DropItem
    CMP #$14                    ; Child Gel
    BEQ @DropItem
    CMP #$1C                    ; Red Keese
    BEQ @DropItem

    ; Advance the world kill cycle: 0 to 9, then back to 0.
    LDA WorldKillCycle
    CLC
    ADC #$01
    CMP #$0A
    BNE :+
    LDA #$00
:
    STA WorldKillCycle

    ; Count the kill for the room, unless the object is a Zora.
    LDA ObjType, X
    CMP #$11                    ; Zora
    BEQ @DropItem
    INC RoomKillCount

@DropItem:
    ; Turn this slot into a dropped item (type $60), to be initialized on its next update.
    LDA #$60
    STA ObjType, X
    STA ObjUninitialized, X     ; Any nonzero value flags it uninitialized.
    LDA #$81                    ; Attribute bits $01 (draws and checks collisions itself) + $80
    STA ObjAttr, X
    JSR Drop_SetUpItem

@Reset:
    JMP Obj_ResetMetastate
