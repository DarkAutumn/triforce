; ============================================================================
; FILE:     bank7/object-update-table.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5265-5518
; SUBJECTS: (none)
; TOPIC:    Per-object-type tables (behaviour attribute bits, packed HP) and the per-frame object
;           update dispatch: first-frame init, spawn cloud / death sparkle metastates, the update
;           routine of each type, and the end of a death (kill cycle, room kill count, dropped item).
; SEE ALSO: bank7/object-init-table.asm (InitObject reads both tables; InitObject_JumpTable);
;           bank7/mode5-play-update.asm (UpdateMode5Play calls UpdateObject for slots $0B..$01 and
;           tests attribute bits $01/$04); bank7/object-animation.asm (AnimateAndDrawMetaObject);
;           bank4/drop-items.asm (ExtractHitPointValue, SetUpDroppedItem, NoDropMonsterTypes);
;           bank1/monster-collisions.asm (CheckMonsterCollisions: attribute bits $20, $40);
;           bank1/shove-knockback.asm (BeginShove: attribute bit $80);
;           bank7/walker-collision-screen-edge.asm (attribute bit $10);
;           bank1/object-animation-draw.asm (attribute bits $02, $08).
; KEY FACTS:
;   - ObjectTypeToAttributes bits: $01 the type's own update draws it and checks its collisions (the
;     object loop does neither); $02 drawn half width; $04 the object loop does not draw it; $08 no
;     sprite attribute lookup when drawn; $10 reverse when blocked; $20 Link's weapons are not tested
;     against it; $40 half width for collisions; $80 it does not turn around after hitting Link.
;   - No type in ObjectTypeToAttributes has bit $10 (only the unused entry 0, $FF), and no code found
;     sets it later, so the "reverse when blocked" branches of the walker code are not taken.
;   - Types $01-$06 (Lynel, Moblin, Goriya) have attribute $00: the object loop draws them as generic
;     walkers and checks their collisions. Octoroks ($07-$0A) have $05 (ObjectTypeToAttributes).
;   - Weapons cannot touch (bit $20): Digdogger $38/$39, Trap $49/$4A, monster shots $53-$5C.
;   - HP is one nibble per type in ObjectTypeToHpPairs (even type = high nibble, odd = low); ObjHP is
;     the nibble x $10. Nibble 0 (Gel, Keese): any weapon hit kills, even one of damage 0 (the
;     boomerang). The table ends at type $4B.
;   - UpdateObject order per slot: uninitialized -> InitObject (no update that frame); metastate <> 0
;     -> cloud or sparkle only; type >= $6A -> UpdateCavePerson (bank 1); else UpdateObject_JumpTable.
;   - Spawn cloud total: slot-number frames (first step, set by InitObject) + 6 + 6 frames; then the
;     metastate returns to 0 (UpdateMetaObjectEnd). Armos ($1E), flying Ghini ($22) and types >= $53
;     get no timer for the first step.
;   - When a death sparkle ends (metastate $14): WorldKillCycle advances 0..9 except for types $5D
;     (dead dummy), $14 (child Gel), $1C (red Keese); RoomKillCount +1 except for those and Zora $11
;     (UpdateMetaObjectEnd).
;   - The dead object then becomes type $60 (dropped item), uninitialized, attribute $81, with
;     Item_ObjMonsterType = the old type; SetUpDroppedItem decides what, if anything, drops.
; CONTENTS: (generated; L = line in this file)
;   L0068 ObjectTypeToAttributes  data  TABLE: behaviour attribute bits per object type; index = object type $00-$5E; value = bits, copied
;   L0092 ObjectTypeToHpPairs     data  TABLE: starting HP, two object types per byte; index = object type / 2; value = HP nibble of the
;   L0107 UpdateObject            code  Runs one object slot for this frame: its init, its spawn cloud or death sparkle, or its update.
;   L0159 UpdateObject_JumpTable  data  TABLE: update routine per object type; index = object type $00-$69; value = routine address.
;   L0272 UpdateMetaObject        code  Draws and times an object's spawn cloud or death sparkle; hands the last step to the end routine.
;   L0282 DoNothing               code  Does nothing (RTS): the update or init routine of object types that have none of their own.
;   L0295 UpdateMetaObjectEnd     code  Ends a spawn cloud (object starts acting) or a death (kill counters, slot becomes a dropped item).
; ============================================================================

; TABLE: behaviour attribute bits per object type; index = object type $00-$5E; value = bits, copied
; to ObjAttr by InitObject.
;   $01 the type's update routine draws the object and checks its collisions itself
;   $02 drawn half width                $04 the object loop does not draw it (tested only when $01 is 0)
;   $08 drawn without the sprite attribute lookup
;   $10 reverse when blocked (no type has it)
;   $20 Link's weapons are not tested against it (it can still hit Link)
;   $40 half width for collisions       $80 does not turn around after hitting Link
; Values by type (names are on the rows of UpdateObject_JumpTable):
;   $00  $01-$06                        $05  $07-$0A
;   $01  $0F $10 $12 $13 $1A $1E        $43  $14 $15
;   $C3  $25 $26                        $89  $27 $28 $2B-$2E $31 $32 $3C $3D
;   $83  $2F                            $C9  $33 $34
;   $A9  $38 $39                        $41  $3A $3B
;   $C1  $41-$45                        $A1  $49 $4A
;   $E3  $53-$57                        $E1  $58-$5C
;   $81  every other type from $0B to $5E;  $FF  type $00 (no object)
; NOTE: Types $5F and up index past the table; InitObject then overwrites their attribute with $81
;       (tile objects and items) or the cave init does (types >= $6A).
ObjectTypeToAttributes:
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
; even type (high nibble) and of the odd type (low nibble). ExtractHitPointValue turns the nibble
; into nibble x $10, which InitObject stores in ObjHP (always a multiple of $10).
; Each type's nibble is on its row of UpdateObject_JumpTable. Nibble 0 (Gels $14/$15, Keese $1B-$1D):
; any weapon hit kills, even one of damage 0 (DealDamage). Nibble $F is given to types that are not
; killed through ordinary HP loss (Boulder, Bubble, Dodongo, Digdogger, Ganon, Trap, ...; each
; type's own code decides, not checked here: (unverified)).
; NOTE: 38 bytes cover types $00-$4B. InitObject does not bound the index, so types $4C and up are
;       given a nibble of the bytes that follow the table (the code of UpdateObject: $48, $A9, $04,
;       $20 give types $4C-$53 the nibbles 4, 8, $A, 9, 0, 4, 2, 0).
ObjectTypeToHpPairs:
    .BYTE $06, $43, $25, $31, $12, $24, $81, $14    ; types $00/$01 .. $0E/$0F
    .BYTE $22, $42, $00, $A9, $8F, $20, $00, $3F    ; $10/$11 .. $1E/$1F
    .BYTE $F9, $FA, $46, $62, $11, $2F, $FF, $FF    ; $20/$21 .. $2E/$2F
    .BYTE $7F, $F6, $2F, $FF, $FF, $22, $46, $F1    ; $30/$31 .. $3E/$3F
    .BYTE $F2, $AA, $AA, $FB, $BF, $F0              ; $40/$41 .. $4A/$4B

; ----------------------------------------------------------------------------
; Runs one object slot for this frame: its init, its spawn cloud or death sparkle, or its update.
; IN:   A = object type (ObjType,X, nonzero); X = object slot ($01-$0B, = CurObjIndex)
; OUT:  [0F] = ObjUninitialized,X as it was on entry; bank 4 switched in (bank 1 for cave persons)
; NOTE: An uninitialized slot only runs InitObject this frame. While ObjMetastate <> 0 the type's
;       update routine does not run (the object neither moves nor is hit). Types >= $6A go to
;       UpdateCavePerson; the rest through UpdateObject_JumpTable.
; ----------------------------------------------------------------------------
UpdateObject:
    PHA
    LDA #$04
    JSR SwitchBank
    PLA

    ; If the object was initialized, go update it.
    LDY ObjUninitialized, X
    STY $0F                     ; [0F] = the "uninitialized" flag as it was.
    BEQ @Update

    ; Types that start with a spawn cloud (type < $53, except Armos and flying Ghini) get object
    ; timer 7 here. InitObject, when it goes through with the init, replaces it with the slot
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
    JMP InitObject

@Update:
    ; The object was already initialized. A still holds the object type.
    ;
    ; Metastate <> 0: only animate the spawn cloud or the death sparkle.
    LDY ObjMetastate, X
    BEQ :+
    JMP UpdateMetaObject
:
    ; Normal update. Types >= $6A are cave persons: their update is in bank 1.
    CMP #$6A
    BCC @NormalObject
    LDA #$01
    JSR SwitchBank
    JMP UpdateCavePerson

@NormalObject:
    ; Jump to the type's update routine (A = object type indexes the table below).
    JSR TableJump

; TABLE: update routine per object type; index = object type $00-$69; value = routine address.
; Row comments: type, name, attribute byte (ObjectTypeToAttributes), HP nibble (ObjectTypeToHpPairs;
; "-" = past the end of that table).
UpdateObject_JumpTable:
    .ADDR DoNothing                             ; $00 no object                attr $FF  HP 0
    .ADDR UpdateLynel                           ; $01 Lynel (red)              attr $00  HP 6
    .ADDR UpdateLynel                           ; $02 Lynel (blue)             attr $00  HP 4
    .ADDR UpdateMoblin                          ; $03 Moblin (red)             attr $00  HP 3
    .ADDR UpdateMoblin                          ; $04 Moblin (blue)            attr $00  HP 2
    .ADDR UpdateGoriya                          ; $05 Goriya (red)             attr $00  HP 5
    .ADDR UpdateGoriya                          ; $06 Goriya (blue)            attr $00  HP 3
    .ADDR UpdateOctorock                        ; $07 Octorok, slow            attr $05  HP 1
    .ADDR UpdateOctorock                        ; $08 Octorok, fast            attr $05  HP 1
    .ADDR UpdateOctorock                        ; $09 blue Octorok, slow       attr $05  HP 2
    .ADDR UpdateOctorock                        ; $0A blue Octorok, fast       attr $05  HP 2
    .ADDR UpdateDarknut                         ; $0B Darknut (red)            attr $81  HP 4
    .ADDR UpdateDarknut                         ; $0C Darknut (blue)           attr $81  HP 8
    .ADDR UpdateTektiteOrBoulder                ; $0D Tektite (blue)           attr $81  HP 1
    .ADDR UpdateTektiteOrBoulder                ; $0E Tektite (red)            attr $81  HP 1
    .ADDR UpdateBlueLeever                      ; $0F Blue Leever              attr $01  HP 4
    .ADDR UpdateRedLeever                       ; $10 Red Leever               attr $01  HP 2
    .ADDR UpdateZora                            ; $11 Zora                     attr $81  HP 2
    .ADDR UpdateVire                            ; $12 Vire                     attr $01  HP 4
    .ADDR UpdateZol                             ; $13 Zol                      attr $01  HP 2
    .ADDR UpdateGel                             ; $14 child Gel (from a Zol)   attr $43  HP 0
    .ADDR UpdateGel                             ; $15 Gel                      attr $43  HP 0
    .ADDR UpdatePolsVoice                       ; $16 Pols Voice               attr $81  HP A
    .ADDR UpdateLikeLike                        ; $17 Like-Like                attr $81  HP 9
    .ADDR UpdateDigdogger                       ; $18 small Digdogger          attr $81  HP 8
    .ADDR DoNothing                             ; $19 (no update routine)      attr $81  HP F
    .ADDR UpdatePeahat                          ; $1A Peahat                   attr $01  HP 2
    .ADDR UpdateKeese                           ; $1B Keese (blue)             attr $81  HP 0
    .ADDR UpdateKeese                           ; $1C Keese (red)              attr $81  HP 0
    .ADDR UpdateKeese                           ; $1D Keese (black)            attr $81  HP 0
    .ADDR UpdateArmos                           ; $1E Armos                    attr $01  HP 3
    .ADDR UpdateBoulderSet                      ; $1F boulder spawner          attr $81  HP F
    .ADDR UpdateTektiteOrBoulder                ; $20 Boulder                  attr $81  HP F
    .ADDR UpdateGhini                           ; $21 Ghini                    attr $81  HP 9
    .ADDR UpdateFlyingGhini                     ; $22 flying Ghini             attr $81  HP F
    .ADDR UpdateBlueWizzrobe                    ; $23 Blue Wizzrobe            attr $81  HP A
    .ADDR UpdateRedWizzrobe                     ; $24 Red Wizzrobe             attr $81  HP 4
    .ADDR UpdatePatraChild                      ; $25 Patra child              attr $C3  HP 6
    .ADDR UpdatePatraChild                      ; $26 Patra child              attr $C3  HP 6
    .ADDR UpdateWallmaster                      ; $27 Wallmaster               attr $89  HP 2
    .ADDR UpdateRope                            ; $28 Rope                     attr $89  HP 1
    .ADDR DoNothing                             ; $29 (no update routine)      attr $81  HP 1
    .ADDR UpdateStalfos                         ; $2A Stalfos                  attr $81  HP 2
    .ADDR UpdateBubble                          ; $2B Bubble                   attr $89  HP F
    .ADDR UpdateBubble                          ; $2C Bubble                   attr $89  HP F
    .ADDR UpdateBubble                          ; $2D Bubble                   attr $89  HP F
    .ADDR UpdateWhirlwind                       ; $2E Whirlwind                attr $89  HP F
    .ADDR UpdatePondFairy                       ; $2F Pond fairy               attr $83  HP F
    .ADDR UpdateGibdo                           ; $30 Gibdo                    attr $81  HP 7
    .ADDR UpdateDodongo                         ; $31 Dodongo                  attr $89  HP F
    .ADDR UpdateDodongo                         ; $32 Dodongo                  attr $89  HP F
    .ADDR UpdateGohma                           ; $33 Gohma                    attr $C9  HP 6
    .ADDR UpdateGohma                           ; $34 Gohma                    attr $C9  HP 2
    .ADDR UpdateRupeeStash                      ; $35 Rupee stash              attr $81  HP F
    .ADDR UpdateGrumble                         ; $36 Grumble Goriya           attr $81  HP F
    .ADDR UpdateZelda                           ; $37 Zelda                    attr $81  HP F
    .ADDR UpdateDigdogger                       ; $38 Digdogger                attr $A9  HP F
    .ADDR UpdateDigdogger                       ; $39 Digdogger                attr $A9  HP F
    .ADDR UpdateLamnola                         ; $3A Lanmola (red)            attr $41  HP 2
    .ADDR UpdateLamnola                         ; $3B Lanmola (blue)           attr $41  HP 2
    .ADDR UpdateManhandla                       ; $3C Manhandla                attr $89  HP 4
    .ADDR UpdateAquamentus                      ; $3D Aquamentus               attr $89  HP 6
    .ADDR UpdateGanon                           ; $3E Ganon                    attr $81  HP F
    .ADDR UpdateGuardFire                       ; $3F Guard fire               attr $81  HP 1
    .ADDR UpdateStandingFire                    ; $40 Standing fire            attr $81  HP F
    .ADDR UpdateMoldorm                         ; $41 Moldorm                  attr $C1  HP 2
    .ADDR UpdateGleeok                          ; $42 Gleeok                   attr $C1  HP A
    .ADDR UpdateGleeok                          ; $43 Gleeok                   attr $C1  HP A
    .ADDR UpdateGleeok                          ; $44 Gleeok                   attr $C1  HP A
    .ADDR UpdateGleeok                          ; $45 Gleeok                   attr $C1  HP A
    .ADDR UpdateGleeokHead                      ; $46 flying Gleeok head       attr $81  HP F
    .ADDR UpdatePatra                           ; $47 Patra                    attr $81  HP B
    .ADDR UpdatePatra                           ; $48 Patra                    attr $81  HP B
    .ADDR UpdateTrap                            ; $49 Trap                     attr $A1  HP F
    .ADDR UpdateTrap                            ; $4A Trap                     attr $A1  HP F
    .ADDR UpdateUnderworldPerson                ; $4B dungeon person           attr $81  HP 0
    .ADDR UpdateUnderworldPerson                ; $4C dungeon person           attr $81  HP -
    .ADDR UpdateUnderworldPerson                ; $4D dungeon person           attr $81  HP -
    .ADDR UpdateUnderworldPerson                ; $4E dungeon person           attr $81  HP -
    .ADDR UpdateUnderworldPerson                ; $4F dungeon person           attr $81  HP -
    .ADDR UpdateUnderworldPerson                ; $50 dungeon person           attr $81  HP -
    .ADDR UpdateUnderworldPersonLifeOrMoney     ; $51 life-or-money person     attr $81  HP -
    .ADDR UpdateUnderworldPerson                ; $52 dungeon person           attr $81  HP -
    .ADDR UpdateMonsterShot                     ; $53 flying rock              attr $E3  HP -
    .ADDR UpdateMonsterShot                     ; $54 monster shot             attr $E3  HP -
    .ADDR UpdateFireball                        ; $55 fireball                 attr $E3  HP -
    .ADDR UpdateFireball                        ; $56 fireball                 attr $E3  HP -
    .ADDR UpdateMonsterShot                     ; $57 monster sword shot       attr $E3  HP -
    .ADDR UpdateMonsterShot                     ; $58 monster magic shot       attr $E1  HP -
    .ADDR UpdateMonsterShot                     ; $59 monster magic shot       attr $E1  HP -
    .ADDR UpdateMonsterShot                     ; $5A monster shot             attr $E1  HP -
    .ADDR UpdateMonsterArrow                    ; $5B monster arrow            attr $E1  HP -
    .ADDR UpdateArrowOrBoomerang                ; $5C monster boomerang        attr $E1  HP -
    .ADDR UpdateDeadDummy                       ; $5D dead dummy               attr $81  HP -
    .ADDR UpdateFluteSecret                     ; $5E flute secret             attr $81  HP -
    .ADDR DoNothing                             ; $5F ladder                   attr $81 (set at init)
    .ADDR UpdateItem                            ; $60 dropped item             attr $81 (set at init)
    .ADDR UpdateDock                            ; $61 dock                     attr $81 (set at init)
    .ADDR UpdateRockOrGravestone                ; $62 rock                     attr $81 (set at init)
    .ADDR UpdateRockWall                        ; $63 rock wall                attr $81 (set at init)
    .ADDR UpdateTree                            ; $64 tree                     attr $81 (set at init)
    .ADDR UpdateRockOrGravestone                ; $65 gravestone               attr $81 (set at init)
    .ADDR UpdateRockOrGravestone                ; $66 armos tile object        attr $81 (set at init)
    .ADDR UpdateRockWall                        ; $67 rock wall                attr $81 (set at init)
    .ADDR UpdateBlock                           ; $68 push block               attr $81 (set at init)
    .ADDR DoNothing                             ; $69 (no update routine)      attr $81 (set at init)

; ----------------------------------------------------------------------------
; Draws and times an object's spawn cloud or death sparkle; hands the last step to the end routine.
; IN:   X = object slot with ObjMetastate <> 0
; NOTE: Returns after drawing unless the low nibble of the metastate has reached 4 (4 or $14).
; ----------------------------------------------------------------------------
UpdateMetaObject:
    JSR AnimateAndDrawMetaObject

    ; Metastate 4 (cloud over) or $14 (sparkle over): go finish.
    LDA ObjMetastate, X
    AND #$0F
    CMP #$04
    BCS UpdateMetaObjectEnd

; Does nothing (RTS): the update or init routine of object types that have none of their own.
DoNothing:
    RTS

; ----------------------------------------------------------------------------
; Ends a spawn cloud (object starts acting) or a death (kill counters, slot becomes a dropped item).
; IN:   X = object slot; ObjMetastate,X = 4 or $14
; OUT:  ObjMetastate,X = 0. After a death: ObjType,X = $60, ObjUninitialized,X <> 0, ObjAttr,X =
;       $81, Item_ObjMonsterType,X = the dead type, WorldKillCycle and RoomKillCount updated.
; NOTE: WorldKillCycle counts 0..9 and wraps; types $5D (dead dummy), $14 (child Gel) and $1C (red
;       Keese) do not advance it or RoomKillCount. A Zora ($11) advances the cycle but not
;       RoomKillCount. Every death goes through SetUpDroppedItem, which decides whether an item
;       appears.
; ----------------------------------------------------------------------------
UpdateMetaObjectEnd:
    ; Metastate 4: the spawn cloud is over. Reset the metastate; the object's own update runs
    ; from the next frame.
    LDA ObjMetastate, X
    AND #$10
    BEQ @Reset

    ; Metastate $14: the death sparkle is over.
    ;
    ; Remember the object type; SetUpDroppedItem chooses the drop by it.
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
    JSR SetUpDroppedItem

@Reset:
    JMP ResetObjMetastate
