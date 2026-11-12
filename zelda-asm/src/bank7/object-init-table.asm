; ============================================================================
; FILE:     bank7/object-init-table.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5520-5749
; SUBJECTS: Obj
; TOPIC:    Obj_Init, run once for every object slot flagged uninitialized: overworld screen-edge
;           entry, start timer, attributes and HP, then the per-type init routine (jump table).
; SEE ALSO: bank7/object-update-table.asm (Obj_Update calls Obj_Init; Obj_TypeAttributes,
;           Obj_TypeHpPairs); bank5/edge-spawn-cell.asm (EdgeSpawn_FindNextCell);
;           bank5/spawn-positions.asm (Spawn_IsFarFromLink); bank4/drop-items.asm
;           (Obj_ExtractStartHp); bank7/flute-secret-pond.asm (Obj_InitTileObjOrItem, FluteSecret_Init,
;           Obj_ResetMetastate); bank7/object-stubs-destroy-monster.asm (bank-1 trampolines);
;           bank1/cave-init-person-text.asm (Cave_Init).
; KEY FACTS:
;   - Init dispatch by object type: >= $6A Cave_Init (bank 1); $5F-$69 Obj_InitTileObjOrItem; $00-$5E
;     through Obj_Init_JumpTable (Obj_Init@CheckTileObj).
;   - Screen-edge entry happens only on the overworld (CurLevel = 0) in rooms with RoomAttrsF bit 3,
;     and never for Zora $11, Armos $1E, Whirlwind $2E, Standing fire $40 or types >= $53 (Obj_Init).
;   - One monster enters at a time: while MonstersFromEdgesLongTimer is nonzero the monster is put back
;     to uninitialized and tried again on a later frame (Obj_Init@UninitMonsterFromEdge).
;   - Each entry attempt reloads that timer with 2 + (Random+1,slot AND 3) = 2-5 (the timer's units are
;     10 frames, see its definition), also when the attempt is refused (Obj_Init@InitMonsterFromEdge).
;   - An entry attempt is refused when the edge cell is within $22 pixels of Link on both axes
;     (Spawn_IsFarFromLink). A monster that enters from the edge has no spawn cloud (ObjMetastate 0).
;   - Edge position: X = cell column * $10, Y = cell row * $10 - 3 (Obj_Init@InitMonsterFromEdge).
;   - ObjTimer starts as the slot number for every type below $53 except Armos $1E and flying Ghini
;     $22, so objects in different slots leave the first spawn-cloud stage on different frames
;     (Obj_Init@NormalSpawn).
;   - ObjHP = the type's nibble of Obj_TypeHpPairs * $10; ObjAttr = Obj_TypeAttributes[type]
;     (Obj_Init@FetchAttrs).
; CONTENTS: (generated; L = line in this file)
;   L0045 Obj_Init            code  Initializes the object in slot CurObjIndex: edge entry, start timer, attributes, HP, type's init.
;   L0177 Obj_Init_JumpTable  data  TABLE: init routine per object type; index = object type $00-$5E; value = routine address, entered
; ============================================================================

; ----------------------------------------------------------------------------
; Initializes the object in slot CurObjIndex: edge entry, start timer, attributes, HP, type's init.
; IN:   CurObjIndex = object slot (1-$0B); ObjType,X set; called by Obj_Update on the first update
;       of a slot whose ObjUninitialized flag was set (Obj_Update has already cleared the flag).
; OUT:  ObjAttr,X and ObjHP,X loaded; ObjTimer,X = slot for cloud-spawning types; the type's init
;       routine has run with X = slot. Or, for a refused edge entry, ObjUninitialized,X nonzero again.
; NOTE: Leaves bank 4 selected (bank 1 for cave persons and the bank-1 types). An edge entry
;       refused by the timer also leaves bank 4; one refused for distance to Link leaves bank 5.
;       The object's own update starts on a later frame.
; ----------------------------------------------------------------------------
Obj_Init:
    LDX CurObjIndex

    ; Monsters that walk in from the screen edges. That applies only when all of these hold:
    ; on the overworld (CurLevel = 0); the room has the "monsters from edges" attribute;
    ; the object is not a Zora, Standing fire, Armos or Whirlwind; its type is below $53
    ; (the types from $53 up are shots, tile objects and persons, which have no spawn cloud).
    ; Otherwise go to the normal init.
    LDA CurLevel
    BNE @NormalSpawn            ; In a dungeon: never from the edge.
    LDA RoomAttrsF
    AND #$08                    ; Bit 3: monsters enter from the screen edges.
    BEQ @NormalSpawn
    LDA ObjType, X
    CMP #$11                    ; Zora
    BEQ @NormalSpawn
    CMP #$40                    ; Standing fire
    BEQ @NormalSpawn
    CMP #$1E                    ; Armos
    BEQ @NormalSpawn
    CMP #$2E                    ; Whirlwind
    BEQ @NormalSpawn
    CMP #$53                    ; Flying rock: first type without a spawn cloud
    BCS @NormalSpawn

@UninitMonsterFromEdge:
    ; If the "monsters from edges" long timer has expired, bring this monster in.
    ; Else set the uninitialized flag again (A = the nonzero timer), so that this
    ; routine runs again for this slot on a later frame.
    LDA MonstersFromEdgesLongTimer
    BEQ @InitMonsterFromEdge
    STA ObjUninitialized, X
    RTS

@InitMonsterFromEdge:
    LDX CurObjIndex
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR EdgeSpawn_FindNextCell   ; Advances CurEdgeSpawnCell to the next usable edge cell.

    ; Turn the cell into the monster's position.
    LDA CurEdgeSpawnCell

    ; Column in the low nibble: X = column * $10.
    PHA
    ASL
    ASL
    ASL
    ASL
    STA ObjX, X

    ; Row in the high nibble: Y = row * $10 - 3.
    PLA
    AND #$F0
    SEC
    SBC #$03
    STA ObjY, X

    ; Reload the long timer with a random 2 to 5 (bits 1-0 of this slot's random byte, plus 2).
    ; This happens whether or not the monster is let in below.
    LDA Random+1, X
    AND #$03
    CLC
    ADC #$02
    STA MonstersFromEdgesLongTimer

    ; If the cell is within $22 pixels of Link on both axes (C set), flag the monster
    ; uninitialized again (the timer is now nonzero) and try again after the timer expires.
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Spawn_IsFarFromLink
    BCS @UninitMonsterFromEdge

    ; OK to enter. A monster from the edge skips the spawn cloud: metastate 0 = normal updates.
    LDA #$00
    STA ObjMetastate, X

@NormalSpawn:
    LDA #$04
    JSR Mmc1_SwitchBank

    ; For the types that start in a spawn cloud (below $53, except Armos and flying Ghini),
    ; set the object timer to the slot number. This replaces the 7 that Obj_Update stored
    ; just before calling here. The first cloud stage ends when the timer reaches 0, so
    ; objects in different slots start moving on different frames.
    LDX CurObjIndex
    LDY ObjType, X
    CPY #$1E                    ; Armos
    BEQ @FetchAttrs
    CPY #$22                    ; Flying Ghini
    BEQ @FetchAttrs
    CPY #$53                    ; Flying rock and above
    BCS @FetchAttrs
    TXA
    STA ObjTimer, X

@FetchAttrs:
    ; Copy the type's attribute byte to the slot.
    LDA Obj_TypeAttributes, Y
    STA ObjAttr, X
    TYA
    STA $00                     ; [00] = object type (read by Obj_ExtractStartHp).

    ; HP values are packed two types to a byte: index = type / 2; even types use the
    ; high nibble, odd types the low nibble. The result is the nibble * $10.
    LSR
    TAY
    LDA Obj_TypeHpPairs, Y
    JSR Obj_ExtractStartHp
    STA ObjHP, X

    ; Types $6A and up are cave persons: init in bank 1.
    LDA $00
    CMP #$6A                    ; First cave person type
    BCC @CheckTileObj
    LDA #$01
    JSR Mmc1_SwitchBank
    JMP Cave_Init

@CheckTileObj:
    ; Types $5F-$69 (ladder, dropped item, tile objects) share one init:
    ; object attribute $81, metastate 0 and timer 0.
    CMP #$5F
    BCC :+
    JMP Obj_InitTileObjOrItem
:
    ; Types $00-$5E: jump to the type's init routine. Util_TableJump takes the entry A (= type)
    ; from the table that follows the JSR and does not return here.
    JSR Util_TableJump

; TABLE: init routine per object type; index = object type $00-$5E; value = routine address, entered
; with X = object slot and bank 4 selected (the bank-1 routines go through trampolines in bank 7).
Obj_Init_JumpTable:
    .ADDR Obj_DoNothing                       ; $00 no object
    .ADDR Walker_Init                      ; $01 blue Lynel
    .ADDR Walker_Init                      ; $02 red Lynel
    .ADDR Walker_Init                      ; $03 blue Moblin
    .ADDR Walker_Init                      ; $04 red Moblin
    .ADDR Walker_Init                      ; $05 blue Goriya
    .ADDR Walker_Init                      ; $06 red Goriya
    .ADDR Octorok_InitSlowOrGhini         ; $07 red Octorok, slow
    .ADDR Octorok_InitFast                ; $08 red Octorok, fast
    .ADDR Octorok_InitSlowOrGhini         ; $09 blue Octorok, slow
    .ADDR Octorok_InitFast                ; $0A blue Octorok, fast
    .ADDR Darknut_Init                     ; $0B red Darknut
    .ADDR Darknut_Init                     ; $0C blue Darknut
    .ADDR Tektite_Init                     ; $0D blue Tektite
    .ADDR Tektite_Init                     ; $0E red Tektite
    .ADDR Leever_Init                      ; $0F blue Leever
    .ADDR Leever_Init                      ; $10 red Leever
    .ADDR Obj_ResetMetastateAndTimer       ; $11 Zora (no cloud, timer 0)
    .ADDR Walker_Init                      ; $12 Vire
    .ADDR Walker_Init                      ; $13 Zol
    .ADDR Walker_Init                      ; $14 child Gel
    .ADDR Gel_Init                         ; $15 Gel
    .ADDR Walker_Init                      ; $16 Pols Voice
    .ADDR Walker_Init                      ; $17 Like-Like
    .ADDR Obj_DoNothing                       ; $18 small Digdogger
    .ADDR Obj_DoNothing                       ; $19 (no routine)
    .ADDR Peahat_Init                      ; $1A Peahat
    .ADDR Keese_InitBlue                   ; $1B blue Keese
    .ADDR Keese_InitRedOrBlack             ; $1C red Keese
    .ADDR Keese_InitRedOrBlack             ; $1D black Keese
    .ADDR Armos_InitOrFlyingGhiniFadeIn          ; $1E Armos
    .ADDR Boulder_InitSpawner                  ; $1F boulder spawner
    .ADDR Boulder_Init                     ; $20 Boulder
    .ADDR Octorok_InitSlowOrGhini         ; $21 Ghini
    .ADDR Armos_InitOrFlyingGhiniFadeIn          ; $22 flying Ghini
    .ADDR Obj_ResetMetastateAndTimer       ; $23 blue Wizzrobe (no cloud, timer 0)
    .ADDR Obj_ResetMetastateAndTimer       ; $24 red Wizzrobe (no cloud, timer 0)
    .ADDR Obj_ResetMetastateAndTimer       ; $25 Patra child (no cloud, timer 0)
    .ADDR Obj_ResetMetastateAndTimer       ; $26 Patra child (no cloud, timer 0)
    .ADDR Obj_ResetMetastateAndTimer       ; $27 Wallmaster (no cloud, timer 0)
    .ADDR Rope_Init                        ; $28 Rope
    .ADDR Obj_DoNothing                       ; $29 (no routine)
    .ADDR Walker_Init                      ; $2A Stalfos
    .ADDR Bubble_Init                      ; $2B Bubble
    .ADDR Bubble_Init                      ; $2C Bubble
    .ADDR Bubble_Init                      ; $2D Bubble
    .ADDR Obj_DoNothing                       ; $2E Whirlwind
    .ADDR PondFairy_Init                   ; $2F Pond fairy
    .ADDR Walker_Init                      ; $30 Gibdo
    .ADDR Dodongo_Init                     ; $31 Dodongo (made as often as the room's monster count says)
    .ADDR Dodongo_Init                     ; $32 Dodongo (always one: ids $32-$61 force the count to 1)
    .ADDR Gohma_Init                       ; $33 Gohma
    .ADDR Gohma_Init                       ; $34 Gohma
    .ADDR RupeeStash_Init                  ; $35 Rupee stash (bank 1)
    .ADDR Grumble_Init                     ; $36 Grumble Goriya (bank 1)
    .ADDR Zelda_Init                       ; $37 Zelda
    .ADDR Digdogger_InitWith3Children                  ; $38 Digdogger
    .ADDR Digdogger_InitWith1Child                  ; $39 Digdogger
    .ADDR Lanmola_Init                     ; $3A red Lanmola
    .ADDR Lanmola_Init                     ; $3B blue Lanmola
    .ADDR Manhandla_Init                   ; $3C Manhandla
    .ADDR Aquamentus_Init                  ; $3D Aquamentus
    .ADDR Ganon_Init                       ; $3E Ganon
    .ADDR Obj_DoNothing                       ; $3F Guard fire
    .ADDR Obj_DoNothing                       ; $40 Standing fire
    .ADDR Moldorm_Init                     ; $41 Moldorm
    .ADDR Gleeok_Init                      ; $42 Gleeok
    .ADDR Gleeok_Init                      ; $43 Gleeok
    .ADDR Gleeok_Init                      ; $44 Gleeok
    .ADDR Gleeok_Init                      ; $45 Gleeok
    .ADDR Gleeok_InitFlyingHead                  ; $46 flying Gleeok head
    .ADDR Patra_Init                       ; $47 Patra
    .ADDR Patra_Init                       ; $48 Patra
    .ADDR Trap_Init                        ; $49 Trap (bank 1)
    .ADDR Trap_Init                        ; $4A Trap (bank 1)
    .ADDR Person_InitUnderworld            ; $4B dungeon person (bank 1)
    .ADDR Person_InitUnderworld            ; $4C dungeon person (bank 1)
    .ADDR Person_InitUnderworld            ; $4D dungeon person (bank 1)
    .ADDR Person_InitUnderworld            ; $4E dungeon person (bank 1)
    .ADDR Person_InitUnderworld            ; $4F dungeon person (bank 1)
    .ADDR Person_InitUnderworld            ; $50 dungeon person (bank 1)
    .ADDR Person_InitLifeOrMoney ; $51 life-or-money man (bank 1)
    .ADDR Person_InitUnderworld            ; $52 dungeon person (bank 1)
    .ADDR MonsterShot_Init                 ; $53 flying rock
    .ADDR MonsterShot_InitType54      ; $54 shot type with its own init
    .ADDR MonsterShot_Init                 ; $55 fireball
    .ADDR MonsterShot_Init                 ; $56 fireball
    .ADDR MonsterShot_Init                 ; $57 monster sword shot
    .ADDR MonsterShot_Init                 ; $58 monster magic shot
    .ADDR MonsterShot_Init                 ; $59 monster magic shot
    .ADDR MonsterShot_Init                 ; $5A shot
    .ADDR Obj_ResetMetastate               ; $5B monster arrow (metastate 0 only)
    .ADDR Obj_ResetMetastate               ; $5C monster boomerang (metastate 0 only)
    .ADDR DeadDummy_StartDeathSparkle                 ; $5D dead dummy: death sound, dying metastate $10
    .ADDR FluteSecret_Init                 ; $5E flute secret
