; ============================================================================
; FILE:     bank7/object-init-table.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5520-5749
; SUBJECTS: (none)
; TOPIC:    InitObject, run once for every object slot flagged uninitialized: overworld screen-edge
;           entry, start timer, attributes and HP, then the per-type init routine (jump table).
; SEE ALSO: bank7/object-update-table.asm (UpdateObject calls InitObject; ObjectTypeToAttributes,
;           ObjectTypeToHpPairs); bank5/edge-spawn-cell.asm (FindNextEdgeSpawnCell);
;           bank5/spawn-positions.asm (IsDistanceSafeToSpawn); bank4/drop-items.asm
;           (ExtractHitPointValue); bank7/flute-secret-pond.asm (InitTileObjOrItem, InitFluteSecret,
;           ResetObjMetastate); bank7/object-stubs-destroy-monster.asm (bank-1 trampolines);
;           bank1/cave-init-person-text.asm (InitCave).
; KEY FACTS:
;   - Init dispatch by object type: >= $6A InitCave (bank 1); $5F-$69 InitTileObjOrItem; $00-$5E
;     through InitObject_JumpTable (InitObject@CheckTileObj).
;   - Screen-edge entry happens only on the overworld (CurLevel = 0) in rooms with RoomAttrsF bit 3,
;     and never for Zora $11, Armos $1E, Whirlwind $2E, Standing fire $40 or types >= $53 (InitObject).
;   - One monster enters at a time: while MonstersFromEdgesLongTimer is nonzero the monster is put back
;     to uninitialized and tried again on a later frame (InitObject@UninitMonsterFromEdge).
;   - Each entry attempt reloads that timer with 2 + (Random+1,slot AND 3) = 2-5 (the timer's units are
;     10 frames, see its definition), also when the attempt is refused (InitObject@InitMonsterFromEdge).
;   - An entry attempt is refused when the edge cell is within $22 pixels of Link on both axes
;     (IsDistanceSafeToSpawn). A monster that enters from the edge has no spawn cloud (ObjMetastate 0).
;   - Edge position: X = cell column * $10, Y = cell row * $10 - 3 (InitObject@InitMonsterFromEdge).
;   - ObjTimer starts as the slot number for every type below $53 except Armos $1E and flying Ghini
;     $22, so objects in different slots leave the first spawn-cloud stage on different frames
;     (InitObject@NormalSpawn).
;   - ObjHP = the type's nibble of ObjectTypeToHpPairs * $10; ObjAttr = ObjectTypeToAttributes[type]
;     (InitObject@FetchAttrs).
; CONTENTS: (generated; L = line in this file)
;   L0045 InitObject            code  Initializes the object in slot CurObjIndex: edge entry, start timer, attributes, HP, type's init.
;   L0177 InitObject_JumpTable  data  TABLE: init routine per object type; index = object type $00-$5E; value = routine address, entered
; ============================================================================

; ----------------------------------------------------------------------------
; Initializes the object in slot CurObjIndex: edge entry, start timer, attributes, HP, type's init.
; IN:   CurObjIndex = object slot (1-$0B); ObjType,X set; called by UpdateObject on the first update
;       of a slot whose ObjUninitialized flag was set (UpdateObject has already cleared the flag).
; OUT:  ObjAttr,X and ObjHP,X loaded; ObjTimer,X = slot for cloud-spawning types; the type's init
;       routine has run with X = slot. Or, for a refused edge entry, ObjUninitialized,X nonzero again.
; NOTE: Leaves bank 4 selected (bank 1 for cave persons and the bank-1 types). An edge entry
;       refused by the timer also leaves bank 4; one refused for distance to Link leaves bank 5.
;       The object's own update starts on a later frame.
; ----------------------------------------------------------------------------
InitObject:
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
    JSR SwitchBank
    JSR FindNextEdgeSpawnCell   ; Advances CurEdgeSpawnCell to the next usable edge cell.

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
    JSR SwitchBank
    JSR IsDistanceSafeToSpawn
    BCS @UninitMonsterFromEdge

    ; OK to enter. A monster from the edge skips the spawn cloud: metastate 0 = normal updates.
    LDA #$00
    STA ObjMetastate, X

@NormalSpawn:
    LDA #$04
    JSR SwitchBank

    ; For the types that start in a spawn cloud (below $53, except Armos and flying Ghini),
    ; set the object timer to the slot number. This replaces the 7 that UpdateObject stored
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
    LDA ObjectTypeToAttributes, Y
    STA ObjAttr, X
    TYA
    STA $00                     ; [00] = object type (read by ExtractHitPointValue).

    ; HP values are packed two types to a byte: index = type / 2; even types use the
    ; high nibble, odd types the low nibble. The result is the nibble * $10.
    LSR
    TAY
    LDA ObjectTypeToHpPairs, Y
    JSR ExtractHitPointValue
    STA ObjHP, X

    ; Types $6A and up are cave persons: init in bank 1.
    LDA $00
    CMP #$6A                    ; First cave person type
    BCC @CheckTileObj
    LDA #$01
    JSR SwitchBank
    JMP InitCave

@CheckTileObj:
    ; Types $5F-$69 (ladder, dropped item, tile objects) share one init:
    ; object attribute $81, metastate 0 and timer 0.
    CMP #$5F
    BCC :+
    JMP InitTileObjOrItem
:
    ; Types $00-$5E: jump to the type's init routine. TableJump takes the entry A (= type)
    ; from the table that follows the JSR and does not return here.
    JSR TableJump

; TABLE: init routine per object type; index = object type $00-$5E; value = routine address, entered
; with X = object slot and bank 4 selected (the bank-1 routines go through trampolines in bank 7).
InitObject_JumpTable:
    .ADDR DoNothing                       ; $00 no object
    .ADDR InitWalker                      ; $01 red Lynel
    .ADDR InitWalker                      ; $02 blue Lynel
    .ADDR InitWalker                      ; $03 red Moblin
    .ADDR InitWalker                      ; $04 blue Moblin
    .ADDR InitWalker                      ; $05 red Goriya
    .ADDR InitWalker                      ; $06 blue Goriya
    .ADDR InitSlowOctorockOrGhini         ; $07 red Octorok, slow
    .ADDR InitFastOctorock                ; $08 red Octorok, fast
    .ADDR InitSlowOctorockOrGhini         ; $09 blue Octorok, slow
    .ADDR InitFastOctorock                ; $0A blue Octorok, fast
    .ADDR InitDarknut                     ; $0B red Darknut
    .ADDR InitDarknut                     ; $0C blue Darknut
    .ADDR InitTektite                     ; $0D Tektite
    .ADDR InitTektite                     ; $0E Tektite
    .ADDR InitLeever                      ; $0F blue Leever
    .ADDR InitLeever                      ; $10 red Leever
    .ADDR ResetObjMetastateAndTimer       ; $11 Zora (no cloud, timer 0)
    .ADDR InitWalker                      ; $12 Vire
    .ADDR InitWalker                      ; $13 Zol
    .ADDR InitWalker                      ; $14 child Gel
    .ADDR InitGel                         ; $15 Gel
    .ADDR InitWalker                      ; $16 Pols Voice
    .ADDR InitWalker                      ; $17 Like-Like
    .ADDR DoNothing                       ; $18 small Digdogger
    .ADDR DoNothing                       ; $19 (no routine)
    .ADDR InitPeahat                      ; $1A Peahat
    .ADDR InitBlueKeese                   ; $1B blue Keese
    .ADDR InitRedOrBlackKeese             ; $1C red Keese
    .ADDR InitRedOrBlackKeese             ; $1D black Keese
    .ADDR InitArmosOrFlyingGhini          ; $1E Armos
    .ADDR InitBoulderSet                  ; $1F boulder spawner
    .ADDR InitBoulder                     ; $20 Boulder
    .ADDR InitSlowOctorockOrGhini         ; $21 Ghini
    .ADDR InitArmosOrFlyingGhini          ; $22 flying Ghini
    .ADDR ResetObjMetastateAndTimer       ; $23 blue Wizzrobe (no cloud, timer 0)
    .ADDR ResetObjMetastateAndTimer       ; $24 red Wizzrobe (no cloud, timer 0)
    .ADDR ResetObjMetastateAndTimer       ; $25 Patra child (no cloud, timer 0)
    .ADDR ResetObjMetastateAndTimer       ; $26 Patra child (no cloud, timer 0)
    .ADDR ResetObjMetastateAndTimer       ; $27 Wallmaster (no cloud, timer 0)
    .ADDR InitRope                        ; $28 Rope
    .ADDR DoNothing                       ; $29 (no routine)
    .ADDR InitWalker                      ; $2A Stalfos
    .ADDR InitBubble                      ; $2B Bubble
    .ADDR InitBubble                      ; $2C Bubble
    .ADDR InitBubble                      ; $2D Bubble
    .ADDR DoNothing                       ; $2E Whirlwind
    .ADDR InitPondFairy                   ; $2F Pond fairy
    .ADDR InitWalker                      ; $30 Gibdo
    .ADDR InitDodongo                     ; $31 Dodongo
    .ADDR InitDodongo                     ; $32 Dodongo
    .ADDR InitGohma                       ; $33 Gohma
    .ADDR InitGohma                       ; $34 Gohma
    .ADDR InitRupeeStash                  ; $35 Rupee stash (bank 1)
    .ADDR InitGrumble                     ; $36 Grumble Goriya (bank 1)
    .ADDR InitZelda                       ; $37 Zelda
    .ADDR InitDigdogger1                  ; $38 Digdogger
    .ADDR InitDigdogger2                  ; $39 Digdogger
    .ADDR InitLamnola                     ; $3A red Lanmola
    .ADDR InitLamnola                     ; $3B blue Lanmola
    .ADDR InitManhandla                   ; $3C Manhandla
    .ADDR InitAquamentus                  ; $3D Aquamentus
    .ADDR InitGanon                       ; $3E Ganon
    .ADDR DoNothing                       ; $3F Guard fire
    .ADDR DoNothing                       ; $40 Standing fire
    .ADDR InitMoldorm                     ; $41 Moldorm
    .ADDR InitGleeok                      ; $42 Gleeok
    .ADDR InitGleeok                      ; $43 Gleeok
    .ADDR InitGleeok                      ; $44 Gleeok
    .ADDR InitGleeok                      ; $45 Gleeok
    .ADDR InitGleeokHead                  ; $46 flying Gleeok head
    .ADDR InitPatra                       ; $47 Patra
    .ADDR InitPatra                       ; $48 Patra
    .ADDR InitTrap                        ; $49 Trap (bank 1)
    .ADDR InitTrap                        ; $4A Trap (bank 1)
    .ADDR InitUnderworldPerson            ; $4B dungeon person (bank 1)
    .ADDR InitUnderworldPerson            ; $4C dungeon person (bank 1)
    .ADDR InitUnderworldPerson            ; $4D dungeon person (bank 1)
    .ADDR InitUnderworldPerson            ; $4E dungeon person (bank 1)
    .ADDR InitUnderworldPerson            ; $4F dungeon person (bank 1)
    .ADDR InitUnderworldPerson            ; $50 dungeon person (bank 1)
    .ADDR InitUnderworldPersonLifeOrMoney ; $51 life-or-money man (bank 1)
    .ADDR InitUnderworldPerson            ; $52 dungeon person (bank 1)
    .ADDR InitMonsterShot                 ; $53 flying rock
    .ADDR _InitMonsterShot_Unknown54      ; $54 shot type with its own init
    .ADDR InitMonsterShot                 ; $55 fireball
    .ADDR InitMonsterShot                 ; $56 fireball
    .ADDR InitMonsterShot                 ; $57 monster sword shot
    .ADDR InitMonsterShot                 ; $58 monster magic shot
    .ADDR InitMonsterShot                 ; $59 monster magic shot
    .ADDR InitMonsterShot                 ; $5A shot
    .ADDR ResetObjMetastate               ; $5B monster arrow (metastate 0 only)
    .ADDR ResetObjMetastate               ; $5C monster boomerang (metastate 0 only)
    .ADDR UpdateDeadDummy                 ; $5D dead dummy: death sound, dying metastate $10
    .ADDR InitFluteSecret                 ; $5E flute secret
