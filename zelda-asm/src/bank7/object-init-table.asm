; ============================================================================
; FILE:     bank7/object-init-table.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5520-5749
; SUBJECTS: (none)
; TOPIC:    InitObject and its table
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 InitObject            code
;   L0147 InitObject_JumpTable  data
; ============================================================================

InitObject:
    LDX CurObjIndex

    ; We want to handle "enemies from the edges of the screen".
    ;
    ; So, if not in UW, or "enemies from edges" attribute is clear,
    ; or the object is (zora, fire, armos, whirlwind), or any object
    ; that does not spawn with a cloud (type >= $53);
    ; 
    ; then go initialize the object.
    LDA CurLevel
    BNE @NormalSpawn
    LDA LevelBlockAttrsByteF
    AND #$08                    ; Enemies from edges
    BEQ @NormalSpawn
    LDA ObjType, X
    CMP #$11                    ; Zora
    BEQ @NormalSpawn
    CMP #$40                    ; Fire
    BEQ @NormalSpawn
    CMP #$1E                    ; Armos
    BEQ @NormalSpawn
    CMP #$2E                    ; Whirlwind
    BEQ @NormalSpawn
    CMP #$53                    ; Flying rock
    BCS @NormalSpawn

@UninitMonsterFromEdge:
    ; If the "monsters from edges" long timer expired, then
    ; go bring the monster in.
    ; Else revert the flag, so this monster becomes uninitialized.
    LDA MonstersFromEdgesLongTimer
    BEQ @InitMonsterFromEdge
    STA ObjUninitialized, X
    RTS

@InitMonsterFromEdge:
    LDX CurObjIndex
    LDA #$05
    JSR SwitchBank
    JSR FindNextEdgeSpawnCell

    ; Extract and set the monster's location.
    LDA CurEdgeSpawnCell

    ; Square column in low nibble. Multiply by 16 to get X.
    PHA
    ASL
    ASL
    ASL
    ASL
    STA ObjX, X

    ; Square row in high nibble. Subtract 3, for the usual offset, to get Y.
    PLA
    AND #$F0
    SEC
    SBC #$03
    STA ObjY, X

    ; Set the monsters-from-edges long timer to a random value
    ; between 2 and 5. So, it will last 20 to 50 frames.
    LDA Random+1, X
    AND #$03
    CLC
    ADC #$02
    STA MonstersFromEdgesLongTimer

    ; If the distance to Link is too close to Link, then
    ; go flag the monster uninitialized again, so we can
    ; try to spawn it again next time.
    LDA #$05
    JSR SwitchBank
    JSR IsDistanceSafeToSpawn
    BCS @UninitMonsterFromEdge

    ; OK to spawn.
    ; In this situation, monsters don't spawn from a cloud.
    LDA #$00
    STA ObjMetastate, X

@NormalSpawn:
    LDA #$04
    JSR SwitchBank

    ; For monsters that spawn in a cloud, set their start time to
    ; the same value as the object slot; so that they all start
    ; moving at different times.
    LDX CurObjIndex
    LDY ObjType, X
    CPY #$1E                    ; Armos
    BEQ @FetchAttrs
    CPY #$22                    ; Flying Ghini
    BEQ @FetchAttrs
    CPY #$53                    ; Flying rock
    BCS @FetchAttrs
    TXA
    STA ObjTimer, X

@FetchAttrs:
    ; Store the object attributes for fast access.
    LDA ObjectTypeToAttributes, Y
    STA ObjAttr, X
    TYA
    STA $00                     ; [00] holds object type.

    ; Hit points are packed two values to a byte.
    ; Divide the object type in half in order to index
    ; into the hit points table.
    LSR
    TAY
    LDA ObjectTypeToHpPairs, Y
    JSR ExtractHitPointValue
    STA ObjHP, X

    ; If the object is a cave, go initialize it.
    LDA $00
    CMP #$6A                    ; Cave 1
    BCC @CheckTileObj
    LDA #$01
    JSR SwitchBank
    JMP InitCave                ; Init cave

@CheckTileObj:
    ; If object is a cave or tile object (type >= $5F),
    ; go assign object attribute $81, and get
    ; it ready to behave autonomously and updating.
    CMP #$5F
    BCC :+
    JMP InitTileObjOrItem
:
    ; Else run the object initialization routine.
    JSR TableJump

InitObject_JumpTable:
    .ADDR DoNothing
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitSlowOctorockOrGhini
    .ADDR InitFastOctorock
    .ADDR InitSlowOctorockOrGhini
    .ADDR InitFastOctorock
    .ADDR InitDarknut
    .ADDR InitDarknut
    .ADDR InitTektite
    .ADDR InitTektite
    .ADDR InitLeever
    .ADDR InitLeever
    .ADDR ResetObjMetastateAndTimer
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR InitGel
    .ADDR InitWalker
    .ADDR InitWalker
    .ADDR DoNothing
    .ADDR DoNothing
    .ADDR InitPeahat
    .ADDR InitBlueKeese
    .ADDR InitRedOrBlackKeese
    .ADDR InitRedOrBlackKeese
    .ADDR InitArmosOrFlyingGhini
    .ADDR InitBoulderSet
    .ADDR InitBoulder
    .ADDR InitSlowOctorockOrGhini
    .ADDR InitArmosOrFlyingGhini
    .ADDR ResetObjMetastateAndTimer
    .ADDR ResetObjMetastateAndTimer
    .ADDR ResetObjMetastateAndTimer
    .ADDR ResetObjMetastateAndTimer
    .ADDR ResetObjMetastateAndTimer
    .ADDR InitRope
    .ADDR DoNothing
    .ADDR InitWalker
    .ADDR InitBubble
    .ADDR InitBubble
    .ADDR InitBubble
    .ADDR DoNothing
    .ADDR InitPondFairy
    .ADDR InitWalker
    .ADDR InitDodongo
    .ADDR InitDodongo
    .ADDR InitGohma
    .ADDR InitGohma
    .ADDR InitRupeeStash
    .ADDR InitGrumble
    .ADDR InitZelda
    .ADDR InitDigdogger1
    .ADDR InitDigdogger2
    .ADDR InitLamnola
    .ADDR InitLamnola
    .ADDR InitManhandla
    .ADDR InitAquamentus
    .ADDR InitGanon
    .ADDR DoNothing
    .ADDR DoNothing
    .ADDR InitMoldorm
    .ADDR InitGleeok
    .ADDR InitGleeok
    .ADDR InitGleeok
    .ADDR InitGleeok
    .ADDR InitGleeokHead
    .ADDR InitPatra
    .ADDR InitPatra
    .ADDR InitTrap
    .ADDR InitTrap
    .ADDR InitUnderworldPerson
    .ADDR InitUnderworldPerson
    .ADDR InitUnderworldPerson
    .ADDR InitUnderworldPerson
    .ADDR InitUnderworldPerson
    .ADDR InitUnderworldPerson
    .ADDR InitUnderworldPersonLifeOrMoney
    .ADDR InitUnderworldPerson
    .ADDR InitMonsterShot
    .ADDR _InitMonsterShot_Unknown54
    .ADDR InitMonsterShot
    .ADDR InitMonsterShot
    .ADDR InitMonsterShot
    .ADDR InitMonsterShot
    .ADDR InitMonsterShot
    .ADDR InitMonsterShot
    .ADDR ResetObjMetastate
    .ADDR ResetObjMetastate
    .ADDR UpdateDeadDummy
    .ADDR InitFluteSecret
