; ============================================================================
; FILE:     bank4/zora-ow-inits.asm      BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1757-1948
; SUBJECTS: (none)
; TOPIC:    Zora spawn/update; inits for Tektite, Boulder, Leever, Octorok, Ghini, Peahat, pond fairy
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 CheckZora                code
;   L0095 TektiteStartingDirs      data
;   L0098 InitBoulderSet           code
;   L0103 InitBoulder              code
;   L0106 InitTektite              code
;   L0120 InitLeever               code
;   L0127 InitSlowOctorockOrGhini  code
;   L0132 InitFastOctorock         code
;   L0154 InitPeahat               code
;   L0160 EndInitFlyer             code
;   L0168 InitPondFairy            code
;   L0179 L10797_Exit              code
;   L0182 UpdateZora               code
; ============================================================================

CheckZora:
    ; If the zora attribute of this room is off, return.
    LDY RoomId
    LDA LevelBlockAttrsA, Y
    AND #$08                    ; Zora attribute
    BEQ @Exit

    ; If zora is active, return.
    LDA ZoraActive
    BNE @Exit

    ; Look for an empty monster slot. If none is found, return.
    JSR FindEmptyMonsterSlot
    BEQ @Exit

    ; Loop $D times looking for a water tile to put a zora on.
    ; Indexed by [0D].
    LDA #$0D
    STA $0D

    ; Switch to the new object slot.
    LDX EmptyMonsterSlot

@LoopTileCandidate:
    ; Turn the random value at this index into a $10 pixel aligned X coordinate.
    ; If it = 0 or $F0, go loop again.
    LDY $0D
    LDA a:Random, Y
    AND #$F0
    STA ObjX, X
    BEQ @NextLoopTileCandidate
    CMP #$F0
    BEQ @NextLoopTileCandidate

    ; Turn the random value into a $10 pixel aligned Y coordinate.
    ; If it < $50 or >= $E0, go loop again.
    LDA a:Random, Y
    ASL
    ASL
    ASL
    ASL
    CMP #$50
    BCC @NextLoopTileCandidate
    CMP #$E0
    BCS @NextLoopTileCandidate

    ; Turn it into a normal object Y coordinate by OR'ing it with $D.
    ORA #$0D
    STA ObjY, X

    ; Get the tile at that location.
    ; If it's not a water tile, then go loop again.
    JSR GetCollidableTileStill
    CMP #$8D
    BCC @NextLoopTileCandidate
    CMP #$99
    BCS @NextLoopTileCandidate

    ; Flag the zora active, and set up the object.
    INC ZoraActive
    LDA #$11                    ; Zora object type
    JMP SetTypeAndClearObject

@NextLoopTileCandidate:
    ; Bottom of the loop. Keep going until loop index [0D] = 0.
    DEC $0D
    BNE @LoopTileCandidate

@Exit:
    RTS

TektiteStartingDirs:
    .BYTE $01, $02, $05, $0A

InitBoulderSet:
    ; Reset active boulder count.
    LDA #$00
    STA ActiveBoulders

InitBoulder:
    JSR ResetObjMetastateAndTimer

InitTektite:
    ; Set a random starting direction.
    LDA Random+1, X
    AND #$03
    TAY
    LDA TektiteStartingDirs, Y
    STA ObjDir, X

    ; Multiply the direction value by 4 to set the object timer.
    ASL
    ASL
    STA ObjTimer, X
    RTS

InitLeever:
    ; Set red leever long timer to 5, for use in determining when
    ; to burst out of the ground.
    LDA #$05
    STA RedLeeverLongTimer
    JSR ResetObjMetastateAndTimer

InitSlowOctorockOrGhini:
    ; Assign QSpeed $20 (0.5 pixels a frame).
    LDA #$20
    BNE :+

InitFastOctorock:
    ; Assign QSpeed $30 (0.75 pixels a frame).
    LDA #$30
:
    STA ObjQSpeedFrac, X

    ; Octorock spawn clouds last longer than other monsters.
    ; Set object timer to ((object slot + 1) * $10).
    INX
    TXA
    DEX
    ASL
    ASL
    ASL
    ASL
    STA ObjTimer, X
    JSR ResetObjState
    STA ObjAnimFrame, X         ; Reset movement frame.
    LDA #$06
    STA ObjAnimCounter, X
    JMP InitWalker

InitPeahat:
    ; Peahats do not use a spawn cloud. Also, go up at first.
    JSR ResetObjMetastateAndTimer
    LDA #$08
    STA ObjDir, X

EndInitFlyer:
    JSR ResetFlyerState
    LDA #$A0
    STA FlyingMaxSpeedFrac
    LDA #$1F
    STA Flyer_ObjSpeed, X
    RTS

InitPondFairy:
    ; Play "item taken" sound effect.
    LDA #$08
    STA Tune1Request

    ; The fairy goes at location ($78, $7D).
    LDA #$78
    STA ObjX, X
    LDA #$7D
    STA ObjY, X

L10797_Exit:
    RTS

UpdateZora:
    ; If we have the magic clock, then return.
    LDA InvClock
    BNE L10797_Exit
    JSR UpdateBurrower

    ; If zora died in the call above, then it happened while in
    ; states 2, 3, or 4. In the code below, the state won't be 0.
    ; So, it won't flag the zora inactive. This is why the zora does
    ; not come back after killing it.
    ;
    ; If state <> 3 or timer <> $FD,
    ; then go see if the object needs to be destroyed.
    LDA ObjState, X
    CMP #$03
    BNE @DestroyWhenDone
    LDA ObjTimer, X
    CMP #$FD
    BNE @DestroyWhenDone

    ; Shoot a fireball; and set the timer so that it expires faster.
    LDA #$55
    JSR ShootFireball
    LDA #$20
    STA ObjTimer, X

@DestroyWhenDone:
    ; If state = 0, then destroy zora object.
    ; Also flag it inactive, so that the object update loop will try
    ; to make another one.
    LDA ObjState, X
    BNE L10797_Exit
    DEC ZoraActive
    JMP DestroyMonster
