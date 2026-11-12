; ============================================================================
; FILE:     bank4/zelda-guard-fire.asm   BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9587-9669
; SUBJECTS: (none)
; TOPIC:    Zelda (type $37): waits for Link to stand in front of her, then starts the ending
;           (mode $13). Guard fire (type $3F): the killable fires around her. The helper that turns
;           a dying object into a dead dummy (type $5D).
; SEE ALSO: bank4/zelda-lanmola-inits.asm (InitZelda and the positions of Zelda and the fires);
;           bank2/mode13-ending-init.asm (mode $13, which this file starts);
;           bank7/object-update-table.asm (UpdateMetaObjectEnd: what type $5D changes at the end of a
;           death); bank7/object-stubs-destroy-monster.asm (UpdateDeadDummy); bank4/patra.asm and
;           bank4/lanmola.asm (other users of SetDeadDummyObjType and L129EA_Exit).
; KEY FACTS:
;   - Zelda triggers when Link's X is $70-$80 and his Y is exactly $95. Nothing else is tested: the
;     code does not ask whether the guard fires are dead (UpdateZelda).
;   - On the trigger Link is halted (ObjState $40), moved to ($88, $88) facing left, song request $06
;     (Zelda's song) is made and Zelda's timer is set to $80 frames (UpdateZelda).
;   - When that timer runs out the game goes to mode $13, submode 0, with the tile map filled with
;     blank tile $24 (UpdateZelda_State1).
;   - A guard fire is an ordinary collision target: it hurts Link on contact (damage byte $80 of
;     type $3F in ObjTypeToDamagePoints: half a heart) and can be killed (HP $10 from
;     ObjectTypeToHpPairs; its invincibility mask is 0). It is never moved by knockback. When it
;     dies its type becomes $5D (UpdateGuardFire).
;   - A type $5D death does not advance WorldKillCycle and does not add to RoomKillCount
;     (SetDeadDummyObjType, UpdateMetaObjectEnd).
; CONTENTS: (generated; L = line in this file)
;   L0042 UpdateZelda          code  Draws Zelda and, once Link stands in front of her, halts him and starts the fanfare.
;   L0081 L129B7_Exit          code  Shared return of UpdateZelda and UpdateZelda_State1.
;   L0090 UpdateZelda_State1   code  Zelda state 1: keeps drawing Link for $80 frames, then switches the game to mode $13.
;   L0120 UpdateGuardFire      code  Animates a guard fire and checks it against Link and his weapons; a killed fire becomes type $5D.
;   L0130 SetDeadDummyObjType  code  Changes the object in slot X to type $5D (dead dummy); its dying metastate carries on.
;   L0135 L129EA_Exit          code  Shared return of UpdateGuardFire (also used by UpdateLamnola).
; ============================================================================

; ----------------------------------------------------------------------------
; Draws Zelda and, once Link stands in front of her, halts him and starts the fanfare.
; IN:   X = Zelda's object slot (1).
; OUT:  on the trigger: ObjState,X = 1, ObjTimer,X = $80; Link halted at ($88, $88) facing left.
; NOTE: State 0 waits for Link's X in $70-$80 and Y = $95 (Zelda is at ($78, $88)). The guard fires
;       are not tested here. Zelda has no collision check of her own.
; ----------------------------------------------------------------------------
UpdateZelda:
    JSR Person_Draw
    LDA ObjState, X
    BNE UpdateZelda_State1

    ; State 0.
    ;
    ; If Link's X < $70 or >= $81, or Link's Y <> $95, then return.
    LDA ObjX
    CMP #$70
    BCC L129B7_Exit
    CMP #$81
    BCS L129B7_Exit
    LDA ObjY
    CMP #$95
    BNE L129B7_Exit

    ; Link is in front of Zelda. Go to state 1, and halt Link.
    INC ObjState, X
    LDA #$40
    STA ObjState

    ; Put Link at ($88, $88), beside Zelda, facing left (2).
    LDA #$88
    STA ObjX
    STA ObjY
    LDA #$02
    STA ObjDir

    ; Song request $06: DriveSong treats this value as one song of its own (Zelda's song), not as
    ; the two bits $02 and $04.
    LDA #$06
    STA SongRequest

    ; State 1 waits $80 frames.
    LDA #$80
    STA ObjTimer, X

; Shared return of UpdateZelda and UpdateZelda_State1.
L129B7_Exit:
    RTS

; ----------------------------------------------------------------------------
; Zelda state 1: keeps drawing Link for $80 frames, then switches the game to mode $13.
; IN:   X = Zelda's object slot.
; OUT:  when ObjTimer,X is 0: GameMode = $13, GameSubmode = 0, IsUpdatingMode = 0, Link's ObjState
;       = 0, ObjX+12 = $20, ObjX+13 = 1, tile map filled with tile $24.
; ----------------------------------------------------------------------------
UpdateZelda_State1:
    ; Link is halted, so draw him from here. Return until Zelda's timer runs out.
    JSR Link_EndMoveAndDraw_Bank4
    LDA ObjTimer, X
    BNE L129B7_Exit

    ; Go to mode $13 (win game): its init routine runs from submode 0. A is 0 here.
    STA IsUpdatingMode
    STA GameSubmode
    STA ObjState                ; Link is no longer halted.
    LDA #$13
    STA GameMode

    ; Mode $13 begins with a closing curtain. The X entries of slots $0C and $0D hold its two
    ; moving columns: $20 (decreasing) and 1 (increasing).
    ; Fill the tile map with blank tile $24 for the background.
    LDA #$20
    STA ObjX+12
    LDA #$01
    STA ObjX+13
    LDA #$24
    STA $0A                     ; [0A] tile to fill with
    JMP FillTileMap

; ----------------------------------------------------------------------------
; Animates a guard fire and checks it against Link and his weapons; a killed fire becomes type $5D.
; IN:   X = object slot.
; NOTE: The fire does not move. It can be killed like a monster; the type change makes its death
;       count as a dead dummy's (no kill cycle step, no room kill).
; ----------------------------------------------------------------------------
UpdateGuardFire:
    LDA #$06                    ; Animation counter reload: 6 frames per animation frame.
    JSR AnimateAndDrawCommonObject
    JSR CheckMonsterCollisions

    ; Metastate 0: not dying, return. Else fall through to change the type to dead dummy.
    LDA ObjMetastate, X
    BEQ L129EA_Exit

; Changes the object in slot X to type $5D (dead dummy); its dying metastate carries on.
SetDeadDummyObjType:
    LDA #$5D
    STA ObjType, X

; Shared return of UpdateGuardFire (also used by UpdateLamnola).
L129EA_Exit:
    RTS
