; ============================================================================
; FILE:     bank4/zelda-guard-fire.asm   BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9587-9669
; SUBJECTS: (none)
; TOPIC:    UpdateZelda, guard fire, dead dummy type
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 UpdateZelda          code
;   L0055 L129B7_Exit          code
;   L0058 UpdateZelda_State1   code
;   L0084 UpdateGuardFire      code
;   L0094 SetDeadDummyObjType  code
;   L0098 L129EA_Exit          code
; ============================================================================

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

    ; Else Link is near enough to Zelda.
    ; Go to state 1, and halt Link.
    INC ObjState, X
    LDA #$40
    STA ObjState

    ; Put Link at location ($88, $88) and facing left.
    LDA #$88
    STA ObjX
    STA ObjY
    LDA #$02
    STA ObjDir

    ; Play the Zelda fanfare.
    LDA #$06
    STA SongRequest

    ; Set a delay of $80 frames at the beginning of the next state.
    LDA #$80
    STA ObjTimer, X

L129B7_Exit:
    RTS

UpdateZelda_State1:
    ; State 1.
    ;
    ; Draw Link, then return if the object timer <> 0.
    JSR Link_EndMoveAndDraw_Bank4
    LDA ObjTimer, X
    BNE L129B7_Exit

    ; Go to mode $13.
    STA IsUpdatingMode
    STA GameSubmode
    STA ObjState
    LDA #$13
    STA GameMode

    ; Mode $13 will start with a curtain furling effect.
    ; Set up the decreasing column ($20), and the increasing column (1).
    ; Prepare the background by filling the tile map with blank tile $24.
    LDA #$20
    STA ObjX+12
    LDA #$01
    STA ObjX+13
    LDA #$24
    STA $0A                     ; [0A] tile to fill with
    JMP FillTileMap

UpdateGuardFire:
    LDA #$06                    ; Animation counter 6
    JSR AnimateAndDrawCommonObject
    JSR CheckMonsterCollisions

    ; If the object has not been killed, then return.
    ; Else change the object type to dead dummy.
    LDA ObjMetastate, X
    BEQ L129EA_Exit

SetDeadDummyObjType:
    LDA #$5D
    STA ObjType, X

L129EA_Exit:
    RTS
