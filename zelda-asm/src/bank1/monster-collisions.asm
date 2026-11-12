; ============================================================================
; FILE:     bank1/monster-collisions.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5457-5590
; SUBJECTS: Person
; TOPIC:    CheckMonsterCollisions weapon dispatch, object midpoint, contact damage table
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 Person_DrawAndCheckCollisions_Common  code
;   L0027 Person_Draw                           code
;   L0038 CheckMonsterCollisions                code  Params:
;   L0113 GetObjectMiddle                       code  Params:
;   L0137 ObjTypeToDamagePoints                 data
; ============================================================================

Person_DrawAndCheckCollisions_Common:
    JSR CheckMonsterCollisions

    ; If the person is flagged dead, then
    ; enable fireballs from the two flames, and reset the metastate.
    LDA ObjMetastate+1
    BEQ Person_Draw
    STA IsPersonFireballsEnabled
    LDA #$00
    STA ObjMetastate+1

Person_Draw:
    ; Either way, draw the object.
    JSR Anim_FetchObjPosForSpriteDescriptor
    JMP DrawObjectMirrored

; Params:
; X: object index
;
; Note:
; This routine might modify the facing direction.
;
CheckMonsterCollisions:
    JSR GetObjectMiddle

    ; If the object is invincible, skip checking collisions with Link's weapons.
    ; But go check for a collision with Link.
    LDA ObjAttr, X
    AND #$20                    ; Invincible
    BNE @CheckLink

    ; If temporarily invincible, return.
    LDA ObjInvincibilityTimer, X
    BNE @Exit

    ; Check for collisions with each weapon.
    LDY #$0F
    JSR CheckMonsterBoomerangOrFoodCollision
    LDY #$0E
    JSR CheckMonsterSwordShotOrMagicShotCollision
    LDY #$10
    JSR CheckMonsterBombOrFireCollision
    LDY #$11
    JSR CheckMonsterBombOrFireCollision
    LDY #$0D
    JSR CheckMonsterSwordCollision
    LDY #$12
    JSR CheckMonsterArrowOrRodCollision

@CheckLink:
    JSR CheckLinkCollision

    ; If the monster is not dying, then go see about monsters
    ; that capture Link.
    LDA ObjType, X
    LDY ObjMetastate, X
    BEQ @CheckCaptors

    ; This monster is dying.
    ;
    ; If it's a goriya that threw a boomerang, then destroy the boomerang.
    CMP #$05
    BEQ :+
    CMP #$06                    ; Red Goriya
    BNE @Exit
:
    LDA ObjState, X
    BPL @Exit
    LDY ObjRefId, X             ; Get the object slot of this goriya's boomerang.
    LDA #$00
    STA ObjType, Y

@Exit:
    RTS

@CheckCaptors:
    ; If the monster is a wallmaster or like-like and there's a hit
    ; (using [0C] flag), then set the monster's capture flag.
    CMP #$27                    ; Wallmaster object type
    BEQ :+
    CMP #$17                    ; LikeLike object type
    BNE @Exit2
:
    LDA $0C
    BEQ @Exit2
    INC ObjCaptureTimer, X

@Exit2:
    RTS

; Params:
; X: object index
;
; Returns:
; [02]: center X
; [03]: center Y
;
GetObjectMiddle:
    ; Start with offset 8 for X and Y in [02] and [03].
    LDA #$08
    STA $02
    STA $03

    ; If object attribute $40 is set, make X offset 4.
    LDA ObjAttr, X
    AND #$40                    ; Half width (for collision detection)
    BEQ :+
    LSR $02                     ; Cut the X offset [02] in half.
:
    ; Add object's X coordinate to [02].
    LDA ObjX, X
    CLC
    ADC $02
    STA $02

    ; Add object's Y coordinate to [03].
    LDA ObjY, X
    CLC
    ADC $03
    STA $03

ObjTypeToDamagePoints:
    .BYTE $60, $02, $01, $80, $80, $01, $80, $80
    .BYTE $80, $80, $80, $01, $02, $80, $80, $01
    .BYTE $80, $80, $01, $01, $80, $80, $02, $01
    .BYTE $02, $00, $80, $80, $80, $80, $01, $80
    .BYTE $80, $01, $01, $02, $01, $02, $02, $80
    .BYTE $80, $80, $80, $00, $00, $00, $00, $00
    .BYTE $02, $01, $01, $02, $02, $00, $00, $00
    .BYTE $02, $02, $02, $02, $01, $01, $04, $80
    .BYTE $80, $80, $01, $01, $01, $01, $01, $02
    .BYTE $02, $01, $01, $00, $00, $00, $00, $00
    .BYTE $00, $00, $00, $80, $80, $80, $01, $02
    .BYTE $02, $04, $04, $80, $01
