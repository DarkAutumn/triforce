; ============================================================================
; FILE:     bank1/object-animation-draw.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4937-5245
; SUBJECTS: (none)
; TOPIC:    Object animation tables, DrawObject routines
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 ObjAnimations                              data
;   L0042 ObjAnimFrameHeap                           data
;   L0070 ObjAnimAttrHeap                            data
;   L0098 SpriteOffsets                              data
;   L0106 AnimateAndDrawObjectWalking                code
;   L0117 DrawObjectMirrored                         code  Params:
;   L0128 DrawObjectNotMirrored                      code  Params:
;   L0151 DrawObjectWithType                         code  Params:
;   L0167 DrawObjectWithAnim                         code  Params:
;   L0185 DrawObjectWithAnimAndSpecificSprites       code
;   L0255 Anim_WriteHorizontallyFlippableSpritePair  code  Params:
;   L0289 Anim_WriteSpritePair                       code  Params:
;   L0307 Anim_WriteSpritePairNotFlashing            code
; ============================================================================

ObjAnimations:
    .BYTE $00, $08, $0B, $0F, $13, $17, $5C, $60
    .BYTE $1B, $1B, $21, $21, $64, $6A, $27, $29
    .BYTE $2B, $35, $3F, $70, $74, $76, $76, $78
    .BYTE $7A, $7E, $80, $49, $82, $84, $86, $4B
    .BYTE $4F, $4F, $51, $51, $88, $8C, $90, $90
    .BYTE $92, $94, $96, $98, $99, $99, $99, $53
    .BYTE $54, $9A, $9B, $9B, $A5, $A5, $AB, $AB
    .BYTE $AC, $AE, $AE, $AF, $AF, $B2, $B8, $B8
    .BYTE $08, $08, $C6, $C6, $C6, $C6, $C6, $C6
    .BYTE $C8, $C8, $C9, $C9, $CA, $CA, $CA, $CA
    .BYTE $CA, $CA, $CA, $CA, $09, $09, $0A, $0A
    .BYTE $0B, $0B, $0B, $0B, $0B, $0B, $CB, $55
    .BYTE $55, $55, $55, $55, $55, $55, $56, $57
    .BYTE $57, $CB, $CC, $58, $58, $58, $58, $58
    .BYTE $58, $58, $58, $58, $59, $59, $59, $59
    .BYTE $5A, $5A, $5A, $5A, $5B, $5B, $5B

ObjAnimFrameHeap:
    .BYTE $00, $04, $08, $0C, $10, $10, $14, $18
    .BYTE $5C, $9E, $44, $CE, $D2, $D6, $DA, $CE
    .BYTE $D2, $D6, $DA, $F0, $F4, $F8, $FC, $F0
    .BYTE $F4, $F8, $FC, $B4, $B0, $B0, $B8, $B2
    .BYTE $B2, $B4, $B0, $B0, $B8, $B2, $B2, $CA
    .BYTE $CC, $CA, $CC, $BC, $BE, $C0, $C0, $C2
    .BYTE $C4, $C0, $C0, $BC, $BE, $BC, $BE, $C0
    .BYTE $C0, $C2, $C4, $C0, $C0, $BC, $BE, $BC
    .BYTE $BE, $EC, $EE, $EC, $EE, $EC, $EE, $BC
    .BYTE $BE, $C6, $C8, $A0, $A8, $A4, $AC, $90
    .BYTE $E8, $E4, $E0, $94, $F3, $C9, $BD, $C1
    .BYTE $98, $9A, $9C, $F8, $B8, $BC, $B0, $B4
    .BYTE $B8, $BC, $B0, $B4, $B8, $AC, $B4, $BC
    .BYTE $B0, $B4, $B8, $AC, $B4, $BC, $B0, $B4
    .BYTE $AC, $AE, $B0, $B2, $A8, $AA, $92, $94
    .BYTE $A0, $A2, $A6, $A4, $A2, $A4, $D8, $DA
    .BYTE $00, $00, $9A, $9C, $9A, $9C, $9A, $9C
    .BYTE $B4, $B8, $BC, $BE, $B4, $B8, $BC, $BE
    .BYTE $FC, $FE, $AC, $9C, $A0, $A4, $A0, $A4
    .BYTE $A8, $8E, $A4, $DC, $E0, $E4, $E8, $EC
    .BYTE $F0, $F4, $F8, $FA, $FE, $F4, $F6, $FE
    .BYTE $FC, $F0, $F8, $B0, $F6, $F0, $D4, $FC
    .BYTE $FE, $F8, $E8, $EA, $E0, $E4, $EC, $EC
    .BYTE $D0, $D4, $D8, $DC, $E0, $E4, $C0, $C8
    .BYTE $C4, $CC, $E8, $EA, $72, $74, $DE, $EE
    .BYTE $F8, $96, $98, $B1

ObjAnimAttrHeap:
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00
    .BYTE $02, $00, $00, $01, $01, $01, $01, $02
    .BYTE $02, $02, $02, $03, $03, $03, $03, $02
    .BYTE $02, $02, $02, $02, $82, $02, $02, $82
    .BYTE $02, $01, $81, $01, $01, $81, $01, $01
    .BYTE $01, $02, $02, $02, $02, $01, $01, $01
    .BYTE $01, $01, $01, $02, $02, $02, $02, $02
    .BYTE $02, $02, $02, $02, $02, $02, $02, $03
    .BYTE $03, $03, $03, $03, $03, $03, $03, $03
    .BYTE $03, $02, $02, $02, $02, $02, $02, $02
    .BYTE $02, $01, $01, $01, $02, $03, $03, $03
    .BYTE $02, $02, $00, $02, $01, $01, $01, $01
    .BYTE $02, $02, $02, $02, $02, $02, $02, $02
    .BYTE $02, $02, $01, $01, $01, $01, $01, $01
    .BYTE $01, $01, $01, $01, $03, $03, $03, $03
    .BYTE $00, $00, $02, $02, $02, $02, $03, $03
    .BYTE $03, $03, $01, $01, $02, $02, $03, $03
    .BYTE $01, $01, $01, $01, $02, $02, $02, $02
    .BYTE $01, $01, $01, $01, $02, $02, $02, $02
    .BYTE $02, $02, $01, $03, $03, $03, $03, $03
    .BYTE $03, $03, $03, $03, $03, $01, $01, $01
    .BYTE $01, $01, $01, $02, $00, $00, $03, $01
    .BYTE $01, $01, $01, $01, $01, $01, $01, $01
    .BYTE $03, $03, $03, $03, $03, $03, $03, $03
    .BYTE $03, $03, $03, $03, $03, $03, $02, $02
    .BYTE $01, $01, $02, $03

SpriteOffsets:
    .BYTE $60, $BC, $64, $B8, $68, $B4, $6C, $B0
    .BYTE $70, $CC, $74, $C8, $78, $C4, $7C, $C0
    .BYTE $80, $DC, $84, $D8, $88, $D4, $8C, $D0
    .BYTE $90, $EC, $94, $E8, $98, $E4, $9C, $E0
    .BYTE $A0, $FC, $A4, $F8, $A8, $F4, $AC, $F0
    .BYTE $60

AnimateAndDrawObjectWalking:
    JSR AnimateObjectWalking
    TYA
    JMP DrawObjectNotMirrored

; Params:
; A: frame image
; X: object index
; [00]: object X
; [01]: object Y
;
DrawObjectMirrored:
    LDY #$01                    ; Set [0C] mirrored.
    BNE :+

; Params:
; A: frame image
; X: object index
; [00]: object X
; [01]: object Y
; [0F]: flip horizontally
;
DrawObjectNotMirrored:
    LDY #$00
:
    STY $0C                     ; Set [0C] not mirrored.

    ; Set animation index = object type + 1.
    LDY ObjType, X

; Params:
; A: frame
; X: object index/cycle sprite index
; Y: object type
; [00]: object X
; [01]: object Y
; [0C]: mirrored
; [0F]: flip horizontally
;
; Note:
; If the object is not mirrored, then it's horizontally flippable.
;
; The animation index of most objects is (object type + 1),
; to make up for the fact that Link uses two animation indexes.
;
DrawObjectWithType:
    INY

; Params:
; A: frame
; X: object index/cycle sprite index
; Y: animation index
; [00]: object X
; [01]: object Y
; [0C]: mirrored
; [0F]: flip horizontally
;
; Note:
; If the object is not mirrored, then it's horizontally flippable.
; But Link uses [0F] for flipping without checking [0C].
;
DrawObjectWithAnim:
    STA $0D                     ; [0D] holds frame.
    STY $0E                     ; [0E] holds animation index.
    STX $08                     ; [08] holds object index/cycle sprite index

    ; Set the left and right sprite record offsets for CurSpriteIndex.
    LDY RollingSpriteIndex
    LDA SpriteOffsets, Y
    STA LeftSpriteOffset
    LDA SpriteOffsets+1, Y

    ; But if it's Link, then hardcode them to sprites $12 and $13.
    CPX #$00
    BNE DrawObjectWithAnimAndSpecificSprites
    LDA #$48                    ; Link's left sprite record offset
    STA LeftSpriteOffset
    LDA #$4C                    ; Link's right sprite record offset

DrawObjectWithAnimAndSpecificSprites:
    STA RightSpriteOffset
    LDY $0E
    LDA #$01                    ; This object has 2 sides.
    STA $07
    LDA #$08                    ; The right side is 8 pixels away from the left.
    STA $0A
    LDA ObjAnimations, Y
    CLC
    ADC $0D                     ; Add frame number.
    TAY
    LDA ObjAnimFrameHeap, Y     ; Look up the tile.
    STA $02                     ; Set the left tile.
    CLC
    ADC #$02
    STA $03                     ; The right tile is two tiles over.

    ; If object is Link, or object index/slot >= $D (weapons, room item),
    ; then go set attributes and write sprites.
    CPX #$00
    BEQ @UseTableAttr
    CPX #$0D
    BCS @UseTableAttr

    ; If the object has a "half-width draw" attribute, then
    ; go draw half-width.
    LDA ObjAttr, X
    AND #$02                    ; "Half-width draw" object attribute
    BNE @DrawHalfWidth

    ; If the object has the "Ignore sprite attribute table" attribute,
    ; then don't look up sprite attributes.
    LDA ObjAttr, X
    AND #$08                    ; "Ignore sprite attribute table" object attribute
    BNE :+

@UseTableAttr:
    LDA ObjAnimAttrHeap, Y      ; Look up the sprite attribute.
    JSR Anim_SetSpriteDescriptorAttributes
:
    ; If it's Link, write the sprites. He's never mirrored.
    CPX #$00
    BEQ Anim_WriteHorizontallyFlippableSpritePair

    ; For other objects, the caller controls mirroring and flipping.
    ;
    ; If mirrored [0C], then draw it mirrored.
    ; Else draw it horizontally flippable, controlled by [0F].
    LDY $0C
    BEQ Anim_WriteHorizontallyFlippableSpritePair
    JMP Anim_WriteMirroredSpritePair

@DrawHalfWidth:
    DEC $07                     ; Mark the object narrow (half-width).
    JMP Anim_WriteSpritePair

; Params:
; [00]: Object X
; [01]: Object Y
; [02]: Left tile
; [03]: Right tile
; [04]: Left attributes
; [05]: Right attributes
; [07]: Has two sides
; [08]: Cycle sprite index
; [0A]: X separation
; [0F]: Flip horizontally
; [0343]: LeftSpriteOffset
; [0344]: RightSpriteOffset
;
Anim_WriteHorizontallyFlippableSpritePair:
    ; If [0F] = 0, write the sprite pair with no further processing.
    LDA $0F
    BEQ Anim_WriteSpritePair

    ; Otherwise, reverse the two sides.
    LDA $02
    PHA
    LDA $03
    STA $02
    PLA
    STA $03

    ; Set the horizontal flip sprite attributes.
    LDA $04
    EOR #$40
    STA $04
    LDA $05
    EOR #$40
    STA $05

; Params:
; [00]: Object X
; [01]: Object Y
; [02]: Left tile
; [03]: Right tile
; [04]: Left attributes
; [05]: Right attributes
; [07]: Has two sides
; [08]: Cycle sprite index
; [0A]: X separation
; [0343]: LeftSpriteOffset
; [0344]: RightSpriteOffset
;
Anim_WriteSpritePair:
    ; If not currently invincible, then leave the attributes alone.
    LDY ObjInvincibilityTimer, X
    BEQ Anim_WriteSpritePairNotFlashing

    ; For both left and right sprites, indexed by Y register:
    LDY #$01
:
    LDA $0004, Y                ; Toss out the palette bits of the sprite attributes.
    AND #$FC
    STA $0004, Y
    LDA ObjInvincibilityTimer, X    ; Patch the bottom 2 bits of the invincibility timer.
    AND #$03
    ORA $0004, Y
    STA $0004, Y                ; This gives us the flashing effect.
    DEY
    BPL :-

Anim_WriteSpritePairNotFlashing:
    LDX LeftSpriteOffset        ; Access the first sprite.
    LDY #$00

@LoopSprite:
    LDA $0002, Y                ; Write the sprite tile.
    STA Sprites+1, X
    LDA $01                     ; Write the sprite Y.
    STA Sprites, X
    LDA $00                     ; Write the sprite X.
    STA Sprites+3, X
    CLC                         ; Separate the second sprite appropriately from the first.
    ADC $0A
    STA $00
    LDA $0004, Y                ; Write the sprite attributes.
    STA Sprites+2, X
    LDX RightSpriteOffset       ; Now point to the other sprite.
    LDA $08                     ; Cycle the current sprite index, if needed.
    BEQ :+
    JSR CycleCurSpriteIndex
:
    INY                         ; Increase the index to access the second sprite's properties.
    DEC $07
    BPL @LoopSprite             ; If there is a right sprite, then go process it.
    LDX $08                     ; Put the object index in X again.
    RTS
