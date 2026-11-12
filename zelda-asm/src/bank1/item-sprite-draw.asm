; ============================================================================
; FILE:     bank1/item-sprite-draw.asm   BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5247-5455
; SUBJECTS: (none)
; TOPIC:    Item frame tiles, item sprite writers
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 Anim_ItemFrameOffsets                      data  Maps an item slot to the offset of its first frame in ItemFrameTiles heap.
;   L0031 Anim_ItemFrameTiles                        data
;   L0046 Anim_WriteStaticItemSpritesWithAttributes  code  Params:
;   L0065 Anim_WriteItemSprites                      code  Params:
;   L0093 Anim_WriteSpecificItemSprites              code  Params:
;   L0154 Anim_WriteMirroredSpritePair               code  Params:
;   L0167 Anim_SetSpriteDescriptorRedPaletteRow      code  Returns:
;   L0179 Anim_SetSpriteDescriptorAttributes         code  Params:
;   L0188 Anim_WriteLevelPaletteSprite               code  Params:
;   L0197 Anim_WriteSprite                           code  Params:
;   L0216 Anim_WriteSpecificSprite                   code  Params:
;   L0227 Anim_EndWriteSprite                        code  Params:
; ============================================================================

; Maps an item slot to the offset of its first frame in ItemFrameTiles heap.
Anim_ItemFrameOffsets:
    .BYTE $00, $03, $07, $0A, $0B, $0C, $0D, $0E
    .BYTE $0F, $11, $12, $13, $14, $15, $16, $17
    .BYTE $18, $17, $18, $17, $19, $1B, $1C, $1D
    .BYTE $1E, $1F, $20, $21, $1C, $22, $22, $26
    .BYTE $27, $28, $29, $2B, $2E

Anim_ItemFrameTiles:
    .BYTE $20, $82, $3C, $34, $70, $72, $74, $28
    .BYTE $86, $3C, $2A, $26, $24, $22, $40, $4A
    .BYTE $8A, $6C, $42, $46, $76, $2C, $4E, $4C
    .BYTE $6A, $50, $52, $66, $32, $2E, $68, $F3
    .BYTE $6E, $F2, $36, $38, $3A, $3C, $56, $48
    .BYTE $78, $20, $82, $7A, $7C, $30, $64, $62

; Params:
; A: sprite attributes
; X: cycle sprite index / object index
; Y: item slot
; [00]: X
; [01]: Y
;
Anim_WriteStaticItemSpritesWithAttributes:
    JSR Anim_SetSpriteDescriptorAttributes
    LDA #$00
    STA $0F                     ; Reset horizontal flipping.
    STA $0C                     ; Reset frame.

; Params:
; X: cycle sprite index / object index
; Y: item slot
; [00]: X
; [01]: Y
; [04]: left sprite attributes
; [05]: right sprite attributes
; [0C]: frame image
; [0F]: flip horizontally
;
; Returns:
; [52]: ProcessedNarrowObj
;
Anim_WriteItemSprites:
    TYA                         ; Save the item slot passed in Y.
    PHA
    LDA #$00
    STA ProcessedNarrowObj
    LDY RollingSpriteIndex      ; Determine the left and right sprite offsets for current sprite index.
    LDA SpriteOffsets, Y
    STA LeftSpriteOffset
    LDA SpriteOffsets+1, Y
    STA RightSpriteOffset
    PLA                         ; Restore the item slot passed in.
    TAY

; Params:
; X: cycle sprite index / object index
; Y: item slot
; [00]: X
; [01]: Y
; [04]: left sprite attributes
; [05]: right sprite attributes
; [0C]: frame
; [0F]: flip horizontally
; [0343]: LeftSpriteOffset
; [0344]: RightSpriteOffset
;
; Returns:
; [52]: ProcessedNarrowObj
;
Anim_WriteSpecificItemSprites:
    STX $08                     ; Store the object index in [08].
    LDA #$01                    ; Assume the object has two sides.
    STA $07
    LDA #$08                    ; Both sides are usually separated by 8 pixels.
    STA $0A
    LDA Anim_ItemFrameOffsets, Y
    CLC
    ADC $0C                     ; Get the frame.
    TAY
    LDA Anim_ItemFrameTiles, Y
    STA $02                     ; [02] gets the tile we just looked up.
    CLC
    ADC #$02                    ; The second tile must be two tiles farther in CHR.
    STA $03                     ; Put it in [03].

    ; If left tile is $F3 or in [$20, $62),
    ; then this is a narrow / half-width object.
    LDA $02
    CMP #$F3
    BEQ @Narrow
    CMP #$20
    BCC @Wide
    CMP #$62
    BCS @Wide

@Narrow:
    LDA LeftAlignHalfWidthObj
    BNE :+                      ; If not left-aligning a half-width object,
    LDA $00                     ; Then add 4 to X to center it.
    CLC
    ADC #$04
    STA $00
:
    INC ProcessedNarrowObj      ; Record that this is a half-width object.
    LDA #$00                    ; And mark it as such.
    STA $07
    JMP Anim_WriteSpritePair    ; Finished. Go write sprites.

@Wide:
    CMP #$6C
    BCC @_Slim                  ; If tile < $6C, go mark this a slim object.
    CMP #$7C
    BCC Anim_WriteMirroredSpritePair
    JMP Anim_WriteHorizontallyFlippableSpritePair    ; Tile >= $7C. Can flip horizontally.

@_Slim:
    LDA #$07
    STA $0A                     ; Both sides of slim objects overlap one pixel.

; Params:
; X: cycle sprite index / object index
; [00]: X
; [01]: Y
; [02]: left tile
; [03]: right tile
; [04]: left sprite attributes
; [05]: right sprite attributes
; [0343]: LeftSpriteOffset
; [0344]: RightSpriteOffset
;
Anim_WriteMirroredSpritePair:
    LDA $02                     ; All wide items that we process have mirrored sides.
    STA $03                     ; Make the right side the same as the left.
    LDA $05                     ; Flip the right side horizontally.
    EOR #$40
    STA $05
    JMP Anim_WriteSpritePair

; Returns:
; A: 2
; [04]: left sprite attributes
; [05]: right sprite attributes
;
Anim_SetSpriteDescriptorRedPaletteRow:
    LDA #$02

; Params:
; A: sprite attributes for both sides
;
; Returns:
; [04]: left side sprite attributes
; [05]: right side sprite attributes
;
; Doesn't change A.
;
Anim_SetSpriteDescriptorAttributes:
    STA $04
    STA $05
    RTS

; Params:
; A: tile number
; X: object index
;
Anim_WriteLevelPaletteSprite:
    LDY #$03
    STY $03

; Params:
; A: tile number
; X: object index
; [03]: sprite attributes
;
Anim_WriteSprite:
    PHA
    LDA ObjInvincibilityTimer, X
    BEQ :+
    LDA FrameCounter
    AND #$03
    STA $03
:
    LDY RollingSpriteIndex
    LDA SpriteOffsets, Y
    TAY
    PLA

; Params:
; A: tile number
; X: object index
; Y: sprite record offset
; [03]: sprite attributes
;
Anim_WriteSpecificSprite:
    STA Sprites+1, Y
    LDA ObjX, X
    STA Sprites+3, Y
    LDA ObjY, X

; Params:
; A: Y coordinate
; Y: sprite record offset
; [03]: sprite attributes
;
Anim_EndWriteSprite:
    STA Sprites, Y
    LDA $03
    STA Sprites+2, Y
    JMP CycleCurSpriteIndex
