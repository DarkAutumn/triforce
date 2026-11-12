; ============================================================================
; FILE:     bank1/item-sprite-draw.asm   BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 5247-5455
; SUBJECTS: Anim
; TOPIC:    Item and weapon sprites: the frame tile table indexed by item slot, the routine that decides
;           from the tile number whether an item is one sprite, a mirrored pair or a flippable pair, and
;           the single-sprite writers used by multi-part monsters.
; SEE ALSO: bank1/object-animation-draw.asm (Anim_WriteSpritePair and the hit flash; SpriteOffsets);
;           bank1/take-item.asm (ItemIdToSlot: item id to item slot); bank7/room-item-draw.asm
;           (DrawItemBySlot chooses the palette row); bank7/sword-and-rod.asm,
;           bank7/sword-shot-magic-shot.asm, bank7/boomerang-return-draw.asm, bank7/bomb.asm (weapon draw
;           calls); bank7/object-animation.asm (death sparkle, slot $24); bank7/status-bar-lift-item.asm
;           (Link lifting an item, slot $21); bank1/modeb-cave-save-sprites.asm (CycleCurSpriteIndex).
; KEY FACTS:
;   - Items are drawn by item slot, not item id: slot = ItemIdToSlot[id] = the item's inventory byte
;     offset from InvItems ($00 sword ... $1F magic shield). Slots $20-$24 exist only as images: $20
;     magic sword, $21 Link lifting an item, $22 sword shot, $23 magic shot, $24 death sparkle.
;     Anim_ItemFrameOffsets
;   - The left tile number decides the shape: $F3 or $20-$61 = one sprite, drawn 4 pixels right of X
;     unless LeftAlignHalfWidthObj is set; below $20 or $62-$6B = mirrored pair 7 pixels apart;
;     $6C-$7B = mirrored pair 8 apart; $7C and up = two halves (tile, tile + 2), flipped by [0F].
;     Anim_WriteSpecificItemSprites
;   - IsDrawnItemNarrow is cleared by Anim_WriteItemSprites and set to 1 when the item drawn is one
;     sprite wide. Anim_WriteItemSprites, Anim_WriteSpecificItemSprites
;   - Frames: sword and rod 0 vertical, 1 horizontal; arrow 0 vertical, 1 horizontal, 2 spark; bomb 0,
;     its dust cloud and the monster spawn cloud 1-3; boomerang 0-2 spin, 3 spark; fairy 0-1; sword
;     shot 0 vertical, 1 horizontal; magic shot 0 vertical, 1 horizontal, 2 = a corner of the spreading
;     sword shot; death sparkle 0-1. Anim_ItemFrameTiles
;   - Item sprites flash like objects while ObjInvincibilityTimer of the slot in X is nonzero
;     (Anim_WriteSpritePair). Anim_WriteSprite does its own flash: attributes = FrameCounter AND 3, a
;     new palette row every frame. Anim_WriteSprite
;   - Anim_WriteSprite and Anim_WriteSpecificSprite write one sprite at the object's ObjX, ObjY and
;     advance RollingSpriteIndex by one. Anim_EndWriteSprite
; CONTENTS: (generated; L = line in this file)
;   L0051 Anim_ItemFrameOffsets                      data  TABLE: where each item's frames start; index = item slot (ItemIdToSlot; $20-$24 are image-only
;   L0069 Anim_ItemFrameTiles                        data  TABLE: left tile of every item frame; index = Anim_ItemFrameOffsets[item slot] + frame; value = tile.
;   L0097 Anim_WriteStaticItemSpritesWithAttributes  code  Draws frame 0 of an item, unflipped, with attributes A on both sprites.
;   L0111 Anim_WriteItemSprites                      code  Draws an item frame in the next rolling sprite records.
;   L0133 Anim_WriteSpecificItemSprites              code  Draws an item frame in the sprite records the caller chose; the tile number selects the shape.
;   L0190 Anim_WriteMirroredSpritePair               code  Writes a sprite pair whose right half is the left tile flipped horizontally.
;   L0202 Anim_SetSpriteDescriptorRedPaletteRow      code  Sets both sprites' attributes to 2: sprite palette row 2 (the red row), no flip, in front.
;   L0206 Anim_SetSpriteDescriptorAttributes         code  Entry with A = the attribute byte for both sprites: [04] := [05] := A. A is unchanged.
;   L0216 Anim_WriteLevelPaletteSprite               code  Writes one sprite of tile A at object X's position with sprite palette row 3 (the per-level row).
;   L0228 Anim_WriteSprite                           code  Writes one sprite of tile A at object X's position in the next rolling sprite record.
;   L0247 Anim_WriteSpecificSprite                   code  Writes one sprite of tile A at object X's position in the sprite record at offset Y.
;   L0255 Anim_EndWriteSprite                        code  Entry with A = the sprite's Y coordinate; tile and X are already in the record at offset Y.
; ============================================================================

; TABLE: where each item's frames start; index = item slot (ItemIdToSlot; $20-$24 are image-only
; slots); value = offset of frame 0 in Anim_ItemFrameTiles.
Anim_ItemFrameOffsets:
    ; $00 sword, $01 bomb, $02 arrow, $03 bow, $04 candle, $05 flute, $06 food, $07 potion
    .BYTE $00, $03, $07, $0A, $0B, $0C, $0D, $0E
    ; $08 rod, $09 raft, $0A book, $0B ring, $0C ladder, $0D magic key, $0E bracelet, $0F letter
    .BYTE $0F, $11, $12, $13, $14, $15, $16, $17
    ; $10 compass, $11 map (letter's tile), $12 level 9 compass, $13 level 9 map, $14 fairy, $15 clock,
    ; $16 rupee, $17 key
    .BYTE $18, $17, $18, $17, $19, $1B, $1C, $1D
    ; $18 heart container, $19 heart, $1A triforce piece, $1B Triforce of Power, $1C 5 rupees (rupee's
    ; tile), $1D boomerang, $1E magic boomerang (same frames), $1F magic shield
    .BYTE $1E, $1F, $20, $21, $1C, $22, $22, $26
    ; $20 magic sword image, $21 Link lifting an item, $22 sword shot, $23 magic shot, $24 death sparkle
    .BYTE $27, $28, $29, $2B, $2E

; TABLE: left tile of every item frame; index = Anim_ItemFrameOffsets[item slot] + frame; value = tile.
; The tile number also selects the shape (Anim_WriteSpecificItemSprites): N = one sprite ($F3,
; $20-$61), S = mirrored pair 7 pixels apart ($62-$6B), M = mirrored pair 8 apart ($6C-$7B),
; F = two halves tile and tile + 2, flippable ($7C and up).
Anim_ItemFrameTiles:
    ; $00 sword: $20 N vertical, $82 F horizontal, $3C N (offset $02; the sword draw code uses only
    ; frames 0 and 1; whether anything draws it is (unverified));
    ; $03 bomb: $34 N, then clouds $70, $72, $74 M; $07 arrow: $28 N vertical
    .BYTE $20, $82, $3C, $34, $70, $72, $74, $28
    ; arrow: $86 F horizontal, $3C N spark; $0A bow $2A; $0B candle $26; $0C flute $24; $0D food $22;
    ; $0E potion $40; $0F rod: $4A N vertical
    .BYTE $86, $3C, $2A, $26, $24, $22, $40, $4A
    ; rod: $8A F horizontal; $11 raft $6C M; $12 book $42; $13 ring $46; $14 ladder $76 M;
    ; $15 magic key $2C; $16 bracelet $4E; $17 letter and maps $4C
    .BYTE $8A, $6C, $42, $46, $76, $2C, $4E, $4C
    ; $18 compasses $6A S; $19 fairy $50, $52 N; $1B clock $66 S; $1C rupee and 5 rupees $32;
    ; $1D key $2E; $1E heart container $68 S; $1F heart $F3 N
    .BYTE $6A, $50, $52, $66, $32, $2E, $68, $F3
    ; $20 triforce piece $6E M; $21 Triforce of Power $F2 F; $22 boomerangs: $36, $38, $3A N spin,
    ; $3C N spark; $26 magic shield $56; $27 magic sword image $48
    .BYTE $6E, $F2, $36, $38, $3A, $3C, $56, $48
    ; $28 Link lifting an item $78 M; $29 sword shot: $20 N vertical, $82 F horizontal; $2B magic shot:
    ; $7A M vertical, $7C F horizontal, $30 N corner of the spreading sword shot; $2E death sparkle:
    ; $64, $62 S
    .BYTE $78, $20, $82, $7A, $7C, $30, $64, $62

; ----------------------------------------------------------------------------
; Draws frame 0 of an item, unflipped, with attributes A on both sprites.
; IN:   A = sprite attributes; X = object slot (nonzero: RollingSpriteIndex advances); Y = item slot;
;       [00] = X; [01] = Y.
; OUT:  as Anim_WriteItemSprites; [0C] = 0; [0F] = 0.
; ----------------------------------------------------------------------------
Anim_WriteStaticItemSpritesWithAttributes:
    JSR Anim_SetSpriteDescriptorAttributes
    LDA #$00
    STA $0F                     ; [0F] := 0: no horizontal flip.
    STA $0C                     ; [0C] := 0: frame 0.

; ----------------------------------------------------------------------------
; Draws an item frame in the next rolling sprite records.
; IN:   X = object slot (nonzero: RollingSpriteIndex advances; also selects the flash timer); Y = item
;       slot; [00] = X; [01] = Y; [04], [05] = left, right attributes; [0C] = frame; [0F] = flip
;       horizontally (used only by tiles $7C and up).
; OUT:  IsDrawnItemNarrow = 1 when one sprite was drawn, else 0; X = object slot; LeftSpriteOffset,
;       RightSpriteOffset set from SpriteOffsets.
; ----------------------------------------------------------------------------
Anim_WriteItemSprites:
    TYA                         ; Keep the item slot on the stack while Y indexes SpriteOffsets.
    PHA
    LDA #$00
    STA IsDrawnItemNarrow
    LDY RollingSpriteIndex
    LDA SpriteOffsets, Y
    STA LeftSpriteOffset
    LDA SpriteOffsets+1, Y
    STA RightSpriteOffset
    PLA
    TAY

; ----------------------------------------------------------------------------
; Draws an item frame in the sprite records the caller chose; the tile number selects the shape.
; IN:   as Anim_WriteItemSprites, plus LeftSpriteOffset and RightSpriteOffset set by the caller.
; OUT:  IsDrawnItemNarrow + 1 when one sprite was drawn (not cleared here); X = object slot.
; NOTE: Shapes by left tile: $F3 or $20-$61 one sprite (X + 4 unless LeftAlignHalfWidthObj); below
;       $20 or $62-$6B mirrored pair 7 pixels apart; $6C-$7B mirrored pair 8 apart; $7C and up two
;       halves (tile, tile + 2), swapped and flipped when [0F] is nonzero.
;       Link's lifting pose (slot $21) is drawn through here into his own records $12, $13.
; ----------------------------------------------------------------------------
Anim_WriteSpecificItemSprites:
    STX $08                     ; [08] := object slot; nonzero also means "advance RollingSpriteIndex".
    LDA #$01                    ; [07] := 1: two sprites, until found otherwise.
    STA $07
    LDA #$08                    ; [0A] := 8 pixels between the two sprites, until found otherwise.
    STA $0A
    LDA Anim_ItemFrameOffsets, Y
    CLC
    ADC $0C                     ; Y := first frame offset + frame.
    TAY
    LDA Anim_ItemFrameTiles, Y
    STA $02                     ; [02] := left tile
    CLC
    ADC #$02
    STA $03                     ; [03] := left tile + 2 (replaced below for mirrored pairs)

    ; Tile $F3 (heart) or $20-$61: one sprite.
    LDA $02
    CMP #$F3
    BEQ @Narrow
    CMP #$20
    BCC @Wide
    CMP #$62
    BCS @Wide

@Narrow:
    ; Center the 8-pixel sprite in the 16-pixel cell (X + 4), unless LeftAlignHalfWidthObj is set
    ; (it is while Link lifts an item).
    LDA LeftAlignHalfWidthObj
    BNE :+
    LDA $00
    CLC
    ADC #$04
    STA $00
:
    INC IsDrawnItemNarrow
    LDA #$00                    ; [07] := 0: one sprite only.
    STA $07
    JMP Anim_WriteSpritePair

@Wide:
    ; A = left tile, below $20 or $62 and up (not $F3).
    CMP #$6C
    BCC @_Slim                  ; Below $6C: mirrored pair, 7 pixels apart.
    CMP #$7C
    BCC Anim_WriteMirroredSpritePair    ; $6C-$7B: mirrored pair, 8 pixels apart.
    JMP Anim_WriteHorizontallyFlippableSpritePair    ; $7C and up: two halves, flippable by [0F].

@_Slim:
    LDA #$07
    STA $0A                     ; [0A] := 7: the two sprites overlap by one pixel column.

; ----------------------------------------------------------------------------
; Writes a sprite pair whose right half is the left tile flipped horizontally.
; IN:   as Anim_WriteSpritePair; [03] is ignored (set to [02] here).
; OUT:  as Anim_WriteSpritePair; [03] = [02]; bit $40 of [05] toggled.
; ----------------------------------------------------------------------------
Anim_WriteMirroredSpritePair:
    LDA $02                     ; Right tile := left tile.
    STA $03
    LDA $05                     ; Toggle the horizontal flip bit of the right sprite.
    EOR #$40
    STA $05
    JMP Anim_WriteSpritePair

; ----------------------------------------------------------------------------
; Sets both sprites' attributes to 2: sprite palette row 2 (the red row), no flip, in front.
; OUT:  A = 2; [04] = [05] = 2.
; ----------------------------------------------------------------------------
Anim_SetSpriteDescriptorRedPaletteRow:
    LDA #$02

; Entry with A = the attribute byte for both sprites: [04] := [05] := A. A is unchanged.
Anim_SetSpriteDescriptorAttributes:
    STA $04
    STA $05
    RTS

; ----------------------------------------------------------------------------
; Writes one sprite of tile A at object X's position with sprite palette row 3 (the per-level row).
; IN:   A = tile; X = object slot.
; NOTE: Sets [03] = 3 and falls into Anim_WriteSprite.
; ----------------------------------------------------------------------------
Anim_WriteLevelPaletteSprite:
    LDY #$03
    STY $03

; ----------------------------------------------------------------------------
; Writes one sprite of tile A at object X's position in the next rolling sprite record.
; IN:   A = tile; X = object slot; [03] = sprite attributes.
; OUT:  RollingSpriteIndex advanced by 1; [03] = FrameCounter AND 3 if the object was flashing.
; NOTE: While ObjInvincibilityTimer,X is nonzero the whole attribute byte is replaced by
;       (FrameCounter AND 3): a different palette row every frame, and any flip or priority bit in
;       [03] is lost.
; ----------------------------------------------------------------------------
Anim_WriteSprite:
    PHA                         ; Keep the tile.
    LDA ObjInvincibilityTimer, X
    BEQ :+
    LDA FrameCounter
    AND #$03
    STA $03
:
    LDY RollingSpriteIndex      ; Y := offset of the next rolling sprite record.
    LDA SpriteOffsets, Y
    TAY
    PLA

; ----------------------------------------------------------------------------
; Writes one sprite of tile A at object X's position in the sprite record at offset Y.
; IN:   A = tile; X = object slot (ObjX, ObjY give the position); Y = offset into Sprites;
;       [03] = sprite attributes.
; OUT:  RollingSpriteIndex advanced by 1 (even though the record was chosen by the caller).
; ----------------------------------------------------------------------------
Anim_WriteSpecificSprite:
    STA Sprites+1, Y
    LDA ObjX, X
    STA Sprites+3, Y
    LDA ObjY, X

; Entry with A = the sprite's Y coordinate; tile and X are already in the record at offset Y.
; Stores Y coordinate and attributes [03], then advances RollingSpriteIndex (CycleCurSpriteIndex).
Anim_EndWriteSprite:
    STA Sprites, Y
    LDA $03
    STA Sprites+2, Y
    JMP CycleCurSpriteIndex
