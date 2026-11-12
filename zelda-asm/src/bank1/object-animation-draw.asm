; ============================================================================
; FILE:     bank1/object-animation-draw.asm BANK: 1 (RAM code: stored at $A500, run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4937-5245
; SUBJECTS: Anim, Sprite
; TOPIC:    How Link and room objects become sprites: the per-type animation tables (first frame offset,
;           tile and attributes of each frame), the rolling sprite record order, and the DrawObject
;           routines that write one or two 8x16 sprites with flipping, mirroring and the hit flash.
; SEE ALSO: bank1/item-sprite-draw.asm (item and weapon sprites; Anim_WriteMirroredSpritePair,
;           Anim_SetSpriteDescriptorAttributes); bank7/object-animation.asm (Anim_AnimateObjWalking picks
;           the frame; Anim_FetchObjPosForSpriteDescriptor sets [00], [01], [0F]);
;           bank1/modeb-cave-save-sprites.asm (Sprite_CycleCurIndex, Sprite_HideObjectSprites);
;           bank7/link-animate-ladder.asm (Link's draw call); bank7/mode5-play-update.asm (the object
;           loop that calls Anim_AnimateAndDrawObjectWalking); bank4/push-block.asm
;           (Sprite_DrawObjMirroredOverLink: sprite records $10 and $11).
; KEY FACTS:
;   - Animation index = object type + 1; index 0 is Link. Anim_ObjFrameOffsets[index] + frame indexes both
;     Anim_ObjFrameTiles (tile) and Anim_ObjFrameAttrs (attributes). Anim_DrawObjectWithType, Anim_DrawObjectWithAnim
;   - An object is two 8x16 sprites 8 pixels apart: left tile from the table, right tile = left + 2.
;     Anim_DrawObjectWithAnimAndSpecificSprites
;   - Walking frames: 0 and 1 sideways (flipped when facing left), 2 down, 3 up; Link's frame gets +4
;     while (his state AND $30) is $10 or $20 (sword or item in use). Anim_AnimateAndDrawObjectWalking
;     (frame from Anim_AnimateObjWalking), Anim_ObjFrameTiles
;   - Link always uses sprite records $12 and $13 (offsets $48, $4C) and does not advance
;     RollingSpriteIndex. Anim_DrawObjectWithAnim, Anim_WriteSpritePairNotFlashing
;   - Every other object takes its records from Sprite_RollingOffsets, a fixed order over records $18-$3F; the
;     index advances once per sprite written and wraps at $28. Sprite_RollingOffsets,
;     Anim_WriteSpritePairNotFlashing
;   - While the slot's ObjInvincibilityTimer is nonzero, the palette bits of its sprites are replaced by
;     (timer AND 3): the object flashes through the four sprite palette rows. Anim_WriteSpritePair
;   - ObjAttr bit $02 draws one sprite only (half width); bit $08 keeps the caller's attributes in [04],
;     [05] instead of the table's. Both are tested only for slots $01-$0C.
;     Anim_DrawObjectWithAnimAndSpecificSprites
;   - Mirrored ([0C] nonzero): both sprites show the left tile, the right one flipped. Not mirrored:
;     [0F] nonzero swaps the halves and flips both. Link is never mirrored.
;     Anim_DrawObjectWithAnimAndSpecificSprites, Anim_WriteHorizontallyFlippableSpritePair
;   - Palette rows in the attribute table: 0 Link's, 1 blue, 2 red, 3 the row that differs per level
;     (colours from the level palette data). Row 1 is used by Lynel $01, Goriya $05, Octorok $09/$0A,
;     Darknut $0C, Tektite $0D, Keese $1B, Wizzrobe $23; row 2 by their other variants. Anim_ObjFrameAttrs
; CONTENTS: (generated; L = line in this file)
;   L0058 Anim_ObjFrameOffsets                       data  TABLE: where each animation's frames start; index = animation index (0 = Link, else object type + 1);
;   L0131 Anim_ObjFrameTiles                         data  TABLE: sprite tile of every animation frame; index = Anim_ObjFrameOffsets[animation index] + frame; value =
;   L0166 Anim_ObjFrameAttrs                         data  TABLE: sprite attributes of every animation frame; index as Anim_ObjFrameTiles; value = sprite
;   L0199 Sprite_RollingOffsets                      data  TABLE: sprite record offsets handed to objects in turn; index = RollingSpriteIndex 0-$27 (entry $28
;   L0216 Anim_AnimateAndDrawObjectWalking           code  Steps a walking object's animation and draws it by its facing direction (not mirrored).
;   L0226 Anim_DrawObjectMirrored                    code  Draws object X with frame A as a mirrored pair: the right sprite is the left tile flipped.
;   L0235 Anim_DrawObjectNotMirrored                 code  Draws object X with frame A as two different halves, flipped horizontally when [0F] is nonzero.
;   L0247 Anim_DrawObjectWithType                    code  Draws an object with the animation of object type Y (animation index Y + 1).
;   L0259 Anim_DrawObjectWithAnim                    code  Draws an object with animation index Y, using Link's own sprite records or the next rolling pair.
;   L0289 Anim_DrawObjectWithAnimAndSpecificSprites  code  Looks up the frame's tile and attributes and writes the object's one or two sprites.
;   L0344 Anim_WriteHorizontallyFlippableSpritePair  code  Writes the sprite pair, first swapping the halves and flipping both when [0F] is nonzero.
;   L0375 Anim_WriteSpritePair                       code  Writes one or two sprites from the descriptor in [00]-[0A]; a slot with invincibility time flashes.
;   L0394 Anim_WriteSpritePairNotFlashing            code  Entry that skips the flash: writes the sprites with [04], [05] as given. X need not be set.
; ============================================================================

; TABLE: where each animation's frames start; index = animation index (0 = Link, else object type + 1);
; value = offset of frame 0 in Anim_ObjFrameTiles and Anim_ObjFrameAttrs.
; Types with the same value share their frames. Type $69 has $CC, one past the end of both heaps.
Anim_ObjFrameOffsets:
    ; Link, type $00, Lynel $01 $02, Moblin $03 $04, Goriya $05 $06
    .BYTE $00, $08, $0B, $0F, $13, $17, $5C, $60
    ; Octorok $07 $08 $09 $0A, Darknut $0B $0C, Tektite $0D $0E
    .BYTE $1B, $1B, $21, $21, $64, $6A, $27, $29
    ; Leever $0F $10, Zora $11, Vire $12, Zol $13, Gel $14 $15, Pols Voice $16
    .BYTE $2B, $35, $3F, $70, $74, $76, $76, $78
    ; Like-Like $17, small Digdogger $18, type $19, Peahat $1A, Keese $1B $1C $1D, Armos $1E
    .BYTE $7A, $7E, $80, $49, $82, $84, $86, $4B
    ; Boulder spawner $1F, Boulder $20, Ghini $21 $22, Wizzrobe $23 $24, Patra child $25 $26
    .BYTE $4F, $4F, $51, $51, $88, $8C, $90, $90
    ; Wallmaster $27, Rope $28, type $29, Stalfos $2A, Bubble $2B $2C $2D, Whirlwind $2E
    .BYTE $92, $94, $96, $98, $99, $99, $99, $53
    ; Pond fairy $2F, Gibdo $30, Dodongo $31 $32, Gohma $33 $34, Rupee stash $35, Grumble $36
    .BYTE $54, $9A, $9B, $9B, $A5, $A5, $AB, $AB
    ; Zelda $37, Digdogger $38 $39, Lanmola $3A $3B, Manhandla $3C, Aquamentus $3D, Ganon $3E
    .BYTE $AC, $AE, $AE, $AF, $AF, $B2, $B8, $B8
    ; Guard fire $3F, Standing fire $40, Moldorm $41, Gleeok $42-$45, Gleeok head $46
    .BYTE $08, $08, $C6, $C6, $C6, $C6, $C6, $C6
    ; Patra $47 $48, Trap $49 $4A, dungeon persons $4B-$4E
    .BYTE $C8, $C8, $C9, $C9, $CA, $CA, $CA, $CA
    ; Dungeon persons $4F-$52, flying rock $53, type $54, fireball $55 $56
    .BYTE $CA, $CA, $CA, $CA, $09, $09, $0A, $0A
    ; Sword shot $57, magic shot $58 $59, shot $5A, monster arrow $5B, monster boomerang $5C,
    ; dead dummy $5D, flute secret $5E
    .BYTE $0B, $0B, $0B, $0B, $0B, $0B, $CB, $55
    ; Ladder $5F, dropped item $60, dock $61, rock $62, rock wall $63, tree $64, gravestone $65,
    ; armos tile object $66
    .BYTE $55, $55, $55, $55, $55, $55, $56, $57
    ; Rock wall $67, push block $68, type $69, cave persons $6A-$6E
    .BYTE $57, $CB, $CC, $58, $58, $58, $58, $58
    ; Cave persons $6F-$76
    .BYTE $58, $58, $58, $58, $59, $59, $59, $59
    ; Cave persons $77-$7D
    .BYTE $5A, $5A, $5A, $5A, $5B, $5B, $5B

; TABLE: sprite tile of every animation frame; index = Anim_ObjFrameOffsets[animation index] + frame; value =
; tile of the left 8x16 sprite (the right sprite shows tile + 2, or the same tile when mirrored).
; Start offsets and the types that use them; (n) = bytes up to the next start, an upper bound on the
; frames (which frames are requested is up to each type's code):
;   $00 Link (8): 0, 1 walking sideways; 2 down; 3 up; 4-7 the same poses with a sword or item in use
;   $08 type $00, guard fire $3F, standing fire $40, and Link's candle fire (1)
;   $09 flying rock $53, type $54 (1)       $0A fireball $55, $56 (1)
;   $0B Lynel $01, shot types $57-$5C (4)   $0F Lynel $02 (4)
;   $13 Moblin $03 (4)                      $17 Moblin $04 (4)
;   $1B Octorok $07, $08 (6)                $21 Octorok $09, $0A (6)
;   $27 Tektite $0D (2)                     $29 Tektite $0E (2)
;   $2B Leever $0F (10)                     $35 Leever $10 (10)
;   $3F Zora $11 (10)                       $49 Peahat $1A (2)
;   $4B Armos $1E (4)                       $4F boulder spawner $1F, Boulder $20 (2)
;   $51 Ghini $21, $22 (2)                  $53 Whirlwind $2E (1)
;   $54 Pond fairy $2F (1)                  $55 types $5E-$64 (1)
;   $56 gravestone $65 (1)                  $57 types $66, $67 (1)
;   $58 cave persons $6A-$72 (1)            $59 cave persons $73-$76 (1)
;   $5A cave persons $77-$7A (1)            $5B cave persons $7B-$7D (1)
;   $5C Goriya $05 (4)                      $60 Goriya $06 (4)
;   $64 Darknut $0B (6)                     $6A Darknut $0C (6)
;   $70 Vire $12 (4)                        $74 Zol $13 (2)
;   $76 Gel $14, $15 (2)                    $78 Pols Voice $16 (2)
;   $7A Like-Like $17 (4)                   $7E small Digdogger $18 (2)
;   $80 type $19 (2)                        $82 Keese $1B (2)
;   $84 Keese $1C (2)                       $86 Keese $1D (2)
;   $88 Wizzrobe $23 (4)                    $8C Wizzrobe $24 (4)
;   $90 Patra child $25, $26 (2)            $92 Wallmaster $27 (2)
;   $94 Rope $28 (2)                        $96 type $29 (2)
;   $98 Stalfos $2A (1)                     $99 Bubble $2B-$2D (1)
;   $9A Gibdo $30 (1)                       $9B Dodongo $31, $32 (10)
;   $A5 Gohma $33, $34 (6)                  $AB Rupee stash $35, Grumble $36 (1)
;   $AC Zelda $37 (2)                       $AE Digdogger $38, $39 (1)
;   $AF Lanmola $3A, $3B (3)                $B2 Manhandla $3C (6)
;   $B8 Aquamentus $3D, Ganon $3E (14)      $C6 Moldorm $41, Gleeok $42-$46 (2)
;   $C8 Patra $47, $48 (1)                  $C9 Trap $49, $4A (1)
;   $CA dungeon persons $4B-$52 (1)         $CB dead dummy $5D, push block $68 (1)
Anim_ObjFrameTiles:
    .BYTE $00, $04, $08, $0C, $10, $10, $14, $18    ; $00-$07
    .BYTE $5C, $9E, $44, $CE, $D2, $D6, $DA, $CE    ; $08-$0F
    .BYTE $D2, $D6, $DA, $F0, $F4, $F8, $FC, $F0    ; $10-$17
    .BYTE $F4, $F8, $FC, $B4, $B0, $B0, $B8, $B2    ; $18-$1F
    .BYTE $B2, $B4, $B0, $B0, $B8, $B2, $B2, $CA    ; $20-$27
    .BYTE $CC, $CA, $CC, $BC, $BE, $C0, $C0, $C2    ; $28-$2F
    .BYTE $C4, $C0, $C0, $BC, $BE, $BC, $BE, $C0    ; $30-$37
    .BYTE $C0, $C2, $C4, $C0, $C0, $BC, $BE, $BC    ; $38-$3F
    .BYTE $BE, $EC, $EE, $EC, $EE, $EC, $EE, $BC    ; $40-$47
    .BYTE $BE, $C6, $C8, $A0, $A8, $A4, $AC, $90    ; $48-$4F
    .BYTE $E8, $E4, $E0, $94, $F3, $C9, $BD, $C1    ; $50-$57
    .BYTE $98, $9A, $9C, $F8, $B8, $BC, $B0, $B4    ; $58-$5F
    .BYTE $B8, $BC, $B0, $B4, $B8, $AC, $B4, $BC    ; $60-$67
    .BYTE $B0, $B4, $B8, $AC, $B4, $BC, $B0, $B4    ; $68-$6F
    .BYTE $AC, $AE, $B0, $B2, $A8, $AA, $92, $94    ; $70-$77
    .BYTE $A0, $A2, $A6, $A4, $A2, $A4, $D8, $DA    ; $78-$7F
    .BYTE $00, $00, $9A, $9C, $9A, $9C, $9A, $9C    ; $80-$87
    .BYTE $B4, $B8, $BC, $BE, $B4, $B8, $BC, $BE    ; $88-$8F
    .BYTE $FC, $FE, $AC, $9C, $A0, $A4, $A0, $A4    ; $90-$97
    .BYTE $A8, $8E, $A4, $DC, $E0, $E4, $E8, $EC    ; $98-$9F
    .BYTE $F0, $F4, $F8, $FA, $FE, $F4, $F6, $FE    ; $A0-$A7
    .BYTE $FC, $F0, $F8, $B0, $F6, $F0, $D4, $FC    ; $A8-$AF
    .BYTE $FE, $F8, $E8, $EA, $E0, $E4, $EC, $EC    ; $B0-$B7
    .BYTE $D0, $D4, $D8, $DC, $E0, $E4, $C0, $C8    ; $B8-$BF
    .BYTE $C4, $CC, $E8, $EA, $72, $74, $DE, $EE    ; $C0-$C7
    .BYTE $F8, $96, $98, $B1                        ; $C8-$CB

; TABLE: sprite attributes of every animation frame; index as Anim_ObjFrameTiles; value = sprite
; attribute byte: bits 1-0 = sprite palette row (0 Link's, 1 blue, 2 red, 3 the row that differs per
; level), bit 7 = flip vertically. Only Octorok frames 1 and 4 have bit 7 ($82 at $1C, $1F; $81 at $22,
; $25). Colour variants share tiles and differ here: row 1 for Lynel $01, Goriya $05, Octorok $09/$0A,
; Darknut $0C, Tektite $0D, Keese $1B, Wizzrobe $23; row 2 for Lynel $02, Moblin $04, Goriya $06,
; Octorok $07/$08, Darknut $0B, Tektite $0E, Leever $10, Keese $1C, Wizzrobe $24; row 3 for Moblin $03,
; Zora $11, Keese $1D. Leever $0F mixes rows 2 and 1 over its 10 frames.
Anim_ObjFrameAttrs:
    .BYTE $00, $00, $00, $00, $00, $00, $00, $00    ; $00-$07
    .BYTE $02, $00, $00, $01, $01, $01, $01, $02    ; $08-$0F
    .BYTE $02, $02, $02, $03, $03, $03, $03, $02    ; $10-$17
    .BYTE $02, $02, $02, $02, $82, $02, $02, $82    ; $18-$1F
    .BYTE $02, $01, $81, $01, $01, $81, $01, $01    ; $20-$27
    .BYTE $01, $02, $02, $02, $02, $01, $01, $01    ; $28-$2F
    .BYTE $01, $01, $01, $02, $02, $02, $02, $02    ; $30-$37
    .BYTE $02, $02, $02, $02, $02, $02, $02, $03    ; $38-$3F
    .BYTE $03, $03, $03, $03, $03, $03, $03, $03    ; $40-$47
    .BYTE $03, $02, $02, $02, $02, $02, $02, $02    ; $48-$4F
    .BYTE $02, $01, $01, $01, $02, $03, $03, $03    ; $50-$57
    .BYTE $02, $02, $00, $02, $01, $01, $01, $01    ; $58-$5F
    .BYTE $02, $02, $02, $02, $02, $02, $02, $02    ; $60-$67
    .BYTE $02, $02, $01, $01, $01, $01, $01, $01    ; $68-$6F
    .BYTE $01, $01, $01, $01, $03, $03, $03, $03    ; $70-$77
    .BYTE $00, $00, $02, $02, $02, $02, $03, $03    ; $78-$7F
    .BYTE $03, $03, $01, $01, $02, $02, $03, $03    ; $80-$87
    .BYTE $01, $01, $01, $01, $02, $02, $02, $02    ; $88-$8F
    .BYTE $01, $01, $01, $01, $02, $02, $02, $02    ; $90-$97
    .BYTE $02, $02, $01, $03, $03, $03, $03, $03    ; $98-$9F
    .BYTE $03, $03, $03, $03, $03, $01, $01, $01    ; $A0-$A7
    .BYTE $01, $01, $01, $02, $00, $00, $03, $01    ; $A8-$AF
    .BYTE $01, $01, $01, $01, $01, $01, $01, $01    ; $B0-$B7
    .BYTE $03, $03, $03, $03, $03, $03, $03, $03    ; $B8-$BF
    .BYTE $03, $03, $03, $03, $03, $03, $02, $02    ; $C0-$C7
    .BYTE $01, $01, $02, $03                        ; $C8-$CB

; TABLE: sprite record offsets handed to objects in turn; index = RollingSpriteIndex 0-$27 (entry $28
; repeats entry 0 so that index + 1 is always valid); value = offset into Sprites.
; The 40 entries are records $18-$3F (offsets $60-$FC), each once: even entries count up from $60 to
; $AC; odd entries cover $B0-$FC, each group of four counting down. A two-sprite object uses entries
; i (left) and i + 1 (right).
Sprite_RollingOffsets:
    .BYTE $60, $BC, $64, $B8, $68, $B4, $6C, $B0
    .BYTE $70, $CC, $74, $C8, $78, $C4, $7C, $C0
    .BYTE $80, $DC, $84, $D8, $88, $D4, $8C, $D0
    .BYTE $90, $EC, $94, $E8, $98, $E4, $9C, $E0
    .BYTE $A0, $FC, $A4, $F8, $A8, $F4, $AC, $F0
    .BYTE $60

; ----------------------------------------------------------------------------
; Steps a walking object's animation and draws it by its facing direction (not mirrored).
; IN:   X = object slot.
; OUT:  X = object slot. Uses [00]-[05], [07], [08], [0A], [0C]-[0F].
; NOTE: Anim_AnimateObjWalking returns Y = frame (0 or 1 sideways, 2 down, 3 up), [00], [01] = the
;       object's position and [0F] = horizontal flip. The play-mode object loop calls this for every
;       object with metastate 0 whose ObjAttr has neither bit $01 nor bit $04: in the type attribute
;       table only types $01-$06 (Lynel, Moblin, Goriya; Obj_TypeAttributes value $00).
; ----------------------------------------------------------------------------
Anim_AnimateAndDrawObjectWalking:
    JSR Anim_AnimateObjWalking
    TYA
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Draws object X with frame A as a mirrored pair: the right sprite is the left tile flipped.
; IN:   A = frame; X = object slot; [00] = sprite X; [01] = sprite Y.
; NOTE: Sets [0C] = 1 and joins Anim_DrawObjectNotMirrored at its unnamed label.
; ----------------------------------------------------------------------------
Anim_DrawObjectMirrored:
    LDY #$01                    ; [0C] := 1: mirrored.
    BNE :+                      ; Always taken.

; ----------------------------------------------------------------------------
; Draws object X with frame A as two different halves, flipped horizontally when [0F] is nonzero.
; IN:   A = frame; X = object slot; [00] = sprite X; [01] = sprite Y; [0F] = flip horizontally.
; NOTE: Sets [0C] = 0, loads Y = ObjType,X and falls into Anim_DrawObjectWithType.
; ----------------------------------------------------------------------------
Anim_DrawObjectNotMirrored:
    LDY #$00                    ; [0C] := 0: not mirrored.
:
    STY $0C
    LDY ObjType, X

; ----------------------------------------------------------------------------
; Draws an object with the animation of object type Y (animation index Y + 1).
; IN:   A = frame; X = object slot; Y = object type; [00], [01] = sprite X, Y; [0C] = mirrored;
;       [0F] = flip horizontally.
; NOTE: Link's candle fire (slots $10, $11) enters here with Y = $40, the standing fire's type.
; ----------------------------------------------------------------------------
Anim_DrawObjectWithType:
    INY

; ----------------------------------------------------------------------------
; Draws an object with animation index Y, using Link's own sprite records or the next rolling pair.
; IN:   A = frame; X = object slot (0 = Link); Y = animation index (0 = Link, else object type + 1);
;       [00], [01] = sprite X, Y; [0C] = mirrored (ignored for Link); [0F] = flip horizontally.
; OUT:  X = object slot; [0D] = frame; [0E] = animation index; [08] = object slot;
;       LeftSpriteOffset, RightSpriteOffset.
; NOTE: Link (X = 0) gets records $12 and $13 (offsets $48, $4C). Any other slot gets
;       Sprite_RollingOffsets[RollingSpriteIndex] and the entry after it.
; ----------------------------------------------------------------------------
Anim_DrawObjectWithAnim:
    STA $0D                     ; [0D] := frame
    STY $0E                     ; [0E] := animation index
    STX $08                     ; [08] := object slot; nonzero also means "advance RollingSpriteIndex"

    ; Left and right sprite record offsets for the current rolling index.
    LDY RollingSpriteIndex
    LDA Sprite_RollingOffsets, Y
    STA LeftSpriteOffset
    LDA Sprite_RollingOffsets+1, Y

    ; Link (slot 0) instead always uses records $12 and $13.
    CPX #$00
    BNE Anim_DrawObjectWithAnimAndSpecificSprites
    LDA #$48                    ; Offset of record $12: Link's left half.
    STA LeftSpriteOffset
    LDA #$4C                    ; Offset of record $13: Link's right half.

; ----------------------------------------------------------------------------
; Looks up the frame's tile and attributes and writes the object's one or two sprites.
; IN:   A = right sprite record offset; LeftSpriteOffset; X = object slot = [08]; [0D] = frame;
;       [0E] = animation index; [00], [01] = sprite X, Y; [0C] = mirrored; [0F] = flip horizontally;
;       [04], [05] = attributes, used only when ObjAttr,X has bit $02 or $08.
; OUT:  X = [08]; sprites written; RollingSpriteIndex advanced once per sprite unless [08] = 0.
; NOTE: Slots $01-$0C: ObjAttr bit $02 = one sprite only (half width) with the caller's [04];
;       bit $08 = two sprites with the caller's [04], [05]. Link and slots $0D and up always take
;       their attributes from Anim_ObjFrameAttrs. Link is never mirrored.
;       Callers from other banks pass records $10 and $11 (offsets $40, $44), which precede Link's
;       records and so are drawn over him (Sprite_DrawObjMirroredOverLink).
; ----------------------------------------------------------------------------
Anim_DrawObjectWithAnimAndSpecificSprites:
    STA RightSpriteOffset
    LDY $0E
    LDA #$01                    ; [07] := 1: two sprites (0 would mean one).
    STA $07
    LDA #$08                    ; [0A] := 8: the right sprite is 8 pixels right of the left one.
    STA $0A
    LDA Anim_ObjFrameOffsets, Y
    CLC
    ADC $0D                     ; Y := first frame offset + frame.
    TAY
    LDA Anim_ObjFrameTiles, Y
    STA $02                     ; [02] := left tile
    CLC
    ADC #$02
    STA $03                     ; [03] := right tile = left tile + 2 (the next 8x16 pair)

    ; Link (slot 0) and slots $0D and up (weapons, room item) have no ObjAttr entry: use the table.
    CPX #$00
    BEQ @UseTableAttr
    CPX #$0D
    BCS @UseTableAttr

    ; ObjAttr bit $02: half width. Go write a single sprite with the attributes already in [04].
    LDA ObjAttr, X
    AND #$02
    BNE @DrawHalfWidth

    ; ObjAttr bit $08: keep the attributes the caller put in [04], [05].
    LDA ObjAttr, X
    AND #$08
    BNE :+

@UseTableAttr:
    LDA Anim_ObjFrameAttrs, Y      ; Attributes of this frame, for both sprites.
    JSR Anim_SetSpriteDescriptorAttributes
:
    ; Link is never mirrored: his halves are only swapped and flipped by [0F].
    CPX #$00
    BEQ Anim_WriteHorizontallyFlippableSpritePair

    ; Other objects: [0C] nonzero = mirrored pair; else flippable by [0F].
    LDY $0C
    BEQ Anim_WriteHorizontallyFlippableSpritePair
    JMP Anim_WriteMirroredSpritePair

@DrawHalfWidth:
    DEC $07                     ; [07] := 0: one sprite only.
    JMP Anim_WriteSpritePair

; ----------------------------------------------------------------------------
; Writes the sprite pair, first swapping the halves and flipping both when [0F] is nonzero.
; IN:   as Anim_WriteSpritePair, plus [0F] = flip horizontally (0 = no).
; OUT:  as Anim_WriteSpritePair; [02], [03] swapped and bit $40 of [04], [05] toggled when flipped.
; ----------------------------------------------------------------------------
Anim_WriteHorizontallyFlippableSpritePair:
    LDA $0F
    BEQ Anim_WriteSpritePair    ; No flip: write the pair as it is.

    ; Swap the left and right tiles.
    LDA $02
    PHA
    LDA $03
    STA $02
    PLA
    STA $03

    ; Toggle the horizontal flip bit ($40) of both attributes.
    LDA $04
    EOR #$40
    STA $04
    LDA $05
    EOR #$40
    STA $05

; ----------------------------------------------------------------------------
; Writes one or two sprites from the descriptor in [00]-[0A]; a slot with invincibility time flashes.
; IN:   X = object slot; [00] = X; [01] = Y; [02], [03] = left, right tile; [04], [05] = left, right
;       attributes; [07] = 1 for two sprites, 0 for one; [08] = nonzero to advance RollingSpriteIndex
;       after each sprite; [0A] = pixels from the left sprite to the right one; LeftSpriteOffset,
;       RightSpriteOffset = offsets into Sprites.
; OUT:  X = [08]; [00] = X + [0A] for each sprite written; [07] = $FF.
; NOTE: Flash: while ObjInvincibilityTimer,X is nonzero the palette bits (1-0) of both attributes are
;       replaced by (timer AND 3). The timer drops every second frame (Obj_DecInvincibilityTimer), so
;       the sprite palette row changes every 2 frames. X may be any slot $00-$13.
; ----------------------------------------------------------------------------
Anim_WriteSpritePair:
    ; Not invincible: keep the attributes.
    LDY ObjInvincibilityTimer, X
    BEQ Anim_WriteSpritePairNotFlashing

    ; For the right (Y = 1) and left (Y = 0) attributes in [05], [04]:
    LDY #$01
@LoopSide:
    LDA $0004, Y                ; Clear the palette bits.
    AND #$FC
    STA $0004, Y
    LDA ObjInvincibilityTimer, X    ; Palette row := low 2 bits of the timer.
    AND #$03
    ORA $0004, Y
    STA $0004, Y
    DEY
    BPL @LoopSide

; Entry that skips the flash: writes the sprites with [04], [05] as given. X need not be set.
Anim_WriteSpritePairNotFlashing:
    LDX LeftSpriteOffset        ; X = offset of the sprite record being written; left one first.
    LDY #$00                    ; Y = 0 left, 1 right: index into [02]/[03] and [04]/[05].

@LoopSprite:
    LDA $0002, Y                ; Tile
    STA Sprites+1, X
    LDA $01                     ; Y coordinate
    STA Sprites, X
    LDA $00                     ; X coordinate
    STA Sprites+3, X
    CLC                         ; [00] += [0A]: X coordinate of the right sprite.
    ADC $0A
    STA $00
    LDA $0004, Y                ; Attributes
    STA Sprites+2, X
    LDX RightSpriteOffset       ; The next pass writes the right sprite's record.
    LDA $08                     ; [08] nonzero: advance RollingSpriteIndex (not for Link, slot 0).
    BEQ :+
    JSR Sprite_CycleCurIndex
:
    INY
    DEC $07
    BPL @LoopSprite             ; [07] was 1: go write the right sprite. Was 0: done.
    LDX $08                     ; X := [08], the object slot.
    RTS
