; ============================================================================
; FILE:     bank4/push-block.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 626-788
; SUBJECTS: PushBlock, Sprite
; TOPIC:    Dungeon push block (object type $68, object slot $0B): waiting to be pushed, sliding one
;           square, done. Sprite helpers that draw a monster in front of Link (Like-Like, Wallmaster).
; SEE ALSO: bank5/uw-floor-push-block.asm (PushBlock_FindAndCreate: where the block is put);
;           bank5/underworld-secrets.asm (Secret_Trigger4BlockShutters, Secret_Trigger5BlockStairs:
;           what a finished push opens; Secret_UpdateRoomAllDead sets IsRoomAllDead);
;           bank7/tile-obj-tiles.asm (Tile_ChangeTileObjTiles); bank7/walker-move.asm (Obj_MoveByQSpeed);
;           bank1/object-animation-draw.asm (Anim_DrawObjectWithAnimAndSpecificSprites);
;           bank4/like-like.asm and bank4/wallmaster.asm (users of the draw-over-Link helpers).
; KEY FACTS:
;   - The block cannot be pushed until IsRoomAllDead is set (no monsters left; Bubbles and types
;     $49 and above do not count). PushBlock_State0Idle
;   - Link must be exactly in line with the block (same X, or Link's Y + 3 = block Y), at most $10
;     pixels from it on the other axis, and hold exactly the one direction that points at the block.
;     PushBlock_State0Idle, PushBlock_PushDirs
;   - The push must last $10 consecutive frames; any frame that fails a condition resets the count.
;     PushBlock_State0Idle, PushBlock_ResetPushTimer
;   - The block can go in any of the four directions, whichever way Link pushes. PushBlock_State0Idle
;   - It slides $10 pixels; then tile $B0 is written at the new place, the "secret" tune plays and
;     BlockPushComplete becomes 1, which the room's secret trigger reads. PushBlock_State1Moving
;   - State 2 does nothing: a block moves once per room entry. PushBlock_State2Done
;   - Sprite records $10 and $11 (offsets $40, $44) come before Link's ($12, $13) in the sprite
;     buffer; the helpers here draw a monster into them. Sprite_DrawObjMirroredOverLink
; CONTENTS: (generated; L = line in this file)
;   L0044 PushBlock_PushDirs                 data  TABLE: direction Link must hold to push the block; index = where Link stands relative to the
;   L0051 PushBlock_Update                   code  Push block (type $68) update: dispatches on the low 2 bits of its state.
;   L0057 PushBlock_Update_JumpTable         data  TABLE: push block state handlers; index = ObjState AND 3; value = routine address.
;   L0074 PushBlock_State0Idle               code  Push block state 0: counts the frames Link pushes it and starts it sliding after $10 frames.
;   L0142 PushBlock_Draw                     code  Draws the push block as sprites (frame image 0) at the object's X and Y - 1.
;   L0149 PushBlock_ResetPushTimer           code  Sets the push timer to 0: Link is not pushing on this frame.
;   L0154 PushBlock_State0Idle_Return        code  Shared exit of PushBlock_State0Idle.
;   L0165 PushBlock_State1Moving             code  Push block state 1: slides the block; after $10 pixels turns it back into tiles and flags it.
;   L0191 PushBlock_State2Done               code  Push block state 2: finished, nothing to do. Also the exit of PushBlock_State1Moving.
;   L0198 Sprite_HideOverLink                code  Hides sprite records $10 and $11, the pair used to draw a monster in front of Link.
;   L0208 Sprite_DrawObjNotMirroredOverLink  code  Draws an object into sprite records $10 and $11 (in front of Link), flippable by [0F].
;   L0218 Sprite_DrawObjMirroredOverLink     code  Draws an object into sprite records $10 and $11 (in front of Link), right half mirroring the left.
; ============================================================================

; TABLE: direction Link must hold to push the block; index = where Link stands relative to the
; block (0 below, 1 above, 2 right, 3 left); value = direction bits.
PushBlock_PushDirs:
    .BYTE $08, $04, $02, $01    ; up, down, left, right

; ----------------------------------------------------------------------------
; Push block (type $68) update: dispatches on the low 2 bits of its state.
; IN:   X = object slot ($0B)
; ----------------------------------------------------------------------------
PushBlock_Update:
    LDA ObjState, X
    AND #$03
    JSR Util_TableJump

; TABLE: push block state handlers; index = ObjState AND 3; value = routine address.
PushBlock_Update_JumpTable:
    .ADDR PushBlock_State0Idle      ; 0: waiting for Link to push
    .ADDR PushBlock_State1Moving    ; 1: sliding
    .ADDR PushBlock_State2Done      ; 2: finished

; ----------------------------------------------------------------------------
; Push block state 0: counts the frames Link pushes it and starts it sliding after $10 frames.
; IN:   X = object slot; Link's ObjX, ObjY, ObjInputDir; IsRoomAllDead
; OUT:  PushBlock_ObjPushTimer,X; on the $10th frame ObjDir,X = Link's input direction, state 1,
;       and the block's 2x2 tiles become floor tiles ($74)
; NOTE: All of these must hold on a frame for it to count, else the count restarts from 0:
;       the room has no monsters left (IsRoomAllDead); Link's X = block X (vertical push) or Link's
;       Y + 3 = block Y (horizontal push); Link is at most $10 pixels away on the other axis;
;       ObjInputDir of Link equals the single direction toward the block.
;       IsRoomAllDead is incremented on every frame the room is empty (Secret_UpdateRoomAllDead), so
;       it reads 0 on one frame in 256 and the count restarts then.
; ----------------------------------------------------------------------------
PushBlock_State0Idle:
    ; If there are monsters in the room, go reset the push timer and return.
    LDA IsRoomAllDead
    BEQ PushBlock_ResetPushTimer

    ; If Link's X equals the block's, then A = Link's (Y + 3) - block's Y, Y = 0 (index of
    ; "Link below"), and go test the difference.
    LDA ObjX
    CMP ObjX, X
    BNE @CheckAlignedY
    LDY #$00
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    JMP @TestDifference

@CheckAlignedY:
    ; If Link's (Y + 3) does not equal the block's Y, then
    ; go reset the push timer.
    LDA ObjY
    CLC
    ADC #$03
    CMP ObjY, X
    BNE PushBlock_ResetPushTimer

    ; A = Link's X - block's X, Y = 2 (index of "Link to the right").
    LDY #$02
    LDA ObjX
    SEC
    SBC ObjX, X

@TestDifference:
    ; If the difference is negative, then Link is above or left of the block:
    ; step Y to the index of that side (1 or 3), and negate the difference.
    BPL :+
    INY
    JSR Math_Negate
:
    ; If the distance >= $11 pixels, then go reset the push timer and return.
    CMP #$11
    BCS PushBlock_ResetPushTimer

    ; If Link's input direction is not the one that points at the block from his side,
    ; then go reset the push timer and return.
    LDA ObjInputDir
    CMP PushBlock_PushDirs, Y
    BNE PushBlock_ResetPushTimer

    ; Link is pushing the block. Increase the push timer.
    INC PushBlock_ObjPushTimer, X

    ; If the push timer still < $10, return.
    LDY PushBlock_ObjPushTimer, X
    CPY #$10
    BCC PushBlock_State0Idle_Return

    ; Link has pushed for $10 frames.
    ; Set the block's direction to Link's input direction (still in A), set state 1,
    ; and change the tiles where the block stands to floor tiles.
    STA ObjDir, X
    INC ObjState, X
    INC ReturnToBank4           ; Switch back to bank 4 when returning from the call below.
    LDA #$74                    ; First of the floor tiles
    JSR Tile_ChangeTileObjTiles

; Draws the push block as sprites (frame image 0) at the object's X and Y - 1.
PushBlock_Draw:
    JSR Anim_FetchObjPosForSpriteDescriptor
    DEC $01                     ; Sprite Y - 1 (a sprite shows one line below its Y value).
    LDA #$00                    ; Frame image 0
    JMP Anim_DrawObjectNotMirrored

; Sets the push timer to 0: Link is not pushing on this frame.
PushBlock_ResetPushTimer:
    LDA #$00
    STA PushBlock_ObjPushTimer, X

; Shared exit of PushBlock_State0Idle.
PushBlock_State0Idle_Return:
    RTS

; ----------------------------------------------------------------------------
; Push block state 1: slides the block; after $10 pixels turns it back into tiles and flags it.
; IN:   X = object slot; ObjDir,X = slide direction
; OUT:  at the end: block tiles ($B0) at the new place, state 2, BlockPushComplete + 1,
;       Tune1Request = $04
; NOTE: Moves at the slot's q-speed, which room entry sets to $20 (0.5 pixel a frame) for slots
;       $01-$0B; the block's creation and init do not change it: $20 frames for the $10 pixels.
; ----------------------------------------------------------------------------
PushBlock_State1Moving:
    ; Move in the block's direction, and draw.
    LDA ObjDir, X
    STA $0F
    JSR Obj_MoveByQSpeed
    JSR PushBlock_Draw

    ; If the block has not moved $10 pixels (grid offset $10 or -$10), then return.
    LDA ObjGridOffset, X
    CMP #$10
    BEQ :+
    CMP #$F0
    BNE PushBlock_State2Done
:
    ; The block has arrived.
    ; Play the "secret" tune, change the tiles at the destination to block tiles,
    ; set state 2, and increment BlockPushComplete.
    LDA #$04
    STA Tune1Request
    INC ReturnToBank4           ; Switch back to bank 4 when returning from the call below.
    LDA #$B0                    ; First of the block tiles
    JSR Tile_ChangeTileObjTiles
    INC ObjState, X
    INC BlockPushComplete

; Push block state 2: finished, nothing to do. Also the exit of PushBlock_State1Moving.
PushBlock_State2Done:
    RTS

; ----------------------------------------------------------------------------
; Hides sprite records $10 and $11, the pair used to draw a monster in front of Link.
; OUT:  Sprites+64 and Sprites+68 (the Y bytes) = $F8
; ----------------------------------------------------------------------------
Sprite_HideOverLink:
    LDA #$F8
    STA Sprites+64
    STA Sprites+68
    RTS

; ----------------------------------------------------------------------------
; Draws an object into sprite records $10 and $11 (in front of Link), flippable by [0F].
; IN:   A = frame image; X = object slot; [00] = X; [01] = Y; [0F] = flip horizontally
; ----------------------------------------------------------------------------
Sprite_DrawObjNotMirroredOverLink:
    LDY #$00                    ; Not mirrored
    BEQ :+                      ; Always taken.

; ----------------------------------------------------------------------------
; Draws an object into sprite records $10 and $11 (in front of Link), right half mirroring the left.
; IN:   A = frame image; X = object slot; [00] = X; [01] = Y
; NOTE: Records $10 and $11 come before Link's ($12, $13) in the sprite buffer, so the object
;       shows over him. Used by Like-Like and Wallmaster while they hold Link.
; ----------------------------------------------------------------------------
Sprite_DrawObjMirroredOverLink:
    LDY #$01
:
    STY $0C                     ; [0C] holds mirrored flag

    ; The animation index is (object type + 1), because Link takes two animation indexes.
    LDY ObjType, X
    INY
    STA $0D                     ; [0D] holds frame image
    STY $0E                     ; [0E] holds animation index
    STX $08                     ; [08] holds object slot
    LDA #$40                    ; Offset of sprite record $10: the left half
    STA LeftSpriteOffset
    LDA #$44                    ; Offset of sprite record $11: the right half
    JMP Anim_DrawObjectWithAnimAndSpecificSprites
