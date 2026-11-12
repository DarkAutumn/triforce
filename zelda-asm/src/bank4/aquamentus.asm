; ============================================================================
; FILE:     bank4/aquamentus.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 5596-5853
; SUBJECTS: Aquamentus, Fireball, Obj, Sprite
; TOPIC:    Aquamentus (type $3D): pacing left and right, three-fireball volleys and their spread,
;           drawing; the boss cry and shove reset shared with Moldorm; a one-sprite writer.
; SEE ALSO: bank4/aquamentus-digdogger-dodongo-inits.asm (Aquamentus_Init: start position, invincibility
;           mask); bank4/monster-shots-fireballs.asm (Fireball_Shoot, Fireball_Update: aiming, speed);
;           bank4/walker-wanderer-common.asm (Sample_PlayBossHitCryIfHit, Sample_PlayBossDeathCryIfDying);
;           bank1/link-collision-harm.asm (shield rule for fireballs, damage);
;           bank7/object-update-table.asm (Obj_TypeAttributes, Obj_TypeHpPairs).
; KEY FACTS:
;   - Aquamentus has HP $60 (Obj_TypeHpPairs, type $3D) and invincibility mask $E2
;     (Aquamentus_Init): the boomerang (bit $02) and fire (bit $20) do nothing; sword, arrow, bomb and
;     magic shot are not masked.
;   - It starts at X $B0, Y $80 (Aquamentus_Init) and only moves horizontally, 1 pixel every 8 frames.
;     It is turned back when found at X < $88 or X >= $C8, so X ranges over $87-$C8
;     (Aquamentus_Move). Y never changes.
;   - Each leg of the pacing is 7 or $F pixels in a random direction: Random,X bit 3 picks the length,
;     bit 0 the direction (Aquamentus_Move).
;   - A volley is three fireballs of type $55 created at (X+4, Y); the next volley comes
;     Random,X OR $70 frames later: $70-$7F or $F0-$FF frames (Aquamentus_Shoot).
;   - Each fireball aims at Link by itself on its first update and waits $10 frames before flying
;     (Fireball_Update); in addition Aquamentus_Shoot moves the lower one down and the upper one up
;     1 pixel every second frame, from creation on, which fans the volley out.
;   - The mouth is drawn open while ObjTimer < $20, the last $20 frames before a volley
;     (Aquamentus_Draw).
;   - With InvClock set, Aquamentus neither moves nor shoots, and fireballs in flight stop spreading
;     (Aquamentus_Update).
;   - Touching Aquamentus costs $0100 = 1 heart, a type $55 fireball $0080 = half a heart, before
;     rings (Damage_ObjTypeDamageToLink entries $3D = $01, $55 = $80).
;   - The collision midpoint is (X+8, Y+8) like any 16x16 object (Obj_GetMiddle; ObjAttr $89 has no
;     half-width bit), although the drawn body is 24x32 pixels (Aquamentus_Draw).
;   - It is never knocked back: Obj_PlayBossDeathCryAndResetShove clears the shove bytes every frame.
; CONTENTS: (generated; L = line in this file)
;   L0054 Aquamentus_Update                  code  Updates Aquamentus for one frame: move, shoot, draw, check collisions, cry when hit or killed.
;   L0070 Obj_PlayBossDeathCryAndResetShove  code  Plays the boss death cry if the monster is dying (ObjMetastate <> 0) and cancels any knockback.
;   L0075 Aquamentus_StepsX                  data  TABLE: Aquamentus's X step per move; index = ObjDir - 1 (0 right, 1 left); value = signed pixels.
;   L0086 Aquamentus_Move                    code  Paces Aquamentus left or right: 1 pixel every 8 frames, in legs of 7 or $F pixels.
;   L0166 Aquamentus_Shoot                   code  Fires a volley of three fireballs when ObjTimer runs out; otherwise spreads the fireballs in flight.
;   L0232 Fireball_ShootType55               code  Creates a fireball of type $55 at the shooter's position (X+4, Y) if a monster slot is free.
;   L0240 Aquamentus_Tiles                   data  TABLE: Aquamentus sprite tiles; index = animation frame * 6 + sprite 0-5; value = tile id.
;   L0245 Aquamentus_SpriteOffsetsY          data  TABLE: Y offsets of Aquamentus's six sprites; index = sprite 0-5; value = pixels.
;   L0249 Aquamentus_SpriteOffsetsX          data  TABLE: X offsets of Aquamentus's six sprites; index = sprite 0-5; value = pixels.
;   L0258 Aquamentus_Draw                    code  Draws Aquamentus as six sprites (3 columns of 8 pixels, 2 rows of 16), animated every $10 frames.
;   L0328 Sprite_WriteBossSprite             code  Writes one sprite of a boss at the next rolling sprite record and advances the record index.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates Aquamentus for one frame: move, shoot, draw, check collisions, cry when hit or killed.
; IN:   X = object slot
; NOTE: With InvClock set the move and shoot steps are skipped. Falls into Obj_PlayBossDeathCryAndResetShove.
; ----------------------------------------------------------------------------
Aquamentus_Update:
    LDA InvClock
    BNE :+
    JSR Aquamentus_Move
    JSR Aquamentus_Shoot
:
    JSR Aquamentus_Draw
    JSR Collision_CheckMonsterVsWeaponsAndLink
    JSR Sample_PlayBossHitCryIfHit

; ----------------------------------------------------------------------------
; Plays the boss death cry if the monster is dying (ObjMetastate <> 0) and cancels any knockback.
; IN:   X = object slot
; OUT:  A = 0; ObjShoveDir,X and ObjShoveDistance,X = 0
; NOTE: Also called by Moldorm.
; ----------------------------------------------------------------------------
Obj_PlayBossDeathCryAndResetShove:
    JSR Sample_PlayBossDeathCryIfDying
    JMP Shove_Reset

; TABLE: Aquamentus's X step per move; index = ObjDir - 1 (0 right, 1 left); value = signed pixels.
Aquamentus_StepsX:
    .BYTE $01, $FF

; ----------------------------------------------------------------------------
; Paces Aquamentus left or right: 1 pixel every 8 frames, in legs of 7 or $F pixels.
; IN:   X = object slot; ObjGridOffset,X = pixels left in the current leg
; NOTE: Uses Random,X once per leg: (value AND $0F) OR 7 = length, bit 0 = direction. On a move
;       frame, X < $88 gives X = $88, facing right, and X >= $C8 gives X = $C7, facing left, each
;       with a new 7-pixel leg whose first step is taken at once. X = $87 and X = $C8 are reached
;       before the correction, so the range is $87-$C8.
; ----------------------------------------------------------------------------
Aquamentus_Move:
    ; Aquamentus uses ObjGridOffset as the pixels left to move in this leg.
    ; If any are left, go move.
    LDA ObjGridOffset, X
    BNE @Move

    ; Pick a new leg: (Random,X AND $0F) OR 7 = 7 or $F pixels.
    LDA Random, X
    PHA
    AND #$0F
    ORA #$07
    STA ObjGridOffset, X
    PLA

    ; Bit 0 of the same random value picks the direction: 1 (right) or 2 (left).
    ; Nothing moves on this frame.
    AND #$01
    TAY
    INY
    STY ObjDir, X
    RTS

@Move:
    ; Move only on frames where the low 3 bits of FrameCounter are 0: 1 frame in 8.
    LDA FrameCounter
    AND #$07
    BNE @Exit

    ; If X < $88 (left limit), then
    ; put it at $88, face right, and start a 7-pixel leg.
    LDA ObjX, X
    CMP #$88
    BCS @CheckRightLimit
    LDA #$88
    STA ObjX, X
    LDA #$01
    STA ObjDir, X
    BNE @SetRemainingDistance

@CheckRightLimit:
    ; If X >= $C8 (past the right limit), then
    ; put it at $C7, face left, and start a 7-pixel leg.
    CMP #$C8
    BCC @ApplySpeed
    LDA #$C7
    STA ObjX, X
    LDA #$02
    STA ObjDir, X

@SetRemainingDistance:
    ; Set the pixels left in the leg to 7.
    LDA #$07
    STA ObjGridOffset, X

@ApplySpeed:
    ; Index the X steps by (direction - 1): 0 for right, 1 for left.
    LDY ObjDir, X
    DEY

    ; Add the step (+1 or -1) to the X coordinate.
    LDA ObjX, X
    CLC
    ADC Aquamentus_StepsX, Y
    STA ObjX, X

    ; One pixel less to go.
    DEC ObjGridOffset, X

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Fires a volley of three fireballs when ObjTimer runs out; otherwise spreads the fireballs in flight.
; IN:   X = object slot
; OUT:  X = CurObjIndex after the spread loop
; NOTE: Volley interval = Random,X OR $70 frames ($70-$7F or $F0-$FF). A fireball needs a free
;       monster slot; when none is free Fireball_Shoot returns Y = 0 and the offset byte is written
;       for slot 0 instead. The spread loop moves every type $55 object in the room, each by its own
;       Aquamentus_ObjFireballOffset, on frames where FrameCounter is even.
; ----------------------------------------------------------------------------
Aquamentus_Shoot:
    ; If the object timer <> 0, then
    ; go spread out the fireballs that are flying.
    LDA ObjTimer, X
    BNE @SpreadOutFireballs

    ; The object timer has run out: shoot.
    ;
    ; Set the timer to Random,X OR $70: $70-$7F or $F0-$FF frames until the next volley.
    LDA Random, X
    ORA #$70
    STA ObjTimer, X

    ; Shoot the middle fireball. Its vertical step is 0. (Y = the new fireball's slot.)
    JSR Fireball_ShootType55
    LDA #$00
    STA Aquamentus_ObjFireballOffset, Y

    ; Shoot the lower fireball. Its vertical step is 1 (down).
    JSR Fireball_ShootType55
    LDA #$01
    STA Aquamentus_ObjFireballOffset, Y

    ; Shoot the upper fireball. Its vertical step is $FF (-1, up).
    JSR Fireball_ShootType55
    LDA #$FF
    STA Aquamentus_ObjFireballOffset, Y
    RTS

@SpreadOutFireballs:
    ; Make the fireballs spread out.
    ;
    ; For each object slot from $B down to 0:
    LDX #$0B

@LoopObject:
    ; Skip objects that are not fireballs of type $55.
    LDA ObjType, X
    CMP #$55
    BNE @NextLoopObject

    ; Skip on frames where FrameCounter is odd: the step is applied every second frame.
    LDA FrameCounter
    LSR
    BCS @NextLoopObject

    ; Add the displacement for this fireball to its Y coordinate.
    LDA ObjY, X
    CLC
    ADC Aquamentus_ObjFireballOffset, X
    STA ObjY, X

@NextLoopObject:
    ; Next slot, down to and including slot 0.
    DEX
    BPL @LoopObject

    ; Restore Aquamentus's slot number.
    LDX CurObjIndex
    RTS

; ----------------------------------------------------------------------------
; Creates a fireball of type $55 at the shooter's position (X+4, Y) if a monster slot is free.
; IN:   X = shooter's object slot
; OUT:  Y = the fireball's slot, or 0 when no slot was free
; ----------------------------------------------------------------------------
Fireball_ShootType55:
    LDA #$55
    JMP Fireball_Shoot

; TABLE: Aquamentus sprite tiles; index = animation frame * 6 + sprite 0-5; value = tile id.
; Sprites 0-2 are the top row from left to right, 3-5 the bottom row. The first tile of each
; frame ($CC) is the face with the mouth shut; Aquamentus_Draw replaces it with $C0 (mouth open).
; Bytes 0-5: frame 0. Bytes 6-11: frame 1 (it differs only in the bottom row).
Aquamentus_Tiles:
    .BYTE $CC, $C4, $C8, $C2, $C6, $CA, $CC, $C4
    .BYTE $C8, $CE, $D0, $D2

; TABLE: Y offsets of Aquamentus's six sprites; index = sprite 0-5; value = pixels.
Aquamentus_SpriteOffsetsY:
    .BYTE $00, $00, $00, $10, $10, $10

; TABLE: X offsets of Aquamentus's six sprites; index = sprite 0-5; value = pixels.
Aquamentus_SpriteOffsetsX:
    .BYTE $00, $08, $10, $00, $08, $10

; ----------------------------------------------------------------------------
; Draws Aquamentus as six sprites (3 columns of 8 pixels, 2 rows of 16), animated every $10 frames.
; IN:   X = object slot
; NOTE: The face tile is $C0 (mouth open) while ObjTimer < $20, else $CC. While the invincibility
;       timer runs, the palette bits cycle, which makes the body flash.
; ----------------------------------------------------------------------------
Aquamentus_Draw:
    ; The animation frame switches every $10 frames: frame 0 while bit 4 of FrameCounter is set,
    ; frame 1 while it is clear. Each frame has 6 tiles.
    ; The last tile of frame 0 is at index 5, the last tile of frame 1 at index $B.
    LDY #$05
    LDA FrameCounter
    AND #$10
    BNE :+
    LDY #$0B
:
    STY $0A                     ; [0A] holds the current tile index in frame tile list.

    ; For each sprite, from 5 to 0, indexed by [0B]:
    LDA #$05
    STA $0B

@LoopSprite:
    LDY $0B                     ; Get current loop index from [0B].

    ; Add the current sprite's offset to the boss's X coordinate.
    ; Store it in [00].
    LDA ObjX, X
    CLC
    ADC Aquamentus_SpriteOffsetsX, Y
    STA $00

    ; Add the current sprite's offset to the boss's Y coordinate.
    ; Store it in [01].
    LDA ObjY, X
    CLC
    ADC Aquamentus_SpriteOffsetsY, Y
    STA $01

    ; Sprite attributes: [03] := (invincibility timer AND 3) XOR 3.
    ;
    ; With the timer at 0 this is 3: sprite palette 3 (the level's palette row), no flipping.
    ; While the timer runs, the palette bits change with it, so the body flashes.
    LDA ObjInvincibilityTimer, X
    AND #$03
    EOR #$03
    STA $03
    LDY $0A

    ; Look up the tile for this sprite.
    ; If it is not the face tile (the first table entry, $CC), go write the sprite.
    LDA Aquamentus_Tiles, Y
    CMP Aquamentus_Tiles
    BNE @WriteSprite

    ; It is the face tile.
    ; If ObjTimer >= $20, use it as it is ($CC, mouth shut). Otherwise use $C0 (mouth open).
    LDY ObjTimer, X
    CPY #$20
    BCS @WriteSprite
    LDA #$C0                    ; Mouth open tile

@WriteSprite:
    JSR Sprite_WriteBossSprite

    ; Bottom of the loop.
    ; Decrement tile index and loop index.
    DEC $0A
    DEC $0B
    BPL @LoopSprite
    RTS

; ----------------------------------------------------------------------------
; Writes one sprite of a boss at the next rolling sprite record and advances the record index.
; IN:   A = tile; [00] = X; [01] = Y; [03] = sprite attributes
; ----------------------------------------------------------------------------
Sprite_WriteBossSprite:
    PHA
    LDY RollingSpriteIndex
    LDA Sprite_RollingOffsets, Y
    TAY
    PLA                         ; Restore the tile
    STA Sprites+1, Y
    LDA $00
    STA Sprites+3, Y
    LDA $01
    JMP Anim_EndWriteSprite
