; ============================================================================
; FILE:     bank4/aquamentus.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 5596-5853
; SUBJECTS: (none)
; TOPIC:    Aquamentus; boss hit reaction, boss sprite writer
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 UpdateAquamentus          code
;   L0032 CheckBossHitReaction      code
;   L0036 AquamentusSpeeds          data
;   L0039 Aquamentus_Move           code
;   L0110 Aquamentus_Shoot          code
;   L0174 ShootFireball55           code  Params:
;   L0178 AquamentusTiles           data
;   L0182 AquamentusSpriteOffsetsY  data
;   L0185 AquamentusSpriteOffsetsX  data
;   L0188 Aquamentus_Draw           code
;   L0269 WriteBossSprite           code  Params:
; ============================================================================

UpdateAquamentus:
    LDA InvClock
    BNE :+
    JSR Aquamentus_Move
    JSR Aquamentus_Shoot
:
    JSR Aquamentus_Draw
    JSR CheckMonsterCollisions
    JSR PlayBossHitCryIfNeeded

CheckBossHitReaction:
    JSR PlayBossDeathCryIfNeeded
    JMP ResetShoveInfo

AquamentusSpeeds:
    .BYTE $01, $FF

Aquamentus_Move:
    ; The monster uses grid offset to mean distance remaining.
    ; If there's more distance to cover, then go move.
    LDA ObjGridOffset, X
    BNE @Move

    ; Set a random distance to move of 7 or $F pixels.
    LDA Random, X
    PHA
    AND #$0F
    ORA #$07
    STA ObjGridOffset, X
    PLA

    ; Randomly head left or right.
    AND #$01
    TAY
    INY
    STY ObjDir, X
    RTS

@Move:
    ; 7 of 8 screen frames return without moving.
    LDA FrameCounter
    AND #$07
    BNE @Exit

    ; If the boss crossed the left limit, then
    ; put it back at the left limit ($88), and face right.
    LDA ObjX, X
    CMP #$88
    BCS @CheckRightLimit
    LDA #$88
    STA ObjX, X
    LDA #$01
    STA ObjDir, X
    BNE @SetRemainingDistance

@CheckRightLimit:
    ; If the boss crossed the right limit, then
    ; put it back at the right limit ($C7), and face left.
    CMP #$C8
    BCC @ApplySpeed
    LDA #$C7
    STA ObjX, X
    LDA #$02
    STA ObjDir, X

@SetRemainingDistance:
    ; Set the distance remaining to 7 pixels.
    LDA #$07
    STA ObjGridOffset, X

@ApplySpeed:
    ; Use (direction - 1) to index the speeds.
    ; This yields 0 for right, 1 for left.
    LDY ObjDir, X
    DEY

    ; Add the speed to the X coordinate.
    LDA ObjX, X
    CLC
    ADC AquamentusSpeeds, Y
    STA ObjX, X

    ; Decrement the distance remaining.
    DEC ObjGridOffset, X

@Exit:
    RTS

Aquamentus_Shoot:
    ; If the object timer <> 0, then
    ; go modify fireballs that might still be flying.
    LDA ObjTimer, X
    BNE @SpreadOutFireballs

    ; The boss's object timer has expired. Time to shoot fireballs.
    ;
    ; Set the timer to a random time, at least $70 frames.
    LDA Random, X
    ORA #$70
    STA ObjTimer, X

    ; Shoot the middle fireball. Its vertical displacement is 0.
    JSR ShootFireball55
    LDA #$00
    STA Aquamentus_ObjFireballOffset, Y

    ; Shoot the lower fireball. Its vertical displacement is 1.
    JSR ShootFireball55
    LDA #$01
    STA Aquamentus_ObjFireballOffset, Y

    ; Shoot the upper fireball. Its vertical displacement is -1.
    JSR ShootFireball55
    LDA #$FF
    STA Aquamentus_ObjFireballOffset, Y
    RTS

@SpreadOutFireballs:
    ; Make the fireballs spread out.
    ;
    ; For each object slot from $B to 1:
    LDX #$0B

@LoopObject:
    ; If type <> $55, loop again.
    LDA ObjType, X
    CMP #$55
    BNE @NextLoopObject

    ; Every other screen frame, loop again.
    LDA FrameCounter
    LSR
    BCS @NextLoopObject

    ; Add the displacement for this fireball to its Y coordinate.
    LDA ObjY, X
    CLC
    ADC Aquamentus_ObjFireballOffset, X
    STA ObjY, X

@NextLoopObject:
    ; Bottom of the loop.
    DEX
    BPL @LoopObject

    ; Restore the boss's slot number.
    LDX CurObjIndex
    RTS

; Params:
; X: object index
;
ShootFireball55:
    LDA #$55
    JMP ShootFireball

AquamentusTiles:
    .BYTE $CC, $C4, $C8, $C2, $C6, $CA, $CC, $C4
    .BYTE $C8, $CE, $D0, $D2

AquamentusSpriteOffsetsY:
    .BYTE $00, $00, $00, $10, $10, $10

AquamentusSpriteOffsetsX:
    .BYTE $00, $08, $10, $00, $08, $10

Aquamentus_Draw:
    ; Every $10 screen frames, switch animation frames.
    ; Each frame has 6 tiles (representing the top of 6 sprites).
    ; The last tile of frame 0 is at index 5.
    ; The last tile of frame 1 is at index $B.
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
    ADC AquamentusSpriteOffsetsX, Y
    STA $00

    ; Add the current sprite's offset to the boss's Y coordinate.
    ; Store it in [01].
    LDA ObjY, X
    CLC
    ADC AquamentusSpriteOffsetsY, Y
    STA $01

    ; Calculate sprite attributes:
    ; [03] := (invincibility timer AND 3) XOR 3
    ;
    ; If the boss is temporarily invincible, then
    ; this makes it flash by cycling all the palette rows.
    ;
    ; Otherwise invincibility timer = 0. So [03] becomes 3:
    ; the attribute value for a normal sprite with palette row 7 (level).
    LDA ObjInvincibilityTimer, X
    AND #$03
    EOR #$03
    STA $03
    LDY $0A

    ; Look up the tile for this sprite.
    ; If it's not the first one (face), then go write the sprite record.
    LDA AquamentusTiles, Y
    CMP AquamentusTiles
    BNE @WriteSprite

    ; Else it's the first one (the face tile).
    ; For timer >= $20, use it as is ($CC). Otherwise, use tile $C0.
    ;
    ; $CC/CD shows the closed mouth face.
    ; $C0/C1 shows the open mouth face.
    LDY ObjTimer, X
    CPY #$20
    BCS @WriteSprite
    LDA #$C0                    ; Open mouth tile

@WriteSprite:
    JSR WriteBossSprite

    ; Bottom of the loop.
    ; Decrement tile index and loop index.
    DEC $0A
    DEC $0B
    BPL @LoopSprite
    RTS

; Params:
; A: tile
; [00]: X
; [01]: Y
; [03]: sprite attributes
;
; Save the tile
WriteBossSprite:
    PHA
    LDY RollingSpriteIndex
    LDA SpriteOffsets, Y
    TAY
    PLA                         ; Restore the tile
    STA Sprites+1, Y
    LDA $00
    STA Sprites+3, Y
    LDA $01
    JMP Anim_EndWriteSprite
