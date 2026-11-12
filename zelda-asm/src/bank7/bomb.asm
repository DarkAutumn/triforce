; ============================================================================
; FILE:     bank7/bomb.asm               BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4806-5030
; SUBJECTS: Anim, Bomb
; TOPIC:    Link's bomb in object slots $10 and $11: the fuse, blast and fade states and their timers, the
;           test that opens a bombable dungeon wall, and the drawing of the bomb and its dust clouds
;           (the cloud writer is also the monster spawn cloud).
; SEE ALSO: bank1/bomb-candle-wield.asm (Bomb_Wield: placing a bomb, the two-slot limit;
;           Bomb_UpdateFlashEffect); bank1/sword-bomb-arrow-collision.asm (Bomb_CheckBombOrFireVsMonster:
;           blast damage and radius); bank4/rock-wall-tree-secrets.asm (RockWall_Update: overworld rock
;           walls); bank5/update-doors.asm (Door_Update: runs TriggeredDoorCmd); bank5/fill-walls.asm
;           (Door_FindTypeByDirBit); bank7/fire.asm (Fire_Update: the other user of slots $10/$11);
;           bank7/object-animation.asm (spawn cloud drawn through Anim_DrawCloud).
; KEY FACTS:
;   - Slots $10 and $11 hold a bomb (ObjState $11-$14) or a candle fire ($2x); Bomb_UpdateBombOrFire picks the
;     routine by the high nibble of ObjState.
;   - Bomb states: $11 placed (ends on the first update), $12 fuse for $30 frames, $13 blast for $18
;     frames, $14 fading clouds for $0C frames, then the slot is freed (Bomb_Update, Bomb_StateTimes).
;     From the first update to the freed slot: $30 + $18 + $0C = $54 (84) frames.
;   - Only state $13 hurts monsters and overworld rock walls; they test ObjState = $13 themselves
;     (Bomb_CheckBombOrFireVsMonster, RockWall_Update). Nothing in this file damages anything.
;   - The explosion sound (effect request bit $10) is requested once, when state $13 begins (Bomb_Update).
;   - A dungeon wall is tested once per bomb, on the frame the state changes $13 -> $14 (the blast's
;     end), not during the blast (Bomb_TryOpenWall).
;   - The wall test needs: a dungeon level, not a cellar (GameMode 9), the bomb's ObjX and ObjY each less
;     than $18 pixels from a wall hotspot, that door not already open (CurOpenedDoors), no door
;     command running (TriggeredDoorCmd = 0), and door type 4 (bombable) on that side
;     (Bomb_TryOpenWall, Bomb_WallHotspotsX, Bomb_WallHotspotsY).
;   - Hotspots (bomb ObjX, ObjY): up wall ($78,$5D), down ($78,$BD), left ($20,$8D), right ($D0,$8D).
;   - A passing test sets TriggeredDoorCmd = 6 (open) and TriggeredDoorDir (Bomb_TryOpenWall).
;   - In states $13 and $14 four clouds are drawn per frame: one on the bomb and three of six outer
;     positions, alternating by frame parity (Bomb_Draw, Bomb_DrawOuterClouds). Drawing only.
; CONTENTS: (generated; L = line in this file)
;   L0055 Bomb_StateTimes              data  TABLE: bomb timer loaded when a state begins; index = low nibble of the state being left - 1;
;   L0060 Bomb_WallHotspotsX           data  TABLE: X of the four bombable-wall hotspots, compared with the bomb's ObjX; index 0 up wall,
;   L0064 Bomb_WallHotspotsY           data  TABLE: Y of the four bombable-wall hotspots, compared with the bomb's ObjY; same index; pixels.
;   L0071 Bomb_UpdateBombOrFire        code  Updates one bomb-or-fire slot: nothing if empty, Bomb_Update for state $1x, else Fire_Update.
;   L0089 Bomb_Update                  code  Steps a bomb through fuse, blast and fade when its timer runs out, then draws it.
;   L0122 Bomb_Update_Return           code  Shared exit of the bomb routines above and below.
;   L0136 Bomb_TryOpenWall             code  On the frame a bomb's blast ends (state becomes $14), opens a bombable dungeon wall near it.
;   L0203 Bomb_Draw                    code  Draws the bomb (state $12) or its dust clouds (states $13, $14) at the bomb's position.
;   L0222 Bomb_DrawBombOrCloudAt       code  Runs the blast's screen flash, then draws the bomb or one cloud at ([00], [01]).
;   L0230 Bomb_DrawBombOrCloudNoFlash  code  Draws the bomb or one cloud at ([00], [01]) with frame image = low nibble of the state - 2.
;   L0242 Anim_DrawCloud               code  Writes the sprites of one frame image of the bomb item slot (0 bomb, 1-3 dust cloud).
;   L0256 Bomb_CloudOffsetsY1          data  TABLE: outer cloud Y offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
;   L0261 Bomb_CloudOffsetsX1          data  TABLE: outer cloud X offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
;   L0266 Bomb_CloudOffsetsY2          data  TABLE: outer cloud Y offsets, odd frames (reached as Bomb_CloudOffsetsY1 + 6); signed pixels
;   L0271 Bomb_CloudOffsetsX2          data  TABLE: outer cloud X offsets, odd frames (reached as Bomb_CloudOffsetsX1 + 6); signed pixels
;   L0280 Bomb_DrawOuterClouds         code  Draws the three outer dust clouds of a blast, using a different set of positions on odd frames.
; ============================================================================

; TABLE: bomb timer loaded when a state begins; index = low nibble of the state being left - 1;
; value = frames. $30 fuse (state $12), $18 blast ($13), $0C fading clouds ($14); the last value is
; loaded for state $15, which frees the slot at once and clears the timer, so $06 is never counted.
Bomb_StateTimes:
    .BYTE $30, $18, $0C, $06

; TABLE: X of the four bombable-wall hotspots, compared with the bomb's ObjX; index 0 up wall,
; 1 down, 2 left, 3 right (the direction bit is Dir_ReverseIndexToDir[index]); value = pixels.
Bomb_WallHotspotsX:
    .BYTE $78, $78, $20, $D0

; TABLE: Y of the four bombable-wall hotspots, compared with the bomb's ObjY; same index; pixels.
Bomb_WallHotspotsY:
    .BYTE $5D, $BD, $8D, $8D

; ----------------------------------------------------------------------------
; Updates one bomb-or-fire slot: nothing if empty, Bomb_Update for state $1x, else Fire_Update.
; IN:   X = object slot ($10 or $11; called once for each, every frame of play)
; ----------------------------------------------------------------------------
Bomb_UpdateBombOrFire:
    ; State 0 = empty slot.
    LDA ObjState, X
    BEQ Bomb_Update_Return

    ; High nibble $10 = bomb. Anything else (fire is $2x) goes to the fire routine.
    AND #$F0
    CMP #$10
    BEQ Bomb_Update
    JMP Fire_Update

; ----------------------------------------------------------------------------
; Steps a bomb through fuse, blast and fade when its timer runs out, then draws it.
; IN:   X = object slot ($10 or $11) holding a bomb (ObjState $11-$14)
; OUT:  ObjState,X advanced and ObjTimer,X reloaded when the timer was 0; state 0 when finished.
; NOTE: $11 -> $12 on the first update (Bomb_Wield leaves the timer 0); $12 lasts $30 frames, $13
;       (the only damaging state) $18 frames, $14 $0C frames; reaching $15 frees the slot.
; ----------------------------------------------------------------------------
Bomb_Update:
    ; While the timer runs, only draw.
    LDA ObjTimer, X
    BNE Bomb_Draw

    ; The timer ran out: load the time of the next state (indexed by the low nibble of the
    ; current one) and advance the state.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA Bomb_StateTimes-1, Y
    STA ObjTimer, X
    INC ObjState, X

    ; Entering state $13 (blast): request the explosion sound effect.
    LDA ObjState, X
    AND #$0F
    PHA
    CMP #$03
    BNE @CheckState5
    LDA #$10                    ; Effect request bit $10 = bomb explosion
    JSR Sfx_Play

@CheckState5:
    PLA

    ; Entering state $15: the bomb is finished. Free the slot (state 0, timer 0) without drawing.
    CMP #$05
    BNE Bomb_TryOpenWall
    JSR Obj_ResetState           ; Returns A = 0.
    STA ObjTimer, X

; Shared exit of the bomb routines above and below.
Bomb_Update_Return:
    RTS

; ----------------------------------------------------------------------------
; On the frame a bomb's blast ends (state becomes $14), opens a bombable dungeon wall near it.
; IN:   X = object slot of the bomb; A = low nibble of its new state
; OUT:  TriggeredDoorCmd = 6 and TriggeredDoorDir = the wall's direction bit when a wall opens;
;       [02] = that direction bit once a hotspot is in range; falls into Bomb_Draw.
; NOTE: Runs once per bomb. Skipped in the overworld (rock walls test the bomb themselves) and in
;       cellars (GameMode 9). The bomb's ObjX and ObjY must each be less than $18 pixels from a
;       hotspot. Only the first hotspot in range (searched right, left, down, up) is considered.
;       No wall opens while another door command is running (TriggeredDoorCmd <> 0).
;       Bank 5 is left switched in when the door type was looked up (door not open, no command).
; ----------------------------------------------------------------------------
Bomb_TryOpenWall:
    ; Any new state but $14: just draw.
    CMP #$04
    BNE Bomb_Draw

    ; Overworld (CurLevel 0): nothing to do here. Rock walls are tile objects that look for a
    ; bomb in state $13 themselves.
    LDA CurLevel
    BEQ Bomb_Draw

    ; Bombs don't open walls in cellars (mode 9).
    LDA GameMode
    CMP #$09
    BEQ Bomb_Draw

    ; Dungeon room: look for a wall hotspot near the bomb, index 3 (right) down to 0 (up).
    LDY #$04

@LoopHotspot:
    DEY
    BMI Bomb_Draw                ; No hotspot in range: just draw.

    ; Next hotspot unless |hotspot X - bomb X| < $18.
    LDA Bomb_WallHotspotsX, Y
    SEC
    SBC ObjX, X
    JSR Math_Abs
    CMP #$18
    BCS @LoopHotspot

    ; Next hotspot unless |hotspot Y - bomb Y| < $18.
    LDA Bomb_WallHotspotsY, Y
    SEC
    SBC ObjY, X
    JSR Math_Abs
    CMP #$18
    BCS @LoopHotspot

    ; [02] = direction bit of this wall: index 0 -> 8 (up), 1 -> 4 (down), 2 -> 2 (left),
    ; 3 -> 1 (right).
    LDA Dir_ReverseIndexToDir, Y
    STA $02

    ; Nothing to do if that door is already open, or while any door command is running.
    AND CurOpenedDoors
    BNE Bomb_Draw
    LDA TriggeredDoorCmd
    BNE Bomb_Draw

    ; The door type on that side must be 4 (bombable wall).
    LDA #$05
    JSR Mmc1_SwitchBank
    JSR Door_FindTypeByDirBit
    CMP #$04
    BNE Bomb_Draw

    ; Start the door command that opens this wall.
    LDA #$06                    ; Door command 6 = open
    STA TriggeredDoorCmd
    LDA $02
    STA TriggeredDoorDir

; ----------------------------------------------------------------------------
; Draws the bomb (state $12) or its dust clouds (states $13, $14) at the bomb's position.
; IN:   X = object slot of the bomb
; NOTE: In states $13/$14 one cloud is drawn on the bomb and three more around it.
; ----------------------------------------------------------------------------
Bomb_Draw:
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Bomb_DrawBombOrCloudAt

    ; State $x2 drew the bomb itself: done.
    LDA ObjState, X
    AND #$0F
    CMP #$02
    BEQ Bomb_Update_Return

    ; Otherwise that was the middle cloud. Go draw the outer ones.
    JMP Bomb_DrawOuterClouds

; ----------------------------------------------------------------------------
; Runs the blast's screen flash, then draws the bomb or one cloud at ([00], [01]).
; IN:   X = object slot of the bomb; [00] = X; [01] = Y
; NOTE: Bomb_UpdateFlashEffect acts only in state $13 (it toggles the grayscale bit of the PPU mask
;       at fixed timer values).
; ----------------------------------------------------------------------------
Bomb_DrawBombOrCloudAt:
    JSR Bomb_UpdateFlashEffect

; ----------------------------------------------------------------------------
; Draws the bomb or one cloud at ([00], [01]) with frame image = low nibble of the state - 2.
; IN:   X = object slot of the bomb; [00] = X; [01] = Y
; NOTE: Image 0 = bomb (state $12), 1 = cloud of state $13, 2 = cloud of state $14.
; ----------------------------------------------------------------------------
Bomb_DrawBombOrCloudNoFlash:
    LDA ObjState, X
    AND #$0F
    SEC
    SBC #$02

; ----------------------------------------------------------------------------
; Writes the sprites of one frame image of the bomb item slot (0 bomb, 1-3 dust cloud).
; IN:   A = frame image; X = object slot; [00] = X; [01] = Y
; OUT:  [0C] = frame image; [0F] = 0; [04] = [05] = 1 (sprite attributes: sprite palette 1)
; NOTE: Also draws the monster spawn cloud: Obj_AnimateMetastate passes metastate 1-3.
; ----------------------------------------------------------------------------
Anim_DrawCloud:
    STA $0C                     ; [0C] frame image
    LDY #$00
    STY $0F                     ; [0F] no horizontal flipping

    ; Sprite attributes 1 (sprite palette 1) for both halves.
    INY
    STY $04
    STY $05
    LDY #$01                    ; Item slot 1 = bomb
    JMP Anim_WriteItemSprites

; TABLE: outer cloud Y offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
; (-13, 0, +14).
Bomb_CloudOffsetsY1:
    .BYTE $F3, $00, $0E

; TABLE: outer cloud X offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
; (-7, +14, +7).
Bomb_CloudOffsetsX1:
    .BYTE $F9, $0E, $07

; TABLE: outer cloud Y offsets, odd frames (reached as Bomb_CloudOffsetsY1 + 6); signed pixels
; (-13, 0, +14).
Bomb_CloudOffsetsY2:
    .BYTE $F3, $00, $0E

; TABLE: outer cloud X offsets, odd frames (reached as Bomb_CloudOffsetsX1 + 6); signed pixels
; (+7, -13, -7).
Bomb_CloudOffsetsX2:
    .BYTE $07, $F3, $F9

; ----------------------------------------------------------------------------
; Draws the three outer dust clouds of a blast, using a different set of positions on odd frames.
; IN:   X = object slot of the bomb (state $13 or $14)
; NOTE: Even frames: (-7,-13), (+14,0), (+7,+14) from the bomb; odd frames: (+7,-13), (-13,0),
;       (-7,+14). Drawing only; the damage test uses the bomb's own position.
; ----------------------------------------------------------------------------
Bomb_DrawOuterClouds:
    ; For each of 3 clouds, Y = 2 down to 0:
    LDY #$02

@LoopCloud:
    TYA                         ; Save the cloud index.
    PHA

    ; On odd frames add 6 to the index, which reaches the second set of offsets (the four
    ; 3-byte tables are contiguous).
    LDA FrameCounter
    LSR
    BCC @Draw
    TYA
    CLC
    ADC #$06
    TAY

@Draw:
    ; [01] = bomb Y + Y offset.
    LDA ObjY, X
    CLC
    ADC Bomb_CloudOffsetsY1, Y
    STA $01

    ; [00] = bomb X + X offset.
    LDA ObjX, X
    CLC
    ADC Bomb_CloudOffsetsX1, Y
    STA $00

    ; Draw the cloud there (no flash update for the outer clouds).
    JSR Bomb_DrawBombOrCloudNoFlash
    PLA                         ; Restore the cloud index.
    TAY

    DEY
    BPL @LoopCloud
    RTS
