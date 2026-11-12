; ============================================================================
; FILE:     bank7/bomb.asm               BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4806-5030
; SUBJECTS: Bomb
; TOPIC:    Link's bomb in object slots $10 and $11: the fuse, blast and fade states and their timers, the
;           test that opens a bombable dungeon wall, and the drawing of the bomb and its dust clouds
;           (the cloud writer is also the monster spawn cloud).
; SEE ALSO: bank1/bomb-candle-wield.asm (WieldBomb: placing a bomb, the two-slot limit;
;           UpdateBombFlashEffect); bank1/sword-bomb-arrow-collision.asm (CheckMonsterBombOrFireCollision:
;           blast damage and radius); bank4/rock-wall-tree-secrets.asm (UpdateRockWall: overworld rock
;           walls); bank5/update-doors.asm (UpdateDoors: runs TriggeredDoorCmd); bank5/fill-walls.asm
;           (FindDoorTypeByDoorBit); bank7/fire.asm (UpdateFire: the other user of slots $10/$11);
;           bank7/object-animation.asm (spawn cloud drawn through DrawCloud).
; KEY FACTS:
;   - Slots $10 and $11 hold a bomb (ObjState $11-$14) or a candle fire ($2x); UpdateBombOrFire picks the
;     routine by the high nibble of ObjState.
;   - Bomb states: $11 placed (ends on the first update), $12 fuse for $30 frames, $13 blast for $18
;     frames, $14 fading clouds for $0C frames, then the slot is freed (UpdateBomb, BombTimes).
;     From the first update to the freed slot: $30 + $18 + $0C = $54 (84) frames.
;   - Only state $13 hurts monsters and overworld rock walls; they test ObjState = $13 themselves
;     (CheckMonsterBombOrFireCollision, UpdateRockWall). Nothing in this file damages anything.
;   - The explosion sound (effect request bit $10) is requested once, when state $13 begins (UpdateBomb).
;   - A dungeon wall is tested once per bomb, on the frame the state changes $13 -> $14 (the blast's
;     end), not during the blast (Bomb_CheckState4).
;   - The wall test needs: a dungeon level, not a cellar (GameMode 9), the bomb's ObjX and ObjY each less
;     than $18 pixels from a wall hotspot, that door not already open (CurOpenedDoors), no door
;     command running (TriggeredDoorCmd = 0), and door type 4 (bombable) on that side
;     (Bomb_CheckState4, BombableWallHotspotsX, BombableWallHotspotsY).
;   - Hotspots (bomb ObjX, ObjY): up wall ($78,$5D), down ($78,$BD), left ($20,$8D), right ($D0,$8D).
;   - A passing test sets TriggeredDoorCmd = 6 (open) and TriggeredDoorDir (Bomb_CheckState4).
;   - In states $13 and $14 four clouds are drawn per frame: one on the bomb and three of six outer
;     positions, alternating by frame parity (DrawBomb, DrawOtherBombClouds). Drawing only.
; CONTENTS: (generated; L = line in this file)
;   L0055 BombTimes                  data  TABLE: bomb timer loaded when a state begins; index = low nibble of the state being left - 1;
;   L0060 BombableWallHotspotsX      data  TABLE: X of the four bombable-wall hotspots, compared with the bomb's ObjX; index 0 up wall,
;   L0064 BombableWallHotspotsY      data  TABLE: Y of the four bombable-wall hotspots, compared with the bomb's ObjY; same index; pixels.
;   L0071 UpdateBombOrFire           code  Updates one bomb-or-fire slot: nothing if empty, UpdateBomb for state $1x, else UpdateFire.
;   L0089 UpdateBomb                 code  Steps a bomb through fuse, blast and fade when its timer runs out, then draws it.
;   L0122 L1F95F_Exit                code  Shared exit of the bomb routines above and below.
;   L0136 Bomb_CheckState4           code  On the frame a bomb's blast ends (state becomes $14), opens a bombable dungeon wall near it.
;   L0203 DrawBomb                   code  Draws the bomb (state $12) or its dust clouds (states $13, $14) at the bomb's position.
;   L0222 DrawBombOrCloudAt          code  Runs the blast's screen flash, then draws the bomb or one cloud at ([00], [01]).
;   L0230 DrawBombOrCloudNoFlashing  code  Draws the bomb or one cloud at ([00], [01]) with frame image = low nibble of the state - 2.
;   L0242 DrawCloud                  code  Writes the sprites of one frame image of the bomb item slot (0 bomb, 1-3 dust cloud).
;   L0256 BombCloudOffsetsY1         data  TABLE: outer cloud Y offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
;   L0261 BombCloudOffsetsX1         data  TABLE: outer cloud X offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
;   L0266 BombCloudOffsetsY2         data  TABLE: outer cloud Y offsets, odd frames (reached as BombCloudOffsetsY1 + 6); signed pixels
;   L0271 BombCloudOffsetsX2         data  TABLE: outer cloud X offsets, odd frames (reached as BombCloudOffsetsX1 + 6); signed pixels
;   L0280 DrawOtherBombClouds        code  Draws the three outer dust clouds of a blast, using a different set of positions on odd frames.
; ============================================================================

; TABLE: bomb timer loaded when a state begins; index = low nibble of the state being left - 1;
; value = frames. $30 fuse (state $12), $18 blast ($13), $0C fading clouds ($14); the last value is
; loaded for state $15, which frees the slot at once and clears the timer, so $06 is never counted.
BombTimes:
    .BYTE $30, $18, $0C, $06

; TABLE: X of the four bombable-wall hotspots, compared with the bomb's ObjX; index 0 up wall,
; 1 down, 2 left, 3 right (the direction bit is ReverseDirections[index]); value = pixels.
BombableWallHotspotsX:
    .BYTE $78, $78, $20, $D0

; TABLE: Y of the four bombable-wall hotspots, compared with the bomb's ObjY; same index; pixels.
BombableWallHotspotsY:
    .BYTE $5D, $BD, $8D, $8D

; ----------------------------------------------------------------------------
; Updates one bomb-or-fire slot: nothing if empty, UpdateBomb for state $1x, else UpdateFire.
; IN:   X = object slot ($10 or $11; called once for each, every frame of play)
; ----------------------------------------------------------------------------
UpdateBombOrFire:
    ; State 0 = empty slot.
    LDA ObjState, X
    BEQ L1F95F_Exit

    ; High nibble $10 = bomb. Anything else (fire is $2x) goes to the fire routine.
    AND #$F0
    CMP #$10
    BEQ UpdateBomb
    JMP UpdateFire

; ----------------------------------------------------------------------------
; Steps a bomb through fuse, blast and fade when its timer runs out, then draws it.
; IN:   X = object slot ($10 or $11) holding a bomb (ObjState $11-$14)
; OUT:  ObjState,X advanced and ObjTimer,X reloaded when the timer was 0; state 0 when finished.
; NOTE: $11 -> $12 on the first update (WieldBomb leaves the timer 0); $12 lasts $30 frames, $13
;       (the only damaging state) $18 frames, $14 $0C frames; reaching $15 frees the slot.
; ----------------------------------------------------------------------------
UpdateBomb:
    ; While the timer runs, only draw.
    LDA ObjTimer, X
    BNE DrawBomb

    ; The timer ran out: load the time of the next state (indexed by the low nibble of the
    ; current one) and advance the state.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA BombTimes-1, Y
    STA ObjTimer, X
    INC ObjState, X

    ; Entering state $13 (blast): request the explosion sound effect.
    LDA ObjState, X
    AND #$0F
    PHA
    CMP #$03
    BNE @CheckState5
    LDA #$10                    ; Effect request bit $10 = bomb explosion
    JSR PlayEffect

@CheckState5:
    PLA

    ; Entering state $15: the bomb is finished. Free the slot (state 0, timer 0) without drawing.
    CMP #$05
    BNE Bomb_CheckState4
    JSR ResetObjState           ; Returns A = 0.
    STA ObjTimer, X

; Shared exit of the bomb routines above and below.
L1F95F_Exit:
    RTS

; ----------------------------------------------------------------------------
; On the frame a bomb's blast ends (state becomes $14), opens a bombable dungeon wall near it.
; IN:   X = object slot of the bomb; A = low nibble of its new state
; OUT:  TriggeredDoorCmd = 6 and TriggeredDoorDir = the wall's direction bit when a wall opens;
;       [02] = that direction bit once a hotspot is in range; falls into DrawBomb.
; NOTE: Runs once per bomb. Skipped in the overworld (rock walls test the bomb themselves) and in
;       cellars (GameMode 9). The bomb's ObjX and ObjY must each be less than $18 pixels from a
;       hotspot. Only the first hotspot in range (searched right, left, down, up) is considered.
;       No wall opens while another door command is running (TriggeredDoorCmd <> 0).
;       Bank 5 is left switched in when the door type was looked up (door not open, no command).
; ----------------------------------------------------------------------------
Bomb_CheckState4:
    ; Any new state but $14: just draw.
    CMP #$04
    BNE DrawBomb

    ; Overworld (CurLevel 0): nothing to do here. Rock walls are tile objects that look for a
    ; bomb in state $13 themselves.
    LDA CurLevel
    BEQ DrawBomb

    ; Bombs don't open walls in cellars (mode 9).
    LDA GameMode
    CMP #$09
    BEQ DrawBomb

    ; Dungeon room: look for a wall hotspot near the bomb, index 3 (right) down to 0 (up).
    LDY #$04

@LoopHotspot:
    DEY
    BMI DrawBomb                ; No hotspot in range: just draw.

    ; Next hotspot unless |hotspot X - bomb X| < $18.
    LDA BombableWallHotspotsX, Y
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$18
    BCS @LoopHotspot

    ; Next hotspot unless |hotspot Y - bomb Y| < $18.
    LDA BombableWallHotspotsY, Y
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$18
    BCS @LoopHotspot

    ; [02] = direction bit of this wall: index 0 -> 8 (up), 1 -> 4 (down), 2 -> 2 (left),
    ; 3 -> 1 (right).
    LDA ReverseDirections, Y
    STA $02

    ; Nothing to do if that door is already open, or while any door command is running.
    AND CurOpenedDoors
    BNE DrawBomb
    LDA TriggeredDoorCmd
    BNE DrawBomb

    ; The door type on that side must be 4 (bombable wall).
    LDA #$05
    JSR SwitchBank
    JSR FindDoorTypeByDoorBit
    CMP #$04
    BNE DrawBomb

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
DrawBomb:
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR DrawBombOrCloudAt

    ; State $x2 drew the bomb itself: done.
    LDA ObjState, X
    AND #$0F
    CMP #$02
    BEQ L1F95F_Exit

    ; Otherwise that was the middle cloud. Go draw the outer ones.
    JMP DrawOtherBombClouds

; ----------------------------------------------------------------------------
; Runs the blast's screen flash, then draws the bomb or one cloud at ([00], [01]).
; IN:   X = object slot of the bomb; [00] = X; [01] = Y
; NOTE: UpdateBombFlashEffect acts only in state $13 (it toggles the grayscale bit of the PPU mask
;       at fixed timer values).
; ----------------------------------------------------------------------------
DrawBombOrCloudAt:
    JSR UpdateBombFlashEffect

; ----------------------------------------------------------------------------
; Draws the bomb or one cloud at ([00], [01]) with frame image = low nibble of the state - 2.
; IN:   X = object slot of the bomb; [00] = X; [01] = Y
; NOTE: Image 0 = bomb (state $12), 1 = cloud of state $13, 2 = cloud of state $14.
; ----------------------------------------------------------------------------
DrawBombOrCloudNoFlashing:
    LDA ObjState, X
    AND #$0F
    SEC
    SBC #$02

; ----------------------------------------------------------------------------
; Writes the sprites of one frame image of the bomb item slot (0 bomb, 1-3 dust cloud).
; IN:   A = frame image; X = object slot; [00] = X; [01] = Y
; OUT:  [0C] = frame image; [0F] = 0; [04] = [05] = 1 (sprite attributes: sprite palette 1)
; NOTE: Also draws the monster spawn cloud: AnimateAndDrawMetaObject passes metastate 1-3.
; ----------------------------------------------------------------------------
DrawCloud:
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
BombCloudOffsetsY1:
    .BYTE $F3, $00, $0E

; TABLE: outer cloud X offsets from the bomb, even frames; index = cloud 0-2; value = signed pixels
; (-7, +14, +7).
BombCloudOffsetsX1:
    .BYTE $F9, $0E, $07

; TABLE: outer cloud Y offsets, odd frames (reached as BombCloudOffsetsY1 + 6); signed pixels
; (-13, 0, +14).
BombCloudOffsetsY2:
    .BYTE $F3, $00, $0E

; TABLE: outer cloud X offsets, odd frames (reached as BombCloudOffsetsX1 + 6); signed pixels
; (+7, -13, -7).
BombCloudOffsetsX2:
    .BYTE $07, $F3, $F9

; ----------------------------------------------------------------------------
; Draws the three outer dust clouds of a blast, using a different set of positions on odd frames.
; IN:   X = object slot of the bomb (state $13 or $14)
; NOTE: Even frames: (-7,-13), (+14,0), (+7,+14) from the bomb; odd frames: (+7,-13), (-13,0),
;       (-7,+14). Drawing only; the damage test uses the bomb's own position.
; ----------------------------------------------------------------------------
DrawOtherBombClouds:
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
    ADC BombCloudOffsetsY1, Y
    STA $01

    ; [00] = bomb X + X offset.
    LDA ObjX, X
    CLC
    ADC BombCloudOffsetsX1, Y
    STA $00

    ; Draw the cloud there (no flash update for the outer clouds).
    JSR DrawBombOrCloudNoFlashing
    PLA                         ; Restore the cloud index.
    TAY

    DEY
    BPL @LoopCloud
    RTS
