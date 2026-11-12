; ============================================================================
; FILE:     bank7/bomb.asm               BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4806-5030
; SUBJECTS: (none)
; TOPIC:    Bomb timers, bombable wall hotspots, UpdateBomb
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0027 BombTimes                  data
;   L0030 BombableWallHotspotsX      data
;   L0033 BombableWallHotspotsY      data
;   L0036 UpdateBombOrFire           code
;   L0047 UpdateBomb                 code
;   L0081 L1F95F_Exit                code
;   L0084 Bomb_CheckState4           code
;   L0148 DrawBomb                   code
;   L0167 DrawBombOrCloudAt          code  Params:
;   L0175 DrawBombOrCloudNoFlashing  code  Params:
;   L0188 DrawCloud                  code  Params:
;   L0200 BombCloudOffsetsY1         data
;   L0203 BombCloudOffsetsX1         data
;   L0206 BombCloudOffsetsY2         data
;   L0209 BombCloudOffsetsX2         data
;   L0212 DrawOtherBombClouds        code
; ============================================================================

BombTimes:
    .BYTE $30, $18, $0C, $06

BombableWallHotspotsX:
    .BYTE $78, $78, $20, $D0

BombableWallHotspotsY:
    .BYTE $5D, $BD, $8D, $8D

UpdateBombOrFire:
    ; If the object's not active, return.
    LDA ObjState, X
    BEQ L1F95F_Exit

    ; Bomb major state = $10. Fire major state = $20.
    AND #$F0
    CMP #$10
    BEQ UpdateBomb
    JMP UpdateFire

UpdateBomb:
    ; This is a bomb.
    ;
    ; If the timer has not expired, then go draw.
    LDA ObjTimer, X
    BNE DrawBomb

    ; The timer has expired. Set another based on the minor state.
    ; Advance the state.
    LDA ObjState, X
    AND #$0F
    TAY
    LDA BombTimes-1, Y
    STA ObjTimer, X
    INC ObjState, X

    ; If the minor state = 3, play the sound effect.
    LDA ObjState, X
    AND #$0F
    PHA
    CMP #$03
    BNE @CheckState5
    LDA #$10
    JSR PlayEffect

@CheckState5:
    PLA

    ; If minor state = 5, then deactivate the bomb, and return.
    CMP #$05
    BNE Bomb_CheckState4
    JSR ResetObjState
    STA ObjTimer, X

L1F95F_Exit:
    RTS

Bomb_CheckState4:
    ; If minor state <> 4, go draw.
    CMP #$04
    BNE DrawBomb

    ; Minor state = 4. Try to break a wall.
    ;
    ; If in OW, go draw. Rock walls have a tile object that checks for bombs.
    LDA CurLevel
    BEQ DrawBomb

    ; Bombs don't blast walls in cellars (mode 9). Go draw.
    LDA GameMode
    CMP #$09
    BEQ DrawBomb

    ; We're in UW. See if the bomb is near a bombable wall.
    LDY #$04

@LoopHotspot:
    DEY
    BMI DrawBomb                ; If none is found, then go draw.

    ; If the bomb's X is not within $18 pixels of the hotspot, then
    ; go check the next hotspot.
    LDA BombableWallHotspotsX, Y
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$18
    BCS @LoopHotspot

    ; If the bomb's Y is not within $18 pixels of the hotspot, then
    ; go check the next hotspot.
    LDA BombableWallHotspotsY, Y
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$18
    BCS @LoopHotspot

    ; Store in [02] the direction corresponding to the hotspot's index.
    LDA ReverseDirections, Y
    STA $02

    ; If it was already opened or triggered, then go draw.
    AND CurOpenedDoors
    BNE DrawBomb
    LDA TriggeredDoorCmd
    BNE DrawBomb

    ; If it's not a bombable wall, go draw.
    LDA #$05
    JSR SwitchBank
    JSR FindDoorTypeByDoorBit
    CMP #$04
    BNE DrawBomb

    ; Trigger this bombable wall to open.
    LDA #$06
    STA TriggeredDoorCmd
    LDA $02
    STA TriggeredDoorDir

DrawBomb:
    ; Draw the bomb or dust cloud.
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR DrawBombOrCloudAt

    ; If minor state = 2, we drew a bomb. So, return.
    LDA ObjState, X
    AND #$0F
    CMP #$02
    BEQ L1F95F_Exit

    ; Otherwise, we drew one dust cloud. Go draw the others.
    JMP DrawOtherBombClouds

; Params:
; X: object index
; [00]: X
; [01]: Y
;
DrawBombOrCloudAt:
    JSR UpdateBombFlashEffect

; Params:
; X: object index
; [00]: X
; [01]: Y
;
DrawBombOrCloudNoFlashing:
    ; Set frame image = minor state - 2.
    LDA ObjState, X
    AND #$0F
    SEC
    SBC #$02

; Params:
; A: frame image (1 to 3)
; X: object index
; [00]: X
; [01]: Y
;
DrawCloud:
    STA $0C                     ; [0C] frame image
    LDY #$00
    STY $0F                     ; [0F] no horizontal flipping

    ; Write sprites with blue sprite palette row and bomb item slot.
    INY
    STY $04
    STY $05
    LDY #$01                    ; Bomb item slot
    JMP Anim_WriteItemSprites

BombCloudOffsetsY1:
    .BYTE $F3, $00, $0E

BombCloudOffsetsX1:
    .BYTE $F9, $0E, $07

BombCloudOffsetsY2:
    .BYTE $F3, $00, $0E

BombCloudOffsetsX2:
    .BYTE $07, $F3, $F9

DrawOtherBombClouds:
    ; For each of 3 clouds, indexed by Y register:
    LDY #$02

@LoopCloud:
    TYA                         ; Save cloud index.
    PHA

    ; Every other screen frame, add 6 to the index to point inside
    ; the second set of coordinates.
    LDA FrameCounter
    LSR
    BCC @Draw
    TYA
    CLC
    ADC #$06
    TAY

@Draw:
    ; Set [01] to cloud position (object Y + Y offset).
    LDA ObjY, X
    CLC
    ADC BombCloudOffsetsY1, Y
    STA $01

    ; Set [00] to cloud position (object X + X offset).
    LDA ObjX, X
    CLC
    ADC BombCloudOffsetsX1, Y
    STA $00

    ; Draw the cloud at this position.
    JSR DrawBombOrCloudNoFlashing
    PLA                         ; Restore cloud index.
    TAY

    ; Loop while >= 0.
    DEY
    BPL @LoopCloud
    RTS
