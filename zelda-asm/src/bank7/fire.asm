; ============================================================================
; FILE:     bank7/fire.asm               BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4681-4804
; SUBJECTS: (none)
; TOPIC:    Candle fire
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0014 UpdateFire           code
;   L0126 GetWideObjectMiddle  code  Params:
;   L0136 L1F91D_Exit          code
; ============================================================================

UpdateFire:
    ; If this is not a walking fire (state $21), skip moving it.
    LDA ObjState, X
    CMP #$21
    BNE @StandingFie

    ; Move as if grid offset were 0.
    LDA ObjGridOffset, X
    PHA
    LDA #$00
    STA ObjGridOffset, X
    LDA ObjDir, X
    STA $0F
    JSR MoveObject
    PLA

    ; Add grid offset after the move to the original.
    ; This makes it continuous, and useful as a distance traveled.
    CLC
    ADC ObjGridOffset, X
    STA ObjGridOffset, X

    ; If the absolute value of grid offset < $10, go draw.
    ; 
    ; Once the absolute value of grid offset = $10,
    ; make the fire stand instead of move.
    ; Set state $22 and last $3F frames.
    JSR Abs
    CMP #$10
    BNE @DrawAndCheckCollisions
    LDA #$3F
    STA ObjTimer, X
    INC ObjState, X

@StandingFie:
    ; Check a standing fire.
    ;
    ; If time has run out, deactivate the item and return.
    LDA ObjTimer, X
    BEQ ResetObjState

    ; If in UW, update the candle to brighten the room if needed.
    LDA CurLevel
    BEQ @DrawAndCheckCollisions
    TXA
    PHA
    LDA #$04
    JSR SwitchBank
    JSR UpdateCandle
    PLA
    TAX

@DrawAndCheckCollisions:
    ; Advance the animation counter, and draw.
    LDA #$04                    ; 4 frames in every animation cycle.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    JSR Anim_SetObjHFlipForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorRedPaletteRow
    LDA #$00                    ; A: frame 0
    STA $0C                     ; [0C]: not mirrored
    LDY #$40                    ; Y: animation index $40 (but will use $41)
    JSR DrawObjectWithType

    ; Now check for a collision with the player.
    ;
    ; First, if the player is invincible, return.
    LDA ObjInvincibilityTimer
    BNE L1F91D_Exit

    ; Save the object index in [00].
    STX $00

    ; Store the player's center point coordinates in [04] and [05].
    LDX #$00
    LDY #$02
    JSR GetWideObjectMiddle

    ; Store the fire's center point coordinates in [02] and [03].
    LDX $00
    LDY #$00
    JSR GetWideObjectMiddle

    ; If the objects don't collide (< $E pixels in X and Y), then return.
    LDY $00
    LDX #$00
    LDA #$0E
    JSR DoObjectsCollide
    BEQ L1F91D_Exit

    ; The player and the fire collide.
    ;
    ; Make the fire shove the player.
    LDX $00
    LDY #$00
    STY $00
    JSR BeginShove

    ; Harm the player $80 points.
    LDA #$00
    STA $0D
    LDA #$80
    STA $0E
    JMP Link_BeHarmed

; Params:
; X: object index
; Y: offset from [02] and [03] to store center X and Y
;
; Returns:
; [02 + Y]: center X
; [03 + Y]: center Y
;
GetWideObjectMiddle:
    LDA ObjX, X
    CLC
    ADC #$08
    STA $0002, Y
    LDA ObjY, X
    CLC
    ADC #$08
    STA $0003, Y

L1F91D_Exit:
    RTS
