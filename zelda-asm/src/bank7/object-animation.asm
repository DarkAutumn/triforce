; ============================================================================
; FILE:     bank7/object-animation.asm   BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5032-5263
; SUBJECTS: (none)
; TOPIC:    Spawn cloud, death sparkle, walking animation
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0021 AnimateAndDrawMetaObject                                code
;   L0068 AnimateLinkBase                                         code  Returns:
;   L0090 AnimateObjectWalking                                    code  Params:
;   L0105 SetUpWalkingSprites                                     code
;   L0128 Anim_SetObjHFlipForSpriteDescriptor                     code  Use the movement frame as the value for horizontal flipping.
;   L0133 SetUpHorizontalWalkingSprites                           code
;   L0157 Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor  code  Params:
;   L0172 Anim_FetchObjPosForSpriteDescriptor                     code  Params:
;   L0186 RollOverAnimCounter                                     code  Roll over the animation counter, and switch the movement frame.
;   L0194 AnimateLinkObjState                                     code
; ============================================================================

AnimateAndDrawMetaObject:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; If the metastate >= $10, go animate and draw the death sparkle.
    LDA ObjMetastate, X
    CMP #$10
    BCS @AnimateSpark

    ; Animate and draw the spawning cloud.
    AND #$0F
    JSR DrawCloud

@CheckMetaObjTimer:
    ; If the object's timer has expire, then go to the next metastate
    ; and last 6 frames.
    LDA ObjTimer, X
    BNE @Exit

@IncMetastate:
    LDA #$06
    STA ObjTimer, X
    INC ObjMetastate, X

@Exit:
    RTS

@AnimateSpark:
    ; If metastate = $10, then go increment metastate and set
    ; object timer without drawing.
    AND #$0F
    BEQ @IncMetastate

    ; Set the frame image in [0C] (1, 0, 1) for the metastate.
    AND #$01
    STA $0C
    LDA #$01                    ; Normal sprite, palette 5 (blue)
    JSR Anim_SetSpriteDescriptorAttributes
    LDY #$24                    ; Death spark
    JSR Anim_WriteItemSprites
    JMP @CheckMetaObjTimer

; Returns:
; Y: animation frame
; [00]: object X
; [01]: object Y
; [0F]: flip horizontally
;
AnimateLinkBase:
    LDA ObjState
    BNE AnimateObjectWalking    ; If Link isn't idle, go animate object.
    LDA GameMode
    CMP #$04
    BEQ AnimateObjectWalking    ; If in mode 4, go animate object.
    CMP #$10
    BEQ AnimateObjectWalking    ; If in mode $10, go animate object.
    LDA ObjInputDir
    BEQ SetUpWalkingSprites     ; If there was no direction input button, skip animating Link.

; Params:
; X: object index
;
; Returns:
; Y: animation frame
; [00]: object X
; [01]: object Y
; [0F]: flip horizontal
;
; As the object walks, the animation counter rolls down.
;
AnimateObjectWalking:
    DEC ObjAnimCounter, X
    BNE SetUpWalkingSprites

    ; Once the counter reaches 0, and if object is the player,
    ; animate Link's object state.
    CPX #$00
    BNE :+
    JSR AnimateLinkObjState
:
    ; Roll over the counter (6 frames).
    LDA #$06
    STA $00
    JSR RollOverAnimCounter

SetUpWalkingSprites:
    ; Set up sprite position in the descriptors.
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Set up horizontal flipping.
    LDA ObjDir, X
    AND #$0C
    BEQ SetUpHorizontalWalkingSprites

    ; Facing up or down.
    LDY #$03                    ; Assume animation frame 3 (up).
    AND #$08
    BNE Anim_SetObjHFlipForSpriteDescriptor
    DEY                         ; If down, then use animation frame 2 (down).

; Use the movement frame as the value for horizontal flipping.
;
; Params:
; X: object index
;
; Returns:
; [0F]: flip horizontal, based on ObjMovementFrame
;
Anim_SetObjHFlipForSpriteDescriptor:
    LDA ObjAnimFrame, X
    STA $0F
    RTS

SetUpHorizontalWalkingSprites:
    ; Facing left or right.
    LDY #$00                    ; Assume animation frame 0 (legs apart).
    LDA ObjAnimFrame, X
    BEQ :+
    INY                         ; If walking frame 1, then animation frame 1 (legs together).
:
    LDA ObjDir, X
    AND #$01
    BNE :+
    INC $0F                     ; Facing left. Flip horizontally.
:
    RTS

; Params:
; A: new value for animation counter, in case it rolls over
; X: object index
;
; Returns:
; A: 0
; [00]: X
; [01]: Y
; [0F]: 0 for no horizontal flipping
;
Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor:
    STA $00
    DEC ObjAnimCounter, X
    BNE Anim_FetchObjPosForSpriteDescriptor
    JSR RollOverAnimCounter

; Params:
; X: object index
;
; Returns:
; A: 0
; [00]: X
; [01]: Y
; [0F]: 0 for no horizontal flipping
;
Anim_FetchObjPosForSpriteDescriptor:
    LDA ObjX, X
    STA $00
    LDA ObjY, X
    STA $01
    LDA #$00                    ; Don't flip horizontally
    STA $0F
    RTS

; Roll over the animation counter, and switch the movement frame.
;
; Params:
; [00]: new counter value
;
RollOverAnimCounter:
    LDA $00
    STA ObjAnimCounter, X
    LDA ObjAnimFrame, X
    EOR #$01
    STA ObjAnimFrame, X
    RTS

AnimateLinkObjState:
    ; If major state = $10
    ;   if minor state = 0, make it 1
    ;   else OR state with $30
    ;   go set movement frame to 1
    LDA ObjState
    AND #$30
    CMP #$10
    BNE @CheckState20
    LDA ObjState
    AND #$0F
    BNE @CombineStateWith30
    BEQ @IncState

@CheckState20:
    ; If major state = $20
    ;   if minor state = 0, make it 1
    ;   else OR state with $30
    ;   go set movement frame to 1
    CMP #$20
    BNE @CheckState30
    LDA ObjState
    AND #$0F
    BNE @CombineStateWith30

@IncState:
    INC ObjState
    JMP @SetMovementFrame1

@CombineStateWith30:
    LDA ObjState
    ORA #$30
    STA ObjState

@SetMovementFrame1:
    ; To animate the state, the animation counter had to have
    ; reached 0. Right after this routine, the counter will be
    ; rolled over, and the movement frame will be switched.
    ;
    ; So, set movement frame to 1 (legs together) at the end of
    ; the animation here, in order to immediately change it to
    ; 0 (legs apart) after this.
    LDA #$01
    STA ObjAnimFrame
    RTS

@CheckState30:
    ; If in state $30, then make Link idle, except for halting.
    CMP #$30
    BNE :+
    LDA ObjState
    AND #$C0
    STA ObjState
:
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF
