; ============================================================================
; FILE:     bank7/sword-shot-magic-shot.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 3450-3761
; SUBJECTS: (none)
; TOPIC:    Sword beam / magic shot
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 SwordShotSpreadBaseAttr     data  Sprite attributes that flip each corner of the spread shot
;   L0026 UpdateSwordShotOrMagicShot  code
;   L0053 DrawSwordShotOrMagicShot    code
;   L0117 HandleShotBlocked           code  Params:
;   L0173 DeactivateLinkShot          code
;   L0176 DeactivateShot              code
;   L0179 SetShotSpreadingState       code
;   L0193 SpreadShot                  code
;   L0330 L1F49F_Exit                 code
; ============================================================================

; Sprite attributes that flip each corner of the spread shot
; differently: H, H-V, V, 0
;
SwordShotSpreadBaseAttr:
    .BYTE $40, $C0, $80, $00

UpdateSwordShotOrMagicShot:
    ; If the shot is not active, return.
    LDA ObjState, X
    BEQ L1F36A_Exit

    ; If low bit is set, go handle shot spreading out.
    LSR
    BCC :+
    JMP SpreadShot
:
    ; Move the shot.
    LDA ObjGridOffset, X
    BNE :+
:
    LDA ObjDir, X
    JSR MoveShot

    ; If movement was blocked, go handle the situation.
    LDA $0F
    BEQ HandleShotBlocked

    ; If grid offset is a multiple of 8, reset it.
    LDA ObjGridOffset, X
    AND #$07
    BNE DrawSwordShotOrMagicShot
    STA ObjGridOffset, X

DrawSwordShotOrMagicShot:
    ; Prepare the sprite position.
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; If the direction is horizontal, move the sprites down 3 pixels.
    LDA ObjDir, X
    PHA
    AND #$03
    BEQ @Flash
    LDA $01
    CLC
    ADC #$03
    STA $01

@Flash:
    PLA

    ; Set the sprite attributes to (base attribute OR (frame counter AND 3)).
    ; This makes the shot flash by cycling all the palettes, one each frame.
    JSR GetOppositeDir          ; Call this only to get reverse direction index.
    LDA FrameCounter
    AND #$03
    ORA RDirectionToWeaponBaseAttribute, Y
    JSR Anim_SetSpriteDescriptorAttributes

    ; Set [0C] to the correct frame for the direction: horizontal or vertical.
    LDA RDirectionToWeaponFrame, Y
    STA $0C

    ; If the direction is left, flip the right facing image by setting [0F] to 1.
    CPY #$02
    BNE :+
    INC $0F
:
    ; If this a monster's shot, choose the item slot for drawing based on object type.
    ;   $57 (sword shot) => $22
    ;   other (presumably $58 or $59, magic shot) => $23
    LDY #$22
    CPX #$0D
    BCS @PlayerShot
    LDA ObjType, X
    CMP #$57
    BEQ @WriteSprites
    BNE @WriteMagicSprites

@PlayerShot:
    ; But if the player shot it, the choice depends on the shot's state.
    ;   high bit set (magic)   => $23
    ;   high bit clear (sword) => $22
    LDA ObjState, X
    ASL
    BCC @WriteSprites

@WriteMagicSprites:
    LDY #$23

@WriteSprites:
    JMP Anim_WriteItemSprites

; Params:
; X: shot object index
;
; If the object is a sword shot, go spread out the shot.
;
HandleShotBlocked:
    LDA ObjState, X
    ASL
    BCC SetShotSpreadingState

    ; This is a magic shot.
    ;
    ; If missing the magic book, go deactivate the shot object.
    LDA InvBook
    BEQ DeactivateShot

    ; Try to activate a fire object. Temporarily mark the candle
    ; unused, so it only has to depend on having an empty slot.
    ;
    ; If successful, Link's state will have been changed to "using item".
    ; But we don't want that in this case.
    ;
    ; For both these reasons, save state and restore it afterward.
    LDA ObjState
    PHA
    LDA UsedCandle
    PHA
    LDA #$00
    STA UsedCandle
    JSR WieldCandle
    PLA
    STA UsedCandle
    PLA
    STA ObjState

    ; State $21 means that a moving fire was just made.
    ;
    ; If it's any another value, then we didn't find an empty slot
    ; to make a fire. In this case, all that's left to do is to go
    ; deactivate the shot object.
    LDA ObjState, X
    CMP #$21
    BNE DeactivateLinkShot

    ; Make state $22 for a standing fire.
    INC ObjState, X

    ; Copy the shot's position and direction to the fire.
    LDY #$0E
    LDA a:ObjX, Y
    STA ObjX, X
    LDA a:ObjY, Y
    STA ObjY, X
    LDA a:ObjDir, Y
    STA ObjDir, X

    ; The fire lasts $4F frames.
    ; Deactivate the shot.
    LDA #$4F
    STA ObjTimer, X

DeactivateLinkShot:
    LDX #$0E

DeactivateShot:
    JMP ResetObjState

SetShotSpreadingState:
    ; MULTI: ObjDir -> ObjSwordShotNegativeOffset
    ;
    ; Set the state to spreading out (low bit is set).
    ;
    ; Set [98][X] to base negative offset -2. This is how far the left
    ; corners will be shown relative to the center point at object X, Y.
    ; The right corners will be shown 2 pixels in the opposite direction
    ; of the center point.
    INC ObjState, X
    LDA #$FE
    STA ObjDir, X
    RTS

SpreadShot:
    ; MULTI: ObjDir -> ObjSwordShotNegativeOffset
    ;
    ; The shot is spreading out.
    ;
    ; Copy base negative offset in [98][X] to [02] and [03].
    ; These will be negated appropriately for each corner.
    LDA ObjDir, X
    STA $02
    STA $03

    ; Reset [0F] to not flip horizontally.
    LDA #$00
    STA $0F

    ; Loop 4 times to draw each corner of the spreading sword shot.
    ; The corners are: top-left, bottom-left, bottom-right, top-right
    LDY #$03

@DrawShotCorner:
    TYA                         ; Save the loop index.
    PHA

    ; Save offset X [02] and Y offset [03].
    LDA $02
    PHA
    LDA $03
    PHA

    ; Calculate and set sprite attributes for this corner:
    ;   base attribute OR (frame counter AND 3)
    ;
    ; The base attribute defines how to flip the sprite.
    ; The frame counter makes the shot flash by cycling each
    ; palette row.
    LDA FrameCounter
    AND #$03
    ORA SwordShotSpreadBaseAttr, Y
    JSR Anim_SetSpriteDescriptorAttributes

    ; Add the object's X (the center of the spread) and the
    ; current offset in [02] to set sprite X in [00].
    LDA ObjX, X
    CLC
    ADC $02
    STA $00

    ; If sprite X >= object X and sprite X >= $FC, skip this corner.
    ; Else calculate the distance from object X to sprite X.
    CMP ObjX, X
    BCC @ReverseSub
    CMP #$FC
    BCS @NextCorner
    SEC
    SBC ObjX, X
    JMP @CheckDistance

@ReverseSub:
    LDA ObjX, X
    SEC
    SBC $00

@CheckDistance:
    ; If distance >= $20, skip this corner.
    CMP #$20
    BCS @NextCorner

    ; Add the object's Y (the center of the spread) and the
    ; current offset in [03] to set sprite Y in [01].
    LDA ObjY, X
    CLC
    ADC $03
    STA $01

    ; If in UW, and (sprite Y < $3E or >= $E8), then skip this corner.
    LDY CurLevel
    BEQ @Draw
    CMP #$3E
    BCC @NextCorner
    CMP #$E8
    BCS @NextCorner

@Draw:
    ; Draw the corner.
    ; Pass [0C] = frame 2 (spread shot), Y = $23 (sword 2).
    LDA #$02
    STA $0C
    LDY #$23
    JSR Anim_WriteItemSprites

@NextCorner:
    ; Prepare to process the next corner.
    ;
    ; First, restore X offset [02] and Y offset [03].
    PLA
    STA $03
    PLA
    STA $02

    ; Take turns negating X offset or Y offset for the next round.
    ; - First turn:  loop index = 3, negate Y offset [03]
    ; - Second turn: loop index = 2, negate X offset [02]
    ; - Third turn:  loop index = 1, negate Y offset [03]
    ; - Fourth turn: doesn't matter
    PLA                         ; Get the loop index.
    PHA
    TAY
    CPY #$01                    ; The first two loop indexes coincide with offset from address 0.
    BNE :+                      ; But for loop index 1,
    LDY #$03                    ; make the offset 3 to negate the Y offset.
:
    LDA $0000, Y
    EOR #$FF
    CLC
    ADC #$01
    STA $0000, Y

    ; Restore the loop index, decrement it, and loop again if >= 0.
    PLA
    TAY
    DEY
    BPL @DrawShotCorner

    ; MULTI: ObjDir -> ObjSwordShotNegativeOffset
    ;
    ; Decrease the base negative offset to get farther away from
    ; the center point.
    DEC ObjDir, X

    ; MULTI: ObjDir -> ObjSwordShotNegativeOffset
    ;
    ; Once the base negative offset reaches $E8, go deactivate the object.
    LDA ObjDir, X
    CMP #$E8
    BNE L1F49F_Exit
    JMP DeactivateLinkShot

L1F49F_Exit:
    RTS
