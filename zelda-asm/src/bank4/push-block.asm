; ============================================================================
; FILE:     bank4/push-block.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 626-788
; SUBJECTS: (none)
; TOPIC:    Dungeon push block states; draw-over-Link sprite helpers
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 BlockPushDirections            data
;   L0026 UpdateBlock                    code
;   L0031 UpdateBlock_JumpTable          data
;   L0036 UpdateBlock0Idle               code
;   L0113 DrawBlock                      code
;   L0119 ResetPushTimer                 code
;   L0124 L10262_Exit                    code
;   L0127 UpdateBlock1Moving             code
;   L0153 UpdateBlock2Done               code
;   L0156 HideSpritesOverLink            code
;   L0162 DrawObjectNotMirroredOverLink  code
;   L0170 DrawObjectMirroredOverLink     code  Params:
; ============================================================================

BlockPushDirections:
    .BYTE $08, $04, $02, $01

UpdateBlock:
    LDA ObjState, X
    AND #$03
    JSR TableJump

UpdateBlock_JumpTable:
    .ADDR UpdateBlock0Idle
    .ADDR UpdateBlock1Moving
    .ADDR UpdateBlock2Done

UpdateBlock0Idle:
    ; If there are monsters in the room, go reset the push timer and return.
    LDA IsRoomAllDead
    BEQ ResetPushTimer

    ; If Link's X is aligned with the block's, then get the difference
    ; between Link's (Y + 3) and the block's Y.
    ; Then go test the difference.
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
    ; If Link's (Y + 3) is not aligned with the block's Y, then
    ; go reset the push timer.
    LDA ObjY
    CLC
    ADC #$03
    CMP ObjY, X
    BNE ResetPushTimer

    ; Link's (Y + 3) is aligned with the block's Y.
    ; Get the difference in X.
    LDY #$02
    LDA ObjX
    SEC
    SBC ObjX, X

@TestDifference:
    ; If the difference in coordinates is negative, then Link is
    ; up or left of the block.
    ;
    ; In that case, increase Y register to index the correct opposite
    ; direction below. Also, negate the difference, so we test
    ; positive values.
    BPL :+
    INY
    JSR Negate
:
    ; If the distance >= $11, then go reset the push timer and return.
    CMP #$11
    BCS ResetPushTimer

    ; At this point, Link is aligned with the block.
    ;
    ; If the input direction is not the required direction based on
    ; Link's placement; then go reset the push timer and return.
    LDA ObjInputDir
    CMP BlockPushDirections, Y
    BNE ResetPushTimer

    ; Link is truly pushing the block.
    ;
    ; Increase the push timer.
    INC PushBlock_ObjPushTimer, X

    ; If the push timer still < $10, return.
    LDY PushBlock_ObjPushTimer, X
    CPY #$10
    BCC L10262_Exit

    ; Link has pushed enough.
    ; Set the block's direction to Link's, set state 1,
    ; and change tiles at the source to a floor tile.
    STA ObjDir, X
    INC ObjState, X
    INC ReturnToBank4           ; Switch back to bank 4 when returning from the call below.
    LDA #$74                    ; Floor tile tile
    JSR ChangeTileObjTiles

DrawBlock:
    JSR Anim_FetchObjPosForSpriteDescriptor
    DEC $01                     ; Decrement sprite Y, to draw at the object's true Y.
    LDA #$00                    ; Frame image 0
    JMP DrawObjectNotMirrored

ResetPushTimer:
    ; Reset the push timer.
    LDA #$00
    STA PushBlock_ObjPushTimer, X

L10262_Exit:
    RTS

UpdateBlock1Moving:
    ; Move in the block's direction, and draw.
    LDA ObjDir, X
    STA $0F
    JSR MoveObject
    JSR DrawBlock

    ; If the block has not moved $10 pixels, then return.
    LDA ObjGridOffset, X
    CMP #$10
    BEQ :+
    CMP #$F0
    BNE UpdateBlock2Done
:
    ; The block has moved enough.
    ; Play the "secret revealed" tune.
    ; Change the tiles at the destination to the block tile.
    ; Set state 2, and increment BlockPushComplete.
    LDA #$04
    STA Tune1Request
    INC ReturnToBank4           ; Switch back to bank 4 when returning from the call below.
    LDA #$B0
    JSR ChangeTileObjTiles
    INC ObjState, X
    INC BlockPushComplete

UpdateBlock2Done:
    RTS

HideSpritesOverLink:
    LDA #$F8
    STA Sprites+64
    STA Sprites+68
    RTS

DrawObjectNotMirroredOverLink:
    LDY #$00                    ; Not mirrored
    BEQ :+

; Params:
; A: frame image
;
; Mirrored
DrawObjectMirroredOverLink:
    LDY #$01
:
    STY $0C                     ; [0C] holds mirrored flag

    ; The animation index is (object type + 1) to account for Link
    ; having two animation indexes.
    LDY ObjType, X
    INY
    STA $0D                     ; [0D] holds frame image
    STY $0E                     ; [0E] holds animation index
    STX $08                     ; [08] holds object index
    LDA #$40                    ; Sprite $10: left sprite under Link
    STA LeftSpriteOffset
    LDA #$44                    ; Sprite $11: right sprite under Link
    JMP DrawObjectWithAnimAndSpecificSprites
