; ============================================================================
; FILE:     bank4/ganon-draw-collisions.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10547-10993
; SUBJECTS: (none)
; TOPIC:    Ganon draw, collisions, ashes, triforce
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0034 GanonBurstDirs                                 data
;   L0037 GanonBurstSpriteAttrs                          data
;   L0041 GanonBurstTiles                                data
;   L0045 Ganon_SetUpBurstRays                           code
;   L0073 Ganon_DrawCloud                                code  Params:
;   L0098 Ganon_DrawBurst                                code
;   L0189 Ganon_GetCurCloudLeft                          code  Returns:
;   L0199 Ganon_GetCurCloudRight                         code  Returns:
;   L0209 Ganon_GetCurCloudTop                           code  Returns:
;   L0219 Ganon_GetCurCloudBottom                        code  Returns:
;   L0226 GanonFrameImages                               data
;   L0231 GanonSpriteOffsetsX                            data
;   L0234 GanonSpriteOffsetsY                            data
;   L0237 GanonSpriteHFlips                              data
;   L0240 Ganon_DrawBody                                 code
;   L0289 Ganon_CheckCollisions                          code
;   L0397 GanonColorTransferRecord                       data
;   L0400 GanonColorSets                                 data
;   L0404 Ganon_AppendPaletteRowTransferRecord_Brown     code
;   L0408 Ganon_AppendPaletteRowTransferRecord_Blue      code
;   L0412 Ganon_AppendPaletteRowTransferRecord_Triforce  code
;   L0448 Ganon_DrawAshes                                code
;   L0453 Ganon_ActivateRoomItem                         code
; ============================================================================

GanonBurstDirs:
    .BYTE $01, $02, $04, $05, $06, $08, $09, $0A

GanonBurstSpriteAttrs:
    .BYTE $00, $00, $40, $00, $80, $C0, $80, $00
    .BYTE $40, $00

GanonBurstTiles:
    .BYTE $00, $00, $EE, $EE, $E8, $30, $30, $E8
    .BYTE $30, $30

Ganon_SetUpBurstRays:
    ; Loop over 8 burst rays, from 7 to 0, indexed by Y register.
    ; Access object slots 9 to 2.
    LDY #$07

@LoopRay:
    ; Start each piece at offset (4, 4) from Ganon's location.
    LDA ObjX, X
    CLC
    ADC #$04
    STA a:ObjX+2, Y
    LDA ObjY, X
    ADC #$04
    STA a:ObjY+2, Y

    ; Set the direction.
    LDA GanonBurstDirs, Y
    STA a:ObjDir+2, Y
    DEY
    BPL @LoopRay
    RTS

; Params:
; [00]: X
; [01]: Y
;
; Save the coordinates.
;
Ganon_DrawCloud:
    LDA $00
    PHA
    LDA $01
    PHA

    ; If cloud distance < 6, then use frame image $C (medium).
    ; Else use frame image $D (low).
    LDA #$0C
    LDY Ganon_ObjCloudDist, X
    CPY #$06
    BCS :+
    LDA #$0D                    ; Low density cloud frame image (tile $74)
:
    LDY #$00                    ; No horizontal flipping.
    STY $0F
    JSR DrawObjectMirrored

    ; Restore the coordinates.
    PLA
    STA $01
    PLA
    STA $00
    RTS

Ganon_DrawBurst:
    ; If cloud distance <> 0, then decrement it once every 8 frames.
    LDA Ganon_ObjCloudDist, X
    BEQ @DrawClouds
    LDA FrameCounter
    AND #$07
    BNE @DrawClouds
    DEC Ganon_ObjCloudDist, X

@DrawClouds:
    ; Draw the four clouds at diagonal corners.
    JSR Ganon_GetCurCloudLeft
    JSR Ganon_GetCurCloudTop
    JSR Ganon_DrawCloud
    JSR Ganon_GetCurCloudRight
    JSR Ganon_DrawCloud
    JSR Ganon_GetCurCloudLeft
    JSR Ganon_GetCurCloudBottom
    JSR Ganon_DrawCloud
    JSR Ganon_GetCurCloudRight
    JSR Ganon_DrawCloud

    ; Draw the top and bottom clouds.
    LDA ObjX, X
    STA $00
    JSR Ganon_GetCurCloudTop
    JSR Ganon_DrawCloud
    JSR Ganon_GetCurCloudBottom
    JSR Ganon_DrawCloud

    ; Draw the left and right clouds.
    JSR Ganon_GetCurCloudLeft
    LDA ObjY, X
    STA $01
    JSR Ganon_DrawCloud
    JSR Ganon_GetCurCloudRight
    JSR Ganon_DrawCloud

    ; Save the object index.
    LDA CurObjIndex
    PHA

    ; Loop over the 8 burst rays in slots 2 to 9, to move and draw them.
    LDX #$02

@LoopRay:
    STX CurObjIndex

    ; If object slot < 5 or = 7, then move every frame.
    ; Else move 3 out of every 4 frames.
    ;
    ; This makes pieces spread out in somewhat of a circle,
    ; by making the pieces at cardinal directions move at
    ; full speed while the diagonal ones move a little slower.
    CPX #$05
    BCC @Move
    CPX #$07
    BEQ @Move
    LDA FrameCounter
    AND #$03
    BEQ :+

@Move:
    JSR BlueWizzrobe_Move
:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; The frame counter will make the sprite flash.
    ; The array has the horizontal and vertical flipping attributes.
    ; Combine them into [03].
    LDA FrameCounter
    AND #$03
    ORA GanonBurstSpriteAttrs, X
    STA $03
    LDA GanonBurstTiles, X
    JSR Anim_WriteSprite

    ; Bottom of the loop.
    ; Increment object slot while <= 9.
    INX
    CPX #$0A
    BCC @LoopRay

    ; Restore the object index.
    PLA
    STA CurObjIndex
    RTS

; Returns:
; [00]: X coordinate
;
Ganon_GetCurCloudLeft:
    LDA ObjX, X
    SEC
    SBC Ganon_ObjCloudDist, X
    STA $00
    RTS

; Returns:
; [00]: X coordinate
;
Ganon_GetCurCloudRight:
    LDA ObjX, X
    CLC
    ADC Ganon_ObjCloudDist, X
    STA $00
    RTS

; Returns:
; [01]: Y coordinate
;
Ganon_GetCurCloudTop:
    LDA ObjY, X
    SEC
    SBC Ganon_ObjCloudDist, X
    STA $01
    RTS

; Returns:
; [01]: Y coordinate
;
Ganon_GetCurCloudBottom:
    LDA ObjY, X
    CLC
    ADC Ganon_ObjCloudDist, X
    STA $01
    RTS

GanonFrameImages:
    .BYTE $06, $08, $07, $09, $00, $00, $01, $01
    .BYTE $02, $02, $03, $03, $04, $00, $05, $01
    .BYTE $04, $04, $05, $05, $00, $04, $01, $05

GanonSpriteOffsetsX:
    .BYTE $00, $10, $00, $10

GanonSpriteOffsetsY:
    .BYTE $00, $00, $10, $10

GanonSpriteHFlips:
    .BYTE $00, $01, $00, $01

Ganon_DrawBody:
    ; Loop over four 16x16 images that make up Ganon.
    ; There are 6 animation frames.
    LDY #$03

@LoopCorner:
    ; Add an offset to Ganon's X, and store the result in [00] -- the sprite's X.
    LDA ObjX, X
    CLC
    ADC GanonSpriteOffsetsX, Y
    STA $00

    ; Add an offset to Ganon's Y, and store the result in [01] -- the sprite's Y.
    LDA ObjY, X
    CLC
    ADC GanonSpriteOffsetsY, Y
    STA $01

    ; Each part has a horizontal flipping flag.
    LDA GanonSpriteHFlips, Y
    STA $0F

    ; Save and copy the loop index (the current part) to [07].
    TYA
    PHA
    STA $07

    ; Multiply the animation frame by 4, because of the 4 parts
    ; that make it up.
    LDA Ganon_ObjAnimationFrame, X
    ASL
    ASL

    ; Add [07] -- the index of the current part.
    ;
    ; Now you have an index into the frame image table, for the
    ; current image part of the current animation frame.
    ADC $07
    TAY
    LDA GanonFrameImages, Y
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the loop index.

    ; Bottom of the part loop.
    TAY
    DEY
    BPL @LoopCorner
    RTS

Ganon_CheckCollisions:
    ; Calculate Ganon's midpoint at offset ($10, $10) from his coordinates.
    ; Store in [02] and [03].
    LDA ObjX, X
    CLC
    ADC #$10
    STA $02
    LDA ObjY, X
    CLC
    ADC #$10
    STA $03

    ; Check for a collision with Link.
    ;
    ; We can't call CheckLinkCollision, because we need to
    ; customize Ganon's midpoint.
    ;
    ; So, we've calculated the midpoint hotspot. Now, skip collision
    ; detection, if Link is invincible. Initialize variables that will
    ; be returned from the partial collision check routine. Then
    ; call it.
    LDA ObjInvincibilityTimer
    BNE @SkipLinkCollision
    STA $06
    STA $09
    STA $0C
    LDY #$00
    STY $00
    JSR CheckLinkCollisionPreinit

@SkipLinkCollision:
    ; If Ganon is not in state 0, then he's brown.
    ; Go check collision with an arrow.
    LDA ObjState, X
    BNE @CheckArrowCollision

    ; If timer <> 0, then Ganon is visible.
    ; He can't be harmed like this. So, return.
    LDA ObjTimer, X
    BNE @Exit
    LDY #$0D                    ; Sword slot
    JSR CheckMonsterSwordCollision

    ; If Ganon has died according to the usual collision check rules
    ; (HP = 0), then:
    ; 1. Restore HP to the initial value $F0
    ; 2. Set state to $FF to note that Ganon is vulnerable to silver arrows
    ; 3. Change palette row 7 to make Ganon brown
    LDA ObjMetastate, X
    BEQ @CheckHarmed
    LDA #$F0
    STA ObjHP, X
    DEC ObjState, X
    JSR Ganon_AppendPaletteRowTransferRecord_Brown

@CheckHarmed:
    ; If Ganon was harmed, then play the sound effect
    ; and set timer to $40, so that he's visible for $40 frames.
    LDA ObjInvincibilityTimer, X
    BEQ @UndoDeath
    JSR PlayBossHitCryIfNeeded
    LDA #$40
    STA ObjTimer, X

@UndoDeath:
    ; Reset metastate, because Ganon cannot die this way.
    JSR ResetObjMetastate

    ; Reset shove info and invincibility timer, because Ganon
    ; doesn't react at all as a usual monster.
    JSR ResetShoveInfoAndInvincibilityTimer

@Exit:
    RTS

@CheckArrowCollision:
    ; If the silver arrow is not in the inventory, then return.
    LDA InvArrow
    CMP #$02
    BNE @Exit

    ; Reset the collision return value [06].
    LDA #$00
    STA $06

    ; If there's no arrow in flight, then return.
    LDY #$12
    LDA a:ObjState, Y
    CMP #$10
    BNE @Exit

    ; Check for collision with an arrow; and return if no collision.
    JSR CheckMonsterArrowOrRodCollision
    LDA $06                     ; [06] collision result
    BEQ @Exit

    ; Set the Ganon phase to "dying".
    INC Ganon_ObjPhase, X

    ; Now his body does flash.
    LDA #$28
    STA ObjInvincibilityTimer, X

    ; Set initial cloud distance 8.
    LDA #$08
    STA Ganon_ObjCloudDist, X
    RTS

GanonColorTransferRecord:
    .BYTE $3F, $1C, $04, $0F, $07, $17, $27, $FF

GanonColorSets:
    .BYTE $07, $17, $30, $16, $2C, $3C, $27, $06
    .BYTE $16

Ganon_AppendPaletteRowTransferRecord_Brown:
    LDY #$02                    ; Brown and white colors at AF64.
    BNE :+

Ganon_AppendPaletteRowTransferRecord_Blue:
    LDY #$05                    ; Red and blue colors at AF67.
    BNE :+

Ganon_AppendPaletteRowTransferRecord_Triforce:
    LDY #$08                    ; Yellow and red colors at AF6A.
:
    TYA                         ; Save the end index of the colors.
    PHA

    ; Get the length, to append to the dynamic transfer buf.
    LDX DynTileBufLen

    ; Copy the 8 bytes of the palette row transfer record template.
    LDY #$00

@CopyBytes:
    LDA GanonColorTransferRecord, Y
    STA DynTileBuf, X
    INX
    INY
    CPY #$08
    BNE @CopyBytes
    STX DynTileBufLen
    PLA                         ; Restore the end index of the colors.
    TAY

    ; Overwrite the last 3 colors in the record, with the ones
    ; ending at the index passed in.
    LDX #$02

@OverwriteColors:
    LDA GanonColorSets, Y
    STA DynTileBuf+4, X
    DEY
    DEX
    BPL @OverwriteColors
    LDX CurObjIndex
    RTS

Ganon_DrawAshes:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$0B
    JMP DrawObjectNotMirrored

Ganon_ActivateRoomItem:
    ; If the room item was already active or taken, then return.
    LDA ObjState+19
    BEQ @Exit
    JSR GetRoomFlagUWItemState
    BNE @Exit

    ; Activate the room item, by setting its state to 0.
    LDA #$00
    STA ObjState+19

    ; Play "item appears" tune.
    LDA #$02
    STA Tune1Request

@Exit:
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF
