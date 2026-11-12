; ============================================================================
; FILE:     bank4/ganon-draw-collisions.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10547-10993
; SUBJECTS: Ganon
; TOPIC:    Ganon (type $3E): body drawing, the death burst (8 rays, 8 clouds), the ashes, his palette
;           changes, his collision rules (sword while invisible, silver arrow while brown) and the
;           release of the room item (Triforce of Power) when he is dead.
; SEE ALSO: bank4/ganon.asm (UpdateGanon: scene phases, movement, shooting, the dying sequence that calls
;           these routines); bank4/patra-ganon-inits.asm (InitGanon: invincibility mask $FA);
;           bank4/wizzrobe.asm (BlueWizzrobe_Move, used for the burst rays);
;           bank1/sword-bomb-arrow-collision.asm (CheckMonsterSwordCollision,
;           CheckMonsterArrowOrRodCollision); bank1/link-collision-harm.asm (CheckLinkCollisionPreinit);
;           bank1/take-item.asm (room item pickup).
; KEY FACTS:
;   - Ganon is 32x32: his collision midpoint is (ObjX+$10, ObjY+$10) and Link is hit when both axis
;     distances to Link's midpoint are < 9. The magic clock and stun do not protect Link from him; only
;     Link's own invincibility timer does. Ganon_CheckCollisions
;   - Blue Ganon (ObjState 0) can only be hurt by the sword (slot $0D, fully extended) and only while his
;     ObjTimer is 0, i.e. while he is invisible. Each hit shows him for $40 frames. Ganon_CheckCollisions
;   - HP is $F0. Sword damage $10/$20/$40 means 15 wooden, 8 white or 4 magic sword hits bring it to 0.
;     Ganon_CheckCollisions
;   - At 0 HP he does not die: HP returns to $F0, ObjState becomes $FF and he turns brown. The sword no
;     longer affects him until the brown state runs out. Ganon_CheckCollisions
;   - Brown Ganon dies only to a flying arrow (slot $12 in state $10) while InvArrow = 2 (silver arrow).
;     A wooden arrow is not even tested. Ganon_CheckCollisions
;   - The arrow hit starts Ganon_ObjDyingCount (1), sets his invincibility timer to $28 (flashing) and
;     the cloud distance to 8. Ganon_CheckCollisions
;   - The burst uses object slots $02-$09 as its 8 rays without testing that they are free.
;     Ganon_SetUpBurstRays, Ganon_DrawBurst
;   - The Triforce of Power is the room item (slot $13); it is activated only if it is inactive and not
;     already taken according to the room flags. Ganon_ActivateRoomItem
;   - The palette routines write 3 colours at fixed offsets 4-6 of DynTileBuf, which is the record they
;     appended only when the buffer was empty. Ganon_AppendPaletteRowTransferRecord_Brown
; CONTENTS: (generated; L = line in this file)
;   L0062 GanonBurstDirs                                 data  TABLE: direction of each death-burst ray; index = ray 0-7 (object slot $02-$09); value = direction
;   L0067 GanonBurstSpriteAttrs                          data  TABLE: sprite attribute bits of each burst ray; index = object slot ($02-$09; entries 0-1 unused);
;   L0073 GanonBurstTiles                                data  TABLE: sprite tile of each burst ray; index = object slot ($02-$09; entries 0-1 unused);
;   L0085 Ganon_SetUpBurstRays                           code  Places the 8 death-burst rays at Ganon's ashes and gives each its direction.
;   L0110 Ganon_DrawCloud                                code  Draws one death-burst cloud at ([00], [01]): frame image $0C at cloud distance >= 6, $0D below.
;   L0144 Ganon_DrawBurst                                code  Draws the death burst: 8 clouds closing in on the ashes and 8 rays flying outward.
;   L0231 Ganon_GetCurCloudLeft                          code  Sets [00] to the X of the burst clouds left of the ashes (ObjX - cloud distance).
;   L0242 Ganon_GetCurCloudRight                         code  Sets [00] to the X of the burst clouds right of the ashes (ObjX + cloud distance).
;   L0253 Ganon_GetCurCloudTop                           code  Sets [01] to the Y of the burst clouds above the ashes (ObjY - cloud distance).
;   L0264 Ganon_GetCurCloudBottom                        code  Sets [01] to the Y of the burst clouds below the ashes (ObjY + cloud distance).
;   L0273 GanonFrameImages                               data  TABLE: frame images of Ganon's body; index = animation frame (0-5) * 4 + quarter (0 top left,
;   L0279 GanonSpriteOffsetsX                            data  TABLE: X offset of each body quarter from ObjX; index = quarter 0-3; value = pixels.
;   L0283 GanonSpriteOffsetsY                            data  TABLE: Y offset of each body quarter from ObjY; index = quarter 0-3; value = pixels.
;   L0288 GanonSpriteHFlips                              data  TABLE: horizontal flip flag of each body quarter; index = quarter 0-3; value = 1 flipped (the right
;   L0296 Ganon_DrawBody                                 code  Draws Ganon's 32x32 body as four 16x16 quarters for the current animation frame.
;   L0356 Ganon_CheckCollisions                          code  Ganon's collision rules: hurts Link on contact, takes sword hits while invisible, arrow when brown.
;   L0458 GanonColorTransferRecord                       data  DATA: template of the PPU transfer record for Ganon's palette: address $3F1C (sprite palette row 3,
;   L0463 GanonColorSets                                 data  TABLE: three sets of the 3 colours written over the template; index = 0-8, a set ends at index
;   L0475 Ganon_AppendPaletteRowTransferRecord_Brown     code  Queues Ganon's brown palette ($07, $17, $30), used while he is vulnerable to the silver arrow.
;   L0480 Ganon_AppendPaletteRowTransferRecord_Blue      code  Queues Ganon's blue palette ($16, $2C, $3C), used when the brown state ends.
;   L0485 Ganon_AppendPaletteRowTransferRecord_Triforce  code  Queues the palette for the ashes and the Triforce of Power ($27, $06, $16) when Ganon bursts.
;   L0526 Ganon_DrawAshes                                code  Draws Ganon's pile of ashes (frame image $0B) at his position.
;   L0537 Ganon_ActivateRoomItem                         code  Makes the room item (the Triforce of Power in Ganon's room) appear, unless active or already taken.
; ============================================================================

; TABLE: direction of each death-burst ray; index = ray 0-7 (object slot $02-$09); value = direction
;        bits (1 right, 2 left, 4 down, 8 up).
GanonBurstDirs:
    .BYTE $01, $02, $04, $05, $06, $08, $09, $0A    ; R, L, D, R+D, L+D, U, R+U, L+U

; TABLE: sprite attribute bits of each burst ray; index = object slot ($02-$09; entries 0-1 unused);
;        value = $40 flip horizontally, $80 flip vertically.
GanonBurstSpriteAttrs:
    .BYTE $00, $00, $40, $00, $80, $C0, $80, $00    ; -, -, R, L, D, R+D, L+D, U
    .BYTE $40, $00                                  ; R+U, L+U

; TABLE: sprite tile of each burst ray; index = object slot ($02-$09; entries 0-1 unused);
;        value = tile ($EE horizontal ray, $E8 vertical ray, $30 diagonal ray).
GanonBurstTiles:
    .BYTE $00, $00, $EE, $EE, $E8, $30, $30, $E8    ; -, -, R, L, D, R+D, L+D, U
    .BYTE $30, $30                                  ; R+U, L+U

; ----------------------------------------------------------------------------
; Places the 8 death-burst rays at Ganon's ashes and gives each its direction.
; IN:   X = Ganon's object slot (ObjX/ObjY already moved to the ashes position).
; OUT:  ObjX, ObjY, ObjDir of object slots $02-$09 overwritten; Y = $FF.
; NOTE: Only position and direction of slots $02-$09 are written; the slots are not tested for being
;       free and their ObjType is left alone. The carry of the X addition is not cleared before the Y
;       addition.
; ----------------------------------------------------------------------------
Ganon_SetUpBurstRays:
    ; Ray index Y = 7..0 maps to object slots $09..$02.
    LDY #$07

@LoopRay:
    ; Start each ray at offset (4, 4) from the ashes.
    LDA ObjX, X
    CLC
    ADC #$04
    STA a:ObjX+2, Y
    LDA ObjY, X
    ADC #$04
    STA a:ObjY+2, Y

    LDA GanonBurstDirs, Y
    STA a:ObjDir+2, Y
    DEY
    BPL @LoopRay
    RTS

; ----------------------------------------------------------------------------
; Draws one death-burst cloud at ([00], [01]): frame image $0C at cloud distance >= 6, $0D below.
; IN:   X = Ganon's object slot; [00] = sprite X; [01] = sprite Y.
; OUT:  [00] and [01] preserved; [0F] = 0.
; ----------------------------------------------------------------------------
Ganon_DrawCloud:
    ; Save the coordinates; the draw routine changes them.
    LDA $00
    PHA
    LDA $01
    PHA

    ; Cloud distance >= 6: frame image $0C. Below 6: frame image $0D.
    LDA #$0C
    LDY Ganon_ObjCloudDist, X
    CPY #$06
    BCS :+
    LDA #$0D
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

; ----------------------------------------------------------------------------
; Draws the death burst: 8 clouds closing in on the ashes and 8 rays flying outward.
; IN:   X = Ganon's object slot.
; OUT:  CurObjIndex restored; X = $0A (not reloaded); ObjX/ObjY of object slots $02-$09 changed.
; NOTE: Tail-jumped to from Ganon_Dying while Ganon_ObjDyingCount is $50-$9F. The cloud distance (8 at the
;       arrow hit) drops by 1 on every frame whose FrameCounter is a multiple of 8, down to 0.
;       Rays in the 4 cardinal directions move 1 pixel every frame; the 4 diagonal rays move 1 pixel on
;       each axis on 3 of every 4 frames.
; ----------------------------------------------------------------------------
Ganon_DrawBurst:
    ; If the cloud distance <> 0, decrement it once every 8 frames.
    LDA Ganon_ObjCloudDist, X
    BEQ @DrawClouds
    LDA FrameCounter
    AND #$07
    BNE @DrawClouds
    DEC Ganon_ObjCloudDist, X

@DrawClouds:
    ; Draw the four clouds at the diagonal corners: top left, top right, bottom left, bottom right.
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

    ; Draw the top and bottom clouds (X = the ashes' X).
    LDA ObjX, X
    STA $00
    JSR Ganon_GetCurCloudTop
    JSR Ganon_DrawCloud
    JSR Ganon_GetCurCloudBottom
    JSR Ganon_DrawCloud

    ; Draw the left and right clouds (Y = the ashes' Y).
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

    ; Slots 2, 3, 4 and 7 hold the cardinal rays (right, left, down, up): move every frame.
    ; The others are diagonal: skip the move when FrameCounter is a multiple of 4, so that
    ; the rays spread roughly in a circle.
    CPX #$05
    BCC @Move
    CPX #$07
    BEQ @Move
    LDA FrameCounter
    AND #$03
    BEQ :+

@Move:
    JSR BlueWizzrobe_Move       ; 1 pixel along each axis of ObjDir.
:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Sprite attributes [03]: the low 2 bits of the frame counter cycle the palette row (flashing);
    ; the table adds the horizontal and vertical flip bits.
    LDA FrameCounter
    AND #$03
    ORA GanonBurstSpriteAttrs, X
    STA $03
    LDA GanonBurstTiles, X
    JSR Anim_WriteSprite

    ; Next slot, up to 9.
    INX
    CPX #$0A
    BCC @LoopRay

    ; Restore the object index (CurObjIndex only; X stays $0A).
    PLA
    STA CurObjIndex
    RTS

; ----------------------------------------------------------------------------
; Sets [00] to the X of the burst clouds left of the ashes (ObjX - cloud distance).
; IN:   X = Ganon's object slot.
; ----------------------------------------------------------------------------
Ganon_GetCurCloudLeft:
    LDA ObjX, X
    SEC
    SBC Ganon_ObjCloudDist, X
    STA $00
    RTS

; ----------------------------------------------------------------------------
; Sets [00] to the X of the burst clouds right of the ashes (ObjX + cloud distance).
; IN:   X = Ganon's object slot.
; ----------------------------------------------------------------------------
Ganon_GetCurCloudRight:
    LDA ObjX, X
    CLC
    ADC Ganon_ObjCloudDist, X
    STA $00
    RTS

; ----------------------------------------------------------------------------
; Sets [01] to the Y of the burst clouds above the ashes (ObjY - cloud distance).
; IN:   X = Ganon's object slot.
; ----------------------------------------------------------------------------
Ganon_GetCurCloudTop:
    LDA ObjY, X
    SEC
    SBC Ganon_ObjCloudDist, X
    STA $01
    RTS

; ----------------------------------------------------------------------------
; Sets [01] to the Y of the burst clouds below the ashes (ObjY + cloud distance).
; IN:   X = Ganon's object slot.
; ----------------------------------------------------------------------------
Ganon_GetCurCloudBottom:
    LDA ObjY, X
    CLC
    ADC Ganon_ObjCloudDist, X
    STA $01
    RTS

; TABLE: frame images of Ganon's body; index = animation frame (0-5) * 4 + quarter (0 top left,
;        1 top right, 2 bottom left, 3 bottom right); value = frame image of that 16x16 quarter.
GanonFrameImages:
    .BYTE $06, $08, $07, $09, $00, $00, $01, $01    ; frame 0 (also the pose of the entry scene), 1
    .BYTE $02, $02, $03, $03, $04, $00, $05, $01    ; frames 2, 3
    .BYTE $04, $04, $05, $05, $00, $04, $01, $05    ; frames 4, 5

; TABLE: X offset of each body quarter from ObjX; index = quarter 0-3; value = pixels.
GanonSpriteOffsetsX:
    .BYTE $00, $10, $00, $10

; TABLE: Y offset of each body quarter from ObjY; index = quarter 0-3; value = pixels.
GanonSpriteOffsetsY:
    .BYTE $00, $00, $10, $10

; TABLE: horizontal flip flag of each body quarter; index = quarter 0-3; value = 1 flipped (the right
;        column).
GanonSpriteHFlips:
    .BYTE $00, $01, $00, $01

; ----------------------------------------------------------------------------
; Draws Ganon's 32x32 body as four 16x16 quarters for the current animation frame.
; IN:   X = Ganon's object slot; Ganon_ObjAnimationFrame,X = frame 0-5.
; OUT:  Y = $FF; [00], [01], [07], [0F] changed.
; ----------------------------------------------------------------------------
Ganon_DrawBody:
    ; Loop over the four quarters, 3 down to 0.
    LDY #$03

@LoopCorner:
    ; [00] = sprite X = Ganon's X + the quarter's offset.
    LDA ObjX, X
    CLC
    ADC GanonSpriteOffsetsX, Y
    STA $00

    ; [01] = sprite Y = Ganon's Y + the quarter's offset.
    LDA ObjY, X
    CLC
    ADC GanonSpriteOffsetsY, Y
    STA $01

    ; [0F] = the quarter's horizontal flip flag.
    LDA GanonSpriteHFlips, Y
    STA $0F

    ; Save the loop index (the quarter) and copy it to [07].
    TYA
    PHA
    STA $07

    ; Index into GanonFrameImages = animation frame * 4 + quarter.
    ; (Carry is clear after the shifts because the frame is at most 5.)
    LDA Ganon_ObjAnimationFrame, X
    ASL
    ASL
    ADC $07
    TAY
    LDA GanonFrameImages, Y
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the loop index.

    TAY
    DEY
    BPL @LoopCorner
    RTS

; ----------------------------------------------------------------------------
; Ganon's collision rules: hurts Link on contact, takes sword hits while invisible, arrow when brown.
; IN:   X = Ganon's object slot; ObjState,X = 0 blue, nonzero brown (counts down from $FF).
; OUT:  [02], [03] = Ganon's midpoint; may start Ganon_ObjDyingCount,X.
; NOTE: Link: tested every frame of the fight unless Link's invincibility timer is running. The entry
;       point used skips the magic clock and stun tests of CheckLinkCollision; its own tests remain
;       (no hit while Link's state is exactly $40, halted, or while IsLinkParalyzed is set).
;       Blue (ObjState 0): only the sword in slot $0D can hit, only while ObjTimer,X = 0 (he is
;       invisible), and only in sword state 2 (fully extended). A hit makes him visible for $40 frames
;       and causes no knockback. When his HP ($F0 at start) is used up, HP is set back to $F0, ObjState
;       becomes $FF (brown) and the death is cancelled.
;       Brown (ObjState <> 0): only an arrow flying in slot $12 (state $10) can hit, and only when
;       InvArrow = 2 (silver arrow). The hit starts the dying sequence; nothing else can kill him.
;       Sword beam, bombs, fire, boomerang and rod are never tested here (and the mask $FA set by
;       InitGanon covers every damage type except sword 1 and arrow 4).
;       The cancelled sword "death" still went through HandleMonsterDied, so WorldKillCount and
;       HelpDropCount (up to $0A) were incremented and the monster death sound was requested.
; ----------------------------------------------------------------------------
Ganon_CheckCollisions:
    ; Ganon's midpoint at offset ($10, $10) from his coordinates, in [02] and [03].
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
    ; CheckLinkCollision would compute the midpoint of a 16x16 object, so the second entry point
    ; CheckLinkCollisionPreinit is used with the midpoint above. That entry skips the first part of
    ; CheckLinkCollision, so do it here: skip the test if Link is invincible, and clear the result
    ; variables [06], [09], [0C] and the defender slot [00] (0 = Link).
    LDA ObjInvincibilityTimer
    BNE @SkipLinkCollision
    STA $06
    STA $09
    STA $0C
    LDY #$00
    STY $00
    JSR CheckLinkCollisionPreinit

@SkipLinkCollision:
    ; State <> 0: Ganon is brown. Go check the arrow.
    LDA ObjState, X
    BNE @CheckArrowCollision

    ; Blue. If timer <> 0, then Ganon is visible (recently hit) and can't be harmed. Return.
    LDA ObjTimer, X
    BNE @Exit
    LDY #$0D                    ; Sword slot
    JSR CheckMonsterSwordCollision

    ; If the hit used up his HP, the damage code set the dying metastate. Instead of dying:
    ; 1. Restore HP to the initial value $F0
    ; 2. Set state to $FF (0 - 1): brown, vulnerable to the silver arrow
    ; 3. Change sprite palette row 7 to brown
    LDA ObjMetastate, X
    BEQ @CheckHarmed
    LDA #$F0
    STA ObjHP, X
    DEC ObjState, X
    JSR Ganon_AppendPaletteRowTransferRecord_Brown

@CheckHarmed:
    ; A sword hit set his invincibility timer (to $10, in BeginShove). If so, play the boss
    ; hit cry and set timer to $40, so that he's visible for $40 frames.
    LDA ObjInvincibilityTimer, X
    BEQ @UndoDeath
    JSR PlayBossHitCryIfNeeded
    LDA #$40
    STA ObjTimer, X

@UndoDeath:
    ; Reset metastate, because Ganon cannot die this way.
    JSR ResetObjMetastate

    ; Reset shove info and invincibility timer: no knockback and no flashing from sword hits.
    JSR ResetShoveInfoAndInvincibilityTimer

@Exit:
    RTS

@CheckArrowCollision:
    ; If the silver arrow (InvArrow = 2) is not in the inventory, then return.
    LDA InvArrow
    CMP #$02
    BNE @Exit

    ; Reset the collision return value [06].
    LDA #$00
    STA $06

    ; If there's no arrow in flight (slot $12, state $10), then return.
    LDY #$12
    LDA a:ObjState, Y
    CMP #$10
    BNE @Exit

    ; Check for collision with the arrow; return if none.
    JSR CheckMonsterArrowOrRodCollision
    LDA $06                     ; [06] collision result
    BEQ @Exit

    ; Start the dying sequence (count 0 -> 1; Ganon_Dying takes over next frame).
    INC Ganon_ObjDyingCount, X

    ; Invincibility timer $28 makes his body flash while dying.
    LDA #$28
    STA ObjInvincibilityTimer, X

    ; Initial distance of the burst clouds: 8 pixels.
    LDA #$08
    STA Ganon_ObjCloudDist, X
    RTS

; DATA: template of the PPU transfer record for Ganon's palette: address $3F1C (sprite palette row 3,
;       palette row 7 overall), 4 bytes, the 4 colours, end marker $FF. Colours 1-3 get replaced.
GanonColorTransferRecord:
    .BYTE $3F, $1C, $04, $0F, $07, $17, $27, $FF

; TABLE: three sets of the 3 colours written over the template; index = 0-8, a set ends at index
;        2, 5 or 8; value = NES colour.
GanonColorSets:
    .BYTE $07, $17, $30, $16, $2C, $3C, $27, $06    ; brown set (0-2), blue set (3-5), Triforce set (6-8)
    .BYTE $16

; ----------------------------------------------------------------------------
; Queues Ganon's brown palette ($07, $17, $30), used while he is vulnerable to the silver arrow.
; IN:   DynTileBufLen = offset where the record goes.
; OUT:  X = CurObjIndex; DynTileBufLen += 8 (one past the record's $FF end marker).
; NOTE: Shared body below. The 3 colours are stored at DynTileBuf+4..+6 regardless of where the
;       8-byte template was appended, so they land in the appended record only when the buffer was
;       empty (DynTileBufLen = 0); otherwise the record keeps the template colours $07, $17, $27.
; ----------------------------------------------------------------------------
Ganon_AppendPaletteRowTransferRecord_Brown:
    LDY #$02                    ; End index of the brown set in GanonColorSets.
    BNE :+

; Queues Ganon's blue palette ($16, $2C, $3C), used when the brown state ends.
Ganon_AppendPaletteRowTransferRecord_Blue:
    LDY #$05                    ; End index of the blue set.
    BNE :+

; Queues the palette for the ashes and the Triforce of Power ($27, $06, $16) when Ganon bursts.
Ganon_AppendPaletteRowTransferRecord_Triforce:
    LDY #$08                    ; End index of the Triforce set.
:
    TYA                         ; Save the end index of the colours.
    PHA

    ; Append at the current end of the dynamic transfer buf.
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
    PLA                         ; Restore the end index of the colours.
    TAY

    ; Overwrite the last 3 colours with the set ending at index Y. The destination is the fixed
    ; offsets 4-6 of DynTileBuf (see the NOTE above).
    LDX #$02

@OverwriteColors:
    LDA GanonColorSets, Y
    STA DynTileBuf+4, X
    DEY
    DEX
    BPL @OverwriteColors
    LDX CurObjIndex
    RTS

; ----------------------------------------------------------------------------
; Draws Ganon's pile of ashes (frame image $0B) at his position.
; IN:   X = Ganon's object slot.
; OUT:  [00], [01] = his position; [0F] = 0 (no flip) before the draw.
; ----------------------------------------------------------------------------
Ganon_DrawAshes:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$0B
    JMP DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Makes the room item (the Triforce of Power in Ganon's room) appear, unless active or already taken.
; OUT:  ObjState+19 (room item slot $13) = 0 when activated; Tune1Request = 2.
; NOTE: Called once, on the frame Ganon_ObjDyingCount reaches $A0; the caller then moves the item to
;       the ashes.
; ----------------------------------------------------------------------------
Ganon_ActivateRoomItem:
    ; If the room item is already active (state 0), or the room flags say it was taken, return.
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

; Filler: 70 bytes of $FF up to the next chunk.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF
