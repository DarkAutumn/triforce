; ============================================================================
; FILE:     bank4/wallmaster.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4106-4535
; SUBJECTS: Wallmaster
; TOPIC:    Wallmaster tables, emerge/grab states
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 WallmasterDirsAndAttrsLeft             data
;   L0027 WallmasterDirsAndAttrsRight            data
;   L0031 WallmasterDirsAndAttrsTop              data
;   L0035 WallmasterDirsAndAttrsBottom           data
;   L0039 WallmasterInitialXs                    data
;   L0042 WallmasterInitialYs                    data
;   L0045 UpdateWallmaster                       code
;   L0176 L_Wallmaster_State1                    code
;   L0334 Wallmaster_PrepareToDraw               code
;   L0370 Wallmaster_CalcStartPosition           code  Params:
;   L0424 Wallmaster_PutSpritesBehindBgIfNeeded  code
;   L0429 Wallmaster_PutSpriteBehindBgIfNeeded   code
; ============================================================================

WallmasterDirsAndAttrsLeft:
    .BYTE $01, $01, $08, $08, $08, $02, $02, $02
    .BYTE $C1, $C1, $C4, $C4, $C4, $C2, $C2, $C2

WallmasterDirsAndAttrsRight:
    .BYTE $42, $42, $48, $48, $48, $41, $41, $41
    .BYTE $82, $82, $84, $84, $84, $81, $81, $81

WallmasterDirsAndAttrsTop:
    .BYTE $C4, $C4, $C2, $C2, $C2, $C8, $C8, $C8
    .BYTE $84, $84, $81, $81, $81, $88, $88, $88

WallmasterDirsAndAttrsBottom:
    .BYTE $48, $48, $42, $42, $42, $44, $44, $44
    .BYTE $08, $08, $01, $01, $01, $04, $04, $04

WallmasterInitialXs:
    .BYTE $00, $F0

WallmasterInitialYs:
    .BYTE $3D, $DD

UpdateWallmaster:
    LDA ObjState, X
    BEQ :+
    JMP L_Wallmaster_State1
:
    ; State 0.
    ;
    ; All wallmasters share ObjTimer[1] to control when they each
    ; emerge from the walls.
    ;
    ; If the timer has not expired, then return.
    LDA ObjTimer+1
    BNE @Exit

    ; If Link is halted, then return.
    LDA ObjState
    CMP #$40
    BNE @State0

@Exit:
    RTS

@State0:
    ; State 0. Idle/Waiting inside wall
    ;
    ; If  (Link's X < $29 or >= $C8)
    ; and (Link's Y < $6D or >= $B5),
    ; then return.
    LDA ObjX
    CMP #$29
    BCC :+
    CMP #$C8
    BCC @CheckLeftAndRight
:
    LDA ObjY
    CMP #$6D
    BCC @Exit
    CMP #$B5
    BCS @Exit

@CheckLeftAndRight:
    ; Check whether Wallmaster should come out of the left or right wall.
    ;
    ; If Link's X <> $20 nor $D0, then go check the top and bottom walls.
    LDA ObjX
    CMP #$20
    BEQ :+
    CMP #$D0
    BNE @CheckTopAndBottom
:
    ; Store Link's X as the major coordinate in [01],
    ; and Link's Y as the minor coordinate [00].
    LDA ObjY
    STA $00
    LDA ObjX
    STA $01

    ; Pass "up" as decreasing direction along wall in [02].
    LDA #$08
    STA $02
    LDY #$20                    ; Link's minimum major coordinate (X at left wall)
    LDA #$00                    ; Offset 0 for first horizontal wall element in instruction block
    JSR Wallmaster_CalcStartPosition

    ; Set Y coordinate to initial minor value returned.
    LDA $04
    STA ObjY, X

    ; Look up and set X coordinate according to the index returned.
    LDA WallmasterInitialXs, Y
    JMP @SetUpToEmerge

@CheckTopAndBottom:
    ; Check whether Wallmaster should come out of the top or bottom wall.
    ;
    ; If Link's Y <> $5D nor $BD, then return.
    LDA ObjY
    CMP #$5D
    BEQ :+
    CMP #$BD
    BNE @Exit
:
    ; Store Link's Y as the major coordinate in [01],
    ; and Link's X as the minor coordinate [00].
    LDA ObjX
    STA $00
    LDA ObjY
    STA $01

    ; Pass "left" as decreasing direction along wall in [02].
    LDA #$02
    STA $02
    LDY #$5D                    ; Link's minimum major coordinate (Y at top wall)
    LDA #$20                    ; Offset $20 for first vertical wall element in instruction block
    JSR Wallmaster_CalcStartPosition

    ; Look up and set Y coordinate according to the index returned.
    LDA WallmasterInitialYs, Y
    STA ObjY, X

    ; Set X coordinate to initial minor value returned.
    LDA $04

@SetUpToEmerge:
    STA ObjX, X

    ; Look up and set the facing direction for the first step.
    LDY Wallmaster_ObjStep, X
    LDA WallmasterDirsAndAttrsLeft, Y
    AND #$0F
    STA ObjDir, X

    ; Wait $60 frames for the next wallmaster. Set this monster's
    ; QSpeed to $18, and animation counter to 8.
    LDA #$60
    STA ObjTimer+1
    LDA #$18
    STA ObjQSpeedFrac, X
    LDA #$08
    STA ObjAnimCounter, X

    ; Reset grid offset, tiles crossed, and movement frame.
    LDA #$00
    STA ObjGridOffset, X
    STA Wallmaster_ObjTilesCrossed, X
    STA ObjAnimFrame, X

    ; Start Wallmaster state 1.
    INC ObjState, X
    RTS

L_Wallmaster_State1:
    ; If the monster has been shoved, then shove it and go draw.
    LDA ObjShoveDir, X
    BEQ @CheckStunned
    JSR Obj_Shove

@DrawAndCheckCollisions:
    JMP @Wallmaster_DrawAndCheckCollisions

@CheckStunned:
    ; If we have the magic clock or the monster is stunned, then
    ; go draw and check collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @DrawAndCheckCollisions
    LDA ObjDir, X
    STA $0F                     ; [0F] moving direction
    JSR MoveObject

    ; If grid offset <> $10 nor $F0 (so not square aligned), then
    ; go draw and check collisions.
    LDA ObjGridOffset, X
    CMP #$10
    BEQ :+
    CMP #$F0
    BNE @Wallmaster_DrawAndCheckCollisions
:
    ; Truncate grid offset.
    LDA #$00
    STA ObjGridOffset, X

    ; Advance the step.
    INC Wallmaster_ObjStep, X

    ; Set facing direction to the one for this new step.
    LDY Wallmaster_ObjStep, X
    LDA WallmasterDirsAndAttrsLeft, Y
    AND #$0F
    STA ObjDir, X

    ; Increase the count of tiles crossed.
    INC Wallmaster_ObjTilesCrossed, X

    ; If the monster has crossed less than 7 tiles, then
    ; go draw and check collisions.
    LDA Wallmaster_ObjTilesCrossed, X
    CMP #$07
    BCC @Wallmaster_DrawAndCheckCollisions

    ; It has reached the end of a trip.
    ;
    ; If Wallmaster did not capture Link, then go set state to 0 and return.
    LDA ObjCaptureTimer, X
    BEQ @SetState

    ; At the end of a trip where Link was captured:
    ; 1. hide the special sprites used by Wallmaster
    ; 2. go to mode 3 "unfurl" to go the level's entrance
    ; 3. reset Link's state, so he's idle
    JSR HideSpritesOverLink
    LDA #$03
    STA GameMode
    LDA #$00
    STA ObjState
    STA IsUpdatingMode
    STA GameSubmode

@SetState:
    STA ObjState, X
    RTS

@Wallmaster_DrawAndCheckCollisions:
    ; If Link was captured, then go draw Link and Wallmaster specially.
    LDA ObjCaptureTimer, X
    BNE @DrawWithCapturedLink
    JSR CheckMonsterCollisions

    ; If after checking object collisions, Link was captured, then
    ; halt Link and keep him from being shoved.
    LDA ObjCaptureTimer, X
    BEQ :+
    LDA #$40
    STA ObjState
    LDA #$00
    STA ObjShoveDir
:
    ; Remember the index of the first sprite that will be used to draw,
    ; because the drawing routine will cycle the sprites.
    LDA RollingSpriteIndex
    PHA                         ; Save the sprite index
    JSR Wallmaster_PrepareToDraw
    LDA ObjAnimFrame, X
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the sprite index

    ; Store the offset of the left sprite record in [00],
    ; and the right one in [01].
    TAY
    LDA SpriteOffsets, Y
    STA $00
    LDA SpriteOffsets+1, Y

@PatchSprites:
    STA $01
    JSR Wallmaster_PutSpritesBehindBgIfNeeded

    ; The default image is the hand open.
    ; If animation frame = 0, then return.
    LDX CurObjIndex
    LDA ObjAnimFrame, X
    BEQ @Exit

    ; Animation frame 1 uses Wallmaster frame image 1, which
    ; uses tiles $9C/$9D on the left and $9E/$9F on the right.
    ; The right side is correctly the right side of a closed hand.
    ; But the left side is Keese. So, we have to patch the left
    ; side to use the common left side of the hand -- $AC/$AD.
    ;
    ; Keep in mind that left and right sprites might be flipped.
    LDY $00
    LDA Sprites+1, Y
    CMP #$9C
    BEQ :+
    LDY $01
:
    LDA #$AC
    STA Sprites+1, Y

@Exit:
    RTS

@DrawWithCapturedLink:
    ; Draw Wallmaster with captured Link.
    ;
    ; First, put Link right where the monster is.
    LDA ObjX, X
    STA ObjX
    LDA ObjY, X
    STA ObjY
    JSR Link_EndMoveAndAnimate_Bank4
    JSR ShowLinkSpritesBehindHorizontalDoors
    LDX CurObjIndex
    JSR Wallmaster_PrepareToDraw

    ; Force animation frame 1: hand closed.
    LDA #$01
    STA ObjAnimFrame, X
    JSR DrawObjectNotMirroredOverLink

    ; Store the offsets of the sprites to show over Link
    ; ($10 and $11) in [00] and [01] for use in patching the right sprite.
    LDA #$40
    STA $00
    LDA #$44

    ; Go patch up the sprites.
    JMP @PatchSprites

Wallmaster_PrepareToDraw:
    LDA #$08                    ; 8 screen frames an animation frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; For the current step, look up the sprite flipping attributes.
    LDY Wallmaster_ObjStep, X
    LDA WallmasterDirsAndAttrsLeft, Y
    AND #$F0
    ORA #$01                    ; Combine them with palette row 5 (blue)
    JSR Anim_SetSpriteDescriptorAttributes

    ; If the sprite attributes for this step have horizontal flipping ($40),
    ; then mask it off, and turn on horizontal flipping with [0F].
    AND #$40
    BEQ :+
    LDA $04
    AND #$8F
    JSR Anim_SetSpriteDescriptorAttributes
    INC $0F
:
    RTS

; Params:
; A: offset of first instruction for left wall (0)
;    or top wall ($20)
; Y: Link's minimum major coordinate
; [00]: Link's coordinate along minor axis
; [01]: Link's coordinate along major axis
; [02]: a decreasing direction along the wall (left or up)
;
; Returns:
; Y: index of initial major coordinate of Wallmaster
; [04]: initial minor coordinate of Wallmaster
;
; Store the initial offset for this axis.
;
Wallmaster_CalcStartPosition:
    STA Wallmaster_ObjStep, X
    STY $03                     ; [03] holds the Link's minimum major coordinate.

    ; If Link is still (input dir = 0), then use distance $24, else $32.
    LDA #$24
    LDY ObjInputDir
    BEQ :+
    LDA #$32
:
    ; If Link is facing in the direction passed in (which decreases), then:
    ; 1. negate the distance
    ; 2. add 8 to the instruction offset to access
    ;    the opposite direction for the same wall
    LDY ObjDir
    CPY $02                     ; [02] reference decreasing direction
    BNE :+
    PHA
    LDA Wallmaster_ObjStep, X
    CLC
    ADC #$08
    STA Wallmaster_ObjStep, X
    PLA
    EOR #$FF
    CLC
    ADC #$01
:
    ; Add the distance and Link's minor coordinate to calculate
    ; Wallmaster's initial minor coordinate to store in [04].
    ; In general, the monster will be put in front of Link.
    CLC
    ADC $00                     ; [00] Link's minor coordinate
    STA $04                     ; [04] Wallmaster's initial minor coordinate

    ; By default, will return index 0 to choose the minimum value
    ; for Wallmaster's initial major coordinate. This will be returned,
    ; if Link is at the wall with a smaller major coordinate.
    LDY #$00

    ; If Link's major coordinate <> the minimum major coordinate, then
    ; he's on the farther wall. Add $10 to the instruction offset 
    ; to access the block for the other direction along the same axis,
    ; and increment the index for the Wallmaster's initial major coordinate.
    LDA $01
    CMP $03                     ; [03] Link's minimum major coordinate
    BEQ :+
    LDA Wallmaster_ObjStep, X
    CLC
    ADC #$10
    STA Wallmaster_ObjStep, X
    INY
:
    RTS

Wallmaster_PutSpritesBehindBgIfNeeded:
    LDY $00
    JSR Wallmaster_PutSpriteBehindBgIfNeeded
    LDY $01

Wallmaster_PutSpriteBehindBgIfNeeded:
    ; Loop twice, from 1 to 0, in order to check the left and right
    ; extents of the sprite (X and X+8).
    LDX #$01

@LoopExtent:
    ; If sprite X + extent >= $E9 or < $18, then
    ; add priority attribute to show this sprite behind the background.
    LDA Sprites+3, Y
    CLC
    ADC SpriteRelativeExtents, X
    CMP #$E9
    BCS @AddPriorityBit
    CMP #$18
    BCS :+

@AddPriorityBit:
    LDA Sprites+2, Y
    ORA #$20
    STA Sprites+2, Y
:
    DEX
    BPL @LoopExtent
    RTS
