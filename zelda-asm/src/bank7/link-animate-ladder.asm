; ============================================================================
; FILE:     bank7/link-animate-ladder.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 3163-3448
; SUBJECTS: Link
; TOPIC:    Ladder tables, Link end-of-move: ladder, warps, drawing, shield
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 LinkToLadderOffsetsX                data
;   L0027 LinkToLadderOffsetsY                data
;   L0030 LinkHeadTiles                       data
;   L0033 LinkHeadMagicShieldTiles            data
;   L0036 LadderRoomsOW                       data
;   L0039 Link_EndMoveAndAnimate_Bank4        code
;   L0044 Link_EndMoveAndDraw_Bank1           code
;   L0050 Link_EndMoveAndAnimate_Bank1        code
;   L0054 Link_EndMoveAndDraw_Bank4           code
;   L0062 Link_EndMoveAndDraw                 code  Description:
;   L0067 Link_EndMoveAndAnimateBetweenRooms  code
;   L0071 Link_EndMoveAndAnimate              code
;   L0308 L1F36A_Exit                         code
; ============================================================================

LinkToLadderOffsetsX:
    .BYTE $00, $00, $F0, $10

LinkToLadderOffsetsY:
    .BYTE $FB, $13, $03, $03

LinkHeadTiles:
    .BYTE $02, $06, $08, $0A

LinkHeadMagicShieldTiles:
    .BYTE $80, $54, $60, $60

LadderRoomsOW:
    .BYTE $17, $18, $19, $27, $4F, $5F

Link_EndMoveAndAnimate_Bank4:
    JSR Link_EndMoveAndAnimate
    LDA #$04
    JMP SwitchBank

Link_EndMoveAndDraw_Bank1:
    JSR Link_EndMoveAndDraw
:
    LDA #$01
    JMP SwitchBank

Link_EndMoveAndAnimate_Bank1:
    JSR Link_EndMoveAndAnimate
    JMP :-

Link_EndMoveAndDraw_Bank4:
    JSR Link_EndMoveAndDraw
    LDA #$04
    JMP SwitchBank

; Description:
; Draw Link without animating by keeping the animation counter fixed.
;
Link_EndMoveAndDraw:
    LDA #$06
    STA ObjAnimCounter
    BNE Link_EndMoveAndAnimate

Link_EndMoveAndAnimateBetweenRooms:
    LDA CurLevel
    BNE L1F1FC_Exit             ; If in UW, then return.

Link_EndMoveAndAnimate:
    ; Switches bank: 5
    ;
    ; If teleporting by whirlwind, then return.
    LDA WhirlwindTeleportingState
    BNE L1F1FC_Exit

    ; If mode 4 or 6 (not a playing mode), then go check warps and animate.
    TAX
    LDA GameMode
    CMP #$06
    BEQ @CheckWarpsAndAnimate
    CMP #$05
    BCC @CheckWarpsAndAnimate

    ; If Link's grid offset = 0, go see whether we need to wield the ladder.
    ; If Link's grid offset is not a multiple of 8, go check warps and animate.
    LDA ObjGridOffset
    BEQ @CheckLadderRoom
    AND #$07
    BEQ @TruncGridOffset

@CheckWarpsAndAnimate:
    JMP @CheckWarps

@TruncGridOffset:
    ; Grid offset is a multiple of 8. So set it to 0.
    ; If not in mode 5, go check warps and animate.
    LDA #$00
    STA ObjGridOffset
    LDY GameMode
    CPY #$05
    BNE @CheckWarpsAndAnimate

    ; Grid offset was a multiple of 8. So, we've stepped a whole tile,
    ; and need to reset the subroom indicator.
    ;
    ; That way, we know not to go into a subroom that we just came out of.
    STA UndergroundExitType

@CheckLadderRoom:
    ; If not in mode 5, go check warps and animate.
    LDA GameMode
    CMP #$05
    BNE @CheckWarpsAndAnimate

    ; If in OW and we're not in this list of 6 rooms, go check warps and animate.
    LDA CurLevel
    BNE @CheckLadder
    LDA RoomId
    LDY #$05

@LoopLadderRoom:
    CMP LadderRoomsOW, Y
    BEQ @CheckLadder
    DEY
    BPL @LoopLadderRoom
    BMI @CheckWarps

@CheckLadder:
    ; If in a doorway or missing the ladder, go check warps and animate.
    LDA DoorwayDir
    BNE @CheckWarps
    LDA InvLadder
    BEQ @CheckWarps

    ; If Link is halted or using the ladder, go check warps and animate.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ @CheckWarps
    LDA LadderSlot
    BNE @CheckWarps

    ; Get the colliding tile in moving direction.
    LDX #$00
    LDA ObjDir
    STA $0F
    JSR GetCollidingTileMoving

    ; If in OW and tile is not water (as in < $8D or >= $99), then
    ; go check warps and animate.
    ; If in UW, and tile <> $F4, go check warps and animate.
    LDY CurLevel
    BEQ @CheckWaterOW
    CMP #$F4
    BEQ @SetUpLadder
    BNE @CheckWarps

@CheckWaterOW:
    CMP #$8D
    BCC @CheckWarps
    CMP #$99
    BCS @CheckWarps

@SetUpLadder:
    ; Look for an empty monster slot.
    ; If none found, or input dir = 0, or input direction <> facing direction,
    ; then go check warps and animate.
    JSR FindEmptyMonsterSlot
    BEQ @CheckWarps
    LDA ObjInputDir
    BEQ @CheckWarps
    LDX EmptyMonsterSlot
    CMP ObjDir
    BNE @CheckWarps

    ; X register now holds the ladder's slot.
    ; Set ladder slot to empty slot found.
    ; Set ladder's direction to input direction.
    STX LadderSlot
    STA ObjDir, X

    ; Add appropriate offset to player's position to set the ladder's position.
    JSR GetOppositeDir          ; Call this for the reverse direction index.
    LDA ObjX
    CLC
    ADC LinkToLadderOffsetsX, Y
    STA ObjX, X
    LDA ObjY
    CLC
    ADC LinkToLadderOffsetsY, Y
    STA ObjY, X

    ; Set the ladder object's type. Reset shove params and invincibility timer.
    LDA #$5F                    ; Ladder
    STA ObjType, X
    JSR ResetShoveInfo
    STA ObjInvincibilityTimer, X

    ; Set the initial state of the ladder.
    LDA #$01
    STA ObjState, X

@CheckWarps:
    ; If in mode 5, check warps.
    LDX #$00                    ; Set X to 0 to refer to Link object.
    LDA GameMode
    CMP #$05
    BNE @Animate
    LDA ObjCollidedTile         ; Save the last collided tile.
    PHA
    LDA #$05
    JSR SwitchBank
    JSR CheckWarps
    LDX #$00                    ; Set X to 0 to refer to Link object.
    PLA
    STA ObjCollidedTile         ; Restore the last collided tile.

@Animate:
    ; Animate Link.
    JSR AnimateLinkBase
    LDA GameMode
    CMP #$09
    BEQ @ShiftDown2Pixels       ; If in mode 9, go draw Link 2 pixels down.
    LDA CurLevel                ; In OW, draw Link 2 pixels down.
    BNE @ChangeLookForState

@ShiftDown2Pixels:
    INC $01
    INC $01

@ChangeLookForState:
    ; Now change Link's look for specific states and items.
    ;
    ; If Link is attacking or using an item,
    ; then add 4 to frame to use the "attack/use item" frames.
    LDA ObjState
    AND #$30
    CMP #$10
    BEQ @WieldingObject
    CMP #$20
    BNE @Draw

@WieldingObject:
    TYA
    CLC
    ADC #$04
    TAY

@Draw:
    TYA                         ; Now frame is in A.
    LDY #$00                    ; Use Link's animation.
    JSR DrawObjectWithAnim

    ; See if there's special processing needed for the shield.
    LDA InvMagicShield
    BNE @HandleMagicShield

    ; No magic shield.
    LDA ObjDir
    CMP #$04
    BNE L1F36A_Exit             ; If not facing down, then return.
    LDX #$01                    ; Get the sprite's tile.
    LDA Sprites+72, X
    CMP #$0B                    ; The tiles for shieldless Link are 8 and $A.
    BCS L1F36A_Exit             ; If it already has a shield, then return.
    PHA                         ; Save the sprite's original tile.
    CLC
    ADC #$50                    ; Add $50 to change 8 to $58 for one frame, and $A to $5A for the other frame.
    JMP @ApplyShieldSprite      ; Go apply the change.

@HandleMagicShield:
    ; Magic shield.
    LDX #$01                    ; Look at the left tile.
    LDA ObjDir
    LSR
    BCC :+                      ; If facing right,
    LDX #$05                    ; then look at the right tile instead.
:
    LDY #$04
    LDA Sprites+72, X           ; Get the sprite's tile.
    PHA                         ; Save the sprite's original tile.

@LoopHeadTile:
    ; Compare the sprite's tile with 4 tiles: 1 for each direction.
    DEY
    BMI @FixHFlip               ; Quit the loop if we haven't found it.
    CMP LinkHeadTiles, Y
    BNE @LoopHeadTile           ; If it didn't match, go check the next one.
    LDA LinkHeadMagicShieldTiles, Y    ; Get the replacement tile.

@ApplyShieldSprite:
    ; Apply the sprite's modification for the shield.
    STA Sprites+72, X

@FixHFlip:
    PLA                         ; Restore the sprite's original tile to A.
    CMP #$0A
    BNE L1F36A_Exit             ; If it's not a down facing tile, then return. No attributes to change.

    ; It's a down facing tile. Make sure it's not flipped,
    ; because there's only 1 frame of downward shield tile.
    LDA Sprites+73, X
    AND #$0F
    STA Sprites+73, X

L1F36A_Exit:
    RTS
