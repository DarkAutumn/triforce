; ============================================================================
; FILE:     bank4/wallmaster.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 4106-4535
; SUBJECTS: Wallmaster
; TOPIC:    Wallmaster (type $27): the step tables of its trips out of a wall and back, the wait state
;           that picks where it emerges, the trip state that grabs Link and returns him to the level's
;           start room, and its drawing.
; SEE ALSO: bank1/monster-collisions.asm (CheckMonsterCollisions sets ObjCaptureTimer on a touch);
;           bank1/link-collision-harm.asm (when a touch counts); bank7/walker-move.asm (MoveObject);
;           bank7/init-mode-dispatch.asm (InitMode3_Sub1: mode 3 puts Link in the level's start room);
;           bank4/push-block.asm (HideSpritesOverLink, DrawObjectNotMirroredOverLink).
; KEY FACTS:
;   - State 0 (in the wall, not drawn, no collisions) ends only when Link stands exactly against a
;     wall: X = $20 or $D0 with Y in $6D-$B4, or Y = $5D or $BD with X in $29-$C7. UpdateWallmaster.
;   - All Wallmasters share ObjTimer+1: one emerging sets it to $60, so the next can emerge no sooner
;     than $60 frames later. None emerges while Link is halted (ObjState $40). UpdateWallmaster.
;   - It emerges from the wall Link touches, $24 pixels along the wall from Link if Link holds no
;     direction, $32 if he does; on the side he faces if he faces along the wall, else on the
;     increasing-coordinate side (below or right). Wallmaster_CalcStartPosition.
;   - A trip is 7 squares at q-speed $18 (0.375 px/frame, about 299 frames): 2 squares out of the
;     wall (onto the line Link stands on), 3 squares ($30 pixels) along it toward Link, 2 back in.
;     Started $24 away it ends $0C past Link's position; started $32 away, 2 short of it.
;     WallmasterDirsAndAttrsLeft, L_Wallmaster_State1.
;   - Touching Link during a trip (a touch that harms him, found by CheckMonsterCollisions) captures
;     him: Link is halted, drawn in the hand and carried along. Not while the Wallmaster's own
;     ObjInvincibilityTimer runs (the collision routine then returns early). L_Wallmaster_State1.
;   - At the end of a trip with Link captured, GameMode becomes 3 (submode 0), which restarts the
;     level at its start room. Without a capture the Wallmaster returns to state 0. L_Wallmaster_State1.
;   - While carrying Link it runs no collision checks, so Link's weapons cannot hit it.
;     L_Wallmaster_State1.
;   - The clock or a stun stops the trip; a knockback replaces that frame's step. L_Wallmaster_State1.
;   - A sprite of it goes behind the background when its X or X + 8 is below $18 or at least $E9,
;     which hides it inside the side walls. Wallmaster_PutSpriteBehindBgIfNeeded.
; CONTENTS: (generated; L = line in this file)
;   L0055 WallmasterDirsAndAttrsLeft             data  TABLE: Wallmaster steps from the left wall; index = step; value = attributes (high), direction (low).
;   L0060 WallmasterDirsAndAttrsRight            data  TABLE: Wallmaster steps from the right wall; index = step - $10; value as above.
;   L0065 WallmasterDirsAndAttrsTop              data  TABLE: Wallmaster steps from the top wall; index = step - $20; value as above.
;   L0070 WallmasterDirsAndAttrsBottom           data  TABLE: Wallmaster steps from the bottom wall; index = step - $30; value as above.
;   L0075 WallmasterInitialXs                    data  TABLE: Wallmaster start X inside a side wall; index = 0 left wall, 1 right wall; value = X (pixels).
;   L0079 WallmasterInitialYs                    data  TABLE: Wallmaster start Y inside the top or bottom wall; index = 0 top, 1 bottom; value = Y (pixels).
;   L0091 UpdateWallmaster                       code  Updates a Wallmaster (type $27): in state 0 waits for Link to touch a wall, then starts a trip.
;   L0231 L_Wallmaster_State1                    code  Wallmaster state 1: makes the 7-square trip, grabs Link on touch, and at its end takes him away.
;   L0398 Wallmaster_PrepareToDraw               code  Steps the Wallmaster's animation (8 frames per image) and sets its position and attributes.
;   L0434 Wallmaster_CalcStartPosition           code  Picks the Wallmaster's trip (first step index) and where along the wall it starts.
;   L0490 Wallmaster_PutSpritesBehindBgIfNeeded  code  Puts each of the Wallmaster's two sprites behind the background if it is near a screen side.
;   L0500 Wallmaster_PutSpriteBehindBgIfNeeded   code  Sets a sprite's behind-background bit if its X or X + 8 is below $18 or at least $E9.
; ============================================================================

; The four tables below are one array of $40 bytes indexed by Wallmaster_ObjStep. Each row of 8 is
; one trip: low nibble = direction of that step (one square, $10 pixels), high nibble = sprite
; attributes ($80 vertical flip, $40 horizontal flip). A trip uses entries 0-6; entry 7 is only
; loaded as the facing direction when the trip ends.

; TABLE: Wallmaster steps from the left wall; index = step; value = attributes (high), direction (low).
WallmasterDirsAndAttrsLeft:
    .BYTE $01, $01, $08, $08, $08, $02, $02, $02    ; $00: right 2, up 3, left 2
    .BYTE $C1, $C1, $C4, $C4, $C4, $C2, $C2, $C2    ; $08 (Link faces up): right 2, down 3, left 2

; TABLE: Wallmaster steps from the right wall; index = step - $10; value as above.
WallmasterDirsAndAttrsRight:
    .BYTE $42, $42, $48, $48, $48, $41, $41, $41    ; $10: left 2, up 3, right 2
    .BYTE $82, $82, $84, $84, $84, $81, $81, $81    ; $18 (Link faces up): left 2, down 3, right 2

; TABLE: Wallmaster steps from the top wall; index = step - $20; value as above.
WallmasterDirsAndAttrsTop:
    .BYTE $C4, $C4, $C2, $C2, $C2, $C8, $C8, $C8    ; $20: down 2, left 3, up 2
    .BYTE $84, $84, $81, $81, $81, $88, $88, $88    ; $28 (Link faces left): down 2, right 3, up 2

; TABLE: Wallmaster steps from the bottom wall; index = step - $30; value as above.
WallmasterDirsAndAttrsBottom:
    .BYTE $48, $48, $42, $42, $42, $44, $44, $44    ; $30: up 2, left 3, down 2
    .BYTE $08, $08, $01, $01, $01, $04, $04, $04    ; $38 (Link faces left): up 2, right 3, down 2

; TABLE: Wallmaster start X inside a side wall; index = 0 left wall, 1 right wall; value = X (pixels).
WallmasterInitialXs:
    .BYTE $00, $F0

; TABLE: Wallmaster start Y inside the top or bottom wall; index = 0 top, 1 bottom; value = Y (pixels).
WallmasterInitialYs:
    .BYTE $3D, $DD

; ----------------------------------------------------------------------------
; Updates a Wallmaster (type $27): in state 0 waits for Link to touch a wall, then starts a trip.
; IN:   X = object slot
; OUT:  on emerging: ObjX,X, ObjY,X, ObjDir,X, Wallmaster_ObjStep,X set; ObjState,X = 1;
;       ObjTimer+1 = $60; ObjQSpeedFrac,X = $18.
; NOTE: In state 0 it is not drawn and checks no collisions. It emerges only when ObjTimer+1 is 0
;       (shared by all Wallmasters), Link's state is not $40, and Link's X = $20 or $D0 with
;       $6D <= Y < $B5, or Link's Y = $5D or $BD with $29 <= X < $C8.
; ----------------------------------------------------------------------------
UpdateWallmaster:
    LDA ObjState, X
    BEQ :+
    JMP L_Wallmaster_State1
:
    ; State 0: waiting inside the wall.
    ;
    ; All Wallmasters share ObjTimer+1 (the timer of object slot 1) to space out their trips.
    ; If it has not run out, then return.
    LDA ObjTimer+1
    BNE @Exit

    ; If Link is halted (state $40), then return.
    LDA ObjState
    CMP #$40
    BNE @State0

@Exit:
    RTS

@State0:
    ; If  (Link's X < $29 or >= $C8)
    ; and (Link's Y < $6D or >= $B5),
    ; then return: Link is in a corner region of the room.
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
    ; Check whether the Wallmaster should come out of the left or right wall.
    ;
    ; If Link's X is neither $20 (against the left wall) nor $D0 (against the right wall),
    ; then go check the top and bottom walls.
    LDA ObjX
    CMP #$20
    BEQ :+
    CMP #$D0
    BNE @CheckTopAndBottom
:
    ; Store Link's X as the major coordinate in [01],
    ; and Link's Y as the minor coordinate (along the wall) in [00].
    LDA ObjY
    STA $00
    LDA ObjX
    STA $01

    ; Pass "up" as the decreasing direction along the wall in [02].
    LDA #$08
    STA $02
    LDY #$20                    ; Link's major coordinate at the near wall (X at the left wall)
    LDA #$00                    ; Step 0: base of the left and right wall trips
    JSR Wallmaster_CalcStartPosition

    ; Set the Y coordinate to the minor coordinate returned.
    LDA $04
    STA ObjY, X

    ; Look up the X coordinate by the wall index returned: $00 left, $F0 right.
    LDA WallmasterInitialXs, Y
    JMP @SetUpToEmerge

@CheckTopAndBottom:
    ; Check whether the Wallmaster should come out of the top or bottom wall.
    ;
    ; If Link's Y is neither $5D (against the top wall) nor $BD (against the bottom wall),
    ; then return.
    LDA ObjY
    CMP #$5D
    BEQ :+
    CMP #$BD
    BNE @Exit
:
    ; Store Link's Y as the major coordinate in [01],
    ; and Link's X as the minor coordinate (along the wall) in [00].
    LDA ObjX
    STA $00
    LDA ObjY
    STA $01

    ; Pass "left" as the decreasing direction along the wall in [02].
    LDA #$02
    STA $02
    LDY #$5D                    ; Link's major coordinate at the near wall (Y at the top wall)
    LDA #$20                    ; Step $20: base of the top and bottom wall trips
    JSR Wallmaster_CalcStartPosition

    ; Look up the Y coordinate by the wall index returned: $3D top, $DD bottom.
    LDA WallmasterInitialYs, Y
    STA ObjY, X

    ; Set the X coordinate to the minor coordinate returned.
    LDA $04

@SetUpToEmerge:
    STA ObjX, X

    ; Look up and set the facing direction for the first step.
    LDY Wallmaster_ObjStep, X
    LDA WallmasterDirsAndAttrsLeft, Y
    AND #$0F
    STA ObjDir, X

    ; No other Wallmaster may emerge for $60 frames. Set this one's q-speed to $18
    ; (0.375 px/frame), and its animation counter to 8.
    LDA #$60
    STA ObjTimer+1
    LDA #$18
    STA ObjQSpeedFrac, X
    LDA #$08
    STA ObjAnimCounter, X

    ; Reset the grid offset, squares crossed, and animation frame (0 = open hand).
    LDA #$00
    STA ObjGridOffset, X
    STA Wallmaster_ObjTilesCrossed, X
    STA ObjAnimFrame, X

    ; Go to state 1: the trip.
    INC ObjState, X
    RTS

; ----------------------------------------------------------------------------
; Wallmaster state 1: makes the 7-square trip, grabs Link on touch, and at its end takes him away.
; IN:   X = object slot
; OUT:  at the end of the trip ObjState,X = 0; with Link captured also GameMode = 3, GameSubmode = 0,
;       IsUpdatingMode = 0 and Link's ObjState = 0.
; NOTE: Capture = ObjCaptureTimer,X nonzero, set by CheckMonsterCollisions when the touch harms
;       Link. From then on Link is halted and moved with the hand, and no collisions are checked.
;       Mode 3 restarts the level at its start room. The clock or a stun freezes the trip.
;       No capture while the Wallmaster's ObjInvincibilityTimer is nonzero (just hit by a weapon):
;       CheckMonsterCollisions then returns before testing Link.
; ----------------------------------------------------------------------------
L_Wallmaster_State1:
    ; If the monster is being knocked back, then move it that way and go draw.
    LDA ObjShoveDir, X
    BEQ @CheckStunned
    JSR Obj_Shove

@DrawAndCheckCollisions:
    JMP @Wallmaster_DrawAndCheckCollisions

@CheckStunned:
    ; With the clock, or while stunned, don't move:
    ; go draw and check collisions.
    LDA InvClock
    ORA ObjStunTimer, X
    BNE @DrawAndCheckCollisions
    LDA ObjDir, X
    STA $0F                     ; [0F] moving direction
    JSR MoveObject

    ; If grid offset is neither $10 nor -$10 (a whole square has not been crossed), then
    ; go draw and check collisions.
    LDA ObjGridOffset, X
    CMP #$10
    BEQ :+
    CMP #$F0
    BNE @Wallmaster_DrawAndCheckCollisions
:
    ; A square was crossed. Reset the grid offset.
    LDA #$00
    STA ObjGridOffset, X

    ; Advance the step.
    INC Wallmaster_ObjStep, X

    ; Set the facing direction to the one for this new step.
    LDY Wallmaster_ObjStep, X
    LDA WallmasterDirsAndAttrsLeft, Y
    AND #$0F
    STA ObjDir, X

    ; Increase the count of squares crossed.
    INC Wallmaster_ObjTilesCrossed, X

    ; If the monster has crossed fewer than 7 squares, then
    ; go draw and check collisions.
    LDA Wallmaster_ObjTilesCrossed, X
    CMP #$07
    BCC @Wallmaster_DrawAndCheckCollisions

    ; It has reached the end of a trip (it is back inside the wall).
    ;
    ; If the Wallmaster did not capture Link, then go set state 0 (A = 0) and return.
    LDA ObjCaptureTimer, X
    BEQ @SetState

    ; At the end of a trip where Link was captured:
    ; 1. hide the two sprites that drew the hand over Link
    ; 2. go to mode 3 "unfurl", whose init puts Link in the level's start room
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
    ; If Link was captured, then go draw Link and the Wallmaster specially.
    ; No collisions are checked from then on.
    LDA ObjCaptureTimer, X
    BNE @DrawWithCapturedLink
    JSR CheckMonsterCollisions

    ; If the collision check just captured Link, then
    ; halt Link (state $40) and cancel the knockback that the touch started.
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

    ; The default image (animation frame 0) is the open hand.
    ; If animation frame = 0, then return.
    LDX CurObjIndex             ; The call above used X.
    LDA ObjAnimFrame, X
    BEQ @Exit

    ; Animation frame 1 uses Wallmaster frame image 1, which
    ; uses tiles $9C/$9D on the left and $9E/$9F on the right.
    ; The right side is correctly the right side of a closed hand.
    ; But the left side is another monster's tile (which one: unverified). So the sprite that got
    ; tile $9C is patched
    ; to the left side that both hand images share: $AC/$AD.
    ;
    ; The left and right sprites might be swapped by flipping: if the first record
    ; does not hold tile $9C, the second is patched without a test.
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
    ; Draw the Wallmaster with captured Link.
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

    ; The hand was drawn with sprite records $10 and $11, which lie in front of Link's.
    ; Store their offsets ($40 and $44) in [00] and [01] for the patching code.
    LDA #$40
    STA $00
    LDA #$44

    ; Go patch up the sprites.
    JMP @PatchSprites

; ----------------------------------------------------------------------------
; Steps the Wallmaster's animation (8 frames per image) and sets its position and attributes.
; IN:   X = object slot
; OUT:  [00], [01] = X, Y; [04], [05] = sprite attributes of the current step with sprite palette 1;
;       [0F] = 1 when the step is flipped horizontally (the $40 bit is then cleared in [04], [05]).
; ----------------------------------------------------------------------------
Wallmaster_PrepareToDraw:
    LDA #$08                    ; 8 frames per animation frame
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; For the current step, look up the sprite flipping attributes (high nibble).
    LDY Wallmaster_ObjStep, X
    LDA WallmasterDirsAndAttrsLeft, Y
    AND #$F0
    ORA #$01                    ; Combine them with sprite attributes 1: sprite palette 1 (row 5)
    JSR Anim_SetSpriteDescriptorAttributes

    ; If the sprite attributes for this step have horizontal flipping ($40),
    ; then mask it off, and ask for horizontal flipping with [0F] instead
    ; (the call above leaves A unchanged).
    AND #$40
    BEQ :+
    LDA $04
    AND #$8F
    JSR Anim_SetSpriteDescriptorAttributes
    INC $0F
:
    RTS

; ----------------------------------------------------------------------------
; Picks the Wallmaster's trip (first step index) and where along the wall it starts.
; IN:   X = object slot; A = base step (0 left/right walls, $20 top/bottom walls);
;       Y = Link's major coordinate at the near wall ($20 left, $5D top);
;       [00] = Link's coordinate along the wall (minor); [01] = Link's major coordinate;
;       [02] = the decreasing direction along the wall (8 up, or 2 left)
; OUT:  Y = 0 near wall (left or top), 1 far wall (right or bottom);
;       [04] = the Wallmaster's start coordinate along the wall;
;       Wallmaster_ObjStep,X = base, +8 if Link faces direction [02], +$10 at the far wall.
; NOTE: The start is $24 pixels from Link along the wall when Link holds no direction
;       (ObjInputDir = 0), else $32: on the decreasing side if Link faces that way, otherwise on the
;       increasing side. The trip then runs back toward Link.
; ----------------------------------------------------------------------------
Wallmaster_CalcStartPosition:
    STA Wallmaster_ObjStep, X
    STY $03                     ; [03] holds Link's major coordinate at the near wall.

    ; If Link holds no direction (input dir = 0), then use distance $24, else $32.
    LDA #$24
    LDY ObjInputDir
    BEQ :+
    LDA #$32
:
    ; If Link is facing in the direction passed in (which decreases), then:
    ; 1. negate the distance
    ; 2. add 8 to the step index to use the trip that runs
    ;    the other way along the same wall
    LDY ObjDir
    CPY $02                     ; [02] reference decreasing direction
    BNE @AddMinor
    PHA
    LDA Wallmaster_ObjStep, X
    CLC
    ADC #$08
    STA Wallmaster_ObjStep, X
    PLA
    EOR #$FF
    CLC
    ADC #$01
@AddMinor:
    ; Add the signed distance to Link's minor coordinate to get the
    ; Wallmaster's start coordinate along the wall, and store it in [04].
    CLC
    ADC $00                     ; [00] Link's minor coordinate
    STA $04                     ; [04] Wallmaster's start minor coordinate

    ; By default, return index 0: the near wall (left or top),
    ; where Link's major coordinate is the smaller one.
    LDY #$00

    ; If Link's major coordinate <> the near-wall value, then
    ; he's at the far wall (right or bottom). Add $10 to the step index
    ; to use that wall's trips, and return index 1.
    LDA $01
    CMP $03                     ; [03] Link's major coordinate at the near wall
    BEQ :+
    LDA Wallmaster_ObjStep, X
    CLC
    ADC #$10
    STA Wallmaster_ObjStep, X
    INY
:
    RTS

; ----------------------------------------------------------------------------
; Puts each of the Wallmaster's two sprites behind the background if it is near a screen side.
; IN:   [00], [01] = offsets of the two sprite records in Sprites
; OUT:  X = $FF (the object slot is lost)
; ----------------------------------------------------------------------------
Wallmaster_PutSpritesBehindBgIfNeeded:
    LDY $00
    JSR Wallmaster_PutSpriteBehindBgIfNeeded
    LDY $01

; ----------------------------------------------------------------------------
; Sets a sprite's behind-background bit if its X or X + 8 is below $18 or at least $E9.
; IN:   Y = offset of the sprite record in Sprites
; OUT:  X = $FF
; ----------------------------------------------------------------------------
Wallmaster_PutSpriteBehindBgIfNeeded:
    ; Loop twice, X = 1 then 0, to test the left and right
    ; extents of the sprite (X and X + 8; SpriteRelativeExtents).
    LDX #$01

@LoopExtent:
    ; If sprite X + extent >= $E9 or < $18, then
    ; set the priority bit ($20) to show this sprite behind the background.
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
