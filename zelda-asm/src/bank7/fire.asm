; ============================================================================
; FILE:     bank7/fire.asm               BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 4681-4804
; SUBJECTS: Fire, Obj
; TOPIC:    Link's fire (candle flame, or the fire the magic book leaves where a magic shot ends) in object
;           slots $10 and $11: drifting, standing, lighting a dark room, burning out, and hurting Link.
; SEE ALSO: bank1/bomb-candle-wield.asm (Candle_Wield: slot search, blue candle rule, initial values);
;           bank7/bomb.asm (Bomb_UpdateBombOrFire sends states $2x here; bombs share the two slots);
;           bank7/sword-shot-magic-shot.asm (SwordShot_HandleBlocked: the book's fire, state $22 for $4F frames);
;           bank4/shoot-helpers-candle.asm (Candle_UpdateRoomLight: dark room fade); bank4/rock-wall-tree-secrets.asm
;           (Tree_Update: burning a tree); bank1/sword-bomb-arrow-collision.asm
;           (Bomb_CheckBombOrFireVsMonster: fire against monsters); bank1/shove-knockback.asm (Shove_Begin);
;           bank1/link-collision-harm.asm (Link_BeHarmed).
; KEY FACTS:
;   - State $21 = moving fire: goes in the direction Link faced at q-speed $20 (0.5 pixel per frame)
;     until it has gone $10 pixels (31-32 frames). State $22 = standing fire. Fire_Update
;   - A candle fire stands for $3F frames (ObjTimer), then its slot is freed; start to end about 95
;     frames. The book's fire starts in state $22 with $4F frames. Fire_Update, SwordShot_HandleBlocked
;   - At most two fires, or one fire and one bomb: Candle_Wield needs slot $10 or $11 empty.
;   - Blue candle (InvCandle = 1): refused once HasUsedCandle is set, so one fire per room visit; the red
;     candle is limited only by the two slots. Candle_Wield
;   - A dark dungeon room starts to light up only when a fire is standing (state $22), never while it
;     still moves and never on the overworld (CurLevel = 0). Fire_Update@StandingFire, Candle_UpdateRoomLight
;   - The fire hurts Link in both states: within $0E pixels in X and Y (midpoints) and with Link's
;     invincibility timer at 0, he is knocked back and loses half a heart ($0080; a quarter with the
;     blue ring, an eighth with the red). Fire_Update@DrawAndCheckCollisions
;   - A tree burns only when a standing fire with ObjTimer < 2 (its last frames) is within $10 pixels of
;     it. Tree_Update (bank4/rock-wall-tree-secrets.asm)
;   - Against monsters a fire in either state deals $10 damage of type $20 within $0E pixels.
;     Bomb_CheckBombOrFireVsMonster (bank1/sword-bomb-arrow-collision.asm)
; CONTENTS: (generated; L = line in this file)
;   L0044 Fire_Update         code  Updates one fire of Link's (candle or book): moves it, burns it out, lights the room, hurts Link.
;   L0152 Obj_GetWideMiddle   code  Stores the midpoint of a 16-pixel-wide object: X + 8, Y + 8.
;   L0163 Fire_Update_Return  code  Shared return of Fire_Update and Obj_GetWideMiddle.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates one fire of Link's (candle or book): moves it, burns it out, lights the room, hurts Link.
; IN:   X = object slot $10 or $11; ObjState,X = $21 (moving) or $22 (standing).
; OUT:  ObjState,X = 0 when the standing fire's ObjTimer has run out.
; NOTE: Moving: 0.5 pixel per frame for $10 pixels, then standing for $3F frames. The dark room
;       brightens only from the standing state. Link is hurt by his own fire in both states.
; ----------------------------------------------------------------------------
Fire_Update:
    ; Only state $21 moves.
    LDA ObjState, X
    CMP #$21
    BNE @StandingFire

    ; Move in ObjDir with the grid offset at 0, so that Obj_MoveByQSpeed's limit ($10 from a grid
    ; point) never stops it. The q-speed is $20 (Candle_Wield): 1 pixel every second frame.
    LDA ObjGridOffset, X
    PHA
    LDA #$00
    STA ObjGridOffset, X
    LDA ObjDir, X
    STA $0F
    JSR Obj_MoveByQSpeed
    PLA

    ; Add this frame's movement to the saved grid offset: it becomes the signed distance gone.
    CLC
    ADC ObjGridOffset, X
    STA ObjGridOffset, X

    ; While the distance is not $10 pixels, go draw.
    ; At $10 pixels the fire stops: state $22 with ObjTimer = $3F frames.
    JSR Math_Abs
    CMP #$10
    BNE @DrawAndCheckCollisions
    LDA #$3F
    STA ObjTimer, X
    INC ObjState, X

@StandingFire:
    ; Standing fire. When its timer has run out, free the slot (state 0) and return.
    LDA ObjTimer, X
    BEQ Obj_ResetState

    ; In a dungeon (CurLevel <> 0) run the room lighting: if the room is dark and not lit yet,
    ; Candle_UpdateRoomLight starts the fade to light and finishes it over the next frames.
    LDA CurLevel
    BEQ @DrawAndCheckCollisions
    TXA
    PHA
    LDA #$04
    JSR Mmc1_SwitchBank
    JSR Candle_UpdateRoomLight
    PLA
    TAX

@DrawAndCheckCollisions:
    ; Draw: the image flips horizontally every 4 frames (ObjAnimFrame), sprite palette row 6.
    LDA #$04                    ; The animation counter restarts at 4 frames.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    JSR Anim_SetObjHFlipForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorRedPaletteRow
    LDA #$00                    ; A: frame 0
    STA $0C                     ; [0C]: not mirrored
    LDY #$40                    ; Y: object type $40 (standing fire); its animation index is $41
    JSR Anim_DrawObjectWithType

    ; Collision with Link. If he is invincible (timer <> 0, which the clock keeps so), return.
    LDA ObjInvincibilityTimer
    BNE Fire_Update_Return

    ; [00] = the fire's object slot.
    STX $00

    ; [04], [05] = Link's midpoint (X + 8, Y + 8).
    LDX #$00
    LDY #$02
    JSR Obj_GetWideMiddle

    ; [02], [03] = the fire's midpoint (X + 8, Y + 8).
    LDX $00
    LDY #$00
    JSR Obj_GetWideMiddle

    ; They touch when the midpoints are less than $0E pixels apart in X and in Y. If not, return.
    LDY $00
    LDX #$00
    LDA #$0E
    JSR Collision_TestPoints
    BEQ Fire_Update_Return

    ; Knock Link back: X = the fire as attacker, [00] = 0 for Link as defender. Shove_Begin gives
    ; Link $18 on his invincibility timer and a knockback of $20 pixels, away from the fire along
    ; his own axis when his grid offset is not 0; at grid offset 0 it takes the axis from [0B],
    ; which here still holds the vertical distance left by Collision_TestPoints (>= 4: vertical).
    LDX $00
    LDY #$00
    STY $00
    JSR Shove_Begin

    ; Harm Link: [0D] = 0 whole hearts, [0E] = $80 = half a heart, before Link_BeHarmed halves it
    ; for each ring grade.
    ; X is still the fire's slot ($10 or $11), so the "attacker is a whirlwind ($2E): no hurt
    ; sound" test in Link_BeHarmed reads ObjType,X beyond that array's 13 entries: for slot $10
    ; it reads RoomObjTemplateType ($35F), for slot $11 the byte after it ($360).
    LDA #$00
    STA $0D
    LDA #$80
    STA $0E
    JMP Link_BeHarmed

; ----------------------------------------------------------------------------
; Stores the midpoint of a 16-pixel-wide object: X + 8, Y + 8.
; IN:   X = object slot; Y = 0 or 2: offset from [02]/[03] of the result.
; OUT:  [02 + Y] = ObjX + 8; [03 + Y] = ObjY + 8.
; ----------------------------------------------------------------------------
Obj_GetWideMiddle:
    LDA ObjX, X
    CLC
    ADC #$08
    STA $0002, Y
    LDA ObjY, X
    CLC
    ADC #$08
    STA $0003, Y

; Shared return of Fire_Update and Obj_GetWideMiddle.
Fire_Update_Return:
    RTS
