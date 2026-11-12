; ============================================================================
; FILE:     bank4/gleeok-head-body.asm   BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 9044-9454
; SUBJECTS: Gleeok
; TOPIC:    Gleeok (types $42-$45): drawing the segments of one neck, weapon and Link collisions of its
;           head and base, cutting a head off (flying head, type $46), the death of the whole boss,
;           the attached head's weaving movement, and the animated 3x2 sprite body.
; SEE ALSO: bank4/gleeok.asm (UpdateGleeok: the per-neck loop that calls this file, neck stretching,
;           fireballs, the flying head's update); bank4/gleeok-manhandla-gohma-inits.asm (InitGleeok,
;           InitGleeokHead: start positions, HP $A0, mask $FE, head delays); bank1/monster-collisions.asm
;           (CheckMonsterCollisions); bank1/weapon-damage-death.asm (DealDamage, HandleMonsterDied);
;           bank1/world-fading.asm (WriteBlankPrioritySprites).
; KEY FACTS:
;   - UpdateGleeok loads one neck at a time into object slots 1-6: slot 1 is its base, slot 5 its
;     head (slot 6 is not used in this file). Only slots 1 and 5 are tested against Link and his
;     weapons; slots 2-4 are drawn only (Gleeok_CheckCollisions).
;   - All necks share slot 5, so they share one head HP byte (ObjHP+5) and one invincibility timer
;     (ObjInvincibilityTimer+5). InitGleeok sets the HP to $A0; each time a head is cut off the byte is
;     set to $60 for the next one (Gleeok_CheckCollisions). Damage to any head counts against it.
;   - While the timer of slot 5 (or of slot 1) runs, $10 = about 32 frames after a hit that hurt, that
;     segment of every neck is tested against neither weapons nor Link (CheckMonsterCollisions).
;   - Touch damage: the base harms Link as type $42-$45 (1 heart). The head's slot 5 has ObjType 0
;     (room entry zeroes the slot arrays and the room makes only slot 1), so HarmLink reads entry 0
;     of ObjTypeToDamagePoints, $60: no whole heart, $60/256 of a heart (Gleeok_CheckCollisions).
;   - The base (slot 1) can be hit and makes the body writhe, but its death is cancelled every time
;     (Gleeok_CheckCollisions@ResetMetastate). Gleeok takes no knockback (ResetShoveInfo).
;   - A cut-off head becomes a flying head, type $46, in slot (neck index + 7), slots $07-$0A, at the
;     head's position; its neck's bit is set in GleeokDeadNeckMask (Gleeok_CheckCollisions).
;   - The boss dies when the number of bits in GleeokDeadNeckMask equals ObjType+1 - $41 (1 to 4
;     necks): slot 1 starts its death sparkle (metastate $11) and slots $02-$0A are emptied, which
;     removes the flying heads; slot $0B is not emptied (Gleeok_CheckCollisions@BossDied).
;   - An attached head moves 1 pixel in X and 1 pixel in Y per move; a neck moves once every 4 frames.
;     X reverses every $30 moves and Y every $18 moves (Gleeok_MoveHead).
;   - A hit on a head or base sets GleeokAnimationTimer and Gleeok_WrithingCounter to 6: the body then
;     changes frame every 7 frames instead of every $11 until the timer has been reloaded 6 times
;     (Gleeok_DrawBody).
;   - The body is not an object: it is 6 sprites at fixed positions, X $74-$84, Y $57 and $67
;     (L_Gleeok_WriteSprites).
; CONTENTS: (generated; L = line in this file)
;   L0065 Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions  code  Draws a Gleeok head or neck base in a fixed front sprite record, then checks its collisions.
;   L0094 Gleeok_DrawHeadAndCheckCollisions               code  Draws the current neck from the head (slot 5) down to the base (slot 1) and checks collisions.
;   L0098 Gleeok_DrawSegmentAndCheckCollisions            code  Loop entry: draws the segment in slot X (tile $DC for the head, $DA for the others).
;   L0128 Gleeok_CheckCollisions                          code  Checks the head (slot 5) or base (slot 1) against Link and his weapons; handles a cut-off head.
;   L0285 Gleeok_MoveHead                                 code  Moves the attached head (slot 5) of the current neck one step of its weaving pattern.
;   L0336 L127FA_Exit                                     code  Shared return of Gleeok_MoveHead and of the segment loop in Gleeok_CheckCollisions.
;   L0340 Gleeok_DecHeadTimer                             code  Counts down the head's start delay by one move instead of moving it (part of Gleeok_MoveHead).
;   L0350 Gleeok_ChangeCoordinateBySpeedFlag              code  Adds 1 to A or subtracts 1 from it, according to a step flag.
;   L0363 GleeokBodyTiles0                                data  TABLE: Gleeok body tiles, animation frame 1; index = row * 3 + column (2 rows of 3); value = tile.
;   L0367 GleeokBodyTiles1                                data  TABLE: Gleeok body tiles, animation frames 0 and 2; index = row * 3 + column; value = tile.
;   L0371 GleeokBodyTiles2                                data  TABLE: Gleeok body tiles, animation frame 3; index = row * 3 + column; value = tile.
;   L0376 GleeokBodyBaseTileOffsets                       data  TABLE: offset of a frame's six tiles from GleeokBodyTiles0; index = GleeokBodyAnimationFrame 0-3;
;   L0380 L_Gleeok_DecAnimTimerAndDraw                    code  Body animation timer still running: count it down and draw the current frame.
;   L0391 Gleeok_DrawBody                                 code  Animates and draws Gleeok's body: 6 sprites in 2 rows of 3 at a fixed place.
;   L0412 L_Gleeok_WriteSprites                           code  Writes the six body sprites of the current animation frame (second entry of Gleeok_DrawBody).
; ============================================================================

; ----------------------------------------------------------------------------
; Draws a Gleeok head or neck base in a fixed front sprite record, then checks its collisions.
; IN:   A = tile; X = object slot (5 = head, 1 = base); GleeokCurNeck = neck index 0-3.
; NOTE: The record offset is neck index * 8 for a head (offsets $00-$18) and that + $20 for a base
;       (offsets $20-$38). These are among the first $10 sprite records, which come first in sprite
;       order, so heads and bases are drawn in front of every other sprite, Link included.
;       Attributes are always $03 (sprite palette 3): these sprites do not flash when hit.
; ----------------------------------------------------------------------------
Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions:
    LDY #$03
    STY $03                     ; [03] holds sprite attributes
    PHA                         ; Save the tile number.

    ; Sprite record offset := neck index * 8.
    LDA GleeokCurNeck
    ASL
    ASL
    ASL

    ; If the segment is not the head (slot 5), add $20 to use the second group of 8 records.
    CPX #$05
    BEQ :+
    CLC
    ADC #$20
:
    TAY
    PLA                         ; Restore the tile number.
    JSR Anim_WriteSpecificSprite
    JMP Gleeok_CheckCollisions

; ----------------------------------------------------------------------------
; Draws the current neck from the head (slot 5) down to the base (slot 1) and checks collisions.
; IN:   object slots 1-5 hold the neck's segment coordinates; GleeokCurNeck = neck index.
; OUT:  X is changed (it ends at 0, or at 5 when a head was cut off).
; NOTE: The loop runs through Gleeok_CheckCollisions@NextLoopSegment. It ends early when the head is
;       cut off: the rest of the neck is then not drawn in that frame.
; ----------------------------------------------------------------------------
Gleeok_DrawHeadAndCheckCollisions:
    LDX #$05                    ; Head object slot

; Loop entry: draws the segment in slot X (tile $DC for the head, $DA for the others).
Gleeok_DrawSegmentAndCheckCollisions:
    LDA #$DA                    ; Neck segment tile
    CPX #$05
    BNE :+
    LDA #$DC                    ; Head tile
:
    CPX #$05
    BEQ Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions
    CPX #$01
    BEQ Gleeok_WriteHeadOrBaseSpriteAndCheckCollisions

    ; Slots 2-4: a middle segment. It takes one of the rolling sprite records, so it is drawn
    ; behind Link and behind the heads and bases.
    JSR Anim_WriteLevelPaletteSprite

; ----------------------------------------------------------------------------
; Checks the head (slot 5) or base (slot 1) against Link and his weapons; handles a cut-off head.
; IN:   X = segment object slot 1-5; GleeokCurNeck = neck index.
; OUT:  continues the segment loop with X - 1, or returns (X < 1, or a head was cut off).
; NOTE: Slots 2-4 are skipped: the middle of a neck cannot be hit and does not hurt Link.
;       A cut-off head sets ObjHP+5 to $60, makes a flying head (type $46) in slot neck index + 7
;       and sets the neck's bit in GleeokDeadNeckMask. When the count of dead necks equals
;       ObjType+1 - $41 the boss dies (@BossDied).
;       HandleMonsterDied runs for every cut-off head, and for every "death" of the base, so each
;       of them adds 1 to WorldKillCount and (up to $0A) to HelpDropCount.
;       CheckMonsterCollisions returns at once, without the Link test, while the slot's
;       invincibility timer runs. Link is harmed by the damage byte of ObjType,X: 1 heart for the
;       base (types $42-$45); for the head, slot 5, whose ObjType is 0, entry 0 of
;       ObjTypeToDamagePoints = $60 ($60/256 of a heart).
; ----------------------------------------------------------------------------
Gleeok_CheckCollisions:
    ; Middle segments (slots 2-4) go straight to the bottom of the loop.
    CPX #$05
    BEQ :+
    CPX #$01
    BEQ :+
    JMP @NextLoopSegment
:
    JSR CheckMonsterCollisions

    ; A nonzero shove direction means BeginShove ran for this segment: a weapon of damage type 1
    ; (the only type outside mask $FE) hit it in this call, also when that hit "killed" it. Make the
    ; body writhe: set the body animation timer to 6 and have its next 6 reloads use 6.
    LDA a:ObjShoveDir, X
    BEQ :+
    LDA #$06
    STA GleeokAnimationTimer
    STA Gleeok_WrithingCounter
:
    JSR ResetShoveInfo          ; No knockback for Gleeok.

    ; The base (slot 1) cannot die: if the hit killed it, the dying metastate is cleared again.
    ; DealDamage leaves ObjHP+1 at 0 or below that hit's damage, so any later hit with at least
    ; that much damage "kills" it again.
    CPX #$01
    BEQ @ResetMetastate
    JSR PlayBossHitCryIfNeeded  ; Sample $02 when the invincibility timer is $10.

    ; If the head (slot 5) is not dying, go on with the next segment.
    LDA ObjMetastate, X
    BEQ @NextLoopSegment

    ; The head was cut off.
    ;
    ; Slot 5 holds the head of whichever neck is being processed, so this HP is what the next head
    ; to be hit has: $60 instead of the initial $A0.
    LDA #$60
    STA ObjHP, X
    TXA                         ; Save the segment object slot.
    PHA

    ; X is always 5 here (slot 1 branched away above, slots 2-4 skipped the collision check), so
    ; this branch is never taken.
    CPX #$05
    BNE @DestroyNeck

    ; The flying head goes in slot (neck index + 7): $07-$0A.
    LDA GleeokCurNeck
    CLC
    ADC #$07
    TAX

    ; Flag it uninitialized, so that InitGleeokHead runs for it on its first update.
    LDA #$FF
    STA ObjUninitialized, X

    ; It starts where the attached head was.
    LDA ObjX+5
    STA ObjX, X
    LDA ObjY+5
    STA ObjY, X

    ; Object type $46: flying Gleeok head.
    LDA #$46
    STA ObjType, X

@DestroyNeck:
    PLA                         ; Restore the segment object slot (5).
    TAX

    ; Hide this neck's head sprite (offset neck index * 8) and base sprite (that + $20): Y = $F8.
    LDA GleeokCurNeck
    ASL
    ASL
    ASL
    TAY
    LDA #$F8
    STA Sprites, Y
    STA Sprites+32, Y

    ; Set this neck's bit (LevelMasks = 1 << index) in the dead neck mask. UpdateGleeok skips
    ; necks whose bit is set. Copy the mask to [00].
    LDY GleeokCurNeck
    LDA LevelMasks, Y
    ORA GleeokDeadNeckMask
    STA GleeokDeadNeckMask
    STA $00

    ; Count the set bits among the low four bits of the mask; the count ends up in [00].
    LDA #$00
    LDY #$04
:
    LSR $00
    ADC #$00
    DEY
    BNE :-
    STA $00

    ; The number of necks this Gleeok started with is (object type - $41): 1 for type $42 up to 4
    ; for type $45.
    LDA ObjType+1
    SEC
    SBC #$41
    CMP $00

    ; All necks dead: the boss dies.
    BEQ @BossDied

    ; Else clear the dying metastate of slot 5 and return. The segment loop ends here, so the
    ; rest of this neck is not drawn in this frame.
    JMP ResetObjMetastate

@ResetMetastate:
    JSR ResetObjMetastate

@NextLoopSegment:
    ; Bottom of the segment loop: next lower slot, until slot 1 has been done.
    DEX
    CPX #$01
    BCC L127FA_Exit
    JMP Gleeok_DrawSegmentAndCheckCollisions

@BossDied:
    ; The last neck died, so the whole Gleeok dies.
    ;
    ; Blank the first $10 sprite records: the attached heads and bases.
    JSR WriteBlankPrioritySprites
    JSR PlayBossDeathCry

    ; Slot 1 holds the Gleeok object itself. Metastate $11 puts it in its death sparkle, which ends
    ; as an ordinary monster death (kill counted, drop made).
    LDA #$11
    STA ObjMetastate+1

    ; Empty object slots $02 to $0A (Y = 1 to 9). That removes the flying heads in slots $07-$0A.
    ; Slot $0B is not touched, so a fireball there stays.
    LDY #$01

@DestroyFireballs:
    LDA #$00
    STA ObjType+1, Y
    INY
    CPY #$0A
    BCC @DestroyFireballs
    RTS

; ----------------------------------------------------------------------------
; Moves the attached head (slot 5) of the current neck one step of its weaving pattern.
; IN:   Gleeok_CurHeadInfo block = the current neck's head movement data; ObjX+5, ObjY+5.
; OUT:  ObjX+5 and ObjY+5 each changed by 1 pixel; counters and step flags updated.
; NOTE: UpdateGleeok calls this for one neck per frame (neck index = FrameCounter AND 3), so a head
;       moves once every 4 frames. Nothing happens while Gleeok_CurHeadDelay is nonzero; the delay
;       drops by 1 per call (necks 0-3 start with 0, $0C, $18, $24: InitGleeok).
;       Every 4th move both direction counters go up by 1: at $0C the X step reverses ($30 moves),
;       at 6 the Y step reverses ($18 moves). The counters start at 6 and 3, so the first reversals
;       come after half those spans. No random numbers are used.
; ----------------------------------------------------------------------------
Gleeok_MoveHead:
    ; Wait out the head's start delay.
    LDA Gleeok_CurHeadDelay
    BNE Gleeok_DecHeadTimer

    ; X: +1 pixel when the flag is 0, -1 when it is $FF.
    LDA ObjX+5
    LDY Gleeok_CurHeadSpeedFlagX
    JSR Gleeok_ChangeCoordinateBySpeedFlag
    STA ObjX+5

    ; Y: likewise.
    LDA ObjY+5
    LDY Gleeok_CurHeadSpeedFlagY
    JSR Gleeok_ChangeCoordinateBySpeedFlag
    STA ObjY+5

    ; Count the moves. Return on three of every four.
    INC Gleeok_CurHeadDirChangeCounter
    LDA Gleeok_CurHeadDirChangeCounter
    CMP #$04
    BCC L127FA_Exit

    ; Fourth move: reset the move counter and advance both direction counters.
    LDA #$00
    STA Gleeok_CurHeadDirChangeCounter

    ; Horizontal counter: at $0C reset it and reverse the X step.
    INC Gleeok_CurHeadDirCounterH
    LDA Gleeok_CurHeadDirCounterH
    CMP #$0C
    BCC @CheckVertical
    LDA #$00
    STA Gleeok_CurHeadDirCounterH
    LDA Gleeok_CurHeadSpeedFlagX
    EOR #$FF
    STA Gleeok_CurHeadSpeedFlagX

@CheckVertical:
    ; Vertical counter: at 6 reset it and reverse the Y step.
    INC Gleeok_CurHeadDirCounterV
    LDA Gleeok_CurHeadDirCounterV
    CMP #$06
    BCC L127FA_Exit
    LDA #$00
    STA Gleeok_CurHeadDirCounterV
    LDA Gleeok_CurHeadSpeedFlagY
    EOR #$FF
    STA Gleeok_CurHeadSpeedFlagY

; Shared return of Gleeok_MoveHead and of the segment loop in Gleeok_CheckCollisions.
L127FA_Exit:
    RTS

; Counts down the head's start delay by one move instead of moving it (part of Gleeok_MoveHead).
Gleeok_DecHeadTimer:
    DEC Gleeok_CurHeadDelay
    RTS

; ----------------------------------------------------------------------------
; Adds 1 to A or subtracts 1 from it, according to a step flag.
; IN:   A = coordinate; Y = step flag, loaded last so that Z reflects it: 0 = add, nonzero = subtract.
; OUT:  A = coordinate + 1 or - 1.
; NOTE: The routine tests the Z flag left by the caller's LDY, not Y itself.
; ----------------------------------------------------------------------------
Gleeok_ChangeCoordinateBySpeedFlag:
    BNE @Subtract
    CLC
    ADC #$01
    RTS

@Subtract:
    SEC
    SBC #$01
    RTS

; TABLE: Gleeok body tiles, animation frame 1; index = row * 3 + column (2 rows of 3); value = tile.
;        L_Gleeok_WriteSprites reads all three tables as one, starting at GleeokBodyTiles0 + offset.
GleeokBodyTiles0:
    .BYTE $C0, $C4, $C8, $C2, $C6, $CA

; TABLE: Gleeok body tiles, animation frames 0 and 2; index = row * 3 + column; value = tile.
GleeokBodyTiles1:
    .BYTE $CC, $C4, $CE, $C2, $C6, $D0

; TABLE: Gleeok body tiles, animation frame 3; index = row * 3 + column; value = tile.
GleeokBodyTiles2:
    .BYTE $D2, $D6, $D8, $D4, $C6, $D0

; TABLE: offset of a frame's six tiles from GleeokBodyTiles0; index = GleeokBodyAnimationFrame 0-3;
;        value = byte offset (6 = GleeokBodyTiles1, 0 = GleeokBodyTiles0, $0C = GleeokBodyTiles2).
GleeokBodyBaseTileOffsets:
    .BYTE $06, $00, $06, $0C

; Body animation timer still running: count it down and draw the current frame.
L_Gleeok_DecAnimTimerAndDraw:
    DEC GleeokAnimationTimer
    JMP L_Gleeok_WriteSprites

; ----------------------------------------------------------------------------
; Animates and draws Gleeok's body: 6 sprites in 2 rows of 3 at a fixed place.
; OUT:  X, Y, [06], [07] changed. Six rolling sprite records used.
; NOTE: The frame (0-3) advances when GleeokAnimationTimer is found at 0; the timer is then armed
;       with $10, or with 6 while Gleeok_WrithingCounter is nonzero (the counter then drops by 1).
;       So a frame lasts $11 frames normally and 7 while writhing (6 reloads after a hit).
; ----------------------------------------------------------------------------
Gleeok_DrawBody:
    LDA GleeokAnimationTimer
    BNE L_Gleeok_DecAnimTimerAndDraw

    ; The timer ran out. Arm it with $10, or with 6 while writhing after a hit.
    LDA #$10
    LDY Gleeok_WrithingCounter
    BEQ :+
    DEC Gleeok_WrithingCounter
    LDA #$06
:
    STA GleeokAnimationTimer

    ; Next animation frame, 0 to 3 and around.
    LDA GleeokBodyAnimationFrame
    CLC
    ADC #$01
    AND #$03
    STA GleeokBodyAnimationFrame

; Writes the six body sprites of the current animation frame (second entry of Gleeok_DrawBody).
L_Gleeok_WriteSprites:
    ; For each row, 0 to 1, in [06]:
    LDA #$00
    STA $06

    ; X := offset of the current frame's first tile.
    LDY GleeokBodyAnimationFrame
    LDX GleeokBodyBaseTileOffsets, Y

@LoopDrawRow:
    ; For each column, 0 to 2, in [07]:
    LDA #$00
    STA $07

@LoopDrawColumn:
    ; Take the next rolling sprite record.
    LDY RollingSpriteIndex
    LDA SpriteOffsets, Y
    TAY

    ; Sprite Y := $57 + row * $10.
    LDA $06
    ASL
    ASL
    ASL
    ASL
    ADC #$57
    STA Sprites, Y

    ; Tile for this spot.
    LDA GleeokBodyTiles0, X
    STA Sprites+1, Y

    ; Attributes: sprite palette 3 normally. While the heads' shared invincibility timer (slot 5)
    ; runs, the palette is the timer's low two bits, so the body flashes.
    LDA ObjInvincibilityTimer+5
    BNE :+
    LDA #$03
:
    AND #$03
    STA Sprites+2, Y

    ; Sprite X := $74 + column * 8.
    LDA $07
    ASL
    ASL
    ASL
    ADC #$74
    STA Sprites+3, Y

    ; Next tile, next rolling sprite record.
    INX
    JSR CycleCurSpriteIndex

    ; Bottom of the column loop: until column = 3.
    INC $07
    LDA $07
    CMP #$03
    BCC @LoopDrawColumn

    ; Bottom of the row loop: until row = 2.
    INC $06
    LDA $06
    CMP #$02
    BCC @LoopDrawRow
    RTS

; DATA: filler ($FF), 43 bytes, no label; nothing refers to it.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF
