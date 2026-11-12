; ============================================================================
; FILE:     bank5/mode8-continue-mode10-stairs.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2184-2335
; SUBJECTS: Link, Mode10Stairs, Mode8Continue
; TOPIC:    Update of mode 8 (the CONTINUE / SAVE / RETRY menu shown after death) and of mode $10
;           (Link sinking into a cave or dungeon entrance, or taking stairs, before the next mode).
; SEE ALSO: bank5/mode11-death.asm (the death sequence ends by entering mode 8);
;           bank5/submenu-states.asm (Up+A on controller 2 in the submenu also enters mode 8);
;           bank5/mode-inits-obj-lists.asm (Mode10Stairs_Init sets StairsTargetY);
;           bank5/warps-stairs-caves.asm (Warp_StartStairsMode sets ModeAfterStairs and enters mode $10);
;           bank7/update-mode-dispatch.asm (trampolines; Mode4Enter_StepOutside uses
;           Link_AnimateAndDrawBehindBackground for the walk out of an entrance);
;           bank5/enter-room.asm (Link_DrawBetweenRooms calls Link_PutSpritesBehindBackground).
; KEY FACTS:
;   - Mode 8: Select moves the cursor 0 -> 1 -> 2 -> 0 (GameSubmode holds the choice); Start confirms.
;     The chosen line then flashes for $40 frames before anything happens.
;     Mode8Continue_Update_Full.
;   - The choices lead to: 0 CONTINUE -> mode 3; 1 SAVE -> mode $0D; 2 RETRY -> mode 0, entered
;     directly in its update half at submode 1. Mode8Continue_ChoiceModes,
;     Mode8Continue_Update_Full@HandleActivated.
;   - Whatever the choice, hearts are set to 3 full ones (HeartValues low nibble = 2, HeartPartial =
;     $FF) with the heart containers kept; Link's state and the clock are cleared (Link_ResetStateAndClock)
;     and all sound is silenced. Mode8Continue_Update_Full@HandleActivated.
;   - Mode $10: if the tile under Link is $24 (a cave or dungeon entrance), Link moves down 1 pixel
;     every 4 frames until ObjY = StairsTargetY ($10 pixels, 61-64 frames), behind the background.
;     For any other tile (stairs) the mode ends on its first update frame. Mode10Stairs_Update_Full.
;   - Mode $10 ends by setting GameMode = ModeAfterStairs. Mode10Stairs_Update_Full.
; CONTENTS: (generated; L = line in this file)
;   L0041 Mode8Continue_CursorSpriteValues     data  DATA: bytes 1-3 of the mode 8 cursor sprite (sprite 0): tile $F3, attributes $02, X $40.
;   L0045 Mode8Continue_CursorYs               data  TABLE: Y of the mode 8 cursor sprite; index = choice (0 continue, 1 save, 2 retry); value = pixels.
;   L0049 Mode8Continue_ChoiceModes            data  TABLE: game mode entered for each mode 8 choice; index = choice; value = GameMode.
;   L0054 Mode8Continue_FlashTransferRecord    data  DATA: transfer record that flashes the chosen line: PPU address $23D2 (patched per choice), control
;   L0058 Mode8Continue_FlashAttrAddrsLo       data  TABLE: low byte of the attribute address ($23xx) of each mode 8 line; index = choice.
;   L0068 Mode8Continue_Update_Full            code  Mode 8 update: moves the cursor with Select, confirms with Start, flashes, then leaves the mode.
;   L0176 Mode10Stairs_Update_Full             code  Mode $10 update: sinks Link into a cave or dungeon entrance, then switches to ModeAfterStairs.
;   L0195 Link_AnimateAndDrawBehindBackground  code  Animates and draws Link, then puts his sprites behind the background (falls through).
;   L0202 Link_PutSpritesBehindBackground      code  Sets the behind-background priority bit ($20) in the attributes of Link's two sprites.
; ============================================================================

; DATA: bytes 1-3 of the mode 8 cursor sprite (sprite 0): tile $F3, attributes $02, X $40.
Mode8Continue_CursorSpriteValues:
    .BYTE $F3, $02, $40

; TABLE: Y of the mode 8 cursor sprite; index = choice (0 continue, 1 save, 2 retry); value = pixels.
Mode8Continue_CursorYs:
    .BYTE $4F, $67, $7F

; TABLE: game mode entered for each mode 8 choice; index = choice; value = GameMode.
Mode8Continue_ChoiceModes:
    .BYTE $03, $0D, $00         ; continue -> 3 (unfurl), save -> $0D (save), retry -> 0 (demo/title)

; DATA: transfer record that flashes the chosen line: PPU address $23D2 (patched per choice), control
; $43 = one byte repeated 3 times, attribute byte $00 (patched to $00 or $55), end marker.
Mode8Continue_FlashTransferRecord:
    .BYTE $23, $D2, $43, $00, $FF

; TABLE: low byte of the attribute address ($23xx) of each mode 8 line; index = choice.
Mode8Continue_FlashAttrAddrsLo:
    .BYTE $D2, $DA, $E2

; ----------------------------------------------------------------------------
; Mode 8 update: moves the cursor with Select, confirms with Start, flashes, then leaves the mode.
; IN:   GameSubmode = choice 0-2; bit 7 set once Start was pressed. ButtonsPressed.
; OUT:  When done: GameMode from Mode8Continue_ChoiceModes, hearts = 3, IsUpdatingMode and GameSubmode
;       reset (retry: GameSubmode = 1 and IsUpdatingMode = 1).
; NOTE: From Start to the mode change takes $40 frames (ObjTimer[1]); input is ignored meanwhile.
; ----------------------------------------------------------------------------
Mode8Continue_Update_Full:
    LDA GameSubmode
    ASL

    ; Bit 7 of the submode means Start was already pressed: go flash the chosen line.
    BCS @AnimateSelection
    LDA ButtonsPressed
    AND #$10                    ; Start
    BNE @ActivateOption
    LDA ButtonsPressed
    AND #$20                    ; Select
    BEQ @DrawCursor
    LDA #$01                    ; Select was pressed: request tune 0 bit $01 as the cursor sound.
    STA Tune0Request

    ; The submode is the choice. Advance it, wrapping from 2 to 0.
    INC GameSubmode
    LDA GameSubmode
    CMP #$03
    BNE @DrawCursor
    LDA #$00
    STA GameSubmode

@DrawCursor:
    LDY #$02                    ; Set tile, attributes and X of the cursor sprite (sprite 0).
:
    LDA Mode8Continue_CursorSpriteValues, Y
    STA Sprites+1, Y
    DEY
    BPL :-

    ; Set the cursor's Y for the choice.
    LDY GameSubmode
    LDA Mode8Continue_CursorYs, Y
    STA Sprites
    RTS

@ActivateOption:
    LDA GameSubmode             ; Mark that Start was pressed: set bit 7 of the submode.
    ORA #$80
    STA GameSubmode
    LDA #$40                    ; Flash the chosen line for $40 frames.
    STA ObjTimer+1
    RTS

@AnimateSelection:
    LDA ObjTimer+1
    BEQ @HandleActivated        ; When the timer expires, go act on the choice.

    ; Copy the 5-byte flash transfer record to the dynamic transfer buffer.
    LDY #$04
:
    LDA Mode8Continue_FlashTransferRecord, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Patch the low byte of the PPU address for the chosen line.
    LDA GameSubmode
    AND #$03
    TAY
    LDA Mode8Continue_FlashAttrAddrsLo, Y
    STA DynTileBuf+1

    ; Patch the attribute byte: $00 or $55 by bit 2 of the timer, so the palette of the line
    ; switches every 4 frames.
    LDY #$00
    LDA ObjTimer+1
    AND #$04
    BEQ :+
    LDY #$55
:
    STY DynTileBuf+3
    RTS

@HandleActivated:
    LDA GameSubmode             ; Drop bit 7, leaving the choice.
    AND #$03
    STA GameSubmode
    JSR Link_ResetStateAndClock        ; ObjState = 0 and InvClock = 0.

    ; Set the next game mode by the choice: 0 continue -> 3, 1 save -> $0D, 2 retry -> 0.
    LDY GameSubmode
    LDA Mode8Continue_ChoiceModes, Y
    STA GameMode
    LDA HeartValues             ; Keep the heart containers (high nibble); 2 full hearts plus a
    AND #$F0                    ; full partial heart = 3 hearts.
    ORA #$02
    STA HeartValues
    LDA #$FF
    STA HeartPartial
    JSR Mode_End             ; IsUpdatingMode = 0, GameSubmode = 0. Y still holds the choice.
    CPY #$02                    ; Retry:
    BNE :+
    DEY
    STY GameSubmode             ; start mode 0 at submode 1, already in its update half.
    INC IsUpdatingMode
:
    JMP Audio_SilenceAll

; ----------------------------------------------------------------------------
; Mode $10 update: sinks Link into a cave or dungeon entrance, then switches to ModeAfterStairs.
; IN:   ObjCollidedTile (slot 0) = tile under Link, found by the mode's init; StairsTargetY.
; OUT:  GameMode = ModeAfterStairs with IsUpdatingMode and GameSubmode reset, when the walk ends.
; NOTE: Entrance tile $24: 1 pixel down every 4 frames (FrameCounter & 3 = 0) until ObjY equals
;       StairsTargetY, which the init set $10 pixels below: 61 to 64 frames, depending on the
;       phase of FrameCounter. Any other tile: no walk, the mode ends at once.
; ----------------------------------------------------------------------------
Mode10Stairs_Update_Full:
    LDA ObjCollidedTile
    CMP #$24                    ; Cave or dungeon entrance tile
    BNE :+                      ; Any other tile (stairs): end the mode without the walk.

    ; Move once every 4 frames.
    LDA FrameCounter
    AND #$03
    BNE Link_AnimateAndDrawBehindBackground    ; Not a moving frame: only draw.
    INC ObjY                    ; Move Link down 1 pixel.
    LDA ObjY
    CMP StairsTargetY
    BNE Link_AnimateAndDrawBehindBackground    ; Not at the end of the walk yet: draw.
:
    LDA ModeAfterStairs         ; Go to the mode chosen when the stairs or entrance was found.
    STA GameMode
    JSR Mode_End

; Animates and draws Link, then puts his sprites behind the background (falls through).
Link_AnimateAndDrawBehindBackground:
    JSR Link_EndMoveAndAnimate

; ----------------------------------------------------------------------------
; Sets the behind-background priority bit ($20) in the attributes of Link's two sprites.
; NOTE: The sprites are sprite slots $12 and $13 (attribute bytes Sprites+74 and Sprites+78).
; ----------------------------------------------------------------------------
Link_PutSpritesBehindBackground:
    LDA Sprites+74
    ORA #$20
    STA Sprites+74
    LDA Sprites+78
    ORA #$20
    STA Sprites+78
    RTS
