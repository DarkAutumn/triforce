; ============================================================================
; FILE:     bank5/mode8-continue-mode10-stairs.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2184-2335
; SUBJECTS: (none)
; TOPIC:    Continue/Save/Retry menu; stairs descent
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0020 Mode8BaseSpriteValues               data
;   L0023 Mode8SpriteYs                       data
;   L0026 Mode8SelectionToMode                data
;   L0029 Mode8FlashTransferRecord            data
;   L0032 Mode8FlashAttrsAddrLo               data
;   L0035 UpdateMode8ContinueQuestion_Full    code
;   L0143 UpdateMode10Stairs_Full             code
;   L0161 AnimateAndDrawLinkBehindBackground  code
;   L0164 PutLinkBehindBackground             code
; ============================================================================

Mode8BaseSpriteValues:
    .BYTE $F3, $02, $40

Mode8SpriteYs:
    .BYTE $4F, $67, $7F

Mode8SelectionToMode:
    .BYTE $03, $0D, $00

Mode8FlashTransferRecord:
    .BYTE $23, $D2, $43, $00, $FF

Mode8FlashAttrsAddrLo:
    .BYTE $D2, $DA, $E2

UpdateMode8ContinueQuestion_Full:
    LDA GameSubmode
    ASL

    ; If the flag is set in submode, then go animate and
    ; handle the selection.
    BCS @AnimateSelection
    LDA ButtonsPressed
    AND #$10
    BNE @ActivateOption         ; If Start is pressed, go handle it.
    LDA ButtonsPressed
    AND #$20
    BEQ @DrawCursor             ; If Select was pressed,
    LDA #$01                    ; Play a short sound for it (same as rupee taken).
    STA Tune0Request

    ; The selection is tracked by the submode.
    ; Increase it. Wrap around, if needed.
    INC GameSubmode
    LDA GameSubmode
    CMP #$03
    BNE @DrawCursor
    LDA #$00
    STA GameSubmode

@DrawCursor:
    LDY #$02                    ; Set tile, attributes, and X for selection sprite (0).
:
    LDA Mode8BaseSpriteValues, Y
    STA Sprites+1, Y
    DEY
    BPL :-

    ; Set Y for selection sprite.
    LDY GameSubmode
    LDA Mode8SpriteYs, Y
    STA Sprites
    RTS

@ActivateOption:
    LDA GameSubmode             ; Show in the submode that Start was pressed by setting high bit.
    ORA #$80
    STA GameSubmode
    LDA #$40                    ; Flash the selection for $40 frames.
    STA ObjTimer+1
    RTS

@AnimateSelection:
    LDA ObjTimer+1
    BEQ @HandleActivated        ; When the timer expires, go handle the selection.

    ; Copy to dynamic tile buf the transfer record
    ; for flashing NT attributes.
    LDY #$04
:
    LDA Mode8FlashTransferRecord, Y
    STA DynTileBuf, Y
    DEY
    BPL :-

    ; Patch the transfer record to write the low byte of the
    ; appropriate address of the attributes depending on
    ; the selection.
    LDA GameSubmode
    AND #$03
    TAY
    LDA Mode8FlashAttrsAddrLo, Y
    STA DynTileBuf+1

    ; Every 4 frames, depending on ObjTimer,
    ; change the NT attribute byte to another palette.
    LDY #$00
    LDA ObjTimer+1
    AND #$04
    BEQ :+
    LDY #$55
:
    STY DynTileBuf+3
    RTS

@HandleActivated:
    LDA GameSubmode             ; Mask the submode, so that it only holds the selection.
    AND #$03
    STA GameSubmode
    JSR ResetPlayerState

    ; Set the next game mode according to the selection.
    ; 3: Continue
    ; D: Save
    ; 0: Retry
    LDY GameSubmode
    LDA Mode8SelectionToMode, Y
    STA GameMode
    LDA HeartValues             ; Make the player start again with 3 full hearts.
    AND #$F0
    ORA #$02
    STA HeartValues
    LDA #$FF
    STA HeartPartial
    JSR EndGameMode
    CPY #$02                    ; If player chose Retry,
    BNE :+
    DEY
    STY GameSubmode             ; Start the next mode in submode 1 and updating.
    INC IsUpdatingMode
:
    JMP SilenceAllSound

UpdateMode10Stairs_Full:
    LDA ObjCollidedTile
    CMP #$24
    BNE :+                      ; If player touched a stairs tile instead of a cave or dungeon entrance, go end the mode without animating Link.

    ; Update the position once every 4 frames.
    LDA FrameCounter
    AND #$03
    BNE AnimateAndDrawLinkBehindBackground    ; If it's not time to change position, go draw.
    INC ObjY                    ; Move Link down 1 pixel.
    LDA ObjY
    CMP StairsTargetY
    BNE AnimateAndDrawLinkBehindBackground    ; If Link hasn't reached the end of the walk, go draw.
:
    LDA ModeAfterStairs              ; Else go to the target mode.
    STA GameMode
    JSR EndGameMode

AnimateAndDrawLinkBehindBackground:
    JSR Link_EndMoveAndAnimate

PutLinkBehindBackground:
    LDA Sprites+74              ; Change priority of Link sprites $12 and $13 to show them behind the background.
    ORA #$20
    STA Sprites+74
    LDA Sprites+78
    ORA #$20
    STA Sprites+78
    RTS
