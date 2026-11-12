; ============================================================================
; FILE:     bank1/whirlwind.asm          BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1745-2010
; SUBJECTS: (none)
; TOPIC:    Recorder whirlwind: summon, pickup, destinations, drop-off
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0026 InitWhirlwind                     code  Params:
;   L0032 SetUpWhirlwind                    code  Params:
;   L0042 WhirlwindPrevRoomIdList           data
;   L0045 UpdateWhirlwind_Full              code
;   L0156 DestroyWhirlwind                  code
;   L0170 SummonWhirlwind                   code
;   L0236 AdvanceTeleportingLevelIndex      code
;   L0246 L6104_Exit                        code
;   L0249 DrawWhirlwind                     code
;   L0266 TeleportYs                        data
;   L0269 CheckInitWhirlwindAndBeginUpdate  code
; ============================================================================

; Params:
; A: Y coordinate
; X: object index
;
InitWhirlwind:
    STA ObjY

; Params:
; X: object index
;
SetUpWhirlwind:
    ; Set the whirlwind's Y to Link's, X to 0, and type $2E.
    LDA ObjY
    STA ObjY, X
    LDA #$00
    STA ObjX, X
    LDA #$2E
    STA ObjType, X
    RTS

WhirlwindPrevRoomIdList:
    .BYTE $36, $3B, $73, $44, $0A, $21, $41, $6C

UpdateWhirlwind_Full:
    ; Get Link's halted state.
    LDA ObjState
    AND #$40
    TAY

    ; Add 2 to whirlwind X.
    LDA #$02
    CLC
    ADC ObjX, X
    STA ObjX, X

    ; If Link is not halted or teleporting state = 0, then
    ; go check for collision with Link.
    CPY #$40
    BNE @CheckCollision
    LDY WhirlwindTeleportingState
    BEQ @CheckCollision

    ; Else Link is halted (state $40) and teleporting state <> 0.
    ;
    ; Set Link's X to whirlwind's.
    STA ObjX

    ; If teleporting state = 1 or whirlwind's X <> $80, go check
    ; the right screen edge and draw.
    DEY
    BEQ @CheckRightEdge
    CMP #$80
    BNE @CheckRightEdge

    ; Teleporting state = 2 and whirlwind's X = $80.
    ;
    ; Reset Link's state, teleporting state, and whirlwind object type.
    ASL
    STA ObjState
    STA WhirlwindTeleportingState
    STA ObjType, X

    ; Update the player position marker, and draw one last time.
    TXA
    PHA
    JSR UpdatePlayerPositionMarker
    PLA
    TAX
    JMP @Draw

@CheckCollision:
    ; Link is not halted or teleporting state = 0.
    ; Try to pick up Link.
    ;
    ; If the whirlwind does not collide with Link, then go check
    ; for the screen's right edge.
    JSR CheckLinkCollision
    LDA $06
    BEQ @CheckRightEdge

    ; Collided with Link.
    ;
    ; Make Link face right. Reset shove info and subroom type.
    LDA #$01
    STA ObjDir
    LSR
    STA ObjShoveDir
    STA ObjShoveDistance
    STA UndergroundExitType

    ; Halt Link (state $40), and hide him.
    LDA #$40
    STA ObjState
    LDA #$F8
    STA Sprites+72
    STA Sprites+76

    ; Find and set the room that will behave as the previous room
    ; when scrolling right to the room with the level's entrance.
    LDA TeleportingLevelIndex
    AND #$07
    TAY
    LDA WhirlwindPrevRoomIdList, Y
    STA WhirlwindPrevRoomId

    ; Set teleporting state 1.
    INC WhirlwindTeleportingState

@CheckRightEdge:
    ; If whirlwind's X < $F0, go draw.
    LDA ObjX, X
    CMP #$F0
    BCC @Draw

    ; The whirlwind has reached the right edge of the screen.
    ;
    ; Destroy the object.
    JSR DestroyWhirlwind

    ; If teleporting state <> 0, we picked up Link.
    ; So, go to the next mode.
    ;
    ; Either way, draw one last time.
    LDA WhirlwindTeleportingState
    BEQ @Draw
    TXA
    PHA
    JSR GoToNextModeFromPlay
    PLA
    TAX

@Draw:
    JMP DrawWhirlwind

DestroyWhirlwind:
    LDA #$00
    STA ObjType, X
    STA ObjShoveDir, X
    STA ObjShoveDistance, X
    STA ObjTimer, X
    STA ObjState, X
    STA ObjInvincibilityTimer, X
    LDA #$FF
    STA ObjUninitialized, X
    LDA #$01
    STA ObjMetastate, X
    RTS

SummonWhirlwind:
    ; If not in mode 5, return.
    LDA GameMode
    CMP #$05
    BNE L6104_Exit

    ; Advance the teleporting index, and store the mask for its level in [00].
    JSR AdvanceTeleportingLevelIndex
    LDA TeleportingLevelIndex
    AND #$07                    ; Truncate to wrap the index around.
    TAY
    LDA LevelMasks, Y           ; Note that this is from element 1 instead of 0.
    STA $00

@LoopTriforcePiece:
    ; If there are no triforce pieces, then return.
    LDA InvTriforce
    BEQ L6104_Exit

    ; If we have gotten this triforce piece, then go try to make a
    ; whirlwind.
    BIT $00
    BNE @MakeWhirlwind

    ; We have not gotten this piece.
    ; First, advance the teleporting index again.
    JSR AdvanceTeleportingLevelIndex

    ; If the direction Link is facing is an increasing one, then
    ; rotate the level mask in [00] left to correspond to the higher
    ; level number. Then go test this new mask.
    LDA ObjDir
    AND #$09
    BEQ @DecreasingDir
    ASL $00
    BCC @LoopTriforcePiece
    ROL $00
    JMP @LoopTriforcePiece

@DecreasingDir:
    ; If instead the direction is a decreasing one, then
    ; rotate right to correspond to the lower level number.
    ; Then go test this new mask.
    LSR $00
    BCC @LoopTriforcePiece
    ROR $00
    JMP @LoopTriforcePiece

@MakeWhirlwind:
    ; If already summoned or in the middle of teleporting, then
    ; return and don't summon another one.
    LDA SummonedWhirlwind
    ORA WhirlwindTeleportingState
    BNE L6104_Exit

    ; Return if we can't find an empty monster slot.
    JSR FindEmptyMonsterSlot
    BEQ L6104_Exit

    ; An empty slot was found. Flag that summoned the whirlwind.
    ; Switch to the empty slot, and go set up a whirlwind object.
    INC SummonedWhirlwind
    TYA
    TAX
    JMP SetUpWhirlwind

AdvanceTeleportingLevelIndex:
    ; If Link is facing an increasing direction (right or down), then
    ; increase the index. Otherwise, decrease it.
    INC TeleportingLevelIndex
    LDA ObjDir
    AND #$09
    BNE L6104_Exit
    DEC TeleportingLevelIndex
    DEC TeleportingLevelIndex

L6104_Exit:
    RTS

DrawWhirlwind:
    ; Frames last 1 screen frame.
    LDA #$01
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; The whirlwind flashes by using a palette row based on the
    ; screen frame counter.
    LDA FrameCounter
    AND #$03
    JSR Anim_SetSpriteDescriptorAttributes

    ; Flip horizontally based on movement frame, which is
    ; switched based on animation counter.
    JSR Anim_SetObjHFlipForSpriteDescriptor
    LDA #$00                    ; Whirlwind has only 1 animation frame image.
    JMP DrawObjectNotMirrored

TeleportYs:
    .BYTE $8D, $AD, $8D, $8D, $AD, $8D, $AD, $5D

CheckInitWhirlwindAndBeginUpdate:
    LDA WhirlwindTeleportingState
    BEQ :+                      ; If not teleporting, go start updating the mode.

    ; Since we're already teleporting, the source room was
    ; already done. Set up the whirlwind object in destination room.
    ;
    ; Teleporting state 2 drops off Link.
    INC WhirlwindTeleportingState
    LDA #$40                    ; Halt Link (state $40).
    STA ObjState
    LDX #$09                    ; Whirlwind object goes in slot 9 in destination room.
    LDA TeleportingLevelIndex   ; Get Y coordinate of whirlwind in this destination.
    AND #$07
    TAY
    LDA TeleportYs, Y
    JSR InitWhirlwind
:
    JMP BeginUpdateMode
