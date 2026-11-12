; ============================================================================
; FILE:     bank4/like-like.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6816-6913
; SUBJECTS: (none)
; TOPIC:    Like-Like: capture, magic shield eating
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0012 UpdateLikeLike  code
; ============================================================================

UpdateLikeLike:
    ; If the monster captured Link, then go handle it.
    LDA ObjCaptureTimer, X
    BNE @HandleCaptured
    LDA #$80                    ; Turn rate $80
    JSR UpdateCommonWanderer

    ; Like-like has a 4 animation frame cycle unlike most monsters
    ; which have 2.
    DEC ObjAnimCounter, X
    BNE @Draw
    LDA #$08
    STA ObjAnimCounter, X
    LDY ObjAnimFrame, X
    INY
    TYA
    AND #$03
    STA ObjAnimFrame, X

@Draw:
    ; Draw.
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirrored

    ; Check collisions. If Link was not captured, then return.
    JSR CheckMonsterCollisions
    LDA ObjCaptureTimer, X
    BEQ @Exit

    ; The monster captured Link.
    ; Put the monster at the same location as him.
    LDA ObjX
    STA ObjX, X
    LDA ObjY
    STA ObjY, X

    ; Reset Link's timer, metastate, and shove info.
    LDA #$00
    STA ObjTimer
    STA ObjMetastate
    STA ObjShoveDir
    STA ObjShoveDistance

    ; Restart the monster's movement frame cycle: frame 0, counter 4
    STA ObjAnimFrame, X
    LDA #$04
    STA ObjAnimCounter, X

    ; Now Link can't move.
    INC LinkParalyzed

@Exit:
    RTS

@HandleCaptured:
    ; Animate up to frame 3. Then stay there.
    LDA #$02
    CMP ObjAnimFrame, X
    BCC @IncCaptureTime
    DEC ObjAnimCounter, X
    BNE @IncCaptureTime
    ASL
    STA ObjAnimCounter, X
    INC ObjAnimFrame, X

@IncCaptureTime:
    ; Increment capture time.
    INC ObjCaptureTimer, X

    ; Once capture time >= $60 screen frames, remove the magic shield.
    LDA ObjCaptureTimer, X
    CMP #$60
    BCC @DrawAfterCapture
    LDA #$00
    STA InvMagicShield

    ; Set a value above $60 over and over.
    LDA #$C0
    STA ObjCaptureTimer, X

@DrawAfterCapture:
    ; In this state, we have to draw Like-like over Link.
    ; So, don't use the usual sprite writing code that cycles sprites.
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirroredOverLink

    ; Check object collisions.
    ; If this monster hasn't died, then return.
    JSR CheckMonsterCollisions
    LDA ObjMetastate, X
    BEQ @Exit

    ; But once it has died, then let Link move again.
    LDA #$00
    STA LinkParalyzed
    JMP HideSpritesOverLink
