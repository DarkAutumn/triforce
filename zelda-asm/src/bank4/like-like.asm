; ============================================================================
; FILE:     bank4/like-like.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 6816-6913
; SUBJECTS: (none)
; TOPIC:    Like-Like (object type $17): wanders toward Link, holds him when it touches him, and removes
;           the magic shield after holding him long enough.
; SEE ALSO: bank1/monster-collisions.asm (CheckMonsterCollisions sets ObjCaptureTimer on a touch);
;           bank1/link-collision-harm.asm (CheckLinkCollision: no monster can touch a held Link);
;           bank7/mode5-play-update.asm (UpdatePlayer discards direction input while held);
;           bank4/walker-wanderer-common.asm (UpdateCommonWanderer); bank4/push-block.asm
;           (DrawObjectMirroredOverLink, HideSpritesOverLink); bank5/underworld-secrets.asm (clears
;           IsLinkParalyzed when the room is found empty); bank4/drop-items.asm (NoDropMonsterTypes).
; KEY FACTS:
;   - Capture starts on the frame its touch harms Link: CheckMonsterCollisions (bank 1) increments
;     ObjCaptureTimer for type $17; UpdateLikeLike then moves the Like-Like onto Link, clears Link's
;     knockback, ObjTimer and metastate, and increments IsLinkParalyzed.
;   - While held, Link's direction input is discarded (UpdatePlayer, bank 7); his buttons still work,
;     and the Like-Like stays where it is (UpdateLikeLike@HandleCaptured does not move it).
;   - The capture timer counts 1 per frame from 1; when it reaches $60 (95 frames after the capture
;     frame) InvMagicShield is set to 0, and again every frame after (UpdateLikeLike@IncCaptureTime).
;   - Link is freed when this Like-Like dies (UpdateLikeLike clears IsLinkParalyzed when its metastate
;     becomes nonzero); bank 5 also clears it when the room has no monsters left. No shield refund.
;   - Nothing pauses the hold: UpdateLikeLike@HandleCaptured tests neither the stun timer nor InvClock,
;     so a boomerang stun does not stop the capture timer.
;   - While IsLinkParalyzed is set CheckLinkCollision (bank 1) reports no collision for any monster or
;     shot, so a held Link takes no further damage.
;   - HP $90 (nibble 9 in ObjectTypeToHpPairs); touch damage byte $01 = 1 heart before rings
;     (ObjTypeToDamagePoints); q-speed $20 = 0.5 pixels per frame (room-entry default); turn rate $80.
;   - Invincibility mask is never set here (stays 0: every weapon type hurts it). Type $17 is in
;     NoDropMonsterTypes: it never drops an item.
; CONTENTS: (generated; L = line in this file)
;   L0046 UpdateLikeLike  code  Updates a Like-Like: wanders toward Link; once it touches him, holds him and eats the magic shield.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a Like-Like: wanders toward Link; once it touches him, holds him and eats the magic shield.
; IN:   X = object slot.
; OUT:  writes IsLinkParalyzed, InvMagicShield, ObjCaptureTimer,X, and Link's ObjTimer, ObjMetastate,
;       ObjShoveDir, ObjShoveDistance.
; NOTE: The touch that starts the capture also harms Link (1 heart before rings). After that no
;       monster or shot can touch him until the Like-Like dies. The magic shield is lost when the
;       capture timer reaches $60: 95 frames after the frame of the capture. IsLinkParalyzed is
;       incremented, not set, but only goes 0 -> 1: a held Link cannot be touched again. It is
;       cleared when the holding Like-Like dies.
; ----------------------------------------------------------------------------
UpdateLikeLike:
    ; Nonzero capture timer = this Like-Like is holding Link.
    LDA ObjCaptureTimer, X
    BNE @HandleCaptured
    LDA #$80                    ; Turn rate $80 (see Wanderer_ObjTurnRate)
    JSR UpdateCommonWanderer

    ; Like-Like cycles through 4 animation frames, 8 frames each.
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
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirrored

    ; Check collisions. CheckMonsterCollisions increments the capture timer of a Like-Like whose touch
    ; harmed Link on this frame. If that did not happen, return.
    JSR CheckMonsterCollisions
    LDA ObjCaptureTimer, X
    BEQ @Exit

    ; Link was just captured. Put the Like-Like exactly on him.
    LDA ObjX
    STA ObjX, X
    LDA ObjY
    STA ObjY, X

    ; Cancel Link's knockback from the touch, and clear his timer and metastate.
    ; (His invincibility timer from the touch is left running.)
    LDA #$00
    STA ObjTimer
    STA ObjMetastate
    STA ObjShoveDir
    STA ObjShoveDistance

    ; Restart the Like-Like's animation for the capture: frame 0, 4 frames per step.
    STA ObjAnimFrame, X
    LDA #$04
    STA ObjAnimCounter, X

    ; Link's direction input is discarded while this is nonzero.
    INC IsLinkParalyzed

@Exit:
    RTS

@HandleCaptured:
    ; Holding Link. Advance the animation every 4 frames (A = 2, doubled) until frame 3, then stay.
    LDA #$02
    CMP ObjAnimFrame, X
    BCC @IncCaptureTime
    DEC ObjAnimCounter, X
    BNE @IncCaptureTime
    ASL
    STA ObjAnimCounter, X
    INC ObjAnimFrame, X

@IncCaptureTime:
    INC ObjCaptureTimer, X

    ; Below $60 nothing else happens. The timer was 1 on the capture frame, so $60 comes 95 frames
    ; later.
    LDA ObjCaptureTimer, X
    CMP #$60
    BCC @DrawAfterCapture
    LDA #$00                    ; Remove the magic shield (the small shield remains).
    STA InvMagicShield

    ; Hold the timer at $C0, so the shield is zeroed again on every later frame of the capture.
    LDA #$C0
    STA ObjCaptureTimer, X

@DrawAfterCapture:
    ; Draw the Like-Like in the two sprites reserved for drawing over Link, not through the usual
    ; sprite-cycling writer.
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR DrawObjectMirroredOverLink

    ; Check collisions (Link's weapons can hit it; Link himself cannot be touched while held).
    ; If the Like-Like is still alive, return.
    JSR CheckMonsterCollisions
    LDA ObjMetastate, X
    BEQ @Exit

    ; It is dying: free Link and hide the two sprites that covered him.
    LDA #$00
    STA IsLinkParalyzed
    JMP HideSpritesOverLink
