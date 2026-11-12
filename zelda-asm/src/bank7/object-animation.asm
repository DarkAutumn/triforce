; ============================================================================
; FILE:     bank7/object-animation.asm   BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5032-5263
; SUBJECTS: Anim
; TOPIC:    Drawing and timing of the spawn cloud and the death sparkle (ObjMetastate); the walking
;           animation counter shared by Link and walking monsters; the stepping of Link's item-use
;           state by that counter; helpers that put an object's position in the sprite descriptor.
; SEE ALSO: bank7/object-update-table.asm (UpdateMetaObject calls AnimateAndDrawMetaObject and ends
;           the metastates); bank7/bomb.asm (DrawCloud); bank7/object-init-table.asm (InitObject sets
;           the first cloud step's time); bank7/link-animate-ladder.asm (caller of AnimateLinkBase);
;           bank1/object-animation-draw.asm (AnimateAndDrawObjectWalking, DrawObjectNotMirrored);
;           bank1/item-sprite-draw.asm (Anim_WriteItemSprites);
;           bank1/bomb-candle-wield.asm (PlaceWeaponForPlayerStateAndAnim starts Link's state $10).
; KEY FACTS:
;   - Spawn cloud: metastates 1, 2, 3 draw cloud images 1, 2, 3. A step ends when the slot's ObjTimer
;     is 0; each later step lasts 6 frames (AnimateAndDrawMetaObject). The object's own update does
;     not run during the cloud (see UpdateObject).
;   - Death sparkle: metastate $10 draws nothing and ends on its first update; $11, $12, $13 last 6
;     frames each and draw sparkle images 1, 0, 1 (AnimateAndDrawMetaObject).
;   - The walking animation frame (ObjAnimFrame, 0 or 1) flips every 6 frames for anything animated
;     by AnimateObjectWalking (Link and the monsters drawn by the generic walker drawing).
;   - Link's legs move only while a direction is held, or his ObjState is not 0, or the game mode is
;     4 (enter room) or $10 (stairs) (AnimateLinkBase).
;   - Link's item-use state advances only when his ObjAnimCounter reaches 0: $10 -> $11 -> $31 -> idle,
;     and $20 -> $21 -> $31 -> idle; the reload is 6 frames, so each step after the first lasts 6
;     frames unless other code sets the counter (AnimateLinkObjState, AnimateObjectWalking).
;   - Ending the item-use state keeps bits $C0 of Link's ObjState (the halt bit $40 survives)
;     (AnimateLinkObjState).
;   - Walking frame selection: facing left/right uses frame 0 or 1 (ObjAnimFrame), flipped when facing
;     left; down is frame 2, up frame 3, flipped on alternate animation frames
;     (SetUpWalkingSprites, SetUpHorizontalWalkingSprites).
; CONTENTS: (generated; L = line in this file)
;   L0054 AnimateAndDrawMetaObject                                code  Draws an object's spawn cloud or death sparkle and advances ObjMetastate when its timer is 0.
;   L0102 AnimateLinkBase                                         code  Advances Link's walking animation when he is moving or busy, then picks his walking frame.
;   L0121 AnimateObjectWalking                                    code  Counts down an object's animation counter, flips its walking frame every 6 frames, picks the frame.
;   L0136 SetUpWalkingSprites                                     code  Picks the walking frame and flip for the object's facing direction; entry that skips the counter.
;   L0157 Anim_SetObjHFlipForSpriteDescriptor                     code  Sets the horizontal flip [0F] from the object's animation frame (0 or 1).
;   L0163 SetUpHorizontalWalkingSprites                           code  Second half of SetUpWalkingSprites for an object facing left or right.
;   L0182 Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor  code  Counts down an object's animation counter with a caller-chosen period, then fetches its position.
;   L0193 Anim_FetchObjPosForSpriteDescriptor                     code  Copies an object's position into the sprite descriptor and clears the horizontal flip.
;   L0206 RollOverAnimCounter                                     code  Reloads an object's animation counter and toggles its animation frame between 0 and 1.
;   L0223 AnimateLinkObjState                                     code  Steps Link's sword/item-use state: $10 -> $11 -> $31 -> idle, $20 -> $21 -> $31 -> idle.
; ============================================================================

; ----------------------------------------------------------------------------
; Draws an object's spawn cloud or death sparkle and advances ObjMetastate when its timer is 0.
; IN:   X = object slot with ObjMetastate <> 0
; OUT:  ObjMetastate,X incremented and ObjTimer,X = 6 when the step ended; [00], [01] = position.
; NOTE: Cloud: metastates 1-3 draw cloud images 1-3 of the bomb item slot. Sparkle: $10 is not drawn
;       and advances at once (whatever the timer holds); $11-$13 draw images 1, 0, 1 of item slot
;       $24. Every step that begins here lasts 6 frames. The caller (UpdateMetaObject) handles the
;       end values 4 and $14.
; ----------------------------------------------------------------------------
AnimateAndDrawMetaObject:
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Metastate >= $10 is the death sparkle.
    LDA ObjMetastate, X
    CMP #$10
    BCS @AnimateSpark

    ; Spawn cloud: the metastate (1-3) is the cloud's frame image.
    AND #$0F
    JSR DrawCloud

@CheckMetaObjTimer:
    ; While the object's timer runs, stay in this metastate.
    LDA ObjTimer, X
    BNE @Exit

@IncMetastate:
    ; Go to the next metastate; it lasts 6 frames.
    LDA #$06
    STA ObjTimer, X
    INC ObjMetastate, X

@Exit:
    RTS

@AnimateSpark:
    ; Metastate $10: advance to $11 without drawing and without waiting for the timer.
    AND #$0F
    BEQ @IncMetastate

    ; Frame image [0C] = low bit of the metastate: 1, 0, 1 for $11, $12, $13.
    AND #$01
    STA $0C
    LDA #$01                    ; Sprite attributes 1: sprite palette 1, not flipped
    JSR Anim_SetSpriteDescriptorAttributes
    LDY #$24                    ; Item slot $24 = death sparkle
    JSR Anim_WriteItemSprites
    JMP @CheckMetaObjTimer

; ----------------------------------------------------------------------------
; Advances Link's walking animation when he is moving or busy, then picks his walking frame.
; IN:   X = 0 (Link's object slot; the caller must have set it)
; OUT:  Y = walking frame (0/1 sideways, 2 down, 3 up); [00] = ObjX; [01] = ObjY; [0F] = 1 to flip
;       horizontally
; NOTE: The counter runs when Link's ObjState <> 0 (using an item, halted), in game modes 4 and $10,
;       or when a direction is held (ObjInputDir <> 0). Otherwise the frame stays as it is.
; ----------------------------------------------------------------------------
AnimateLinkBase:
    LDA ObjState
    BNE AnimateObjectWalking    ; Link is not idle: animate.
    LDA GameMode
    CMP #$04
    BEQ AnimateObjectWalking    ; Mode 4 (enter room): animate.
    CMP #$10
    BEQ AnimateObjectWalking    ; Mode $10 (stairs): animate.
    LDA ObjInputDir
    BEQ SetUpWalkingSprites     ; No direction held: keep the current frame.

; ----------------------------------------------------------------------------
; Counts down an object's animation counter, flips its walking frame every 6 frames, picks the frame.
; IN:   X = object slot
; OUT:  Y = walking frame (0/1 sideways, 2 down, 3 up); [00] = ObjX; [01] = ObjY; [0F] = 1 to flip
;       horizontally
; NOTE: For Link (X = 0) the counter reaching 0 also steps his item-use state (AnimateLinkObjState).
;       The counter is reloaded with 6.
; ----------------------------------------------------------------------------
AnimateObjectWalking:
    DEC ObjAnimCounter, X
    BNE SetUpWalkingSprites

    ; The counter reached 0. For Link, step his item-use state first.
    CPX #$00
    BNE :+
    JSR AnimateLinkObjState
:
    ; Reload the counter with 6 frames and switch the walking frame.
    LDA #$06
    STA $00
    JSR RollOverAnimCounter

; Picks the walking frame and flip for the object's facing direction; entry that skips the counter.
SetUpWalkingSprites:
    ; [00], [01] = position; [0F] = 0.
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; Facing left or right (no bit of $0C set in ObjDir): go choose frame 0 or 1.
    LDA ObjDir, X
    AND #$0C
    BEQ SetUpHorizontalWalkingSprites

    ; Facing up or down.
    LDY #$03                    ; Frame 3 = facing up.
    AND #$08
    BNE Anim_SetObjHFlipForSpriteDescriptor
    DEY                         ; Frame 2 = facing down.

; ----------------------------------------------------------------------------
; Sets the horizontal flip [0F] from the object's animation frame (0 or 1).
; IN:   X = object slot
; OUT:  [0F] = ObjAnimFrame,X; A = the same
; NOTE: The up and down walking frames animate by flipping on alternate animation frames.
; ----------------------------------------------------------------------------
Anim_SetObjHFlipForSpriteDescriptor:
    LDA ObjAnimFrame, X
    STA $0F
    RTS

; Second half of SetUpWalkingSprites for an object facing left or right.
SetUpHorizontalWalkingSprites:
    LDY #$00                    ; Frame 0 when ObjAnimFrame = 0.
    LDA ObjAnimFrame, X
    BEQ :+
    INY                         ; Frame 1 when ObjAnimFrame = 1.
:
    LDA ObjDir, X
    AND #$01
    BNE :+
    INC $0F                     ; Not facing right (direction bit $01 clear): flip horizontally.
:
    RTS

; ----------------------------------------------------------------------------
; Counts down an object's animation counter with a caller-chosen period, then fetches its position.
; IN:   A = frames to reload the counter with when it reaches 0; X = object slot
; OUT:  A = 0; [00] = ObjX; [01] = ObjY; [0F] = 0 (no horizontal flip); ObjAnimFrame,X toggled when
;       the counter rolled over
; ----------------------------------------------------------------------------
Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor:
    STA $00
    DEC ObjAnimCounter, X
    BNE Anim_FetchObjPosForSpriteDescriptor
    JSR RollOverAnimCounter

; ----------------------------------------------------------------------------
; Copies an object's position into the sprite descriptor and clears the horizontal flip.
; IN:   X = object slot
; OUT:  A = 0; [00] = ObjX; [01] = ObjY; [0F] = 0
; ----------------------------------------------------------------------------
Anim_FetchObjPosForSpriteDescriptor:
    LDA ObjX, X
    STA $00
    LDA ObjY, X
    STA $01
    LDA #$00                    ; Don't flip horizontally
    STA $0F
    RTS

; ----------------------------------------------------------------------------
; Reloads an object's animation counter and toggles its animation frame between 0 and 1.
; IN:   X = object slot; [00] = new counter value (frames)
; ----------------------------------------------------------------------------
RollOverAnimCounter:
    LDA $00
    STA ObjAnimCounter, X
    LDA ObjAnimFrame, X
    EOR #$01
    STA ObjAnimFrame, X
    RTS

; ----------------------------------------------------------------------------
; Steps Link's sword/item-use state: $10 -> $11 -> $31 -> idle, $20 -> $21 -> $31 -> idle.
; OUT:  ObjState (Link's) advanced; ObjAnimFrame (Link's) = 1 unless the state ended or was not
;       an item-use state
; NOTE: Called only when Link's ObjAnimCounter reaches 0 (AnimateObjectWalking), so each step lasts
;       as long as the counter was loaded with: 6 frames by default. Only bits $30 select the
;       step; ending the state clears bits $3F and keeps $C0 (so a halted Link stays halted).
;       A state with bits $30 clear (0, $40) is left alone.
; ----------------------------------------------------------------------------
AnimateLinkObjState:
    ; Bits $30 of the state = $10:
    ;   low nibble 0: increment the state ($10 -> $11);
    ;   else: OR the state with $30 ($11 -> $31).
    LDA ObjState
    AND #$30
    CMP #$10
    BNE @CheckState20
    LDA ObjState
    AND #$0F
    BNE @CombineStateWith30
    BEQ @IncState

@CheckState20:
    ; Bits $30 of the state = $20: the same two steps ($20 -> $21 -> $31).
    CMP #$20
    BNE @CheckState30
    LDA ObjState
    AND #$0F
    BNE @CombineStateWith30

@IncState:
    INC ObjState
    JMP @SetMovementFrame1

@CombineStateWith30:
    LDA ObjState
    ORA #$30
    STA ObjState

@SetMovementFrame1:
    ; The caller toggles the animation frame right after this routine. Setting it to 1 here
    ; makes it 0 for the step that begins now.
    LDA #$01
    STA ObjAnimFrame
    RTS

@CheckState30:
    ; Bits $30 of the state = $30: the use is over. Keep only bits $C0 (idle, or still halted).
    CMP #$30
    BNE :+
    LDA ObjState
    AND #$C0
    STA ObjState
:
    RTS

; DATA: filler ($FF), 11 bytes, not referenced.
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF
