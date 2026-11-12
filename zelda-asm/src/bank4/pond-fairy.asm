; ============================================================================
; FILE:     bank4/pond-fairy.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3374-3563
; SUBJECTS: PondFairy
; TOPIC:    Pond fairy ($2F): refills Link's hearts when he stands at the pond's edge, while eight hearts
;           orbit around the pond.
; SEE ALSO: bank4/zora-ow-inits.asm (InitPondFairy: position and sound); bank4/fairy-flyer-common.asm
;           (DrawFairy); bank5/cave-cellar-enter.asm (World_FillHearts: the refill itself);
;           bank4/orbit-rotation-math.asm (DecreaseObjectAngle, RotateObjectLocation).
; KEY FACTS:
;   - The fairy triggers when Link's Y = $AD and $70 <= X <= $80 while its state is 0: Link is halted
;     (ObjState = $40) and World_IsFillingHearts is set (UpdatePondFairy).
;   - States (ObjState+1): 0 waiting; 1 hearts filling; 2 a further $50 frames of orbiting hearts;
;     3 done, Link released. Nothing returns it to 0, so it heals once per room visit
;     (UpdatePondFairy, PondFairy_HandleOtherStates).
;   - Link stays halted from the trigger until state 3: the whole refill plus $50 (80) frames
;     (PondFairy_HandleOtherStates).
;   - The code reads the fairy's state, timer and position from object slot $01, not from X
;     (UpdatePondFairy, PondFairy_MoveHearts).
;   - The eight orbiting hearts use object slots $02-$09 for their state, angle and position without
;     being objects (no ObjType is set) (PondFairy_MoveHearts).
;   - Each heart starts $1C pixels above the fairy at angle $18 and loses $60/256 of an angle step per
;     frame (32 steps a circle: one revolution in about 85 frames); the next heart starts every 4
;     steps of the first one (PondFairy_MoveHearts, PondHeartStartAngles).
;   - With the slots' q-speed of $20 (set for slots $01-$0B on room entry, InitMode_EnterRoom) a heart
;     moves up to 4 pixels a frame; computed from this arithmetic the circle spans X $49-$B3 and
;     Y $60-$CB, about 53 pixels in radius, with the fairy at ($78, $7D) (RotateObjectLocation).
;   - While Link is halted (state $40) no monster or shot can hurt him by touch: CheckLinkCollision
;     returns at once in that state (UpdatePondFairy).
; CONTENTS: (generated; L = line in this file)
;   L0043 UpdatePondFairy              code  Updates the pond fairy ($2F): draws it; starts the heart refill when Link stands at the pond's edge.
;   L0075 PondFairy_HandleOtherStates  code  Pond fairy states 1-3 (A = state): waits for the refill, then $50 frames, then releases Link.
;   L0135 PondHeartStartAngles         data  TABLE: when each later orbiting heart appears; index = heart's object slot - 3 (slots $03-$09);
;   L0146 PondFairy_MoveHearts         code  Starts, moves and draws the eight hearts that orbit the pond while the fairy heals Link.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates the pond fairy ($2F): draws it; starts the heart refill when Link stands at the pond's edge.
; IN:   X = object slot (the code itself reads slot $01's state).
; NOTE: Trigger: fairy state 0, Link's Y = $AD and X in $70-$80. Link is halted (state $40) until the
;       hearts are full and $50 more frames have passed. The fairy ends in state 3 and never rearms.
; ----------------------------------------------------------------------------
UpdatePondFairy:
    JSR DrawFairy

    ; If the fairy (object slot $01) is not in state 0, then go handle states 1 to 3.
    LDA ObjState+1
    BNE PondFairy_HandleOtherStates

    ; State 0.
    ;
    ; If Link's Y <> $AD (standing at the lower edge of the pond), then return.
    LDA ObjY
    CMP #$AD
    BNE @Exit

    ; If Link's X < $70 or >= $81, then return.
    LDA ObjX
    CMP #$70
    BCC @Exit
    CMP #$81
    BCS @Exit

    ; Link is at the edge of the pond. Advance to state 1, halt Link (state $40),
    ; and start filling hearts (any nonzero value; World_FillHearts does the filling).
    INC ObjState+1
    LDA #$40
    STA ObjState
    STA World_IsFillingHearts

@Exit:
    RTS

; Pond fairy states 1-3 (A = state): waits for the refill, then $50 frames, then releases Link.
PondFairy_HandleOtherStates:
    CMP #$01
    BNE @CheckState2

    ; State 1: hearts are filling.
    ;
    ; If still filling hearts, then go draw Link and the orbiting hearts.
    LDA World_IsFillingHearts
    BNE @DrawLinkAndHearts

    ; The hearts are full. Go to state 2, and set the fairy's timer (slot $01) to $50 frames,
    ; to keep the hearts orbiting a little longer.
    INC ObjState+1
    LDA #$50
    STA ObjTimer+1

@CheckState2:
    ; If state <> 2 (so, it's 3: done), then return.
    ;
    ; When falling through from state 1 above, A = $50 (the timer value just set),
    ; so this returns too; state 2 is first handled next frame.
    CMP #$02
    BNE @Exit

    ; State = 2.
    ;
    ; If the fairy's timer (slot $01) <> 0, then go draw Link and hearts.
    LDA ObjTimer+1
    BNE @DrawLinkAndHearts

    ; Else go to state 3, and release Link (state 0). Link and the hearts are drawn once more.
    INC ObjState+1
    LDA #$00
    STA ObjState

@DrawLinkAndHearts:
    ; Draw Link here, standing still: reset his input direction so that he doesn't animate.
    LDY #$00
    STY ObjInputDir

    ; Save Link's state ($40, halted) and draw him with state 0.
    LDA ObjState
    PHA
    STY ObjState
    JSR Link_EndMoveAndAnimate_Bank4

    ; Restore Link's state.
    PLA
    STA ObjState
    JSR PondFairy_MoveHearts

    ; Restore the fairy's object slot in X.
    LDX CurObjIndex

@Exit:
    RTS

; TABLE: when each later orbiting heart appears; index = heart's object slot - 3 (slots $03-$09);
;        value = whole angle of the first heart (slot $02) at which this heart starts (angle steps).
;        The first heart starts at $18 and counts down, so they appear in slot order, 4 steps apart.
PondHeartStartAngles:
    .BYTE $14, $10, $0C, $08, $04, $00, $1C

; ----------------------------------------------------------------------------
; Starts, moves and draws the eight hearts that orbit the pond while the fairy heals Link.
; OUT:  X = $0A (the caller restores it).
; NOTE: The hearts keep their state, angle and position in object slots $02-$09 but are not objects.
;       Each starts $1C pixels above the fairy (slot $01) at angle $18 and loses $0060 of angle per
;       frame (whole part 0-$1F), moving by RotateObjectLocation with 6 bits of sine and cosine.
;       The step also scales with the slot's q-speed, which this routine does not set.
; ----------------------------------------------------------------------------
PondFairy_MoveHearts:
    ; Loop over the hearts in object slots $02 to $09.
    LDX #$02

@LoopHeart:
    ; If this heart's state <> 0, it is out: go move it.
    ; Else it waits for its turn to appear.
    LDA ObjState, X
    BNE @MoveHeart

    ; This heart's state = 0.
    ;
    ; Slot $02 holds the first heart: go make it appear at once.
    CPX #$02
    BEQ @SetUpHeart

    ; Another slot. If the first heart (slot $02) is not out yet, go start over.
    ; This cannot happen: slot $02 is handled first and set to state 1.
    LDA ObjState+2
    BEQ @LoopHeart

    ; Look up this heart's starting angle with index (object slot - 3). If the first heart's
    ; whole angle is not equal to it, then go on to the next heart.
    TXA
    SEC
    SBC #$03
    TAY
    LDA ObjAngleWhole+2
    CMP PondHeartStartAngles, Y
    BNE @NextLoopHeart

@SetUpHeart:
    ; Bring this heart out: state 1.
    INC ObjState, X

    ; Direction $80. Nothing called here for a heart reads it (DecreaseObjectAngle,
    ; RotateObjectLocation and the position fetch do not). The Patra children get the same assignment.
    LDA #$80
    STA ObjDir, X

    ; Starting angle $18 (the top of the circle).
    LDA #$18
    STA ObjAngleWhole, X

    ; Start at the fairy's X (slot $01).
    LDA ObjX+1
    STA ObjX, X

    ; Start $1C pixels above the fairy's Y.
    LDA ObjY+1
    SEC
    SBC #$1C
    STA ObjY, X

@MoveHeart:
    ; Decrease the 16-bit angle by $0060 ([0B] = high byte 0, A = low byte $60).
    LDA #$00
    STA $0B
    LDA #$60
    JSR DecreaseObjectAngle

    ; Move the heart along the circle for its new angle, using the top 6 bits of the
    ; sine and of the cosine (the more bits, the larger the step and the circle).
    ; The routine stores the new X itself and returns the new Y in A.
    LDA #$06
    TAY
    JSR RotateObjectLocation
    STA ObjY, X
    TXA                         ; Save the loop index (the heart's slot).
    PHA
    JSR Anim_SetSpriteDescriptorRedPaletteRow
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; [00], [01] now hold the heart's position. Draw with X = the pond fairy's object slot,
    ; so the sprite is frame image 0 (A = 0) of the pond fairy's object type.
    LDX CurObjIndex
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the loop index.
    TAX

@NextLoopHeart:
    ; Next object slot, up to $09.
    INX
    CPX #$0A
    BNE @LoopHeart
    RTS
