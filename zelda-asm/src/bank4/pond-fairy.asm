; ============================================================================
; FILE:     bank4/pond-fairy.asm         BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3374-3563
; SUBJECTS: (none)
; TOPIC:    Pond fairy healing and orbiting hearts
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0015 UpdatePondFairy              code
;   L0046 PondFairy_HandleOtherStates  code
;   L0106 PondHeartStartAngles         data
;   L0109 PondFairy_MoveHearts         code
; ============================================================================

UpdatePondFairy:
    JSR DrawFairy

    ; If not in state 0, then go handle states 1 to 3.
    LDA ObjState+1
    BNE PondFairy_HandleOtherStates

    ; State 0.
    ;
    ; If Link is not at the edge of the pond (Y=$AD), then return.
    LDA ObjY
    CMP #$AD
    BNE @Exit

    ; If Link's X < $70 or >= $81, then return.
    LDA ObjX
    CMP #$70
    BCC @Exit
    CMP #$81
    BCS @Exit

    ; Link is at the edge of the pond. Advance to state 1,
    ; halt Link, and start filling hearts.
    INC ObjState+1
    LDA #$40
    STA ObjState
    STA World_IsFillingHearts

@Exit:
    RTS

PondFairy_HandleOtherStates:
    CMP #$01
    BNE @CheckState2

    ; State 1. Flying and filling hearts.
    ;
    ; If still filling hearts, then go draw Link and hearts.
    LDA World_IsFillingHearts
    BNE @DrawLinkAndHearts

    ; If not filling hearts anymore, then go to state 2,
    ; and set Link's object timer to $50, to keep showing hearts
    ; a little while after we finish filling hearts.
    INC ObjState+1
    LDA #$50
    STA ObjTimer+1

@CheckState2:
    ; If state <> 2 (so, it's 3), then return.
    ;
    ; Note that the first time here is when we fall thru from 8DBC.
    ; In this case, A has $50 -- the value that Link's object timer
    ; was set to, and we'll return.
    CMP #$02
    BNE @Exit

    ; State = 2.
    ;
    ; If Link's timer <> 0, then go draw Link and hearts.
    LDA ObjTimer+1
    BNE @DrawLinkAndHearts

    ; Else go to state 3, and unhalt Link.
    INC ObjState+1
    LDA #$00
    STA ObjState

@DrawLinkAndHearts:
    ; Reset the input direction, so that Link doesn't animate
    ; when drawing him.
    LDY #$00
    STY ObjInputDir

    ; Save and reset Link's state.
    LDA ObjState
    PHA
    STY ObjState
    JSR Link_EndMoveAndAnimate_Bank4

    ; Restore Link's state.
    PLA
    STA ObjState
    JSR PondFairy_MoveHearts

    ; Restore the object slot.
    LDX CurObjIndex

@Exit:
    RTS

PondHeartStartAngles:
    .BYTE $14, $10, $0C, $08, $04, $00, $1C

PondFairy_MoveHearts:
    ; Loop over each heart from object slot 2 to 9.
    ;
    ; Even though they are not standalone objects, their info
    ; is stored in those object slots.
    LDX #$02

@LoopHeart:
    ; If the current heart's state <> 0, then go move it.
    ; Else we have to wait for it to come out at the right time.
    LDA ObjState, X
    BNE @MoveHeart

    ; The current heart's state = 0.
    ;
    ; If object slot = 2, then go make it appear. This is the first
    ; heart to show up.
    CPX #$02
    BEQ @SetUpHeart

    ; Object slot <> 2. If the state of the heart in slot 2 = 0, then
    ; go start over.
    ;
    ; The first heart -- in slot 2 -- is updated first, and set to state 1.
    ; So, you can't jump from here.
    LDA ObjState+2
    BEQ @LoopHeart

    ; Look up the starting angle of this heart with the index
    ; (object slot - 3). If the first heart has not reached this angle,
    ; then loop again.
    TXA
    SEC
    SBC #$03
    TAY
    LDA ObjAngleWhole+2
    CMP PondHeartStartAngles, Y
    BNE @NextLoopHeart

@SetUpHeart:
    ; Set up this heart. Set its state to 1.
    INC ObjState, X

    ; UNKNOWN: I haven't seen anything depend on the heart's
    ; direction, let alone this value. Then again, Patra Child,
    ; which is also orbits, has an assignment like this one.
    LDA #$80
    STA ObjDir, X

    ; Starting angle $18 (N).
    LDA #$18
    STA ObjAngleWhole, X

    ; This heart is right above the fairy. Set its X the same as the fairy's.
    LDA ObjX+1
    STA ObjX, X

    ; This heart is right above the fairy. Set its Y $1C pixels above the fairy.
    LDA ObjY+1
    SEC
    SBC #$1C
    STA ObjY, X

@MoveHeart:
    ; Decrease the angle by $00.60.
    LDA #$00
    STA $0B
    LDA #$60
    JSR DecreaseObjectAngle

    ; Rotate the heart's location by the change in angle.
    ; Use 6 for the number of bits of sine/cosine to use
    ; for a large ring.
    LDA #$06
    TAY
    JSR RotateObjectLocation
    STA ObjY, X
    TXA                         ; Save the loop index.
    PHA
    JSR Anim_SetSpriteDescriptorRedPaletteRow
    JSR Anim_FetchObjPosForSpriteDescriptor

    ; The variable for the currently updating object slot has to be
    ; set to the heart's object slot for this drawing routine to work.
    LDX CurObjIndex
    JSR DrawObjectNotMirrored
    PLA                         ; Restore the loop index.
    TAX

@NextLoopHeart:
    ; Bottom of the loop.
    ; Increase the object slot.
    INX
    CPX #$0A
    BNE @LoopHeart
    RTS
