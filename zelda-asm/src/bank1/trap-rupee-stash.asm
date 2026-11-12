; ============================================================================
; FILE:     bank1/trap-rupee-stash.asm   BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 2366-2677
; SUBJECTS: (none)
; TOPIC:    Blade traps; rupee stash room; filler
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0022 RupeeStashXs                   data
;   L0026 RupeeStashYs                   data
;   L0030 InitRupeeStash_Full            code
;   L0058 InitOneSimpleObject            code  Params:
;   L0067 TrapXs                         data
;   L0070 TrapYs                         data
;   L0073 InitTrap_Full                  code
;   L0112 TrapAllowedDirs                data
;   L0115 UpdateTrap_Full                code
;   L0261 L_Trap_DrawAndCheckCollisions  code
;   L0264 UpdateRupeeStash_Full          code
; ============================================================================

RupeeStashXs:
    .BYTE $78, $70, $80, $60, $70, $80, $90, $70
    .BYTE $80, $78

RupeeStashYs:
    .BYTE $70, $80, $80, $90, $90, $90, $90, $A0
    .BYTE $A0, $B0

InitRupeeStash_Full:
    ; When initializing, this object makes other objects that
    ; each have the same object attributes and type as this one.
    LDA ObjAttr, X
    STA $01                     ; [01] holds object attributes
    LDA #$35
    STA $00                     ; [01] holds object type $35 (rupee stash)

    ; Loop over $A rupee stash objects to make
    ; -- each one now representing an individual rupee
    LDX #$0A

@LoopRupee:
    JSR InitOneSimpleObject

    ; Look up and set the coordinates for one rupee stash/rupee.
    LDA RupeeStashXs-1, X
    STA ObjX, X
    LDA RupeeStashYs-1, X
    STA ObjY, X
    DEX
    BNE @LoopRupee
    RTS

; Params:
; [00]: object type
; [01]: object attributes
;
InitOneSimpleObject:
    LDA $00
    STA ObjType, X
    LDA #$00
    STA ObjUninitialized, X
    LDA $01
    STA ObjAttr, X
    RTS

TrapXs:
    .BYTE $20, $20, $D0, $D0, $40, $B0

TrapYs:
    .BYTE $5D, $BD, $5D, $BD, $8D, $8D

InitTrap_Full:
    ; Copy this parent/generator object's attributes to [01].
    LDA ObjAttr, X
    STA $01

    ; Assume we're initializing trap object type $49 that makes 6 traps.
    LDY #$05

    ; The individual trap objects to make will be of type $49.
    ; Store it in [00].
    LDA #$49
    STA $00

    ; If the object type we're initializing is not $49 (as in it's $4A),
    ; then we'll make 4 traps.
    CMP ObjType, X
    BEQ @LoopTrap
    LDY #$03

@LoopTrap:
    ; Loop over each trap to make, from the last index (5 or 3) to 0.
    TYA
    CLC

    ; Add the loop index to the current object slot, and switch the X register to it.
    ADC CurObjIndex
    TAX

    ; Look up and set the location for this iteration's individual trap.
    LDA TrapXs, Y
    STA ObjX, X
    LDA TrapYs, Y
    STA ObjY, X
    JSR InitOneSimpleObject
    DEX
    DEY
    BPL @LoopTrap
    RTS

TrapAllowedDirs:
    .BYTE $05, $09, $06, $0A, $01, $02

UpdateTrap_Full:
    LDA ObjState, X
    BNE @UpdateMovingStates

    ; State 0. Sensing.
    ;
    ; If the absolute vertical distance between Link and the trap >= $E,
    ; then go see if the horizontal distance is shorter.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$0E
    BCS @CheckHorizontal

    ; Determine the horizontal direction toward Link;
    ; and if Link and the trap are at the same X, then we can't move
    ; along this axis. Go see about moving vertically.
    LDY #$01
    LDA ObjX
    CMP ObjX, X
    BEQ @CheckHorizontal
    BCS :+
    LDY #$02
:
    ; Remember the original X coordinate.
    LDA ObjX, X

@GoToState1:
    STA ObjMovingLimit, X

    ; Set facing direction to the horizontal one that we determined.
    TYA
    STA ObjDir, X

    ; If the direction we determined is not allowed for this trap, then
    ; go draw and check collisions.
    AND TrapAllowedDirs-1, X
    BEQ @DrawAndCheckCollisions

    ; Advance to state 1 with q-speed $70 (1.75 pixels a frame) (fast).
    INC ObjState, X
    LDA #$70
    STA ObjQSpeedFrac, X

@DrawAndCheckCollisions:
    JMP L_Trap_DrawAndCheckCollisions

@CheckHorizontal:
    ; If the horizontal distance between Link and the trap >= $E,
    ; then this trap definitely wasn't triggered.
    ; Go draw and check collisions.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$0E
    BCS @DrawAndCheckCollisions

    ; Determine the vertical direction toward Link;
    ; and if Link and the trap are at the same Y, then we can't move
    ; along this axis. Go draw and check object collisions.
    LDY #$04
    LDA ObjY
    CMP ObjY, X
    BEQ @DrawAndCheckCollisions
    BCS :+
    LDY #$08
:
    ; We'll advance to state 1. Go remember the original Y coordinate,
    ; set the vertical direction we determined, and other preparations.
    LDA ObjY, X
    BNE @GoToState1

@UpdateMovingStates:
    ; State 1 and 2.
    ;
    ; Move in the direction it's facing.
    LDA ObjDir, X
    STA $0F
    JSR MoveObject

    ; Truncate the grid offset to square length ($10).
    LDA ObjGridOffset, X
    AND #$0F
    BNE :+
    STA ObjGridOffset, X
:
    ; The trap cannot be harmed. So, only check for collision with Link.
    JSR CheckLinkCollision

    ; If moving horizontally, then set target coordinate in [00] to $78
    ; and current coordinate to the trap's X.
    LDY ObjX, X
    LDA #$78
    STA $00
    LDA ObjDir, X
    AND #$0C
    BEQ @Finish

    ; Else moving vertically. Set target coordinate in [00] to $90
    ; and current coordinate to the trap's Y.
    LDY ObjY, X
    LDA #$90
    STA $00

@Finish:
    ; If in state 2, then go finish handling it.
    LDA ObjState, X
    AND #$01
    BEQ @FinishState2

    ; Finish handling state 1.
    ;
    ; Get the distance between the current coordinate and the target.
    TYA
    SEC
    SBC $00
    JSR Abs

    ; If distance >= 5, go draw.
    CMP #$05
    BCS @Draw

    ; Else reverse direction, and advace to state 2
    ; with q-speed $20 (0.5 pixels a frame) (slow).
    LDA ObjDir, X
    JSR GetOppositeDir
    STA ObjDir, X
    LDA #$20
    STA ObjQSpeedFrac, X
    INC ObjState, X

@Draw:
    JMP L_Trap_DrawAndCheckCollisions

@FinishState2:
    ; Finish handling state 2.
    ;
    ; If the current coordinate = original coordinate, then go to state 0.
    TYA
    CMP ObjMovingLimit, X
    BNE L_Trap_DrawAndCheckCollisions
    LDA #$00
    STA ObjState, X

L_Trap_DrawAndCheckCollisions:
    JMP Person_DrawAndCheckCollisions_Common

UpdateRupeeStash_Full:
    TXA                         ; Save the current object slot.
    PHA

    ; If Link is not close enough (< 9 pixels in both axes), then
    ; go draw.
    LDA ObjY
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$09
    BCS @DrawRupee
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$09
    BCS @DrawRupee

    ; Else Link is close enough. Add 1 rupee, and destroy this object.
    JSR TakeOneRupee
    JSR DestroyMonster

    ; Once you've taken one, it counts as taking them all
    ; once you leave the room.
    LDA #$00
    STA RoomObjCount
    JMP @ExitRestoreX

@DrawRupee:
    ; Draw a rupee.
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDX #$16                    ; Rupee item slot
    LDY #$16                    ; Rupee item slot
    JSR DrawItemInInventory

@ExitRestoreX:
    PLA                         ; Restore the current object slot.
    TAX
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF
