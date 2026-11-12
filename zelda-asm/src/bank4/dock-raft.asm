; ============================================================================
; FILE:     bank4/dock-raft.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3860-3974
; SUBJECTS: (none)
; TOPIC:    Dock tile object: raft boarding, ride, landing
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0013 RaftDirections  data
;   L0016 UpdateDock      code
; ============================================================================

RaftDirections:
    .BYTE $04, $08

UpdateDock:
    ; If missing the raft, then return.
    LDA InvRaft
    BEQ @Exit
    LDA ObjState, X
    BNE @HandleOtherStates

    ; State 0
    ;
    ; Choose an X coordinate for the dock:
    ; $80 in room $55
    ; $60 in the other room ($3F)
    LDA #$80
    LDY RoomId
    CPY #$55
    BEQ :+
    LDA #$60
:
    ; If the player is not aligned with the dock, return.
    CMP ObjX
    BNE @Exit

    ; Set the dock object's X.
    STA ObjX, X

    ; Determine whether Link is coming or going.
    ; If Y = $3D, then he's at the top edge of the screen. Use state 1.
    LDY #$01
    LDA ObjY
    CMP #$3D
    BEQ @SetState

    ; But if Y = $7D, then he's at the dock. Use state 2.
    ; Otherwise, return.
    INY
    CMP #$7D
    BNE @Exit

@SetState:
    STY ObjState, X

    ; Show the raft beneath Link: 6 pixels down.
    CLC
    ADC #$06
    STA ObjY, X

    ; Play the "secret found" tune, and halt Link.
    JSR PlaySecretFoundTune
    LDA #$40
    STA ObjState

    ; Set Link's direction based on the raft's state.
    LDA RaftDirections-1, Y
    STA ObjDir

@Exit:
    RTS

@HandleOtherStates:
    ; Is state = 1?
    LDY ObjState, X
    DEY
    BEQ @State1

    ; State 2. Up
    DEC ObjY, X
    DEC ObjY

    ; If Link's Y = $3D, then leave the room.
    ; Else go draw.
    LDA ObjY
    CMP #$3D
    BNE @Draw
    TXA
    PHA
    JSR GoToNextModeFromPlay
    PLA
    TAX
    JMP @ResetLinkAndRaftStateAndDraw

@State1:
    ; State 1. Down
    INC ObjY, X
    INC ObjY

    ; If Link's Y <> $7F, go draw.
    ; Note that Y is below $7D, which would trigger going up.
    LDA ObjY
    CMP #$7F
    BNE @Draw

    ; Link's grid offset at $7F must be 2 ($7F - $7D).
    LDA #$02
    STA ObjGridOffset

@ResetLinkAndRaftStateAndDraw:
    LDA #$00
    STA ObjState
    STA ObjState, X

@Draw:
    TXA                         ; Save raft object slot.
    PHA
    JSR Link_EndMoveAndAnimate_Bank4
    PLA                         ; Restore raft object slot.
    TAX

    ; This call seems to be used to put the raft's coordinates in [00], [01].
    JSR AnimateObjectWalking
    LDA #$00                    ; Normal sprites, palette row 4 (Link).
    LDY #$09                    ; Raft item slot
    JMP Anim_WriteStaticItemSpritesWithAttributes
