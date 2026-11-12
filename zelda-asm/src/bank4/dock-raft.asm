; ============================================================================
; FILE:     bank4/dock-raft.asm          BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 3860-3974
; SUBJECTS: Dock
; TOPIC:    The dock tile object (type $61) of OW rooms $3F and $55: with the raft it carries Link from
;           the dock up to the top screen edge and into the room above, or from the top edge down to it.
; SEE ALSO: bank5/enter-room.asm (Room_SetUpTileObjOW puts type $61 in slot $0B in rooms $3F and $55);
;           bank7/walker-collision-screen-edge.asm (Mode5Play_GoToNextMode: leaving the room);
;           bank4/flying-ghini-peahat.asm (Sfx_PlaySecretFoundTune); bank1/item-sprite-draw.asm (raft sprite).
; KEY FACTS:
;   - Docks exist only in OW rooms $3F and $55 (Room_SetUpTileObjOW); Dock_Update uses dock X $80 in room
;     $55 and $60 in any other room (that is, $3F).
;   - Nothing happens without the raft (InvRaft = 0): Dock_Update returns at once.
;   - A ride starts only when Link's X equals the dock X exactly and his Y is exactly $7D (at the
;     dock: ride up, state 2) or $3D (top screen edge: ride down, state 1). Dock_Update.
;   - Link is halted (ObjState $40) for the whole ride and faces Dock_RaftLinkDirs (up or down).
;   - The raft and Link move 1 pixel per frame: 64 frames up ($7D to $3D), 66 frames down ($3D to
;     $7F). Dock_Update@HandleOtherStates, @State1.
;   - Riding up ends in Mode5Play_GoToNextMode (the room above is entered facing up). Riding down ends
;     at Y $7F with Link's grid offset set to 2, which is not $7D, so the ride up does not restart.
;   - The raft sprite is drawn only during a ride, 6 pixels below Link's Y. Dock_Update@Draw.
; CONTENTS: (generated; L = line in this file)
;   L0028 Dock_RaftLinkDirs  data  TABLE: direction Link faces during a raft ride; index = dock state - 1; value = direction bits.
;   L0039 Dock_Update        code  Updates the dock (type $61): starts a raft ride when Link stands on it, then carries him 1 px/frame.
; ============================================================================

; TABLE: direction Link faces during a raft ride; index = dock state - 1; value = direction bits.
Dock_RaftLinkDirs:
    .BYTE $04, $08              ; State 1 (ride down from the top edge): down. State 2 (ride up): up.

; ----------------------------------------------------------------------------
; Updates the dock (type $61): starts a raft ride when Link stands on it, then carries him 1 px/frame.
; IN:   X = object slot of the dock ($0B)
; OUT:  ObjState,X = 0 idle, 1 riding down, 2 riding up; Link's ObjState = $40 (halted) during a ride.
; NOTE: Needs InvRaft. The start test is exact: Link's X = dock X ($80 in room $55, else $60) and
;       Link's Y = $7D (ride up) or $3D (ride down). Up takes 64 frames and ends by leaving the room
;       through the top; down takes 66 frames and ends at Y = $7F.
; ----------------------------------------------------------------------------
Dock_Update:
    ; Without the raft, return.
    LDA InvRaft
    BEQ @Exit
    LDA ObjState, X
    BNE @HandleOtherStates

    ; State 0: idle, waiting for Link.
    ;
    ; The dock's X is $80 in room $55, and $60 otherwise (the only other dock room is $3F).
    LDA #$80
    LDY RoomId
    CPY #$55
    BEQ :+
    LDA #$60
:
    ; If Link's X is not exactly the dock's X, return.
    CMP ObjX
    BNE @Exit

    ; Set the dock object's X. It becomes the raft's X.
    STA ObjX, X

    ; If Link's Y = $3D, he is at the top screen edge: use state 1, ride down.
    LDY #$01
    LDA ObjY
    CMP #$3D
    BEQ @SetState

    ; If Link's Y = $7D, he is standing at the dock: use state 2, ride up.
    ; Otherwise, return.
    INY
    CMP #$7D
    BNE @Exit

@SetState:
    STY ObjState, X

    ; Put the raft 6 pixels below Link's Y (A still holds Link's Y).
    CLC
    ADC #$06
    STA ObjY, X

    ; Play the "secret found" tune, and halt Link (ObjState $40: no input, no movement of his own).
    JSR Sfx_PlaySecretFoundTune
    LDA #$40
    STA ObjState

    ; Make Link face the way the raft goes: down in state 1, up in state 2.
    LDA Dock_RaftLinkDirs-1, Y
    STA ObjDir

@Exit:
    RTS

@HandleOtherStates:
    ; State 1 goes down; anything else (state 2) goes up.
    LDY ObjState, X
    DEY
    BEQ @State1

    ; State 2: move the raft and Link up 1 pixel.
    DEC ObjY, X
    DEC ObjY

    ; Until Link's Y = $3D (the top screen edge), only draw.
    ; There, leave the room: Mode5Play_GoToNextMode goes to mode 6 and clears Link's state.
    ; Link still faces up (set when the ride began).
    LDA ObjY
    CMP #$3D
    BNE @Draw
    TXA                         ; Save the dock's object slot (the call returns X = 0).
    PHA
    JSR Mode5Play_GoToNextMode
    PLA
    TAX
    JMP @ResetLinkAndRaftStateAndDraw

@State1:
    ; State 1: move the raft and Link down 1 pixel.
    INC ObjY, X
    INC ObjY

    ; Until Link's Y = $7F, only draw.
    ; The ride ends 2 pixels below $7D, the Y that starts a ride up, so Link is not taken
    ; straight back.
    LDA ObjY
    CMP #$7F
    BNE @Draw

    ; Link's grid offset at $7F is 2 ($7F - $7D): he is 2 pixels past a grid point.
    LDA #$02
    STA ObjGridOffset

@ResetLinkAndRaftStateAndDraw:
    ; Release Link (state 0) and make the dock idle again.
    LDA #$00
    STA ObjState
    STA ObjState, X

@Draw:
    TXA                         ; Save the dock's object slot.
    PHA
    JSR Link_EndMoveAndAnimate_ReturnToBank4    ; Draw Link at his new position.
    PLA                         ; Restore the dock's object slot.
    TAX

    ; Called for its outputs [00], [01] = the raft's X, Y. It also counts down the dock's
    ; ObjAnimCounter and sets [0F], which the sprite writer below resets.
    JSR Anim_AnimateObjWalking
    LDA #$00                    ; Sprite attributes 0: in front of the background, sprite palette 0.
    LDY #$09                    ; Item slot 9: the raft.
    JMP Anim_WriteStaticItemSpritesWithAttributes
