; ============================================================================
; FILE:     bank4/ganon.asm              BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 10284-10545
; SUBJECTS: Ganon
; TOPIC:    Ganon (type $3E): the entry scene (Link holds up a Triforce piece in the dark, then in
;           the light), the fight (invisible and moving, visible after a hit, brown and vulnerable
;           to the silver arrow), and the death sequence that ends with the room item appearing.
; SEE ALSO: bank4/ganon-draw-collisions.asm (Ganon_CheckCollisions: sword and silver arrow rules;
;           Ganon_DrawBody, burst, ashes, palettes, Ganon_ActivateRoomItem);
;           bank4/patra-ganon-inits.asm (Ganon_Init: mask $FA, halts Link, Link's timer $40);
;           bank4/wizzrobe.asm (Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile: Ganon's movement);
;           bank4/monster-shots-fireballs.asm (Fireball_Shoot, Fireball_Update);
;           bank4/shoot-helpers-candle.asm (Candle_UpdateRoomLight: the room brightening).
; KEY FACTS:
;   - Ganon_ScenePhase: 0 = Link halted holding item $1B in the dark; 1 = the same in the light
;     for $C0 frames; 2 = fight (Ganon_Update_JumpTable).
;   - Phase 0 waits for Link's ObjTimer ($40 from Ganon_Init), then fades the room to light with
;     Candle_UpdateRoomLight and ends when FadeCycle's low nibble is 4 (Ganon_ScenePhase0_LiftInDark).
;   - In the fight Ganon has ObjState 0 = blue, nonzero = brown. Blue with ObjTimer 0 he is not
;     drawn, moves 1 pixel per frame like a teleporting Blue Wizzrobe and shoots a fireball (type
;     $56) whenever FrameCounter AND $3F = 0 (Ganon_ScenePhase2_Fight, Ganon_MoveAndShoot).
;   - Blue with ObjTimer nonzero (set to $40 by a sword hit in Ganon_CheckCollisions) he stands
;     still and is drawn; when the timer reaches 1 he jumps to Y = $A0, X = $30 or $B0 by bit 0 of
;     FrameCounter (Ganon_ScenePhase2_Fight, Ganon_RandomizeLocation, Ganon_StartXs).
;   - Brown: ObjState starts at $FF (set in Ganon_CheckCollisions when his HP runs out) and drops
;     by 1 on every odd FrameCounter, about 510 frames; he does not move or shoot; he is drawn
;     every frame while the state is >= $30, then only on odd frames. At 0 he turns blue again
;     at a new start spot (Ganon_UpdateBrownState).
;   - Dying (Ganon_ObjDyingCount nonzero, started by the silver arrow in Ganon_CheckCollisions):
;     +1 per frame; body drawn below $50; at $50 ashes and burst; burst drawn below $A0; at $A0
;     the room item appears at the ashes and RoomKillCount is incremented (Ganon_UpdateDying).
;   - Ganon's object is never destroyed and never becomes a dropped item (Ganon_UpdateDying).
;   - Collisions (Link's touch included) are tested every fight frame, also while Ganon is not
;     drawn (Ganon_ScenePhase2_Fight).
; CONTENTS: (generated; L = line in this file)
;   L0054 Ganon_Update                   code  Updates Ganon: runs the handler of the current scene phase (Ganon_ScenePhase).
;   L0059 Ganon_Update_JumpTable         data  TABLE: scene phase handlers; index = Ganon_ScenePhase.
;   L0071 Ganon_ScenePhase0_LiftInDark   code  Scene phase 0: Link holds up a Triforce piece in the dark; then the room fades to light.
;   L0127 Ganon_ScenePhase1_LiftInLight  code  Scene phase 1: Link holds up the Triforce piece in the lit room until his timer ($C0) runs out.
;   L0144 Ganon_DrawBodyFrame0           code  Draws Ganon with animation frame 0 (the pose used during the entry scene).
;   L0156 Ganon_ScenePhase2_Fight        code  Scene phase 2, the fight: collisions, then the blue (state 0) or brown state, or the death.
;   L0180 Ganon_RandomizeLocation        code  Puts Ganon at Y = $A0 and X = $30 (FrameCounter even) or $B0 (FrameCounter odd).
;   L0195 Ganon_MoveAndShoot             code  Blue Ganon, not visible: steps his animation frame, moves 1 pixel, shoots every $40 frames.
;   L0223 Ganon_StartXs                  data  TABLE: Ganon's start X coordinates; index = FrameCounter AND 1; value = pixels.
;   L0234 Ganon_UpdateBrownState         code  Brown Ganon: counts ObjState down every other frame; at 0 he turns blue at a new start spot.
;   L0256 Ganon_DrawBodyAndReturn        code  Draws Ganon's body and returns; shared by the blue, brown and dying code.
;   L0270 Ganon_UpdateDying              code  Ganon's death: counts Ganon_ObjDyingCount up each frame; body, then ashes and burst, then the item.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates Ganon: runs the handler of the current scene phase (Ganon_ScenePhase).
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Ganon_Update:
    LDA Ganon_ScenePhase
    JSR Util_TableJump

; TABLE: scene phase handlers; index = Ganon_ScenePhase.
Ganon_Update_JumpTable:
    .ADDR Ganon_ScenePhase0_LiftInDark         ; 0: Link holds up the Triforce piece, room dark, then fading in
    .ADDR Ganon_ScenePhase1_LiftInLight         ; 1: Link holds it up in the lit room for $C0 frames
    .ADDR Ganon_ScenePhase2_Fight         ; 2: the fight and Ganon's death

; ----------------------------------------------------------------------------
; Scene phase 0: Link holds up a Triforce piece in the dark; then the room fades to light.
; IN:   X = object slot; ObjTimer (Link's, slot 0) = frames left in the dark ($40 from Ganon_Init).
; OUT:  Ganon_ScenePhase = 1 and Link's ObjTimer = $C0 when the fade is done.
; NOTE: Link is halted (ObjState $40, set by Ganon_Init) through phases 0 and 1. Ganon is not
;       drawn while Link's timer runs; during the fade he is drawn with animation frame 0.
; ----------------------------------------------------------------------------
Ganon_ScenePhase0_LiftInDark:
    LDA #$1B                    ; Item $1B (Triforce piece) is drawn held above Link.
    STA ItemTypeToLift

    ; While Link's timer runs, do nothing except the cry on its last frame.
    LDA ObjTimer
    BNE @CheckTimeToShout

    ; Timer expired: fade the room to light with the candle code (X is saved around the call).
    TXA
    PHA
    JSR Candle_UpdateRoomLight
    PLA
    TAX

    ; Candle_UpdateRoomLight sets IsRoomBrightening; clear it, because the fade is driven from here.
    LDA #$00
    STA IsRoomBrightening

    ; While the fade is still at its first value ($C0: fade to light, step 0), request song $02.
    LDA FadeCycle
    CMP #$C0
    BNE @CheckFadeCycle
    LDA #$02
    STA SongRequest

@CheckFadeCycle:
    ; Until the fade step (low nibble) reaches 4 = done, only draw Ganon.
    LDA FadeCycle
    AND #$0F
    CMP #$04
    BNE Ganon_DrawBodyFrame0

    ; Fade done: Link's timer = $C0 frames for phase 1.
    LDA #$C0
    STA ObjTimer

    ; Go to scene phase 1, and draw Ganon.
    INC Ganon_ScenePhase
    BNE Ganon_DrawBodyFrame0

@CheckTimeToShout:
    ; On the frame Link's timer is 1, request sample $02 (the boss hit cry).
    CMP #$01
    BNE :+
    LDA #$02
    STA SampleRequest
:
    RTS

; ----------------------------------------------------------------------------
; Scene phase 1: Link holds up the Triforce piece in the lit room until his timer ($C0) runs out.
; IN:   X = object slot; ObjTimer (Link's) = frames left.
; OUT:  at the end: Link's ObjState = 0 (control returns), ItemTypeToLift = 0, song $20 (level 9)
;       requested, Ganon_ScenePhase = 2.
; ----------------------------------------------------------------------------
Ganon_ScenePhase1_LiftInLight:
    LDA #$1B                    ; Item $1B (Triforce piece) stays above Link.
    STA ItemTypeToLift

    ; While Link's timer runs, only draw Ganon.
    LDA ObjTimer
    BNE Ganon_DrawBodyFrame0

    ; Timer expired (A = 0): unhalt Link, stop the item lift, request the level 9 song ($20),
    ; and go to scene phase 2. Ganon is drawn this frame too (falls through).
    STA ObjState
    STA ItemTypeToLift
    LDA #$20
    STA SongRequest
    INC Ganon_ScenePhase

; Draws Ganon with animation frame 0 (the pose used during the entry scene).
Ganon_DrawBodyFrame0:
    LDA #$00
    STA Ganon_ObjAnimationFrame, X
    JMP Ganon_DrawBody

; ----------------------------------------------------------------------------
; Scene phase 2, the fight: collisions, then the blue (state 0) or brown state, or the death.
; IN:   X = object slot; ObjState,X = 0 blue, nonzero brown; ObjTimer,X = frames left visible.
; NOTE: Blue: timer 0 = not drawn, moving and shooting (Ganon_MoveAndShoot); timer > 1 = drawn,
;       standing still; timer = 1 = moved to a start spot, not drawn that frame.
; NOTE: Falls through into Ganon_RandomizeLocation when the timer is 1.
; ----------------------------------------------------------------------------
Ganon_ScenePhase2_Fight:
    ; Dying: nothing else runs.
    LDA Ganon_ObjDyingCount, X
    BNE Ganon_UpdateDying
    JSR Ganon_CheckCollisions
    JSR Sample_PlayBossHitCryIfHit

    ; State nonzero: brown.
    LDA ObjState, X
    BNE Ganon_UpdateBrownState

    ; State 0: blue. Timer 0: move around unseen and shoot.
    LDA ObjTimer, X
    BEQ Ganon_MoveAndShoot

    ; Timer > 1: visible after a hit; only draw (collisions were tested above).
    ; Timer = 1: fall through and put Ganon at a start spot before he moves again next frame.
    CMP #$01
    BNE Ganon_DrawBodyAndReturn

; ----------------------------------------------------------------------------
; Puts Ganon at Y = $A0 and X = $30 (FrameCounter even) or $B0 (FrameCounter odd).
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Ganon_RandomizeLocation:
    LDA #$A0
    STA ObjY, X
    LDA FrameCounter
    AND #$01
    TAY
    LDA Ganon_StartXs, Y
    STA ObjX, X
    RTS

; ----------------------------------------------------------------------------
; Blue Ganon, not visible: steps his animation frame, moves 1 pixel, shoots every $40 frames.
; IN:   X = object slot.
; NOTE: Nothing is drawn here. The fireball (type $56) is made only if a monster slot is free.
; ----------------------------------------------------------------------------
Ganon_MoveAndShoot:
    ; Step the animation frame 0-5 every frame, so that the pose shown when he is hit varies.
    INC Ganon_ObjAnimationFrame, X
    LDA Ganon_ObjAnimationFrame, X
    CMP #$06
    BNE @Move
    LDA #$00
    STA Ganon_ObjAnimationFrame, X

@Move:
    ; Move as a teleporting Blue Wizzrobe does: remaining distance 1 (nonzero) makes the
    ; wizzrobe code pass through blocks and water; it turns toward Link every $40 calls and
    ; moves 1 pixel per call.
    LDA #$01
    STA ObjRemDistance, X
    JSR Wizzrobe_Blue_TurnSometimesAndMoveAndCheckTile

    ; Shoot when the low 6 bits of FrameCounter are 0: once every $40 frames.
    LDA FrameCounter
    AND #$3F
    BNE @Exit
    LDA #$56                    ; Fireball, object type $56
    JSR Fireball_Shoot

@Exit:
    RTS

; TABLE: Ganon's start X coordinates; index = FrameCounter AND 1; value = pixels.
Ganon_StartXs:
    .BYTE $30, $B0

; ----------------------------------------------------------------------------
; Brown Ganon: counts ObjState down every other frame; at 0 he turns blue at a new start spot.
; IN:   X = object slot; ObjState,X = $FF when he turns brown.
; NOTE: About 510 frames from $FF to 0. Drawn every frame while the state is >= $30, on odd
;       frames only below that. He does not move or shoot while brown.
; NOTE: His HP is back at $F0 from the moment he turns brown (Ganon_CheckCollisions), so a missed
;       silver arrow means the whole sword phase again.
; ----------------------------------------------------------------------------
Ganon_UpdateBrownState:
    ; Even FrameCounter: no countdown this frame.
    LDA FrameCounter
    LSR
    BCC @Draw
    DEC ObjState, X

    ; State reached 0: move to a start spot and queue the blue palette; not drawn this frame.
    BNE @Draw
    JSR Ganon_RandomizeLocation
    JMP Ganon_AppendPaletteRowTransferRecord_Blue

@Draw:
    ; State >= $30: draw every frame. Below: draw on odd frames only (flicker before turning blue).
    LDA ObjState, X
    CMP #$30
    BCS Ganon_DrawBodyAndReturn
    LDA FrameCounter
    LSR
    BCC :+

; Draws Ganon's body and returns; shared by the blue, brown and dying code.
Ganon_DrawBodyAndReturn:
    JSR Ganon_DrawBody
:
    RTS

; ----------------------------------------------------------------------------
; Ganon's death: counts Ganon_ObjDyingCount up each frame; body, then ashes and burst, then the item.
; IN:   X = object slot; Ganon_ObjDyingCount,X nonzero (set to 1 by the silver arrow hit).
; OUT:  at $50: palette, position moved to the ashes, burst set up, death cry, song $02;
;       at $A0: room item (slot $13) activated at the ashes, RoomKillCount + 1.
; NOTE: Count < $50: body drawn (the arrow hit sets his invincibility timer to $28; that this
;       makes the body flash is (unverified)). $50-$9F: ashes and burst. From $A0: ashes only.
;       The count stops at $FF.
; ----------------------------------------------------------------------------
Ganon_UpdateDying:
    INC Ganon_ObjDyingCount, X

    ; Hold the count at $FF instead of wrapping to 0.
    LDA Ganon_ObjDyingCount, X
    BNE :+
    LDA #$FF
    STA Ganon_ObjDyingCount, X
:
    ; Count < $50: only draw the body.
    CMP #$50
    BCC Ganon_DrawBodyAndReturn

    ; Count > $50: ashes (and burst).
    BNE @HandleAshes

    ; Count = $50: queue the palette for the ashes and the room item.
    JSR Ganon_AppendPaletteRowTransferRecord_Triforce

    ; Move the object to where the ashes are drawn. There is no CLC: carry is still set from the
    ; CMP #$50 above (the palette routine ends its copy loop on an equal CPY and does not change
    ; it afterwards), so X gains 8; that addition clears carry unless X wraps, so Y gains 8.
    LDA ObjX, X
    ADC #$07
    STA ObjX, X
    LDA ObjY, X
    ADC #$08
    STA ObjY, X
    JSR Ganon_SetUpBurstRays
    JSR Sample_PlayBossDeathCry
    LDA #$02                    ; Song $02, the same request as when the room lit up.
    STA SongRequest

@HandleAshes:
    JSR Ganon_DrawAshes

    ; Count < $A0: draw the burst (clouds and rays), and return from there.
    LDA Ganon_ObjDyingCount, X
    CMP #$A0
    BCC Ganon_DrawBurst

    ; Count > $A0: nothing more.
    BNE @Exit

    ; Count = $A0: make the room item appear (if it was not taken before) ...
    JSR Ganon_ActivateRoomItem

    ; ... at the ashes (slot $13 = 19 is the room item's slot).
    LDA ObjX, X
    STA ObjX+19
    LDA ObjY, X
    STA ObjY+19

    ; Count Ganon as one kill in this room.
    INC RoomKillCount

@Exit:
    RTS
