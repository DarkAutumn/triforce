; ============================================================================
; FILE:     bank4/zora-ow-inits.asm      BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 1757-1948
; SUBJECTS: (none)
; TOPIC:    Zora: spawning on a random water tile and its update (surface, shoot one fireball, dive,
;           respawn). Object init routines of Tektite, Boulder and boulder spawner, Leever, Octorok,
;           Ghini, Peahat (and the shared flyer init tail) and the pond fairy.
; SEE ALSO: bank4/leever.asm (UpdateBurrower: the state cycle, tables and drawing the Zora shares with
;           the blue Leever); bank4/monster-shots-fireballs.asm (ShootFireball, UpdateFireball);
;           bank7/mode5-play-update.asm (calls CheckZora every frame in the overworld);
;           bank7/object-init-table.asm (InitObject: HP, spawn-cloud timer, InitObject_JumpTable);
;           bank7/object-update-table.asm (a killed Zora does not count in RoomKillCount);
;           bank4/walker-wanderer-common.asm (InitWalker); bank4/fairy-flyer-common.asm (ResetFlyerState);
;           bank4/tektite-boulder.asm, bank4/octorok.asm, bank4/flying-ghini-peahat.asm,
;           bank4/pond-fairy.asm (the updates of the monsters initialized here).
; KEY FACTS:
;   - A Zora exists only in overworld rooms with bit 3 of LevelBlockAttrsA set, one at a time
;     (IsZoraActive), and needs an empty monster slot. CheckZora
;   - It appears on a random square with X in $10-$E0 and Y in $5D-$DD whose lower left tile is
;     reported as $8D-$98 (water; $8D and $91 are reported as $26 and never qualify), chosen from up
;     to 13 random bytes per frame; high nibble = X, low nibble = Y. CheckZora
;   - Cycle per surfacing: mound $20 frames, surfaced 65 frames ($0F + 34 + $10), mound $60 frames =
;     193 frames; then the object is destroyed and a new one is created in the same frame on another
;     random water square (about 195 frames from mound to mound). UpdateZora, BlueLeeverStateTimes
;   - It can be hit, and can hurt Link by touch, only while surfaced (states 2-4, 65 frames).
;     Burrower_AnimateDrawAndCheckCollisions (bank4/leever.asm), UpdateZora
;   - One fireball (type $55) per surfacing, on the third frame of state 3, i.e. $0F + 2 = 17 frames
;     after it surfaces. UpdateZora
;   - Each respawn is a new object with full HP ($20): damage that did not kill is lost when it dives.
;     UpdateZora, CheckZora
;   - Killing the Zora leaves IsZoraActive set, so none returns until the room is entered again.
;     UpdateZora
;   - With the clock the Zora update does nothing at all: no drawing, no collisions, no state change
;     (its object timer still runs down). UpdateZora
;   - Spawn clouds: Octoroks and Ghinis get a first cloud step of (slot + 1) * $10 frames; Tektites 4,
;     8, $14 or $28 frames; Boulders, Leevers, Peahats and the Zora have no cloud. InitFastOctorock,
;     InitTektite, InitBoulder, InitLeever, InitPeahat
;   - Q-speeds set at init: slow Octorok (types $07, $09), Ghini and Leever $20 (0.5 px/frame); fast
;     Octorok (types $08, $0A) $30 (0.75 px/frame; its update later stores $40, see
;     bank4/octorok.asm). InitSlowOctorockOrGhini, InitFastOctorock
;   - Every Leever init sets RedLeeverLongTimer to 5 (41-50 frames). InitLeever
; CONTENTS: (generated; L = line in this file)
;   L0075 CheckZora                code  Creates the room's Zora (type $11) on a random water tile when the room allows one and none exists.
;   L0147 TektiteStartingDirs      data  TABLE: starting directions of a Tektite or Boulder; index = Random+1,X AND 3; value = direction bits.
;   L0154 InitBoulderSet           code  Init of the boulder spawner (type $1F): zeroes the boulder count, then inits like a Boulder.
;   L0162 InitBoulder              code  Init of a Boulder (type $20): no spawn cloud, then a random direction and delay as for a Tektite.
;   L0172 InitTektite              code  Init of a Tektite (types $0D, $0E): random starting direction and a first delay of 4-$28 frames.
;   L0192 InitLeever               code  Init of a Leever (types $0F blue, $10 red): no spawn cloud; then as a slow Octorok.
;   L0202 InitSlowOctorockOrGhini  code  Init of a slow Octorok (types $07, $09) or a Ghini (type $21): q-speed $20 (0.5 pixels a frame).
;   L0217 InitFastOctorock         code  Init of a fast Octorok (types $08, $0A): q-speed $30; its tail sets timer, state and facing.
;   L0241 InitPeahat               code  Init of a Peahat (type $1A): no spawn cloud, facing up, flyer state reset, slow start.
;   L0254 EndInitFlyer             code  Finishes a flyer's init: clears its flight fields, top speed $A0, starting speed $1F.
;   L0268 InitPondFairy            code  Init of the pond fairy (type $2F): requests square-1 tune $08 and places it at ($78, $7D).
;   L0279 L10797_Exit              code  Shared RTS of InitPondFairy and UpdateZora.
;   L0299 UpdateZora               code  Updates the Zora: runs the burrower state cycle, shoots one fireball per surfacing, removes it on dive.
; ============================================================================

; ----------------------------------------------------------------------------
; Creates the room's Zora (type $11) on a random water tile when the room allows one and none exists.
; IN:   nothing (called once per frame at the end of the play update when CurLevel = 0; the caller
;       does not test the game mode, so it also runs in modes $B and $C).
; OUT:  on success: IsZoraActive = 1 and slot EmptyMonsterSlot holds an uninitialized Zora. Uses [0D],
;       [0F]; X, Y, A not preserved.
; NOTE: Room condition: bit 3 ($08) of LevelBlockAttrsA for RoomId. One Zora at a time (IsZoraActive).
; NOTE: RNG: up to 13 candidate squares per frame from the bytes Random+$0D down to Random+1. High
;       nibble = X ($10-$E0 accepted), low nibble * $10 = Y ($50-$D0 accepted, then OR $0D). The first
;       byte read, Random+$0D ($25), lies one past the 13-byte Random array; no code that writes $25
;       was found (unverified), and a 0 there fails the X test. If no candidate qualifies, the search
;       is repeated on the next frame with new random bytes.
; NOTE: Tile test: the tile under (X, Y + $0B), the lower left tile of the 16-pixel square, must be
;       $8D-$98 as returned by GetCollidableTileStill. That routine reports the overworld tiles of
;       WalkableTiles (among them $8D and $91) as $26, so those two never qualify.
; NOTE: The slot is the highest empty one of $0B-$01, so a full room has no Zora.
; ----------------------------------------------------------------------------
CheckZora:
    ; If the room's Zora bit is clear, return.
    LDY RoomId
    LDA LevelBlockAttrsA, Y
    AND #$08                    ; Overworld room attribute: Zora lives here
    BEQ @Exit

    ; If a Zora exists (or was killed in this room: see UpdateZora), return.
    LDA IsZoraActive
    BNE @Exit

    ; Look for an empty monster slot. If none is found (Z set), return.
    JSR FindEmptyMonsterSlot
    BEQ @Exit

    ; Try up to $0D random squares, counted down in [0D], looking for a water tile.
    LDA #$0D
    STA $0D

    ; Switch to the new object slot.
    LDX EmptyMonsterSlot

@LoopTileCandidate:
    ; X := high nibble of the random byte at index [0D], a multiple of $10.
    ; If it is 0 or $F0 (the outer columns), try the next candidate.
    LDY $0D
    LDA a:Random, Y
    AND #$F0
    STA ObjX, X
    BEQ @NextLoopTileCandidate
    CMP #$F0
    BEQ @NextLoopTileCandidate

    ; Y := low nibble of the same random byte * $10.
    ; If it is < $50 or >= $E0, try the next candidate.
    LDA a:Random, Y
    ASL
    ASL
    ASL
    ASL
    CMP #$50
    BCC @NextLoopTileCandidate
    CMP #$E0
    BCS @NextLoopTileCandidate

    ; Make it an object Y coordinate (square row top - 3): $5D, $6D, ... $DD.
    ORA #$0D
    STA ObjY, X

    ; Get the tile at that location. Water tiles are $8D-$98; anything else: next candidate.
    JSR GetCollidableTileStill
    CMP #$8D
    BCC @NextLoopTileCandidate
    CMP #$99
    BCS @NextLoopTileCandidate

    ; Flag the Zora active, and set up the object: type $11, state 0, timer 0, uninitialized.
    ; Its init runs on the next frame's object loop and gives it full HP ($20).
    INC IsZoraActive
    LDA #$11                    ; Zora object type
    JMP SetTypeAndClearObject

@NextLoopTileCandidate:
    ; Loop until [0D] = 0 (index 0, the byte Random itself, is never used).
    DEC $0D
    BNE @LoopTileCandidate

@Exit:
    RTS

; TABLE: starting directions of a Tektite or Boulder; index = Random+1,X AND 3; value = direction bits.
; $01 right, $02 left, $05 right+down, $0A left+up. InitTektite also uses value * 4 as the first delay.
TektiteStartingDirs:
    .BYTE $01, $02, $05, $0A

; ----------------------------------------------------------------------------
; Init of the boulder spawner (type $1F): zeroes the boulder count, then inits like a Boulder.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
InitBoulderSet:
    LDA #$00
    STA ActiveBoulderCount

; ----------------------------------------------------------------------------
; Init of a Boulder (type $20): no spawn cloud, then a random direction and delay as for a Tektite.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
InitBoulder:
    JSR ResetObjMetastateAndTimer    ; Metastate 0: skip the spawn cloud.

; ----------------------------------------------------------------------------
; Init of a Tektite (types $0D, $0E): random starting direction and a first delay of 4-$28 frames.
; IN:   X = object slot.
; OUT:  ObjDir,X = one of TektiteStartingDirs; ObjTimer,X = direction * 4 (4, 8, $14 or $28 frames).
; NOTE: RNG: Random+1,X AND 3. The metastate is left alone, so a Tektite keeps its spawn cloud and
;       this timer is the length of the cloud's first step.
; ----------------------------------------------------------------------------
InitTektite:
    LDA Random+1, X
    AND #$03
    TAY
    LDA TektiteStartingDirs, Y
    STA ObjDir, X

    ; Object timer := direction * 4.
    ASL
    ASL
    STA ObjTimer, X
    RTS

; ----------------------------------------------------------------------------
; Init of a Leever (types $0F blue, $10 red): no spawn cloud; then as a slow Octorok.
; IN:   X = object slot.
; NOTE: Sets RedLeeverLongTimer to 5 (41-50 frames) for every Leever initialized, blue ones too; red
;       Leevers do not surface before it runs out.
; NOTE: Falls into InitSlowOctorockOrGhini: q-speed $20, state 0, object timer (slot + 1) * $10 frames.
; ----------------------------------------------------------------------------
InitLeever:
    LDA #$05
    STA RedLeeverLongTimer
    JSR ResetObjMetastateAndTimer    ; Metastate 0: skip the spawn cloud.

; ----------------------------------------------------------------------------
; Init of a slow Octorok (types $07, $09) or a Ghini (type $21): q-speed $20 (0.5 pixels a frame).
; IN:   X = object slot.
; NOTE: Continues in InitFastOctorock's body: timer (slot + 1) * $10, state 0, then InitWalker.
; ----------------------------------------------------------------------------
InitSlowOctorockOrGhini:
    LDA #$20                    ; Q-speed $20 = 0.5 pixels a frame
    BNE :+                      ; Always taken.

; ----------------------------------------------------------------------------
; Init of a fast Octorok (types $08, $0A): q-speed $30; its tail sets timer, state and facing.
; IN:   X = object slot (A = q-speed when entered at the unnamed label from the routine above).
; OUT:  ObjQSpeedFrac,X set; ObjTimer,X = (slot + 1) * $10; ObjState,X = 0; ObjAnimFrame,X = 0;
;       ObjAnimCounter,X = 6; facing direction chosen by InitWalker.
; NOTE: The metastate is not reset here, so for Octoroks and Ghinis the timer is the length of the
;       spawn cloud's first step: $20 frames in slot 1 up to $C0 frames in slot $0B. For Leevers
;       (metastate already 0) it is the wait before UpdateBurrower first advances the state.
; NOTE: $30 = 0.75 pixels a frame. It lasts only until _TryShooting first stores the walking speed
;       UpdateOctorock passes for a fast Octorok, $40 (see bank4/moblin-lynel-shooting.asm).
; ----------------------------------------------------------------------------
InitFastOctorock:
    LDA #$30                    ; Q-speed $30 = 0.75 pixels a frame
:
    STA ObjQSpeedFrac, X

    ; Object timer := (object slot + 1) * $10.
    INX
    TXA
    DEX
    ASL
    ASL
    ASL
    ASL
    STA ObjTimer, X
    JSR ResetObjState           ; State 0; returns A = 0.
    STA ObjAnimFrame, X         ; Animation frame 0.
    LDA #$06
    STA ObjAnimCounter, X       ; 6 frames until the first animation frame change.
    JMP InitWalker              ; Choose a facing direction relative to the chase target.

; ----------------------------------------------------------------------------
; Init of a Peahat (type $1A): no spawn cloud, facing up, flyer state reset, slow start.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
InitPeahat:
    JSR ResetObjMetastateAndTimer    ; Metastate 0: skip the spawn cloud.
    LDA #$08                    ; Up
    STA ObjDir, X

; ----------------------------------------------------------------------------
; Finishes a flyer's init: clears its flight fields, top speed $A0, starting speed $1F.
; IN:   X = object slot.
; OUT:  Flyer_ObjPosFrac, Flyer_ObjTurns, Flyer_ObjDistTraveled, Flyer_ObjFlyingState and
;       ObjInvincibilityTimer of the slot = 0; Flyer_ObjSpeed,X = $1F; FlyerMaxSpeedFrac = $A0.
; NOTE: FlyerMaxSpeedFrac is one byte for the whole room. Also the end of the flying Ghini's init
;       (jumped to from bank4/ghini-armos.asm).
; ----------------------------------------------------------------------------
EndInitFlyer:
    JSR ResetFlyerState
    LDA #$A0
    STA FlyerMaxSpeedFrac
    LDA #$1F
    STA Flyer_ObjSpeed, X
    RTS

; ----------------------------------------------------------------------------
; Init of the pond fairy (type $2F): requests square-1 tune $08 and places it at ($78, $7D).
; IN:   X = object slot.
; NOTE: The metastate is not reset here, so the fairy starts with a spawn cloud. Tune $08 is the
;       one PlayCharacterSfx (bank1/underworld-person-init.asm) requests when a person appears.
; ----------------------------------------------------------------------------
InitPondFairy:
    LDA #$08
    STA Tune1Request

    ; The fairy goes at location ($78, $7D).
    LDA #$78
    STA ObjX, X
    LDA #$7D
    STA ObjY, X

; Shared RTS of InitPondFairy and UpdateZora.
L10797_Exit:
    RTS

; ----------------------------------------------------------------------------
; Updates the Zora: runs the burrower state cycle, shoots one fireball per surfacing, removes it on dive.
; IN:   X = object slot.
; NOTE: With the clock (InvClock) nothing runs: the Zora is not drawn, cannot be hit and cannot hurt.
; NOTE: Cycle (UpdateBurrower with the blue Leever's tables; the Zora does not move): state 1 mound
;       $20 frames; states 2, 3, 4 surfaced (one body image, front or back) $0F, 34 and $10 frames;
;       state 5 mound $60 frames; 193 frames in all. Weapons and contact with Link are tested only
;       in states 2-4 (65 frames).
; NOTE: Shoots a fireball (type $55) on the third frame of state 3 (timer $FF has reached $FD), then
;       cuts the state's timer to $20. No stun test here; ShootFireball fails silently when no
;       monster slot is empty.
; NOTE: Reaching state 0 destroys the object and clears IsZoraActive; CheckZora makes a new Zora
;       later in the same frame if it finds water (first state-1 frame 2 frames later, so about 195
;       frames per cycle). The new object gets full HP ($20): damage does not carry over a dive.
; NOTE: A killed Zora leaves IsZoraActive at 1 (the dying object never comes back here), so no
;       other Zora appears until the room is entered again (room entry zeroes IsZoraActive).
; ----------------------------------------------------------------------------
UpdateZora:
    ; If we have the clock, then return without updating or drawing.
    LDA InvClock
    BNE L10797_Exit
    JSR UpdateBurrower          ; Timer, state cycle, drawing; collisions in states 2-4.

    ; If the Zora was killed in that call (possible only in states 2-4), the state is not 0 below, so
    ; IsZoraActive stays set; from the next frame on the death sequence runs instead of this routine.
    ;
    ; If state <> 3 or timer <> $FD, then skip shooting.
    LDA ObjState, X
    CMP #$03
    BNE @DestroyWhenDone
    LDA ObjTimer, X
    CMP #$FD                    ; State 3 starts with timer $FF, so this is its third frame.
    BNE @DestroyWhenDone

    ; Shoot a fireball, and shorten the rest of state 3 to $20 frames.
    LDA #$55                    ; Fireball object type
    JSR ShootFireball
    LDA #$20
    STA ObjTimer, X

@DestroyWhenDone:
    ; If state = 0 (the cycle wrapped from state 5: the Zora has dived), destroy the object and
    ; clear the active flag, so that CheckZora makes another one.
    LDA ObjState, X
    BNE L10797_Exit
    DEC IsZoraActive
    JMP DestroyMonster
