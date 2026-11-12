; ============================================================================
; FILE:     bank5/warps-stairs-caves.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7255-7407
; SUBJECTS: (none)
; TOPIC:    The transitions Link starts by standing on a tile in mode 5: dungeon stairs to a cellar,
;           overworld stairs and openings to a cave, a shortcut cave or a dungeon level. Also banks the
;           room's kill count when Link leaves a room.
; SEE ALSO: bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate, the only caller of CheckWarps);
;           bank5/mode8-continue-mode10-stairs.asm and bank5/mode-inits-obj-lists.asm (mode $10, the
;           walk down the stairs, which ends in ModeAfterStairs); bank5/cave-cellar-enter.asm (modes 9
;           and $B: arriving in the cellar or cave); bank5/cave-cellar-exits.asm (leaving them);
;           bank5/room-history-ow.asm and bank5/room-history-uw.asm (SaveKillCountOW, SaveKillCountUW);
;           bank7/init-mode-dispatch.asm (InitMode3_Sub1 reads LevelEntranceRoomId);
;           bank7/tile-collision.asm (GetCollidableTileStill).
; KEY FACTS:
;   - The check runs only in mode 5 with Link's grid offset 0, UndergroundExitType 0, Y low nibble $D
;     and X a multiple of $10 (a multiple of 8 in OW room $22) (CheckWarps).
;   - UndergroundExitType stays nonzero after Link comes out of a cave, cellar or dungeon until he
;     completes one grid step, so the entrance he stands on does not take him back in (CheckWarps).
;   - UW: tiles $70-$73 under Link are stairs: the cellar whose attribute A or B names this room
;     becomes RoomId and mode $10 leads to mode 9; sound keeps playing (CheckWarps).
;   - OW: tile $24, tile $88 or tiles $70-$73 under Link start a transition chosen by the room's cave
;     index (LevelBlockAttrsB bits 7-2): 1-9 load that level (mode 2), $14 shortcut cave (mode $C),
;     any other value $10 or more a cave (mode $B) (HandleWarpOW).
;   - LevelEntranceRoomId is set here, to the OW room of the entrance, only when a level is entered;
;     caves and cellars do not touch it (HandleWarpOW@LoadLevel).
;   - Entering a cave, shortcut cave or level silences all sound and clears FluteTimer; entering a
;     cellar does neither (SetTargetMode).
;   - Every transition here goes through mode $10 first; ModeAfterStairs holds the mode that follows
;     (SetTargetMode).
;   - The room's kill count is banked before each of these transitions (SaveKillCount).
; CONTENTS: (generated; L = line in this file)
;   L0054 CheckWarps     code  Starts a stairs, cave or level-entrance transition when Link stands on such a tile.
;   L0130 SetTargetMode  code  Starts mode $10 (stairs) and records the mode to enter when it ends.
;   L0150 SaveKillCount  code  Banks the current room's kill count into the room's saved state (OW or UW version).
;   L0158 L1746E_Exit    code  Shared return of CheckWarps, SaveKillCount and HandleWarpOW.
;   L0174 HandleWarpOW   code  Overworld part of CheckWarps: enters the cave, shortcut cave or level under Link.
; ============================================================================

; ----------------------------------------------------------------------------
; Starts a stairs, cave or level-entrance transition when Link stands on such a tile.
; IN:   Link's ObjX, ObjY, ObjGridOffset; UndergroundExitType; RoomId; CurLevel
; OUT:  on a transition: GameMode = $10, ModeAfterStairs set, Link's state, knockback and
;       invincibility timer cleared (EndPrepareMode); ObjCollidedTile (slot 0) overwritten
; NOTE: Called from Link_EndMoveAndAnimate only, after Link has moved, and only when GameMode = 5.
;       The caller saves ObjCollidedTile before the call and restores it after.
; NOTE: The tile tested is the one at (ObjX, ObjY + $0B): with Y = $xD that is the lower left tile
;       of the 16-pixel square Link stands on (GetCollidableTileStill).
; NOTE: Requires grid offset 0, so a grid point that Link passes without a frame at offset 0 is not
;       tested.
; NOTE: OW room $22 is tested by number: there X only has to be a multiple of 8 (upstream: the
;       level 6 entrance is wide; which level uses the room is unverified here).
; ----------------------------------------------------------------------------
CheckWarps:
    ; Return if Link just came out of a cave, dungeon or cellar and has not
    ; finished a grid step since, or if he is between grid points.
    LDA UndergroundExitType
    ORA ObjGridOffset
    BNE L1746E_Exit

    ; OW room $22: X must be a multiple of 8.
    LDA CurLevel
    BNE @EnsureSquareX
    LDA RoomId
    CMP #$22
    BNE @EnsureSquareX
    LDA ObjX
    AND #$07
    BNE L1746E_Exit
    BEQ @EnsureSquareY

@EnsureSquareX:
    ; Every other OW room and all of UW: X must be a multiple of $10.
    LDA ObjX
    AND #$0F
    BNE L1746E_Exit

@EnsureSquareY:
    ; Y must be (a multiple of $10) + $D: Link aligned with a row of squares.
    LDA ObjY
    AND #$0F
    CMP #$0D
    BNE L1746E_Exit

    ; Get the tile under Link (no direction: X = 0 is still Link's slot).
    JSR GetCollidableTileStill

    ; OW is handled separately.
    LDA ObjCollidedTile
    LDY CurLevel
    BEQ HandleWarpOW

    ; In UW: only stairs tiles $70-$73 count.
    CMP #$70
    BCC L1746E_Exit
    CMP #$74
    BCS L1746E_Exit

    ; Link is on dungeon stairs. Bank the kills and remember this room.
    JSR SaveKillCount
    LDA RoomId
    STA CellarSourceRoomId      ; Used to pick the cellar's left or right ladder on arrival.

    ; Find the first cellar room in the level's cellar list whose attribute A (room of
    ; the left stairs) or B (room of the right stairs) is the current room.
    ; NOTE: the search has no end test; it relies on a match existing.
    LDX #$FF
:
    INX
    LDA LevelInfo_CellarRoomIdArray, X
    TAY
    LDA RoomId
    CMP LevelBlockAttrsA, Y
    BEQ :+
    CMP LevelBlockAttrsB, Y
    BNE :-
:
    ; The cellar becomes the current room; after the stairs comes mode 9.
    STY RoomId
    LDA #$09

; ----------------------------------------------------------------------------
; Starts mode $10 (stairs) and records the mode to enter when it ends.
; IN:   A = mode after the stairs: 9 cellar, $B cave, $C shortcut cave, 2 load level
; OUT:  ModeAfterStairs = A; GameMode = $10 with submode 0, not yet updating; Link's state,
;       knockback and invincibility timer = 0 (EndPrepareMode)
; NOTE: Unless A = 9, all sound is silenced and Tune1Request and FluteTimer are set to 0. A cellar
;       keeps the dungeon's sound.
; ----------------------------------------------------------------------------
SetTargetMode:
    STA ModeAfterStairs

    ; Not a cellar: silence all sound (SilenceAllSound returns A = 0).
    CMP #$09
    BEQ :+
    JSR SilenceAllSound
    STA Tune1Request
    STA FluteTimer
:
    LDA #$10
    STA GameMode
    JSR MaskCurPpuMaskGrayscale
    JMP EndPrepareMode

; ----------------------------------------------------------------------------
; Banks the current room's kill count into the room's saved state (OW or UW version).
; NOTE: Called before each stairs or entrance transition here and when Link leaves a room by its
;       edge (mode 6 init).
; ----------------------------------------------------------------------------
SaveKillCount:
    LDA CurLevel
    BNE :+
    JMP SaveKillCountOW
:
    JSR SaveKillCountUW

; Shared return of CheckWarps, SaveKillCount and HandleWarpOW.
L1746E_Exit:
    RTS

; ----------------------------------------------------------------------------
; Overworld part of CheckWarps: enters the cave, shortcut cave or level under Link.
; IN:   A = tile under Link (OW tiles listed in WalkableTiles arrive as $26)
; OUT:  UndergroundEntranceTile = A, even when A is no entrance; on a transition as SetTargetMode,
;       and for a level also CurLevel and LevelEntranceRoomId
; NOTE: Entrance tiles: $24 (the black opening: Link later sinks into it and rises out of it),
;       $88 (what square uses it is unverified) and the stairs tiles $70-$73.
; NOTE: The destination comes from the room's cave index, LevelBlockAttrsB bits 7-2: below $10 it
;       is the level number (index 0 is not excluded: it would load level 0, the OW); $14 is the
;       shortcut cave; anything else is an ordinary cave.
; NOTE: The $70 written to ObjCollidedTile for stairs does not survive: the caller of CheckWarps
;       restores ObjCollidedTile, and mode $10's init reads the tile under Link again.
; ----------------------------------------------------------------------------
HandleWarpOW:
    ; Remember the tile, so the exit later knows whether Link walked into an
    ; opening ($24) or took stairs.
    STA UndergroundEntranceTile

    ; Return unless the tile is $24, $88 or $70-$73.
    CMP #$24
    BEQ @GoUnderground
    CMP #$88
    BEQ @GoUnderground
    CMP #$70
    BCC L1746E_Exit
    CMP #$74
    BCS L1746E_Exit

    ; Any stairs tile is recorded as $70 (see the NOTE above).
    LDA #$70
    STA ObjCollidedTile

@GoUnderground:
    JSR SaveKillCount

    ; Get the room's cave index, still shifted left by 2.
    LDY RoomId
    LDA LevelBlockAttrsB, Y
    AND #$FC

    ; Below $40 (cave index below $10): a level entrance.
    CMP #$40
    BCC @LoadLevel

    ; $50 (cave index $14): shortcut cave, mode $C. Anything else: cave, mode $B.
    LDY #$0B
    CMP #$50
    BNE :+
    INY
:
    TYA
    JMP SetTargetMode

@LoadLevel:
    ; The cave index is the level number. Remember the entrance room: mode 3 puts
    ; Link back in it when the OW is loaded again.
    LSR
    LSR
    STA CurLevel
    LDA RoomId
    STA LevelEntranceRoomId
    LDA #$02                    ; After the stairs: mode 2, load the level.
    BNE SetTargetMode
