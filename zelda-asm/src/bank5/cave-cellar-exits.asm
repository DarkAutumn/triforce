; ============================================================================
; FILE:     bank5/cave-cellar-exits.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 3059-3217
; SUBJECTS: Mode, ModeALeaveCellar, Warp
; TOPIC:    Leaving a cellar (up its stairs), a cave (off the screen edge) or the shortcut cave (down one
;           of its three stairs): picks the destination room and starts mode $A.
; SEE ALSO: bank7/walker-move.asm (Walker_Move calls Warp_CheckSubroomExit for Link in modes 9, $B, $C);
;           bank7/walker-collision-screen-edge.asm (Link_CheckScreenEdge, Link_ScreenEdgeBounds);
;           bank1/modeb-cave-save-sprites.asm (Person_BlockLinkMovingUp); bank5/cave-cellar-enter.asm (mode $A);
;           bank5/warps-stairs-caves.asm (the way in; also jumps to Mode_EndPrepareMode).
; KEY FACTS:
;   - Cellar (mode 9): Link leaves when his Y < $40 and Up is held. Destination room =
;     LevelBlockAttrsA[cellar room] if his X < $80, else LevelBlockAttrsB[cellar room] (Warp_CheckSubroomExit).
;   - He comes out at X = high nibble of LevelBlockAttrsC[cellar room] * $10, Y = low nibble * $10 + $0D,
;     the same spot whichever of the two rooms he goes to (Warp_CheckSubroomExit).
;   - Cave (modes $B, $C): Link cannot move up while his Y < $8E (Person_BlockLinkMovingUp is called whether
;     or not a person is there). Reaching a screen edge bound while moving toward it leaves the cave:
;     mode $A with RoomId unchanged (Warp_CheckCaveEdgeExit).
;   - Shortcut cave (mode $C): with grid offset 0 and Y = $9D, X = $50, $80 or $B0 selects stairs 1, 2
;     or 3. Destination = LevelInfo_CellarRoomIdArray[(i + n) AND 3], i = index of the current room in
;     that array, n = stairs number (Warp_CheckSubroomExit).
;   - Every exit clears Link's state, knockback and invincibility timer and the moving direction [0F]
;     (Mode_EndPrepareMode).
;   - Warp_CheckCaveEdgeExit calls Link_CheckScreenEdge directly, so the mode 5, ladder and grid offset 0 conditions
;     that guard it in Walker_CheckTileCollision do not apply in a cave.
; CONTENTS: (generated; L = line in this file)
;   L0043 Warp_CheckSubroomExit          code  Tests whether Link is leaving the cellar or cave he is in, and starts mode $A if so.
;   L0173 ModeALeaveCellar_GoFromCellar  code  Makes room A the current room, marks it visited and starts mode $A (leave cellar or cave).
;   L0178 ModeALeaveCellar_GoFromCave    code  Starts mode $A without changing RoomId: Link leaves a cave back into the room it belongs to.
;   L0188 Mode_EndPrepareMode            code  Ends the current mode's update so the new GameMode starts with its init next frame.
;   L0199 Mode_EndPrepareMode_Return     code  Shared exit of Warp_CheckSubroomExit and Mode_EndPrepareMode.
;   L0209 Warp_CheckCaveEdgeExit         code  Lets Link leave a cave by walking off a screen edge: turns the edge's mode change into mode $A.
; ============================================================================

; ----------------------------------------------------------------------------
; Tests whether Link is leaving the cellar or cave he is in, and starts mode $A if so.
; IN:   [0F] = Link's moving direction; GameMode = 9 (cellar), $B (cave) or $C (shortcut cave).
; OUT:  [0F] unchanged, or 0 when blocked (cave, moving up while Y < $8E) or leaving. When leaving:
;       GameMode = $A, RoomId = destination (cellar and shortcut exits), Link's X and Y set (cellar).
; NOTE: Cellar exit: Y < $40 with Up held; left half (X < $80) and right half lead to different rooms.
;       Shortcut stairs: Y = $9D, X = $50 / $80 / $B0, grid offset 0.
; ----------------------------------------------------------------------------
Warp_CheckSubroomExit:
    LDA GameMode
    CMP #$09
    BNE @InCave

    ; In a cellar.
    ;
    ; If Link's Y >= $40 or Up is not held, return.
    LDA ObjY
    CMP #$40
    BCS Potion_Wield_Return
    LDA ObjInputDir
    AND #$08
    BEQ Potion_Wield_Return

    ; Look for this room's ID in the 6-element cellar room array.
    ; The index found is not used (Y is overwritten below); the loop has no end test, so it relies on
    ; the room being in the array.
    LDY #$06
    LDA RoomId
    PHA                         ; Save the cellar's room ID.
:
    DEY
    CMP LevelInfo_CellarRoomIdArray, Y
    BNE :-

    ; Determine the destination room ID, and go there.
    ;
    ; If Link's X < $80 look in level block attributes A, else B.
    ;
    ; In cellar rooms, level block attributes A and B hold nothing but
    ; the destination room IDs (left stairs, right stairs).
    TAY                         ; Y = the cellar's room ID.
    LDA ObjX
    CMP #$80
    BCS @GetRightSideRoom
    LDA LevelBlockAttrsA, Y
    JMP :+

@GetRightSideRoom:
    LDA LevelBlockAttrsB, Y
:
    JSR ModeALeaveCellar_GoFromCellar

    ; Set Link's position in the destination room from attribute C of the cellar room:
    ; X = high nibble * $10; Y = low nibble * $10 + $0D.
    PLA
    TAY
    LDA LevelBlockAttrsC, Y
    PHA
    AND #$F0
    STA ObjX
    PLA
    ASL
    ASL
    ASL
    ASL
    ORA #$0D
    STA ObjY
    RTS

@InCave:
    ; In a cave (mode $B or $C; A = game mode).
    ;
    ; Stop upward movement while Link's Y < $8E: the upper part of the room,
    ; where the person and wares are. This runs whether or not a person is present.
    PHA
    JSR Person_BlockLinkMovingUp
    PLA

    ; If mode is not $C (shortcuts), go check the screen edge.
    CMP #$0C
    BNE Warp_CheckCaveEdgeExit

    ; In a shortcut cave (mode $C).
    ;
    ; If grid offset <> 0, return.
    LDA ObjGridOffset
    BNE Mode_EndPrepareMode_Return

    ; If Link's Y <> $9D, go check the screen edge.
    LDA ObjY
    CMP #$9D
    BNE Warp_CheckCaveEdgeExit

    ; See if Link is on one of the 3 shortcut stairs.
    ; X = $50: 1
    ; X = $80: 2
    ; X = $B0: 3
    ;
    ; If he's not on any, then return.
    LDY #$01
    LDA ObjX
    CMP #$50
    BEQ @FoundStairs
    INY
    CMP #$80
    BEQ @FoundStairs
    INY
    CMP #$B0
    BNE Mode_EndPrepareMode_Return

@FoundStairs:
    ; Look for the current room in the shortcut room array (in the overworld's level info this
    ; array holds the shortcut rooms). No end test: relies on the room being in the array.
    STY $00                     ; [00] = stairs number, 1 to 3.
    LDY #$FF
    LDA RoomId
:
    INY
    CMP LevelInfo_CellarRoomIdArray, Y
    BNE :-

    ; Add the stairs number to the index the current room has in the array.
    ;
    ; This yields the index of one of the three shortcut rooms
    ; after the current one.
    TYA
    CLC
    ADC $00

    ; Wrap around within the first 4 entries. Go to that room by way of mode $A.
    AND #$03
    TAY
    LDA LevelInfo_CellarRoomIdArray, Y

; ----------------------------------------------------------------------------
; Makes room A the current room, marks it visited and starts mode $A (leave cellar or cave).
; IN:   A = destination room ID.
; ----------------------------------------------------------------------------
ModeALeaveCellar_GoFromCellar:
    STA RoomId
    JSR RoomFlags_MarkVisited

; Starts mode $A without changing RoomId: Link leaves a cave back into the room it belongs to.
ModeALeaveCellar_GoFromCave:
    LDA #$0A
    STA GameMode

; ----------------------------------------------------------------------------
; Ends the current mode's update so the new GameMode starts with its init next frame.
; OUT:  GameSubmode, IsUpdatingMode, [0F] (moving direction), Link's ObjState, ObjShoveDir,
;       ObjShoveDistance and ObjInvincibilityTimer = 0.
; NOTE: Link's invincibility frames and knockback do not survive a mode change made through here.
; ----------------------------------------------------------------------------
Mode_EndPrepareMode:
    LDA #$00
    STA GameSubmode
    STA IsUpdatingMode
    STA $0F
    STA ObjState
    STA ObjShoveDir
    STA ObjShoveDistance
    STA ObjInvincibilityTimer

; Shared exit of Warp_CheckSubroomExit and Mode_EndPrepareMode.
Mode_EndPrepareMode_Return:
    RTS

; ----------------------------------------------------------------------------
; Lets Link leave a cave by walking off a screen edge: turns the edge's mode change into mode $A.
; NOTE: Link_CheckScreenEdge fires when ObjInputDir <> 0 and Link's coordinate on the axis of its single
;       direction equals Link_ScreenEdgeBounds for that direction; it then increments GameMode and
;       clears IsUpdatingMode, which is the signal tested here. Called this way it skips the mode 5,
;       ladder and grid offset tests made in Walker_CheckTileCollision.
; ----------------------------------------------------------------------------
Warp_CheckCaveEdgeExit:
    JSR Link_CheckScreenEdge

    ; If Link reached the screen edge, Link_CheckScreenEdge cleared IsUpdatingMode and incremented
    ; GameMode. Go replace that mode with $A.
    LDA IsUpdatingMode
    BEQ ModeALeaveCellar_GoFromCave
    RTS
