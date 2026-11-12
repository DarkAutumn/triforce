; ============================================================================
; FILE:     bank5/link-handle-input.asm  BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 6919-7253
; SUBJECTS: Dir, Link
; TOPIC:    Link's per-frame input handling in mode 5: A (sword) and B (item) presses, then the rules that
;           turn the held directions into ObjDir/ObjInputDir at a grid point and between grid points,
;           and Link's walking speed.
; SEE ALSO: bank5/border-input-mask.asm (Link_FilterInput: buttons and directions masked near the room
;           border); bank5/doorway.asm (Link_ModifyDirInDoorway); bank5/wield-items.asm (Sword_Wield,
;           Link_WieldItem); bank7/walker-move.asm (Walker_Move, which moves Link right after this routine);
;           bank7/walker-collision-screen-edge.asm (Link_CheckScreenEdge: room change only at grid offset 0);
;           bank7/link-animate-ladder.asm (Link_EndMoveAndAnimate: grid offset +-8 becomes 0);
;           bank7/mode5-play-update.asm (Link_Update, the caller); bank1/qspeed-direction-math.asm
;           (Dir_GetOpposite, Abs, Negate, the q-speed fraction).
; KEY FACTS:
;   - A and B are read only while Link's state is 0. A swings the sword only when SwordBlockedLongTimer
;     and IsSwordBlocked are both 0; B is not subject to that test (Link_HandleInput).
;   - During knockback (ObjShoveDir <> 0) the directions are not processed at all (Link_HandleInput).
;   - Link's grid is 8 pixels. ObjGridOffset = 0 selects the grid-point rules (Link_ModifyDirAtGridPoint),
;     anything else the between-points rules (Link_ModifyDirOnGridLine).
;   - At a grid point one held direction always becomes ObjDir and ObjInputDir, walkable or not
;     (Link_ModifyDirAtGridPoint_HaveInput).
;   - At a grid point with two held directions: none walkable = no movement; one walkable = go that way;
;     both walkable = OW takes the vertical one, UW alternates (turn perpendicular once, then keep going
;     straight) with special cases at X=$20/$D0 and at ($78,$5D) (Link_ModifyDirAtGridPoint_HaveInput).
;   - Between grid points Link can only continue or go back: the same direction continues, the opposite
;     direction is taken at once WITHOUT touching ObjGridOffset (Link_ModifyDirOnGridLine).
;   - Between grid points a single perpendicular press reverses Link when Link_GoStraight is 0, he is
;     less than 4 pixels from the grid point he left and still heading away from it. ObjGridOffset is
;     rewritten for the new heading: negative n -> 8 + n, positive n -> n - 8
;     (Link_ModifyDirOnGridLine, Link_SetInputDir@ReverseDir).
;   - After such a reversal followed by the opposite direction, the offset's sign is opposite to Link's
;     heading and counts toward 0 instead of +-8. Nothing stops it at 0, so a 2-pixel frame can step it
;     from +1 to -1 (or -1 to +1) and Link passes that grid point without any frame at offset 0
;     (Link_ModifyDirOnGridLine NOTE).
;   - With two directions held between grid points the horizontal one is the one considered
;     (Link_ModifyDirOnGridLine, through Dir_GetOpposite).
;   - Link's q-speed is $60 (1.5 pixels per frame: 1, 2, 1, 2 ...); $30 (0.75) in the OW while
;     ObjCollidedTile is $74 or $75 (Link_InitSpeed).
; CONTENTS: (generated; L = line in this file)
;   L0055 Dir_AxisMasks                        data  TABLE: direction bits on the same axis as a direction; index = reverse direction index (0 up, 1 down,
;   L0068 Link_HandleInput                     code  Handles Link's A and B presses, then turns the held directions into his facing and moving direction.
;   L0117 Link_ModifyDirAtGridPoint            code  Grid-point rules: picks Link's direction from the held directions and which of them are walkable.
;   L0153 Link_HandleInput_Return              code  Shared return of Link_HandleInput and Link_ModifyDirAtGridPoint.
;   L0157 Link_ModifyDirAtGridPoint_HaveInput  code  Grid-point rules, second part: at least one direction is held; Y = their count, X = 0.
;   L0272 Link_InitSpeed                       code  Sets Link's walking q-speed: $60 (1.5 pixels per frame), or $30 (0.75) on OW tiles $74 and $75.
;   L0333 Link_ModifyDirOnGridLine             code  Between-grid-points rules: Link may only continue, go back, or (near the start of a step) reverse.
;   L0360 Link_SetDirAndInputDir               code  Sets Link's facing direction and input direction to A.
;   L0364 Link_SetInputDir                     code  Sets Link's input direction to A and returns; cases 4-6 of Link_ModifyDirOnGridLine follow.
; ============================================================================

; TABLE: direction bits on the same axis as a direction; index = reverse direction index (0 up, 1 down,
;        2 left, 3 right); value = $0C (up|down) for a vertical direction, $03 (left|right) for a
;        horizontal one.
Dir_AxisMasks:
    .BYTE $0C, $0C, $03, $03

; ----------------------------------------------------------------------------
; Handles Link's A and B presses, then turns the held directions into his facing and moving direction.
; IN:   ObjInputDir (slot 0) = direction bits held on the controller (1 right, 2 left, 4 down, 8 up)
; OUT:  ObjDir, ObjInputDir, ObjQSpeedFrac and sometimes ObjGridOffset of slot 0; X = 0 on most paths
; NOTE: Called by Link_Update directly before Walker_Move, on every frame Link_Update runs with
;       Link not halted ((ObjState AND $C0) <> $40).
; NOTE: A and B are tested only in state 0 and only on the frame they go down (ButtonsPressed).
;       Link_FilterInput may have cleared them or the perpendicular directions near the room border.
; NOTE: While Link is shoved (ObjShoveDir <> 0) the routine returns before looking at the directions.
; ----------------------------------------------------------------------------
Link_HandleInput:
    ; Buttons are handled only in state 0 (not swinging or using an item).
    LDA ObjState
    BNE @CheckMovement
    JSR Link_FilterInput

    ; A newly pressed: swing the sword, unless a Bubble has blocked it
    ; (timed block or the block that lasts until cured).
    LDA SwordBlockedLongTimer
    ORA IsSwordBlocked
    BNE :+
    LDA ButtonsPressed
    AND #$80                    ; A button
    BEQ :+
    JSR Sword_Wield
:
    ; B newly pressed: use the selected item.
    LDA ButtonsPressed
    AND #$40                    ; B button
    BEQ @CheckMovement
    JSR Link_WieldItem

@CheckMovement:
    ; If Link is being shoved, return: knockback decides the movement.
    LDX #$00
    LDA ObjShoveDir
    BNE Link_HandleInput_Return

    ; In UW, restrict the input direction inside a doorway.
    LDA CurLevel
    BEQ :+
    JSR Link_ModifyDirInDoorway
:
    ; Choose the rules: at a grid point (grid offset = 0) or between
    ; grid points (grid offset <> 0).
    LDA ObjGridOffset
    BEQ Link_ModifyDirAtGridPoint
    JMP Link_ModifyDirOnGridLine

; ----------------------------------------------------------------------------
; Grid-point rules: picks Link's direction from the held directions and which of them are walkable.
; IN:   A = 0 (the grid offset); X = 0; ObjInputDir = held direction bits
; OUT:  [0B] = number of held directions; [0C] = number of them that are walkable; [0F] = last held
;       direction found; [0D] = last walkable held direction found; Link_GoStraight = 0
; NOTE: The directions are tested in the order right, left, down, up, so with a horizontal and a
;       vertical direction held "last found" is the vertical one.
; NOTE: Walkable = the tile Tile_GetAheadOfObj finds ahead of Link in that direction is below
;       RoomFirstUnwalkableTile. Doors, room bounds and tile objects are not considered here.
; ----------------------------------------------------------------------------
Link_ModifyDirAtGridPoint:
    STA $0B                     ; [0B] counts the held directions.
    STA $0C                     ; [0C] counts the held directions that are walkable.
    STA Link_GoStraight         ; Cleared at every grid point; set again below for diagonal input.

    ; For each direction (reverse direction index 3 to 0: right, left, down, up)
    ; that is held, count it and test the tile ahead.
    LDY #$03

@LoopDir:
    LDA ObjInputDir
    AND Dir_ReverseIndexToDir, Y
    BEQ @NextLoopDir            ; Not held.
    STA $0F                     ; [0F] = this held direction (also the argument of the tile lookup).
    TYA                         ; Save the reverse direction index.
    PHA
    INC $0B
    JSR Tile_GetAheadOfObj
    CMP RoomFirstUnwalkableTile
    BCS :+                      ; Tile >= first unwalkable tile: blocked.
    LDA $0F
    STA $0D                     ; [0D] = this walkable held direction.
    INC $0C
:
    PLA                         ; Restore the reverse direction index.
    TAY

@NextLoopDir:
    DEY
    BPL @LoopDir

    ; No direction held: return. ObjDir, ObjInputDir (0) and the speed stay as they are.
    LDY $0B
    BNE Link_ModifyDirAtGridPoint_HaveInput

; Shared return of Link_HandleInput and Link_ModifyDirAtGridPoint.
Link_HandleInput_Return:
    RTS

; Grid-point rules, second part: at least one direction is held; Y = their count, X = 0.
Link_ModifyDirAtGridPoint_HaveInput:
    ; One direction held: it becomes ObjDir and ObjInputDir whether or not it is walkable,
    ; and Link_GoStraightWhenDiagInput is cleared (X = 0).
    LDA $0F
    CPY #$01
    BEQ @SetLinkDirAndSpeed

    ; More than one direction held.
    ;
    ; None walkable: set ObjInputDir to 0 (A = 0), leave ObjDir alone, and return.
    LDA $0C
    BNE :+
    JMP Link_SetInputDir
:
    ; At least one of them is walkable. Link will keep one direction until the
    ; next grid point even if a perpendicular direction is pressed on the way.
    TAY                         ; Y = walkable count.
    INC Link_GoStraight

    ; Exactly one walkable, or in OW: take the last walkable direction found ([0D];
    ; in OW with both walkable that is the vertical one) and clear
    ; Link_GoStraightWhenDiagInput (X = 0).
    LDX #$00
    LDA $0D
    CPY #$01
    BEQ @SetLinkDirAndSpeed
    LDY CurLevel
    BEQ @SetLinkDirAndSpeed

    ; In UW with both held directions walkable.
    ;
    ; The general rule alternates: the first such grid point turns Link to the held
    ; direction perpendicular to his facing and sets Link_GoStraightWhenDiagInput;
    ; at later grid points, while the flag is set, he keeps his facing.
    ; Two position tests come first.

    ; Special case 1: Link's X is $20 or $D0, the leftmost and rightmost X the UW room
    ; bounds allow outside a doorway (the columns in front of the left and right doors,
    ; which are entered at Y = $8D: Doorway_RequiredCoords).
    ;   At Y = $85 (8 pixels above the door row) facing down: keep the facing direction
    ;   and clear the flag.
    ;   Otherwise: turn perpendicular, whatever the flag says.
    ; (The intent, making diagonal input lead into the side doors, is unverified.)
    LDY ObjX
    CPY #$20
    BEQ :+
    CPY #$D0
    BNE @CheckGoStraightFlag
:
    LDY ObjY
    CPY #$85
    BNE @TurnPerpendicular
    LDA ObjDir
    AND #$04                    ; Facing down?
    BEQ @TurnPerpendicular

@SetLinkDirToObjDir:
    LDA ObjDir
    BNE @SetLinkDirAndSpeed     ; Always taken when ObjDir <> 0.

@CheckGoStraightFlag:
    ; Flag clear: turn perpendicular (and set the flag).
    LDA ObjDir
    LDX Link_GoStraightWhenDiagInput
    BEQ @TurnPerpendicular

    ; Flag set (X <> 0): keep the facing direction, flag unchanged ...
    LDY CurLevel
    BEQ @SetLinkDirAndSpeed     ; Never taken: the OW left this path above.

    ; ... except in special case 2: Link at X = $78, Y = $5D (the top floor row, at the X
    ; the up and down doors require: Doorway_RequiredCoords) and facing horizontally. There
    ; he turns perpendicular (up or down) even though the flag is set. Facing vertically
    ; there, he keeps his facing. The bottom door ($78, $BD) has no such case.
    LDY ObjX
    CPY #$78
    BNE @SetLinkDirAndSpeed
    LDY ObjY
    CPY #$5D
    BNE @SetLinkDirAndSpeed
    AND #$03                    ; A = ObjDir: is it horizontal?
    BEQ @SetLinkDirToObjDir     ; Vertical: keep it.

@TurnPerpendicular:
    ; Take the held direction that is perpendicular to ObjDir:
    ; remove from the input the bits on ObjDir's own axis.
    LDA ObjDir
    INX                         ; Flag + 1: nonzero (X was 0, or the set flag in special case 2).
    JSR Dir_GetOpposite          ; Only for Y = reverse direction index of ObjDir.
    LDA ObjInputDir
    PHA
    AND Dir_AxisMasks, Y            ; The input bits on ObjDir's axis ...
    STA $0C
    PLA
    EOR $0C                     ; ... removed from the input.

@SetLinkDirAndSpeed:
    ; ObjDir = ObjInputDir = A; Link_GoStraightWhenDiagInput = X; then set the speed.
    STX Link_GoStraightWhenDiagInput
    JSR Link_SetDirAndInputDir
    LDX #$00

; ----------------------------------------------------------------------------
; Sets Link's walking q-speed: $60 (1.5 pixels per frame), or $30 (0.75) on OW tiles $74 and $75.
; IN:   ObjCollidedTile (slot 0) = tile found by Link's last tile lookup. Coming from the grid-point
;       rules that is the tile ahead in the last held direction tested (order right, left, down,
;       up), which need not be the direction chosen.
; OUT:  ObjQSpeedFrac (slot 0); [00] = the same value; ObjPosFrac = 0 when the speed changes to $30
; NOTE: The q-speed is added to ObjPosFrac four times a frame, one pixel per carry, so $60 gives 1.5
;       pixels per frame: 1 and 2 pixels on alternate frames.
; NOTE: Upstream calls tiles $74/$75 mountain stairs (unverified here). The slow speed applies in
;       the OW only. ObjPosFrac is not reset when the speed goes back to $60.
; NOTE: Also called when Link enters a room (Room_EnterCommon chunk) and, by branch, from
;       Link_ModifyDirOnGridLine when Link keeps his direction.
; ----------------------------------------------------------------------------
Link_InitSpeed:
    LDA #$60                    ; Default q-speed.
    STA $00
    LDA CurLevel
    BNE @SetSpeed               ; UW: always $60.

    ; OW: on tile $74 or $75 use q-speed $30.
    LDA ObjCollidedTile
    CMP #$74
    BEQ :+
    CMP #$75
    BNE @SetSpeed
:
    LDA #$30
    STA $00
    CMP ObjQSpeedFrac
    BEQ @SetSpeed               ; Already slow: keep the position fraction.
    LDA #$00                    ; Just became slow: restart the position fraction.
    STA ObjPosFrac

@SetSpeed:
    LDA $00
    STA ObjQSpeedFrac
:
    RTS

; ----------------------------------------------------------------------------
; Between-grid-points rules: Link may only continue, go back, or (near the start of a step) reverse.
; IN:   X = 0; ObjGridOffset <> 0; ObjInputDir = held direction bits; ObjDir = facing
; OUT:  ObjDir, ObjInputDir, ObjGridOffset, ObjQSpeedFrac changed as listed below
; NOTE: The input is first reduced to one direction by Dir_GetOpposite, which takes the lowest set
;       bit: right, else left, else down, else up. The cases, in order:
;       1. Nothing held: return. Link stops between grid points.
;       2. Same as ObjDir: keep going (speed is set again).
;       3. Opposite of ObjDir: ObjDir = ObjInputDir = that direction, return. ObjGridOffset is NOT
;          changed and the speed is not set again.
;       4. Perpendicular with Link_GoStraight set: keep going (speed is set again).
;       5. Perpendicular with |ObjGridOffset| >= 4: return; Walker_Move keeps moving Link in ObjDir.
;       6. Perpendicular with |ObjGridOffset| < 4: if the offset's sign agrees with the heading
;          (negative when facing left or up, positive when facing right or down: Link is still
;          walking away from the grid point he left), reverse: ObjDir = opposite, and the offset
;          is rewritten as negative n -> 8 + n (-3 -> +5), positive n -> n - 8 (+3 -> -5). If the
;          sign disagrees, return without changing anything.
;       In cases 4 to 6 ObjInputDir keeps the held bits; Walker_Move moves Link in ObjDir whenever
;       the grid offset is nonzero and ObjInputDir is nonzero, so in case 6 he steps back that frame.
; NOTE: Case 6 keeps the offset consistent: it is now measured from the grid point 8 pixels beyond
;       the one Link left, in the direction he was heading before the reversal, and reaches +-8 at
;       the point he left.
;       Case 3 does not: after "reverse (case 6), then the original direction (case 3)" Link heads
;       for that farther grid point with an offset whose sign is opposite to his heading, counting
;       toward 0. The offset is only stopped at +-8 and only reset when it is a multiple of 8 at the
;       end of a frame, so when a 2-pixel frame takes it from +1 to -1 (or -1 to +1) Link crosses
;       that grid point with no frame at offset 0: the grid-point rules, the warp check and the
;       screen-edge test (all of which require offset 0) are skipped there, and the offset runs on
;       to +-8, eight pixels farther.
; NOTE: A plain turn-back (case 3 without a preceding case 6) also leaves the offset counting toward
;       0, but it lands exactly on 0: adding the q-speed to ObjPosFrac undoes the subtractions frame
;       for frame, so Link retraces the same positions back to the grid point he left. The crossing
;       above needs the far grid point, which a normal walk reaches only because the +-8 stop
;       drops the second pixel of a 2-pixel frame.
; ----------------------------------------------------------------------------
Link_ModifyDirOnGridLine:
    ; Case 1: nothing held.
    LDA ObjInputDir
    BEQ :-

    ; Reduce the input to a single direction.
    ; Y = its reverse direction index, A = the direction.
    JSR Dir_GetOpposite
    LDA Dir_ReverseIndexToDir, Y

    ; Case 2: same as the facing direction.
    CMP ObjDir
    BEQ Link_InitSpeed

    ; Case 3: opposite of the facing direction (the two together make a full axis).
    ORA ObjDir
    CMP #$03                    ; right | left
    BEQ :+
    CMP #$0C                    ; down | up
    BNE :++                     ; Perpendicular: go to cases 4-6, below Link_SetInputDir.
:
    LDA Dir_ReverseIndexToDir, Y

; ----------------------------------------------------------------------------
; Sets Link's facing direction and input direction to A.
; IN:   A = direction
; ----------------------------------------------------------------------------
Link_SetDirAndInputDir:
    STA ObjDir

; Sets Link's input direction to A and returns; cases 4-6 of Link_ModifyDirOnGridLine follow.
Link_SetInputDir:
    STA ObjInputDir
    RTS
:
    ; The single input direction is perpendicular to the facing direction.
    ;
    ; Case 4: told at the last grid point to go straight (diagonal input).
    LDA Link_GoStraight
    BNE Link_InitSpeed

    ; Case 5: 4 or more pixels (half a grid cell) from the grid point: return.
    LDA ObjGridOffset
    JSR Math_Abs
    PHA
    LDA ObjDir
    JSR Dir_GetOpposite
    STA $01                     ; [01] = the opposite of the facing direction.
    PLA
    CMP #$04
    BCS @Exit

    ; Case 6: reverse only while the offset's sign agrees with the heading.
    ; Facing left or up: the offset must be negative.
    ; Facing right or down: the offset must be positive.
    ; Otherwise Link already faces the grid point he started from: return.
    LDA ObjDir
    AND #$0A                    ; Left or up?
    BEQ :+
    LDA ObjGridOffset
    BPL @Exit
    BMI @ReverseDir
:
    LDA ObjGridOffset
    BMI @Exit

@ReverseDir:
    LDA $01
    STA ObjDir

    ; Rewrite the grid offset for the same position seen from the opposite heading:
    ;   negative offset n: 8 + n      (-1 -> 7, -3 -> 5)
    ;   positive offset n: -8 + n     (1 -> -7, 3 -> -5)
    ; computed as base - (-n) with base 8 or $F8.
    LDA #$08
    LDY ObjGridOffset
    BMI :+
    LDA #$F8
:
    PHA
    TYA
    JSR Math_Negate
    STA $01
    PLA
    SEC
    SBC $01
    STA ObjGridOffset

@Exit:
    RTS
