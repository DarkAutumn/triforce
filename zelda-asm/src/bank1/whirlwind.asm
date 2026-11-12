; ============================================================================
; FILE:     bank1/whirlwind.asm          BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1745-2010
; SUBJECTS: Whirlwind
; TOPIC:    The recorder's whirlwind (object type $2E): summoning it, choosing the destination level,
;           picking Link up, the scroll to the destination room and the drop-off there.
; SEE ALSO: bank7/flute.asm (Flute_Wield decides between a flute secret and Whirlwind_Summon);
;           bank7/init-mode5-play.asm (calls Whirlwind_InitDropOffAndBeginUpdate when play starts in a room);
;           bank5/mode7-scroll-init.asm (Mode7Scroll_Init_Sub0 swaps RoomId for WhirlwindPrevRoomId);
;           bank1/link-collision-harm.asm (Link_CheckMonsterTouch, the pickup test);
;           bank7/object-stubs-destroy-monster.asm (Whirlwind_Update trampoline, Obj_FindEmptyMonsterSlot).
; KEY FACTS:
;   - No whirlwind without at least one Triforce piece, and only in mode 5: Whirlwind_Summon.
;   - Each flute use steps TeleportingLevelIndex by +1 when Link faces right or up (ObjDir AND $09),
;     by -1 when he faces left or down, then keeps stepping the same way past levels whose piece
;     is missing: Whirlwind_StepLevelIndex, Whirlwind_Summon@TestLevelPiece.
;   - Destination = level (TeleportingLevelIndex AND 7) + 1, levels 1-8 only. The index is kept
;     between rooms and it steps even when no whirlwind can be made: Whirlwind_Summon.
;   - One whirlwind per room visit (IsWhirlwindSummoned) and none while one carries Link; it needs a
;     free object slot in $01-$0B: Whirlwind_Summon@MakeWhirlwind.
;   - It is created at X = 0 on Link's Y without touching ObjMetastate: a slot still in metastate 1
;     (unused since room entry, or freed by Obj_DestroyMonster) first plays the spawn cloud. Then it
;     moves right 2 pixels per frame and is destroyed at X >= $F0, 120 frames after it starts
;     moving: Whirlwind_SetUp, Whirlwind_Update_Full.
;   - Pickup uses Link_CheckMonsterTouch (midpoints less than 9 pixels apart on both axes), so it fails
;     while Link is invincible, stunned, halted, paralyzed or holds the clock. A missed whirlwind
;     just leaves at the right edge: Whirlwind_Update_Full@CheckCollision.
;   - The pickup deals no damage but passes through Link_HarmByObjType: Link's ObjInvincibilityTimer becomes
;     $18, and Link_BeHarmed zeroes WorldKillCount, HelpDropCount and IsHelpDropBomb:
;     Whirlwind_Update_Full@CheckCollision.
;   - The room scrolls right into the room east of Whirlwind_PrevRoomIds[level - 1]: rooms $37,
;     $3C, $74, $45, $0B, $22, $42, $6D for levels 1-8, in both quests: Whirlwind_PrevRoomIds.
;   - In the destination room the whirlwind is put in object slot $09 (whatever is there) at X = 0,
;     Y = Whirlwind_DropOffYs[level - 1]. After its spawn cloud Link (state $40) rides it for 64 frames and
;     is released at X = $80; in a trace that was 86 frames after play started in the room:
;     Whirlwind_InitDropOffAndBeginUpdate, Whirlwind_Update_Full.
; CONTENTS: (generated; L = line in this file)
;   L0056 Whirlwind_SetLinkYAndSetUp           code  Moves Link to Y = A, then falls through to set up a whirlwind on that Y in object slot X.
;   L0066 Whirlwind_SetUp                      code  Makes object slot X a whirlwind (type $2E) at the left screen edge on Link's Y.
;   L0080 Whirlwind_PrevRoomIds                data  TABLE: room the screen scrolls in from when the whirlwind delivers Link; index = level - 1
;   L0098 Whirlwind_Update_Full                code  Whirlwind update: moves right 2 pixels per frame, picks Link up, carries him off or drops him.
;   L0217 Whirlwind_Destroy                    code  Frees the whirlwind's object slot and resets its per-slot fields.
;   L0242 Whirlwind_Summon                     code  Flute use on the overworld: picks the next destination level and makes a whirlwind.
;   L0313 Whirlwind_StepLevelIndex             code  Steps TeleportingLevelIndex: +1 when Link faces right or up, -1 when he faces left or down.
;   L0322 Whirlwind_StepLevelIndex_Return      code  Shared exit of Whirlwind_Summon and Whirlwind_StepLevelIndex.
;   L0329 Whirlwind_Draw                       code  Draws the whirlwind: one image, palette row cycling every frame.
;   L0347 Whirlwind_DropOffYs                  data  TABLE: Y coordinate of Link and the whirlwind in the destination room; index = level - 1
;   L0359 Whirlwind_InitDropOffAndBeginUpdate  code  On starting play in a room: sets up the drop-off if a whirlwind carries Link; then Mode_BeginUpdate.
; ============================================================================

; ----------------------------------------------------------------------------
; Moves Link to Y = A, then falls through to set up a whirlwind on that Y in object slot X.
; IN:   A = Y coordinate for Link and the whirlwind; X = object slot
; OUT:  writes ObjY (Link), then as Whirlwind_SetUp
; ----------------------------------------------------------------------------
Whirlwind_SetLinkYAndSetUp:
    STA ObjY

; Makes object slot X a whirlwind (type $2E) at the left screen edge on Link's Y.
; IN:   X = object slot
; OUT:  ObjX,X = 0; ObjY,X = Link's Y; ObjType,X = $2E
; NOTE: ObjMetastate,X and ObjUninitialized,X are left as they are. In a slot unused since room
;       entry (metastate 1, uninitialized) the whirlwind is first initialized (ObjTimer = slot
;       number; its init routine does nothing) and plays the spawn cloud: the rest of stage 1, then
;       stages 2 and 3 of 6 frames each. Whirlwind_Update_Full runs only after that.
Whirlwind_SetUp:
    LDA ObjY
    STA ObjY, X
    LDA #$00
    STA ObjX, X
    LDA #$2E                    ; Object type $2E: whirlwind.
    STA ObjType, X
    RTS

; TABLE: room the screen scrolls in from when the whirlwind delivers Link; index = level - 1
;        (TeleportingLevelIndex AND 7); value = overworld room id west of the destination room.
;        The scroll goes right, so Link arrives in value + 1: $37, $3C, $74, $45, $0B, $22, $42, $6D.
;        The table is the same in Q1 and Q2. That these are the rooms of the level entrances is not
;        shown by this code (unverified here).
Whirlwind_PrevRoomIds:
    .BYTE $36, $3B, $73, $44, $0A, $21, $41, $6C

; ----------------------------------------------------------------------------
; Whirlwind update: moves right 2 pixels per frame, picks Link up, carries him off or drops him.
; IN:   X = object slot of the whirlwind
; OUT:  WhirlwindTeleportingState 0 -> 1 at pickup, 2 -> 0 at release; may start mode 6
; NOTE: Runs only while ObjMetastate,X = 0 (not during the spawn cloud). Three cases per frame,
;       after X += 2:
;       - Link's state has bit $40 (halted) and WhirlwindTeleportingState <> 0: Link's X follows the
;         whirlwind. In state 2, at X = $80 (the 64th update), Link is released (ObjState = 0), the
;         state returns to 0 and the whirlwind's ObjType is zeroed on the spot.
;       - otherwise: Link_CheckMonsterTouch; on a hit Link is turned right, halted and hidden, and the
;         state becomes 1. The test fails while Link is invincible, stunned, halted, paralyzed or
;         has the clock.
;       - at X >= $F0 (the 120th update) the whirlwind is destroyed; with state <> 0 the game goes
;         to mode 6 (leave room), which leads to the scroll to the right.
; ----------------------------------------------------------------------------
Whirlwind_Update_Full:
    ; Y = $40 when Link's state has bit $40 (halted), else 0.
    LDA ObjState
    AND #$40
    TAY

    ; Move right 2 pixels. A keeps the new X.
    LDA #$02
    CLC
    ADC ObjX, X
    STA ObjX, X

    ; If Link is not halted or teleporting state = 0, then
    ; go check for collision with Link.
    CPY #$40
    BNE @CheckCollision
    LDY WhirlwindTeleportingState
    BEQ @CheckCollision

    ; Link is halted and teleporting state <> 0: the whirlwind carries Link.
    ; Set Link's X to the whirlwind's.
    STA ObjX

    ; If teleporting state = 1 (carrying Link away) or the whirlwind's X <> $80, go check
    ; the right screen edge and draw.
    DEY
    BEQ @CheckRightEdge
    CMP #$80
    BNE @CheckRightEdge

    ; Teleporting state = 2 and the whirlwind's X = $80: release Link here.
    ; A = $80, so ASL makes 0. Link becomes idle, teleporting ends, and the
    ; whirlwind's slot is freed by zeroing its type (Whirlwind_Destroy is not called).
    ASL
    STA ObjState
    STA WhirlwindTeleportingState
    STA ObjType, X

    ; Update the status bar's position marker for the new room, and draw one last time.
    TXA
    PHA
    JSR StatusBar_UpdateLinkPositionMarker
    PLA
    TAX
    JMP @Draw

@CheckCollision:
    ; Link is not halted or teleporting state = 0: try to pick up Link.
    ;
    ; Link_CheckMonsterTouch reports no collision while Link is invincible, stunned, halted
    ; or paralyzed, or holds the clock. A whirlwind (type < $53) that touches Link goes
    ; through Link_HarmByObjType: Shove_Begin gives Link a knockback (cancelled below) and
    ; ObjInvincibilityTimer = $18 (kept); he takes 0 damage (Damage_ObjTypeDamageToLink[$2E] = 0)
    ; without the hurt sound, and WorldKillCount, HelpDropCount and IsHelpDropBomb are zeroed.
    ;
    ; If [06] = 0 (no collision), go check the right screen edge.
    JSR Link_CheckMonsterTouch
    LDA $06
    BEQ @CheckRightEdge

    ; Collided with Link.
    ;
    ; Make Link face right (the direction of the coming room scroll). Reset his shove
    ; direction and distance, and UndergroundExitType.
    LDA #$01
    STA ObjDir
    LSR
    STA ObjShoveDir
    STA ObjShoveDistance
    STA UndergroundExitType

    ; Halt Link (state $40), and hide his two sprites (Y = $F8).
    LDA #$40
    STA ObjState
    LDA #$F8
    STA Sprites+72
    STA Sprites+76

    ; Look up the room that stands in as the current room during the scroll:
    ; the one west of the destination level's entrance room.
    LDA TeleportingLevelIndex
    AND #$07
    TAY
    LDA Whirlwind_PrevRoomIds, Y
    STA WhirlwindPrevRoomId

    ; Set teleporting state 1 (carrying Link away).
    INC WhirlwindTeleportingState

@CheckRightEdge:
    ; If whirlwind's X < $F0, go draw.
    LDA ObjX, X
    CMP #$F0
    BCC @Draw

    ; The whirlwind has reached the right edge of the screen: destroy it.
    JSR Whirlwind_Destroy

    ; If teleporting state <> 0, the whirlwind is carrying Link:
    ; go to the next mode (6, leave room).
    ;
    ; Either way, draw one last time.
    LDA WhirlwindTeleportingState
    BEQ @Draw
    TXA
    PHA
    JSR Mode5Play_GoToNextMode
    PLA
    TAX

@Draw:
    JMP Whirlwind_Draw

; ----------------------------------------------------------------------------
; Frees the whirlwind's object slot and resets its per-slot fields.
; IN:   X = object slot
; OUT:  ObjType, shove, timer, state and invincibility timer = 0; ObjUninitialized = $FF;
;       ObjMetastate = 1
; ----------------------------------------------------------------------------
Whirlwind_Destroy:
    LDA #$00
    STA ObjType, X
    STA ObjShoveDir, X
    STA ObjShoveDistance, X
    STA ObjTimer, X
    STA ObjState, X
    STA ObjInvincibilityTimer, X
    LDA #$FF
    STA ObjUninitialized, X
    LDA #$01
    STA ObjMetastate, X
    RTS

; ----------------------------------------------------------------------------
; Flute use on the overworld: picks the next destination level and makes a whirlwind.
; IN:   ObjDir = Link's facing direction; InvTriforce
; OUT:  TeleportingLevelIndex stepped; on success IsWhirlwindSummoned = 1 and a whirlwind object
;       in the highest free slot of $0B-$01. Uses [00] (the destination level's bit).
; NOTE: Returns at once outside mode 5. The index steps +1 (Link faces right or up) or -1 (left or
;       down), then on in the same direction until it names a level whose Triforce piece Link
;       has (bit level - 1 of InvTriforce); with no piece it steps once and nothing appears. The
;       index has stepped even if no whirlwind is made (one already summoned in this room,
;       teleporting in progress, or no free slot).
; ----------------------------------------------------------------------------
Whirlwind_Summon:
    ; If not in mode 5, return.
    LDA GameMode
    CMP #$05
    BNE Whirlwind_StepLevelIndex_Return

    ; Step the level index once, and store the mask for its level in [00]:
    ; Util_BitMasks[n] = bit n, the InvTriforce bit of level n + 1.
    JSR Whirlwind_StepLevelIndex
    LDA TeleportingLevelIndex
    AND #$07                    ; The index wraps around within 0-7.
    TAY
    LDA Util_BitMasks, Y
    STA $00

@TestLevelPiece:
    ; If Link has no Triforce pieces, then return.
    LDA InvTriforce
    BEQ Whirlwind_StepLevelIndex_Return

    ; If Link has this level's Triforce piece, then go try to make a
    ; whirlwind.
    BIT $00
    BNE @MakeWhirlwind

    ; Link does not have this piece.
    ; Step the level index again in the same direction.
    JSR Whirlwind_StepLevelIndex

    ; If Link faces right or up (the increasing directions), then
    ; rotate the level mask in [00] left (8-bit rotation: bit 7 wraps to bit 0)
    ; to match the next higher level. Then go test this new mask.
    LDA ObjDir
    AND #$09                    ; 1 right, 8 up
    BEQ @DecreasingDir
    ASL $00
    BCC @TestLevelPiece
    ROL $00
    JMP @TestLevelPiece

@DecreasingDir:
    ; Link faces left or down: rotate the mask right (bit 0 wraps to bit 7)
    ; to match the next lower level. Then go test this new mask.
    LSR $00
    BCC @TestLevelPiece
    ROR $00
    JMP @TestLevelPiece

@MakeWhirlwind:
    ; If a whirlwind was already summoned in this room or Link is being carried, then
    ; return and don't summon another one.
    LDA IsWhirlwindSummoned
    ORA WhirlwindTeleportingState
    BNE Whirlwind_StepLevelIndex_Return

    ; Return if there is no empty slot in $0B-$01.
    JSR Obj_FindEmptyMonsterSlot
    BEQ Whirlwind_StepLevelIndex_Return

    ; An empty slot was found (in Y). Flag the whirlwind summoned (cleared with the rest of
    ; $0300-$051F when a room is entered), and go set up a whirlwind object in that slot.
    INC IsWhirlwindSummoned
    TYA
    TAX
    JMP Whirlwind_SetUp

; ----------------------------------------------------------------------------
; Steps TeleportingLevelIndex: +1 when Link faces right or up, -1 when he faces left or down.
; IN:   ObjDir = Link's facing direction
; OUT:  TeleportingLevelIndex changed by 1 (callers use it AND 7)
; ----------------------------------------------------------------------------
Whirlwind_StepLevelIndex:
    INC TeleportingLevelIndex
    LDA ObjDir
    AND #$09                    ; 1 right, 8 up: keep the increment.
    BNE Whirlwind_StepLevelIndex_Return
    DEC TeleportingLevelIndex   ; Left or down: +1 - 2 = -1.
    DEC TeleportingLevelIndex

; Shared exit of Whirlwind_Summon and Whirlwind_StepLevelIndex.
Whirlwind_StepLevelIndex_Return:
    RTS

; ----------------------------------------------------------------------------
; Draws the whirlwind: one image, palette row cycling every frame.
; IN:   X = object slot
; ----------------------------------------------------------------------------
Whirlwind_Draw:
    ; Animation frames last 1 frame.
    LDA #$01
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor

    ; The whirlwind flashes: its sprite palette row is FrameCounter AND 3.
    LDA FrameCounter
    AND #$03
    JSR Anim_SetSpriteDescriptorAttributes

    ; Flip horizontally based on movement frame, which is
    ; switched based on animation counter.
    JSR Anim_SetObjHFlipForSpriteDescriptor
    LDA #$00                    ; Whirlwind has only 1 animation frame image.
    JMP Anim_DrawObjectNotMirrored

; TABLE: Y coordinate of Link and the whirlwind in the destination room; index = level - 1
;        (TeleportingLevelIndex AND 7); value = pixels. Link is released on this Y at X = $80.
Whirlwind_DropOffYs:
    .BYTE $8D, $AD, $8D, $8D, $AD, $8D, $AD, $5D

; ----------------------------------------------------------------------------
; On starting play in a room: sets up the drop-off if a whirlwind carries Link; then Mode_BeginUpdate.
; OUT:  when WhirlwindTeleportingState <> 0: state incremented (1 -> 2), Link halted (state $40) at
;       Y = Whirlwind_DropOffYs[level - 1], whirlwind in object slot $09 at X = 0 on the same Y
; NOTE: Only ObjType, ObjX and ObjY of slot $09 are written, whatever the room put there. Room
;       entry left the slot uninitialized in metastate 1, so the whirlwind takes 1 frame to
;       initialize and 9 + 6 + 6 frames of spawn cloud before it moves; Link stays halted at X = 0
;       meanwhile. In a trace Link was released 86 frames after play started.
; ----------------------------------------------------------------------------
Whirlwind_InitDropOffAndBeginUpdate:
    LDA WhirlwindTeleportingState
    BEQ @BeginUpdate            ; If not teleporting, go start updating the mode.

    ; Link was carried here: the pickup room is behind us. Set up the whirlwind
    ; object in the destination room.
    ;
    ; Teleporting state 2 drops off Link.
    INC WhirlwindTeleportingState
    LDA #$40                    ; Halt Link (state $40).
    STA ObjState
    LDX #$09                    ; The whirlwind goes in object slot $09 in the destination room.
    LDA TeleportingLevelIndex   ; Get the Y coordinate for this destination level.
    AND #$07
    TAY
    LDA Whirlwind_DropOffYs, Y
    JSR Whirlwind_SetLinkYAndSetUp
@BeginUpdate:
    JMP Mode_BeginUpdate
