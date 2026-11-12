; ============================================================================
; FILE:     bank4/gleeok.asm             BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 8505-9043
; SUBJECTS: Flyer, Gleeok
; TOPIC:    Gleeok (types $42-$45): the per-frame update that loads each living neck into object slots
;           1-6, bends the neck, lets its head shoot, and stores it back; and the cut-off flying head
;           (type $46), which cannot be hurt.
; SEE ALSO: bank4/gleeok-head-body.asm (Gleeok_MoveHead, head and segment drawing, collisions, a head
;           being cut off, the boss dying, Gleeok_DrawBody); bank4/gleeok-manhandla-gohma-inits.asm
;           (Gleeok_Init, Gleeok_InitFlyingHead, start values); bank4/fairy-flyer-common.asm (Flyer_Move,
;           Flyer_State0SpeedUp, Flyer_State2Chase, Flyer_State3Wander); bank4/bubble-keese.asm (Flyer_SetStateAndTurns);
;           bank4/monster-shots-fireballs.asm (Fireball_Shoot, Fireball_Update).
; KEY FACTS:
;   - Number of necks = type of object slot 1 - $41 ($42 has 1 neck, $45 has 4); neck n is skipped once bit n
;     of GleeokDeadNeckMask is set. Gleeok_Update
;   - A neck is 5 drawn segments worked on in object slots 1 (base) to 5 (head); its X, Y and head
;     movement block live in per-neck arrays and are copied in and out every frame.
;     Gleeok_Update@LoadNeckBytes, Gleeok_Update@SaveNeckBytes, Gleeok_FetchNeckAddrs
;   - ObjHP and ObjInvincibilityTimer of slot 5 are not part of the copied neck data, so all attached
;     heads share them. Gleeok_Update@LoadNeckBytes
;   - Each neck moves, and its head may shoot, only on frames where (FrameCounter AND 3) = neck index:
;     once every 4 frames. Every living neck is drawn and collision-tested every frame. Gleeok_Update
;   - An attached head shoots fireball type $56 when Random+5 < $20 (1 in 8 of its move frames) and
;     object slot $0B is empty; a flying head does so on frames where its distance travelled is even,
;     Random+1,X < $20 and slot $0B is empty. So one Gleeok fireball exists at a time (it takes slot
;     $0B). Gleeok_Update, Gleeok_UpdateFlyingHead
;   - The flying head cannot be harmed, shoved or killed: after its collision test its metastate,
;     shove info and invincibility timer are zeroed every frame. Gleeok_UpdateFlyingHead
;   - Flying head flight states: 0 speed up, 1 decide, 2 chase Link, 3 wander. Decide picks chase when
;     Random+1,X < $D0 (13 in 16), else wander. It has no slow-down or rest state.
;     Gleeok_FlyingHeadFlight_JumpTable, Flyer_GleeokHeadDecideState
;   - Gleeok_StretchNeck subtracts ObjX+3,X from itself, so its horizontal distance is always 0:
;     table entries 2, 5 and 8 are never used, so Gleeok_ContractSegmentX and Gleeok_ContractSegmentXOrY
;     never run. Gleeok_StretchNeck, Gleeok_StretchNeck_JumpTable
;   - Neck spacing limits per axis from d = |head - base| / 4: tier 1 = min(d, 4), tier 2 = tier 1 + 4,
;     tier 3 = min(tier 2 + 4, $B) pixels. Gleeok_CalcSegmentLimits
; CONTENTS: (generated; L = line in this file)
;   L0075 Gleeok_UpdateFlyingHead            code  Updates a cut-off flying Gleeok head (type $46): flies, may shoot a fireball, draws, harms Link.
;   L0110 Gleeok_ControlFlyingHeadFlight     code  Runs the flying Gleeok head's current flight state.
;   L0115 Gleeok_FlyingHeadFlight_JumpTable  data  TABLE: flight state handlers of the flying Gleeok head; index = Flyer_ObjFlyingState.
;   L0127 Flyer_GleeokHeadDecideState        code  Flying Gleeok head flight state 1: picks chase (Random+1,X < $D0) or wander; turn counter 6.
;   L0140 Gleeok_NeckXAddrsLo                data  TABLE: low address bytes of each neck's segment X array; index = neck 0-3; Gleeok_NeckXs0-3.
;   L0144 Gleeok_NeckXAddrsHi                data  TABLE: high address bytes of each neck's segment X array; index = neck 0-3.
;   L0148 Gleeok_NeckYAddrsLo                data  TABLE: low address bytes of each neck's segment Y array; index = neck 0-3; Gleeok_NeckYs0-3.
;   L0152 Gleeok_NeckYAddrsHi                data  TABLE: high address bytes of each neck's segment Y array; index = neck 0-3.
;   L0156 Gleeok_NeckHeadInfoAddrsLo         data  TABLE: low address bytes of each neck's head movement block; index = neck 0-3; Gleeok_HeadInfo0-3.
;   L0160 Gleeok_NeckHeadInfoAddrsHi         data  TABLE: high address bytes of each neck's head movement block; index = neck 0-3.
;   L0172 Gleeok_Update                      code  Updates Gleeok (types $42-$45): draws the body, then moves, draws and hit-tests every living neck.
;   L0256 Gleeok_FetchNeckAddrs              code  Loads the addresses of the current neck's X, Y and head movement arrays.
;   L0274 Gleeok_MoveNeck_DivideBy4Unsigned  code  Gleeok_MoveNeck branches here when the head is not left of the base: A = distance / 4.
;   L0287 Gleeok_MoveNeck                    code  Bends the current neck (object slots 1-5) so its segments follow the head.
;   L0303 Gleeok_MoveNeck_StoreRefDistance   code  Both paths of Gleeok_MoveNeck join here with A = signed (head X - base X) / 4.
;   L0453 Gleeok_CalcSegmentLimits           code  Derives the three neck segment spacing limits of one axis from a quarter head-to-base distance.
;   L0490 Gleeok_StretchNeck                 code  Moves one middle neck segment 2 pixels away from or toward the next segment, or leaves it.
;   L0533 Gleeok_StretchNeck_JumpTable       data  TABLE: segment handlers of Gleeok_StretchNeck; index = h + 3 * v (h = horizontal limits reached,
;   L0550 Gleeok_ExpandSegment               code  Moves a neck segment 2 pixels away from the next segment, horizontally or vertically at random.
;   L0564 Gleeok_StepSegmentYMinus2          code  Shared tail: turns the prepared segment Y + 2 into segment Y - 2.
;   L0571 Gleeok_StoreSegmentY               code  Shared tail: stores the Y register as the Y of the segment in slot 2+X.
;   L0575 Gleeok_IgnoreSegment               code  Segment handler that does nothing (also the return of the handlers above).
;   L0583 Gleeok_ContractSegmentY            code  Moves a neck segment 2 pixels toward the next segment vertically.
;   L0594 Gleeok_ExpandSegmentX              code  Horizontal half of Gleeok_ExpandSegment: X + 2 if this segment's X >= the next one's, else - 2.
;   L0609 Gleeok_ContractSegmentX            code  Moves a neck segment 2 pixels toward the next segment horizontally.
;   L0618 Gleeok_StepSegmentXMinus2          code  Shared tail: turns the prepared segment X + 2 into segment X - 2.
;   L0625 Gleeok_StoreSegmentX               code  Shared tail: stores the Y register as the X of the segment in slot 2+X.
;   L0633 Gleeok_ContractSegmentXOrY         code  Moves a neck segment toward the next one, horizontally or vertically at random.
; ============================================================================

; ----------------------------------------------------------------------------
; Updates a cut-off flying Gleeok head (type $46): flies, may shoot a fireball, draws, harms Link.
; IN:   X = object slot (7 + neck index; set where the head is cut off).
; NOTE: Shoots fireball type $56 on a frame where bit 0 of Flyer_ObjDistTraveled is clear,
;       Random+1,X < $20 and object slot $0B is empty. The head cannot be harmed: whatever
;       Collision_CheckMonsterVsWeaponsAndLink did to its metastate, shove info and invincibility timer is undone.
; ----------------------------------------------------------------------------
Gleeok_UpdateFlyingHead:
    JSR Gleeok_ControlFlyingHeadFlight
    JSR Flyer_Move

    ; Shoot only if the distance travelled is even, Random+1,X < $20 (1 in 8),
    ; and object slot $0B is empty. Fireball_Shoot takes the highest empty monster slot,
    ; so the fireball goes into slot $0B and blocks further shots until it is gone.
    LDA Flyer_ObjDistTraveled, X
    AND #$01
    BNE :+
    LDA Random+1, X
    CMP #$20
    BCS :+
    LDA ObjType+11              ; Slot $0B in use?
    BNE :+
    LDA #$56                    ; Fireball type $56.
    JSR Fireball_Shoot
:
    LDA #$01                    ; Animation counter reload: switch frames every frame.
    JSR Anim_AdvanceAnimCounterAndSetObjPosForSpriteDescriptor
    LDA ObjAnimFrame, X
    JSR Anim_DrawObjectMirrored
    JSR Collision_CheckMonsterVsWeaponsAndLink

    ; Undo any hit: metastate 0 (alive), no shove, no invincibility frames.
    ; Obj_ResetMetastate returns A = 0, which the next two stores use.
    JSR Obj_ResetMetastate
    JSR Shove_SetDirAndDistance
    STA ObjInvincibilityTimer, X
    RTS

; ----------------------------------------------------------------------------
; Runs the flying Gleeok head's current flight state.
; IN:   X = object slot; Flyer_ObjFlyingState,X = 0-3.
; ----------------------------------------------------------------------------
Gleeok_ControlFlyingHeadFlight:
    LDA Flyer_ObjFlyingState, X
    JSR Util_TableJump

; TABLE: flight state handlers of the flying Gleeok head; index = Flyer_ObjFlyingState.
Gleeok_FlyingHeadFlight_JumpTable:
    .ADDR Flyer_State0SpeedUp                 ; 0: +1 speed per frame up to FlyerMaxSpeedFrac, then state 1
    .ADDR Flyer_GleeokHeadDecideState   ; 1: choose chase or wander
    .ADDR Flyer_State2Chase                   ; 2: turn toward Link, $10 frames between turns, then state 1
    .ADDR Flyer_State3Wander                  ; 3: turn at random, then state 1

; ----------------------------------------------------------------------------
; Flying Gleeok head flight state 1: picks chase (Random+1,X < $D0) or wander; turn counter 6.
; IN:   X = object slot.
; OUT:  Flyer_ObjFlyingState,X = 2 (chase, 208 in 256) or 3 (wander); Flyer_ObjTurns,X = 6
;       (5 turns $10 frames apart, then back to this state).
; ----------------------------------------------------------------------------
Flyer_GleeokHeadDecideState:
    LDY #$02
    LDA Random+1, X
    CMP #$D0
    BCC :+
    INY
:
    JMP Flyer_SetStateAndTurns

; The six tables below give, per neck, the RAM address of its 6-byte arrays.
; Gleeok_FetchNeckAddrs reads them.

; TABLE: low address bytes of each neck's segment X array; index = neck 0-3; Gleeok_NeckXs0-3.
Gleeok_NeckXAddrsLo:
    .BYTE $38, $52, $6C, $95    ; $0438, $0452, $046C, $0395

; TABLE: high address bytes of each neck's segment X array; index = neck 0-3.
Gleeok_NeckXAddrsHi:
    .BYTE $04, $04, $04, $03

; TABLE: low address bytes of each neck's segment Y array; index = neck 0-3; Gleeok_NeckYs0-3.
Gleeok_NeckYAddrsLo:
    .BYTE $45, $5F, $79, $BD    ; $0445, $045F, $0479, $03BD

; TABLE: high address bytes of each neck's segment Y array; index = neck 0-3.
Gleeok_NeckYAddrsHi:
    .BYTE $04, $04, $04, $03

; TABLE: low address bytes of each neck's head movement block; index = neck 0-3; Gleeok_HeadInfo0-3.
Gleeok_NeckHeadInfoAddrsLo:
    .BYTE $20, $2D, $81, $A9    ; $0420, $042D, $0381, $03A9

; TABLE: high address bytes of each neck's head movement block; index = neck 0-3.
Gleeok_NeckHeadInfoAddrsHi:
    .BYTE $04, $04, $03, $03

; ----------------------------------------------------------------------------
; Updates Gleeok (types $42-$45): draws the body, then moves, draws and hit-tests every living neck.
; NOTE: Necks are processed from index (type of object slot 1 - $42) down to 0. Each is copied into object
;       slots 1-6 (slot 1 = base, slot 5 = head; slot 6 is copied but not drawn or tested) and
;       Gleeok_CurHeadInfo, updated, and copied back. ObjHP and ObjInvincibilityTimer of slot 5
;       are not copied, so every attached head uses the same HP byte and invincibility timer.
;       A neck moves and may shoot only when (FrameCounter AND 3) = its index (every 4th frame).
;       Its head shoots fireball type $56 when Random+5 < $20 and object slot $0B is empty.
; ----------------------------------------------------------------------------
Gleeok_Update:
    JSR Gleeok_DrawBody

    ; Index of the last neck = object type of slot 1 - $42 (0 for the one-headed Gleeok).
    LDA ObjType+1
    SEC
    SBC #$42
    STA GleeokCurNeck

@LoopNeck:
    ; Skip this neck if its head has been cut off (its bit is set in the dead neck mask).
    LDY GleeokCurNeck
    LDA Util_BitMasks, Y           ; Bit mask 1 << neck index.
    BIT GleeokDeadNeckMask
    BEQ :+
    JMP @NextLoopNeck
:
    ; Get the addresses of this neck's arrays. Returns X = neck index, Y = 5.
    JSR Gleeok_FetchNeckAddrs

@LoadNeckBytes:
    ; For Y from 5 to 0: copy the neck's X and Y into object slots 6 to 1,
    ; and its head movement block into Gleeok_CurHeadInfo.
    LDA ($00), Y
    STA a:ObjX+1, Y
    LDA ($02), Y
    STA a:ObjY+1, Y
    LDA ($04), Y
    STA Gleeok_CurHeadInfo, Y
    DEY
    BPL @LoadNeckBytes

    ; Only the neck whose index (still in X) equals FrameCounter AND 3 moves and may shoot
    ; on this frame.
    LDA FrameCounter
    AND #$03
    STA $00
    CPX $00
    BNE @DrawAndCheckCollisions
    JSR Gleeok_MoveNeck
    JSR Gleeok_MoveHead

    ; The head is in object slot 5: make it the shooter, so the fireball starts at the head.
    LDX #$05

    ; Shoot fireball type $56 if Random+5 < $20 (1 in 8) and object slot $0B is empty.
    LDA Random, X
    CMP #$20
    BCS @DrawAndCheckCollisions
    LDA ObjType+11              ; Slot $0B in use?
    BNE @DrawAndCheckCollisions
    LDA #$56
    JSR Fireball_Shoot

@DrawAndCheckCollisions:
    ; Every living neck, every frame: draw slots 5 to 1 and test the head and the base
    ; against Link and his weapons.
    JSR Gleeok_DrawNeckAndCheckCollisions

    ; Copy the neck from the object slots back to its arrays.
    JSR Gleeok_FetchNeckAddrs

@SaveNeckBytes:
    LDA a:ObjX+1, Y
    STA ($00), Y
    LDA a:ObjY+1, Y
    STA ($02), Y
    LDA Gleeok_CurHeadInfo, Y
    STA ($04), Y
    DEY
    BPL @SaveNeckBytes

@NextLoopNeck:
    ; Next lower neck index, until it goes below 0.
    DEC GleeokCurNeck
    BPL @LoopNeck
    RTS

; ----------------------------------------------------------------------------
; Loads the addresses of the current neck's X, Y and head movement arrays.
; IN:   GleeokCurNeck = neck index.
; OUT:  X = neck index; Y = 5 (last array index); [00:01] = address of the segment X array;
;       [02:03] = address of the segment Y array; [04:05] = address of the head movement block.
; ----------------------------------------------------------------------------
Gleeok_FetchNeckAddrs:
    LDX GleeokCurNeck
    LDA Gleeok_NeckXAddrsLo, X
    STA $00
    LDA Gleeok_NeckXAddrsHi, X
    STA $01
    LDA Gleeok_NeckYAddrsLo, X
    STA $02
    LDA Gleeok_NeckYAddrsHi, X
    STA $03
    LDA Gleeok_NeckHeadInfoAddrsLo, X
    STA $04
    LDA Gleeok_NeckHeadInfoAddrsHi, X
    STA $05
    LDY #$05                    ; Callers copy from index 5 down to 0.
    RTS

; Gleeok_MoveNeck branches here when the head is not left of the base: A = distance / 4.
Gleeok_MoveNeck_DivideBy4Unsigned:
    LSR
    LSR
    JMP Gleeok_MoveNeck_StoreRefDistance

; ----------------------------------------------------------------------------
; Bends the current neck (object slots 1-5) so its segments follow the head.
; NOTE: Runs for one neck every 4th frame, before Gleeok_MoveHead. Five passes: spacing limits
;       from the head-to-base distance; pull each of slots 2-5 (head included) 2 pixels toward
;       the slot below it when at or beyond the tier 3 limit; Gleeok_StretchNeck for slots 2-4;
;       step X of slots 2-4 one pixel toward evenly spaced reference points; step Y of slots
;       4 and 3 one pixel when above or below both neighbours. The base (slot 1) never moves.
; ----------------------------------------------------------------------------
Gleeok_MoveNeck:
    ; Signed horizontal distance from the base (slot 1) to the head (slot 5), as a byte.
    LDA ObjX+5
    SEC
    SBC ObjX+1

    ; Not negative: divide by 4 with two shifts.
    BPL Gleeok_MoveNeck_DivideBy4Unsigned

    ; Negative: negate, divide by 4, negate again.
    JSR Math_Negate
    LSR
    LSR
    JSR Math_Negate

; Both paths of Gleeok_MoveNeck join here with A = signed (head X - base X) / 4.
Gleeok_MoveNeck_StoreRefDistance:
    ; This is the signed X spacing of the reference points used further down.
    STA GleeokSignedRefSegmentDistance

    ; Its absolute value gives the three horizontal spacing limits.
    JSR Math_Abs
    LDX #$00                    ; 0 = horizontal limits.
    JSR Gleeok_CalcSegmentLimits

    ; Absolute vertical distance from the base to the head, divided by 4,
    ; gives the three vertical spacing limits.
    LDA ObjY+5
    SEC
    SBC ObjY+1
    JSR Math_Abs
    LSR
    LSR

    INX                         ; 1 = vertical limits.
    JSR Gleeok_CalcSegmentLimits

    ; Keep each segment within the tier 3 limit of the segment below it (toward the base).
    ; X = 0 to 3: the segment in slot 2+X (the head when X = 3) against slot 1+X.
    LDX #$00

@KeepSegsNearNeighbors:
    ; Absolute horizontal distance between the two segments.
    LDA ObjX+1, X
    SEC
    SBC ObjX+2, X
    JSR Math_Abs

    ; Less than the tier 3 horizontal limit: leave X alone.
    CMP GleeokTertiarySegmentLimits
    BCC @CheckDistanceV

    ; Else move this segment 2 pixels toward the lower one: +2 if it is left of it, else -2.
    LDA ObjX+2, X
    TAY
    INY
    INY
    CMP ObjX+1, X
    BCC :+
    DEY
    DEY
    DEY
    DEY
:
    STY ObjX+2, X

@CheckDistanceV:
    ; Absolute vertical distance between the two segments.
    LDA ObjY+1, X
    SEC
    SBC ObjY+2, X
    JSR Math_Abs

    ; Less than the tier 3 vertical limit: leave Y alone.
    CMP GleeokTertiarySegmentLimits+1
    BCC @NextKeepSegsNearNeighbors

    ; Else move this segment 2 pixels toward the lower one: +2 if it is above it, else -2.
    LDA ObjY+2, X
    TAY
    INY
    INY
    CMP ObjY+1, X
    BCC :+
    DEY
    DEY
    DEY
    DEY
:
    STY ObjY+2, X

@NextKeepSegsNearNeighbors:
    ; Next segment, up to and including the head.
    INX
    CPX #$04
    BNE @KeepSegsNearNeighbors

    ; Spread or gather slots 2, 3, 4 against the next segment toward the head.
    LDX #$00
:
    JSR Gleeok_StretchNeck
    INX
    CPX #$03
    BCC :-

    ; Step X of each middle segment 1 pixel toward its reference point:
    ;   base X + GleeokSignedRefSegmentDistance * (X + 1)
    ; for X = 2 to 0, the segments in slots 4 to 2.
    LDX #$02

@KeepSegsNearRefPoint:
    ; A := base X plus (X + 1) times the signed spacing.
    TXA
    TAY
    LDA ObjX+1
:
    CLC
    ADC GleeokSignedRefSegmentDistance
    DEY
    BPL :-

    ; Reference X >= segment X (unsigned): segment X + 1. Else segment X - 1.
    ; A segment already on its reference point still moves 1 pixel right.
    LDY ObjX+2, X
    INY
    CMP ObjX+2, X
    BCS @NextKeepSegsNearRefPoint
    DEY
    DEY

@NextKeepSegsNearRefPoint:
    STY ObjX+2, X
    DEX
    BPL @KeepSegsNearRefPoint

    ; Keep the segments in slots 4 and 3 vertically between their neighbours:
    ; if a segment's Y is less than both neighbours' Y, add 1; if it is greater than or
    ; equal to both, subtract 1. X = 1, 0.
    LDX #$01

@KeepSegsBetweenNeighbors:
    LDA ObjY+3, X
    CMP ObjY+2, X               ; Lower neighbour (toward the base).
    BCS :+
    CMP ObjY+4, X               ; Upper neighbour (toward the head).
    BCS @NextKeepSegsBetweenNeighbors
    INC ObjY+3, X
    JMP @NextKeepSegsBetweenNeighbors
:
    CMP ObjY+2, X
    BCC @NextKeepSegsBetweenNeighbors
    CMP ObjY+4, X
    BCC @NextKeepSegsBetweenNeighbors
    DEC ObjY+3, X

@NextKeepSegsBetweenNeighbors:
    DEX
    BPL @KeepSegsBetweenNeighbors
    RTS

; ----------------------------------------------------------------------------
; Derives the three neck segment spacing limits of one axis from a quarter head-to-base distance.
; IN:   A = |head - base| / 4 on that axis; X = 0 horizontal, 1 vertical.
; OUT:  GleeokPrimarySegmentLimits,X = min(A, 4); GleeokSecondarySegmentLimits,X = that + 4
;       (4-8); GleeokTertiarySegmentLimits,X = min(secondary + 4, $B) (8-$B). Pixels.
; ----------------------------------------------------------------------------
Gleeok_CalcSegmentLimits:
    CMP #$04
    BCC :+
    LDA #$04
:
    STA GleeokPrimarySegmentLimits, X

    ; Secondary limit: primary + 4. The cap of 8 is never exceeded, since primary <= 4.
    CLC
    ADC #$04
    CMP #$08
    BCC :+
    LDA #$08
:
    STA GleeokSecondarySegmentLimits, X

    ; Tertiary limit: secondary + 4, capped at $B.
    CLC
    ADC #$04
    CMP #$0B
    BCC :+
    LDA #$0B
:
    STA GleeokTertiarySegmentLimits, X
    RTS

; ----------------------------------------------------------------------------
; Moves one middle neck segment 2 pixels away from or toward the next segment, or leaves it.
; IN:   X = 0-2: the segment in object slot 2+X, compared with the one in slot 3+X.
; NOTE: Handler index = h + 3 * v, where h and v count how many of the primary and secondary
;       limits the horizontal and vertical distances reach. The horizontal distance is computed
;       as ObjX+3,X - ObjX+3,X = 0, so h is 1 only when the primary horizontal limit is 0 and
;       never 2 (the secondary limit is at least 4): indexes 2, 5 and 8 cannot occur.
;       So a segment is spread (index 0) only when both primary limits are nonzero (head at
;       least 4 pixels from the base on both axes) and its vertical distance is below the
;       primary vertical limit; it is pulled closer only vertically (indexes 6, 7).
; ----------------------------------------------------------------------------
Gleeok_StretchNeck:
    LDY #$00

    ; Horizontal distance >= primary horizontal limit: index + 1.
    ; Both operands are the same slot, so the distance is always 0 (the vertical test
    ; below uses ObjY+3 and ObjY+2); this is true only when the limit itself is 0.
    LDA ObjX+3, X
    SEC
    SBC ObjX+3, X
    JSR Math_Abs
    CMP GleeokPrimarySegmentLimits
    BCC :+
    INY
:
    ; Horizontal distance >= secondary horizontal limit: index + 1.
    ; Never true: the distance is 0 and the limit is at least 4.
    CMP GleeokSecondarySegmentLimits
    BCC :+
    INY
:
    ; Vertical distance to the next segment >= primary vertical limit: index + 3.
    LDA ObjY+3, X
    SEC
    SBC ObjY+2, X
    JSR Math_Abs
    CMP GleeokPrimarySegmentLimits+1
    BCC :+
    INY
    INY
    INY
:
    ; Vertical distance >= secondary vertical limit: index + 3 more.
    CMP GleeokSecondarySegmentLimits+1
    BCC :+
    INY
    INY
    INY
:
    TYA
    JSR Util_TableJump

; TABLE: segment handlers of Gleeok_StretchNeck; index = h + 3 * v (h = horizontal limits reached,
; v = vertical limits reached). Entries 2, 5 and 8 are never used (h is never 2).
Gleeok_StretchNeck_JumpTable:
    .ADDR Gleeok_ExpandSegment          ; 0: h 0, v 0: close on both axes, move apart
    .ADDR Gleeok_IgnoreSegment          ; 1: h 1, v 0
    .ADDR Gleeok_ContractSegmentX       ; 2: h 2, v 0 (never)
    .ADDR Gleeok_IgnoreSegment          ; 3: h 0, v 1
    .ADDR Gleeok_IgnoreSegment          ; 4: h 1, v 1
    .ADDR Gleeok_ContractSegmentX       ; 5: h 2, v 1 (never)
    .ADDR Gleeok_ContractSegmentY       ; 6: h 0, v 2: too far vertically, move closer
    .ADDR Gleeok_ContractSegmentY       ; 7: h 1, v 2
    .ADDR Gleeok_ContractSegmentXOrY        ; 8: h 2, v 2 (never)

; ----------------------------------------------------------------------------
; Moves a neck segment 2 pixels away from the next segment, horizontally or vertically at random.
; IN:   X = 0-2: the segment in object slot 2+X; the next segment is in slot 3+X.
; NOTE: Bit 7 of Random clear = horizontal, set = vertical. Vertically: +2 if this segment's Y
;       is greater than the next one's, else -2 (also when equal).
; ----------------------------------------------------------------------------
Gleeok_ExpandSegment:
    LDA Random
    BPL Gleeok_ExpandSegmentX

    ; Vertical. Y register := segment Y + 2, kept if the segment's Y is greater than the next one's.
    LDA ObjY+2, X
    TAY
    INY
    INY
    CMP ObjY+3, X
    BEQ Gleeok_StepSegmentYMinus2
    BCS Gleeok_StoreSegmentY

; Shared tail: turns the prepared segment Y + 2 into segment Y - 2.
Gleeok_StepSegmentYMinus2:
    DEY
    DEY
    DEY
    DEY

; Shared tail: stores the Y register as the Y of the segment in slot 2+X.
Gleeok_StoreSegmentY:
    STY ObjY+2, X

; Segment handler that does nothing (also the return of the handlers above).
Gleeok_IgnoreSegment:
    RTS

; ----------------------------------------------------------------------------
; Moves a neck segment 2 pixels toward the next segment vertically.
; IN:   X = 0-2: the segment in object slot 2+X; the next segment is in slot 3+X.
; NOTE: +2 if this segment's Y is less than or equal to the next one's, else -2.
; ----------------------------------------------------------------------------
Gleeok_ContractSegmentY:
    LDA ObjY+2, X
    TAY
    INY
    INY
    CMP ObjY+3, X
    BEQ Gleeok_StoreSegmentY
    BCC Gleeok_StoreSegmentY
    BCS Gleeok_StepSegmentYMinus2

; Horizontal half of Gleeok_ExpandSegment: X + 2 if this segment's X >= the next one's, else - 2.
Gleeok_ExpandSegmentX:
    LDA ObjX+2, X
    TAY
    INY
    INY
    CMP ObjX+3, X
    BCS Gleeok_StoreSegmentX
    BCC Gleeok_StepSegmentXMinus2

; ----------------------------------------------------------------------------
; Moves a neck segment 2 pixels toward the next segment horizontally.
; IN:   X = 0-2: the segment in object slot 2+X; the next segment is in slot 3+X.
; NOTE: +2 if this segment's X is less than the next one's, else -2. Never reached through
;       Gleeok_StretchNeck_JumpTable (see Gleeok_StretchNeck).
; ----------------------------------------------------------------------------
Gleeok_ContractSegmentX:
    LDA ObjX+2, X
    TAY
    INY
    INY
    CMP ObjX+3, X
    BCC Gleeok_StoreSegmentX

; Shared tail: turns the prepared segment X + 2 into segment X - 2.
Gleeok_StepSegmentXMinus2:
    DEY
    DEY
    DEY
    DEY

; Shared tail: stores the Y register as the X of the segment in slot 2+X.
Gleeok_StoreSegmentX:
    STY ObjX+2, X
    RTS

; ----------------------------------------------------------------------------
; Moves a neck segment toward the next one, horizontally or vertically at random.
; NOTE: Bit 7 of Random clear = horizontal, set = vertical. Never reached (table index 8).
; ----------------------------------------------------------------------------
Gleeok_ContractSegmentXOrY:
    LDA Random
    BPL Gleeok_ContractSegmentX
    BMI Gleeok_ContractSegmentY
