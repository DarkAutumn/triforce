; ============================================================================
; FILE:     bank4/gleeok-manhandla-gohma-inits.asm BANK: 4 (switchable, $8000-$BFFF)
; UPSTREAM: Z_04.asm 7638-7830
; SUBJECTS: Gleeok, Gohma, Manhandla
; TOPIC:    Init routines of Gleeok ($42-$45: neck segments, HP, head timing), Manhandla ($3C: four hands
;           and the base in object slots 1-5), Gohma ($33/$34) and the flying Gleeok head ($46).
; SEE ALSO: bank4/gleeok.asm (Gleeok_Update, Gleeok_UpdateFlyingHead); bank4/gleeok-head-body.asm (head and neck
;           collisions, cutting a head off); bank4/manhandla.asm (Manhandla_Update); bank4/gohma.asm
;           (Gohma_Update, Gohma_HandleWeaponCollision); bank4/bubble-keese.asm (Keese_InitBlue);
;           bank7/object-init-table.asm (Obj_Init_JumpTable calls these; HP and attributes are loaded
;           there first); bank7/object-update-table.asm (Obj_TypeHpPairs).
; KEY FACTS:
;   - Gleeok: object slots 1-6 get HP $A0 and ObjInvincibilityMask $FE: only sword-type damage (sword,
;     sword shot, rod) counts (Gleeok_Init). Only slot 1 (neck bottom) and slot 5 (head) are ever
;     collision-tested, and slot 5 serves the head of every neck in turn, so the heads share one HP
;     value (Gleeok_CheckSegmentCollisions in bank4/gleeok-head-body.asm).
;   - Gleeok: every neck starts with 6 stored segments at X $7C, Y $6F-$88 (Gleeok_SegmentStartYs): index 0
;     (slot 1, Y $6F) is the bottom, index 4 (slot 5, Y $83) the head. The heads of necks 0-3 start
;     moving after 0, 12, 24 and 36 of their moves (Gleeok_Init).
;   - Gleeok, Manhandla and Gohma each request their roar sample at init: $10, $40, $20.
;   - Manhandla: slots 1-4 are the hands (up, down, left, right of the base, 16 pixels away), slot 5 the
;     base at the spawn position slot 5 was given on room entry. All five are type $3C with slot 1's
;     HP ($40 from the HP table), direction and attributes (Manhandla_Init, Spawn_AssignPositions).
;   - Manhandla: ObjInvincibilityMask $E2: boomerang and fire do nothing; sword, arrow, bomb and magic
;     shot hurt (Manhandla_Init).
;   - Manhandla: all segments start with speed low byte $80 (0.5 pixel per frame while the high byte is
;     0) in one random 8-way direction (Manhandla_Init).
;   - Gohma: ObjInvincibilityMask $FB: only arrows pass the mask. Starts at ($80, $70). The shoot timer
;     starts at 1, so the first fireball leaves on Gohma's first update (Gohma_Init, Gohma_Update).
;   - Flying Gleeok head: a blue Keese-style flyer with start speed $BF and top speed $E0
;     (Gleeok_InitFlyingHead). FlyerMaxSpeedFrac is one byte for the whole room.
; CONTENTS: (generated; L = line in this file)
;   L0048 Gleeok_SegmentStartYs              data  TABLE: start Y of the 6 stored segments of every Gleeok neck; index = segment (object slot - 1); pixels.
;   L0059 Gleeok_Init                        code  Initializes Gleeok (types $42-$45): neck segment positions, HP and mask of slots 1-6, head timing.
;   L0133 Manhandla_BaseFrameImagesAndAttrs  data  TABLE: start value of Manhandla_ObjFrame; index = segment 0-4 (slots 1-5); frame image and sprite bits.
;   L0139 Manhandla_SegmentOffsetsX          data  TABLE: X offset of a Manhandla segment from the base; index = segment 0-4; signed pixels.
;   L0143 Manhandla_SegmentOffsetsY          data  TABLE: Y offset of a Manhandla segment from the base; index = segment 0-4; signed pixels.
;   L0155 Manhandla_Init                     code  Initializes Manhandla (type $3C): makes object slots 1-4 the hands and slot 5 the base.
;   L0221 Gohma_Init                         code  Initializes Gohma (types $33, $34): arrow-only mask, shoot timer 1, position ($80, $70).
;   L0240 Gleeok_InitFlyingHead              code  Initializes a flying Gleeok head (type $46) as a fast blue Keese-style flyer.
; ============================================================================

; TABLE: start Y of the 6 stored segments of every Gleeok neck; index = segment (object slot - 1); pixels.
;   Index 0 (slot 1) is the neck bottom, index 4 (slot 5) the head. Index 5 (slot 6) is stored and
;   loaded with the neck but never read: Gleeok_MoveNeck, Gleeok_MoveHead and the draw and collision
;   loop (Gleeok_DrawSegmentAndCheckCollisions) address slots 1-5 only. The copy loops are 6 long
;   because they also copy the 6-byte head movement block (Gleeok_CurHeadInfo).
Gleeok_SegmentStartYs:
    .BYTE $6F, $74, $79, $7E, $83, $88

; ----------------------------------------------------------------------------
; Initializes Gleeok (types $42-$45): neck segment positions, HP and mask of slots 1-6, head timing.
; IN:   Nothing is read from X; the loop below uses X as the segment index. Gleeok is object slot 1
;       (Gleeok_Update reads ObjType+1).
; OUT:  X = $FF. All 4 necks are set up whatever the type; Gleeok_Update uses (type - $42 + 1) of them.
; NOTE: Slots 2-6 keep object type 0: they are work slots into which Gleeok_Update loads one neck at a
;       time. ObjHP+5 (head) starts at $A0 and is shared by the heads of all necks.
; ----------------------------------------------------------------------------
Gleeok_Init:
    LDA #$10                    ; Roar sample (Aquamentus, Gleeok, Ganon).
    STA SampleRequest

    ; For each stored segment 5 to 0 (object slots 6 to 1; the head is slot 5):
    LDX #$05
@NextSegment:
    ; X of this segment in all 4 necks = $7C.
    LDA #$7C
    STA Gleeok_NeckXs0, X
    STA Gleeok_NeckXs1, X
    STA Gleeok_NeckXs2, X
    STA Gleeok_NeckXs3, X
    STA ObjX+1, X               ; Overwritten before use: Gleeok_Update reloads slots 1-6 from the arrays.

    ; Y of this segment in all 4 necks from the table.
    LDA Gleeok_SegmentStartYs, X
    STA Gleeok_NeckYs0, X
    STA Gleeok_NeckYs1, X
    STA Gleeok_NeckYs2, X
    STA Gleeok_NeckYs3, X
    STA ObjY+1, X               ; Overwritten before use, as for X above.

    ; HP $A0 in slots 1-6. HP is not part of the per-neck arrays: it stays in the object slots, and
    ; only slot 1 (bottom) and slot 5 (head) are collision-tested.
    LDA #$A0
    STA ObjHP+1, X

    ; Metastate 0 (normal) in slots 1-6; "uninitialized" flag cleared in slots 2-7 (slot 1's flag was
    ; already cleared by Obj_Update before it called this init).
    LDA #$00
    STA ObjMetastate+1, X
    STA ObjUninitialized+2, X

    ; Mask $FE: everything but damage type 1 (sword, sword shot, rod) is ignored.
    LDA #$FE
    STA ObjInvincibilityMask+1, X
    DEX
    BPL @NextSegment

    ; X = $FF. Step flags: nonzero = -1 pixel per move, 0 = +1.
    ;   neck 0: X step -1      neck 1: Y step -1
    ;   neck 2: X step -1      neck 3: Y step -1
    ; The other step flags stay 0.
    STX Gleeok_HeadInfo0+GleeokHead_SpeedFlagX
    STX Gleeok_HeadInfo1+GleeokHead_SpeedFlagY
    STX Gleeok_HeadInfo2+GleeokHead_SpeedFlagX
    STX Gleeok_HeadInfo3+GleeokHead_SpeedFlagY

    ; Vertical direction counter of every head = 3.
    LDA #$03
    STA Gleeok_HeadInfo0+GleeokHead_DirCounterV
    STA Gleeok_HeadInfo1+GleeokHead_DirCounterV
    STA Gleeok_HeadInfo2+GleeokHead_DirCounterV
    STA Gleeok_HeadInfo3+GleeokHead_DirCounterV

    ; Horizontal direction counter of every head = 6.
    ASL
    STA Gleeok_HeadInfo0+GleeokHead_DirCounterH
    STA Gleeok_HeadInfo1+GleeokHead_DirCounterH
    STA Gleeok_HeadInfo2+GleeokHead_DirCounterH
    STA Gleeok_HeadInfo3+GleeokHead_DirCounterH

    ; Start delays, in head moves: neck 0 = 0 (untouched), neck 1 = 12, neck 2 = 24, neck 3 = 36.
    ASL
    STA Gleeok_HeadInfo1+GleeokHead_Delay
    ASL
    STA Gleeok_HeadInfo2+GleeokHead_Delay
    ADC Gleeok_HeadInfo1+GleeokHead_Delay   ; C is clear after the ASL: 24 + 12.
    STA Gleeok_HeadInfo3+GleeokHead_Delay
    RTS

; TABLE: start value of Manhandla_ObjFrame; index = segment 0-4 (slots 1-5); frame image and sprite bits.
;   Bit 0 is the animation frame (toggled later); bit 7 vertical flip, bit 6 horizontal flip.
Manhandla_BaseFrameImagesAndAttrs:
    .BYTE $00, $80, $02, $42, $04   ; up hand, down hand (V flip), left hand, right hand (H flip), base

; TABLE: X offset of a Manhandla segment from the base; index = segment 0-4; signed pixels.
;   Only 2 bytes are stored here; indexes 2-4 read the first 3 bytes of Manhandla_SegmentOffsetsY, so
;   the five values are $00, $00, $F0, $10, $00.
Manhandla_SegmentOffsetsX:
    .BYTE $00, $00

; TABLE: Y offset of a Manhandla segment from the base; index = segment 0-4; signed pixels.
Manhandla_SegmentOffsetsY:
    .BYTE $F0, $10, $00, $00, $00   ; up hand -16, down hand +16, left, right, base

; ----------------------------------------------------------------------------
; Initializes Manhandla (type $3C): makes object slots 1-4 the hands and slot 5 the base.
; IN:   X = object slot being initialized. The code is written for slot 1: it copies slot 1's direction,
;       attributes and HP to slots 1-5.
; OUT:  Slots 1-5: type $3C, ObjInvincibilityMask $E2, metastate 0, initialized, speed low byte $80.
; NOTE: The base stays at the position slot 5 already had on entry: Spawn_AssignPositions gives
;       every slot $01-$09 a spawn position whatever the monster count, so the base sits on slot 5's
;       spawn spot. RNG: Random,X AND 7 picks the common start direction.
; ----------------------------------------------------------------------------
Manhandla_Init:
    LDA #$40                    ; Roar sample (Digdogger, Manhandla, Patra).
    STA SampleRequest

    ; Random 8-way direction.
    LDA Random, X
    AND #$07
    TAY
    LDA Dir_EightWayDirs, Y
    STA ObjDir, X

    ; For each segment 4 to 0 (object slots 5 to 1), indexed by Y:
    LDY #$04
@NextSegment:
    ; Direction of slot 1 (the one chosen above when X = 1).
    LDA ObjDir+1
    STA a:ObjDir+1, Y

    ; Every segment is object type $3C.
    LDA #$3C
    STA ObjType+1, Y

    ; Mask $E2: immune to boomerang (2) and fire ($20); bits $40 and $80 match no damage type.
    LDA #$E2
    STA ObjInvincibilityMask+1, Y
    LDA Manhandla_BaseFrameImagesAndAttrs, Y
    STA Manhandla_ObjFrame+1, Y

    ; No spawn cloud, and no init call of its own.
    LDA #$00
    STA ObjMetastate+1, Y
    STA ObjUninitialized+1, Y

    ; Attributes and HP of slot 1, which Obj_Init loaded for type $3C before calling this.
    LDA ObjAttr+1
    STA ObjAttr+1, Y
    LDA ObjHP+1
    STA ObjHP+1, Y

    ; Position = slot 5's position + this segment's offset. Slot 5 (the base, offset 0) is done first
    ; and keeps its position.
    LDA ObjX+5
    CLC
    ADC Manhandla_SegmentOffsetsX, Y
    STA a:ObjX+1, Y
    LDA ObjY+5
    CLC
    ADC Manhandla_SegmentOffsetsY, Y
    STA a:ObjY+1, Y

    ; Speed low byte $80 (1/256 pixel units: 0.5 pixel per frame while the high byte is 0).
    LDA #$80
    STA Manhandla_ObjSpeedFrac+1, Y
    DEY
    BPL @NextSegment
    RTS

; ----------------------------------------------------------------------------
; Initializes Gohma (types $33, $34): arrow-only mask, shoot timer 1, position ($80, $70).
; IN:   X = object slot.
; OUT:  Timer and metastate cleared by Obj_ResetMetastateAndTimer (no spawn cloud).
; NOTE: Mask $FB lets only damage type 4 (arrow) through; Gohma_HandleWeaponCollision then also wants
;       the open eye. The INC makes the shoot timer 1: room entry zeroes $0300-$051F
;       (Util_ClearRam0300UpTo). Gohma_Update decrements it before testing, so the first fireball is shot
;       on the first update; the reload is $41.
; ----------------------------------------------------------------------------
Gohma_Init:
    LDA #$20                    ; Roar sample (Dodongo, Gohma).
    STA SampleRequest
    LDA #$FB
    STA ObjInvincibilityMask, X
    INC Gohma_ObjShootTimer, X
    LDA #$80
    STA ObjX, X
    LDA #$70
    STA ObjY, X
    JMP Obj_ResetMetastateAndTimer

; ----------------------------------------------------------------------------
; Initializes a flying Gleeok head (type $46) as a fast blue Keese-style flyer.
; IN:   X = object slot.
; OUT:  As Keese_InitBlue (random 8-way direction, flyer state reset), then Flyer_ObjSpeed,X = $BF and
;       FlyerMaxSpeedFrac = $E0.
; NOTE: FlyerMaxSpeedFrac is shared by every flyer in the room.
; ----------------------------------------------------------------------------
Gleeok_InitFlyingHead:
    JSR Keese_InitBlue
    LDA #$E0
    STA FlyerMaxSpeedFrac
    LDA #$BF
    STA Flyer_ObjSpeed, X
    RTS
