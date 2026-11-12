; ============================================================================
; FILE:     bank7/flute-secret-pond.asm  BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 5864-5970
; SUBJECTS: FluteSecret, Obj
; TOPIC:    Shared init tails for objects (tile objects and items, metastate/timer reset) and the
;           flute secret object ($5E): the pond's colour cycle, the stairs it uncovers, and the
;           reverse cycle played when Link leaves the room by scrolling.
; SEE ALSO: bank7/flute.asm (Flute_Wield: which rooms get the secret, per quest; creates the $5E
;           object); bank7/object-init-table.asm (Obj_Init and its table use the init tails);
;           bank7/object-update-table.asm (FluteSecret_Update is the update of type $5E);
;           bank4/rock-wall-tree-secrets.asm (Secret_RevealAndFlagStairsObj); bank5/mode7-scroll-init.asm
;           (Mode7Scroll_Init_Sub0 calls FluteSecret_ReversePondCycle).
; KEY FACTS:
;   - The flute secret object steps SecretColorCycle once every 8 frames (when FrameCounter AND 7 = 4),
;     from 1 up; the room is frozen until FluteTimer runs out, so it starts after the tune
;     (FluteSecret_Update).
;   - Steps 1-10 recolour the water (colour 3 of background palette 3, PPU $3F0F) through
;     FluteSecret_PondCycleColors; step 10 also sets RoomFirstUnwalkableTile = $99, which makes tiles below $99
;     walkable (FluteSecret_CuePondPaletteRow).
;   - Step 11 (81-88 frames after updates resume) puts stairs at X = $60, Y = $90, flags the room's
;     secret as found and frees the object; SecretColorCycle stays $0C (FluteSecret_RevealPondStairs).
;   - The stairs position is the same in every room that has the secret (FluteSecret_RevealPondStairs).
;   - SecretColorCycle <> 0 blocks a second flute secret in the room (Flute_Wield) and delays the next
;     room scroll: mode 7 init submode 0 runs the cycle backwards to 0 first, one step on each
;     frame with FrameCounter bit 2 set (FluteSecret_ReversePondCycle).
;   - Types $5F-$69 (ladder, dropped item, tile objects) all get object attribute $81, timer 0 and
;     metastate 0: no spawn cloud (Obj_InitTileObjOrItem).
; CONTENTS: (generated; L = line in this file)
;   L0049 Obj_InitTileObjOrItem                code  Init of object types $5F-$69 (ladder, dropped item, tile objects): attribute $81, no spawn cloud.
;   L0063 FluteSecret_Init                     code  Init of the flute secret object ($5E): starts the pond colour cycle at 1.
;   L0069 Obj_ResetMetastateAndTimer           code  Init tail: clears the object's timer and metastate (no spawn cloud). Also the init routine of
;   L0075 Obj_ResetMetastate                   code  Init tail: sets ObjMetastate,X = 0, the normal state in which the object's own update routine
;   L0082 FluteSecret_WaterPaletteTransferBuf  data  DATA: PPU transfer record for background palette 3 (the water palette): address $3F0C, count 4,
;   L0088 FluteSecret_PondCycleColors          data  TABLE: colour 3 of the water palette during the pond cycle; index = cycle step 0-$0B; value = NES
;   L0101 FluteSecret_Update                   code  Update of the flute secret object ($5E): steps the pond cycle every 8 frames, then makes stairs.
;   L0127 FluteSecret_CuePondPaletteRow        code  Cues the water palette with the colour of cycle step Y; step $0A also makes the water walkable.
;   L0148 FluteSecret_Return                   code  Shared exit of FluteSecret_Update, FluteSecret_CuePondPaletteRow and FluteSecret_ReversePondCycle.
;   L0157 FluteSecret_RevealPondStairs         code  Moves the flute secret object to X = $60, Y = $90 and turns it into the revealed stairs.
;   L0172 FluteSecret_ReversePondCycle         code  Runs the pond colour cycle one step backwards; called each frame while leaving the room.
; ============================================================================

; ----------------------------------------------------------------------------
; Init of object types $5F-$69 (ladder, dropped item, tile objects): attribute $81, no spawn cloud.
; IN:   X = object slot.
; OUT:  ObjAttr,X = $81 (bit $01: the object checks collisions and draws itself; bit $80: it does
;       not turn around after hitting Link); ObjTimer,X = 0; ObjMetastate,X = 0.
; NOTE: This replaces the attribute Obj_Init copied from Obj_TypeAttributes.
; ----------------------------------------------------------------------------
Obj_InitTileObjOrItem:
    LDA #$81
    STA ObjAttr, X
    BNE Obj_ResetMetastateAndTimer    ; Always taken.

; ----------------------------------------------------------------------------
; Init of the flute secret object ($5E): starts the pond colour cycle at 1.
; IN:   X = object slot.
; OUT:  SecretColorCycle = 1; ObjTimer,X = 0; ObjMetastate,X = 0.
; NOTE: Flute_Wield creates the object during Link's update and the object loop runs later in
;       the same frame, so this runs on the frame the flute is used. The object's update
;       (FluteSecret_Update) cannot run until FluteTimer has run out, because the play-mode update
;       returns at once while it is nonzero.
; ----------------------------------------------------------------------------
FluteSecret_Init:
    LDA #$01
    STA SecretColorCycle

; Init tail: clears the object's timer and metastate (no spawn cloud). Also the init routine of
; Zora, the Wizzrobes, Patra's children and Wallmaster (Obj_Init_JumpTable).
Obj_ResetMetastateAndTimer:
    LDA #$00
    STA ObjTimer, X

; Init tail: sets ObjMetastate,X = 0, the normal state in which the object's own update routine
; runs (1-4 are the spawn cloud, $10-$14 the death sparkle). OUT: A = 0.
Obj_ResetMetastate:
    LDA #$00
    STA ObjMetastate, X
    RTS

; DATA: PPU transfer record for background palette 3 (the water palette): address $3F0C, count 4,
; colours $0F $17 $37 $12, then the $FF terminator. The last colour is replaced before use.
FluteSecret_WaterPaletteTransferBuf:
    .BYTE $3F, $0C, $04, $0F, $17, $37, $12, $FF

; TABLE: colour 3 of the water palette during the pond cycle; index = cycle step 0-$0B; value = NES
; colour number. Step 0 ($12) is the same colour as in the template above; steps $0A and $0B ($37)
; are the same colour as colour 2 of that palette.
FluteSecret_PondCycleColors:
    .BYTE $12, $11, $22, $21, $31, $32, $33, $35    ; steps 0-7
    .BYTE $34, $36, $37, $37                        ; steps 8-$0B

; ----------------------------------------------------------------------------
; Update of the flute secret object ($5E): steps the pond cycle every 8 frames, then makes stairs.
; IN:   X = object slot; SecretColorCycle = 1-$0B while running.
; OUT:  SecretColorCycle one higher on a step frame; palette record cued for steps 1-$0A; at step
;       $0B the stairs appear and the object is freed (SecretColorCycle is left at $0C).
; NOTE: A step happens only when FrameCounter AND 7 = 4. 11 steps: the stairs appear 81 to 88
;       frames after the object starts updating. Nothing here depends on the room; the room
;       test is in Flute_Wield.
; ----------------------------------------------------------------------------
FluteSecret_Update:
    ; Cycle finished ($0C or more): nothing to do.
    LDY SecretColorCycle
    CPY #$0C
    BCS FluteSecret_Return

    ; Act on one frame in every 8.
    LDA FrameCounter
    AND #$07
    CMP #$04
    BNE FluteSecret_Return

    ; Advance the cycle. Y keeps the value from before the increment:
    ; $0B = reveal the stairs; 1-$0A = change the water colour.
    INC SecretColorCycle
    CPY #$0B
    BEQ FluteSecret_RevealPondStairs

; ----------------------------------------------------------------------------
; Cues the water palette with the colour of cycle step Y; step $0A also makes the water walkable.
; IN:   Y = cycle step (0-$0B).
; OUT:  DynTileBuf holds the palette record (8 bytes); RoomFirstUnwalkableTile = $99 when Y = $0A.
; NOTE: The record is written at the start of DynTileBuf, over anything cued there this frame.
;       The walkable limit is raised to $99 both on the way up (step $0A) and on the way down
;       (FluteSecret_ReversePondCycle passing $0A); nothing here lowers it again.
; ----------------------------------------------------------------------------
FluteSecret_CuePondPaletteRow:
    ; Copy the 8-byte palette record to the dynamic transfer buffer, keeping Y.
    TYA
    PHA
    LDY #$07

@CopyBytes:
    LDA FluteSecret_WaterPaletteTransferBuf, Y
    STA DynTileBuf, Y
    DEY
    BPL @CopyBytes
    PLA
    TAY
    LDA FluteSecret_PondCycleColors, Y      ; Replace the record's last colour (colour 3) with this step's colour.
    STA DynTileBuf+6
    CPY #$0A
    BNE FluteSecret_Return             ; Only at step $0A:
    LDA #$99                    ; tiles below $99 become walkable (the limit set on room entry is lower).
    STA RoomFirstUnwalkableTile

; Shared exit of FluteSecret_Update, FluteSecret_CuePondPaletteRow and FluteSecret_ReversePondCycle.
FluteSecret_Return:
    RTS

; ----------------------------------------------------------------------------
; Moves the flute secret object to X = $60, Y = $90 and turns it into the revealed stairs.
; IN:   X = object slot of the flute secret object.
; NOTE: Secret_RevealAndFlagStairsObj draws stairs tiles (first tile $70) at the object's
;       position, frees the slot, plays the secret tune and sets bit 7 of the room's flags.
; ----------------------------------------------------------------------------
FluteSecret_RevealPondStairs:
    LDA #$60
    STA ObjX, X
    LDA #$90
    STA ObjY, X
    JMP Secret_RevealAndFlagStairsObj

; ----------------------------------------------------------------------------
; Runs the pond colour cycle one step backwards; called each frame while leaving the room.
; IN:   SecretColorCycle nonzero (the caller, mode 7 init submode 0, tests it and stays in
;       that submode until it is 0).
; OUT:  On frames with FrameCounter bit 2 set: SecretColorCycle one lower and its colour cued.
; NOTE: Steps on 4 consecutive frames, then pauses 4. From $0C that is 12 steps; the last one
;       (step 0) restores the original colour $12. The room scroll waits for this.
; ----------------------------------------------------------------------------
FluteSecret_ReversePondCycle:
    LDA FrameCounter
    AND #$04
    BEQ FluteSecret_Return
    DEC SecretColorCycle
    LDY SecretColorCycle
    JMP FluteSecret_CuePondPaletteRow
