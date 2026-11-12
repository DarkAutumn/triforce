; ============================================================================
; FILE:     bank1/tile-block-power-triforce.asm BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 2012-2211
; SUBJECTS: Ganon, Item, Link, Unused
; TOPIC:    Link stopped by a sliding push block, rock or gravestone; the Triforce of Power fanfare
;           (level 9, after Ganon); bank-1 copies of Ganon's palette, ashes and room item helpers.
; SEE ALSO: bank7/walker-move.asm (calls Link_CheckSlidingTileObjBlocking for Link every move);
;           bank4/push-block.asm and bank4/rock-wall-tree-secrets.asm (state 1 = the object is sliding);
;           bank1/take-item.asm (Item_Take_GoPowerTriforce jumps to Item_BeginPowerTriforceFanfare);
;           bank7/mode5-play-update.asm (calls Item_UpdatePowerTriforceFanfare each frame in a dungeon);
;           bank4/ganon-draw-collisions.asm (the copies of the palette and ashes routines Ganon uses).
; KEY FACTS:
;   - While a push block ($68), rock ($62), gravestone ($65) or armos tile object ($66, a type the
;     overworld layouts never produce) is in state 1 (sliding to its new square), Link cannot move
;     if he is less than $10 pixels from it on both axes (his Y taken as Y + 3):
;     Link_CheckSlidingTileObjBlocking.
;   - Taking the Triforce of Power halts Link (state $40) for $C0 frames: Item_BeginPowerTriforceFanfare.
;   - During those frames the background palettes alternate every 4 frames between the level's
;     palettes and white: Item_UpdatePowerTriforceFanfare.
;   - InvPowerTriforce becomes 1 only when the $C0 frames end; then Link is freed and the level 9
;     song restarts: Item_UpdatePowerTriforceFanfare@EndFanfare.
;   - Unused_ReplaceGanonBrownPaletteRow_Bank1, Unused_ReplaceGanonBluePaletteRow_Bank1, Unused_DrawAshes_Bank1 and
;     Unused_ActivateRoomItem_Bank1 are never referenced; only Ganon_AppendAshesPaletteRow_Bank1 is used here.
; CONTENTS: (generated; L = line in this file)
;   L0045 Link_CheckSlidingTileObjBlocking          code  Cancels Link's move while a sliding push block, rock or gravestone is within $10 pixels of him.
;   L0099 Item_UpdatePowerTriforceFanfare           code  Runs the Triforce of Power fanfare: flashes the palettes, then gives the item and frees Link.
;   L0152 Item_BeginPowerTriforceFanfare            code  Starts the Triforce of Power fanfare: halts Link for $C0 frames.
;   L0164 Ganon_PaletteRow7TransferRecord_Bank1     data  DATA: PPU transfer record template for sprite palette row 7: address $3F1C, 4 bytes
;   L0170 Ganon_ColorTriples_Bank1                  data  TABLE: 3 color triples for palette row 7; index = end offset of the triple (2, 5, 8);
;   L0177 Unused_ReplaceGanonBrownPaletteRow_Bank1  code  Never referenced: cues palette row 7 with Ganon's brown colors (see Ganon_AppendAshesPaletteRow_Bank1).
;   L0182 Unused_ReplaceGanonBluePaletteRow_Bank1   code  Never referenced: cues palette row 7 with Ganon's blue colors (see Ganon_AppendAshesPaletteRow_Bank1).
;   L0194 Ganon_AppendAshesPaletteRow_Bank1         code  Cues a transfer of sprite palette row 7 with the colors of the ashes and the Triforce of Power.
;   L0243 Unused_DrawAshes_Bank1                    code  Never referenced: draws the pile of ashes (frame image $0B) at object slot X's position.
;   L0252 Unused_ActivateRoomItem_Bank1             code  Never referenced: makes the room item appear (state 0) unless it was already taken.
; ============================================================================

; ----------------------------------------------------------------------------
; Cancels Link's move while a sliding push block, rock or gravestone is within $10 pixels of him.
; IN:   [0F] = Link's movement direction
; OUT:  [0F] = 0 when blocked, else unchanged; X = 0
; NOTE: Tests object slots $0C-$01 for types $68, $62, $65, $66 in state 1 (state 1 is the slide
;       after the push: PushBlock_State1Moving, Rock_Update). Idle and finished objects
;       block through their background tiles instead, not here.
; ----------------------------------------------------------------------------
Link_CheckSlidingTileObjBlocking:
    ; Look through object slots $0C to 1 for a tile object that can block Link's movement.
    LDX #$0C

@LoopTileObj:
    LDA ObjType, X
    CMP #$68                    ; Push block
    BEQ @Found
    CMP #$62                    ; Rock
    BEQ @Found
    CMP #$65                    ; Gravestone
    BEQ @Found
    CMP #$66                    ; Armos tile object (updated like a rock; no room layout makes one).
    BNE @NextTileObj            ; If it doesn't match any, go check the next object.

@Found:
    LDA ObjState, X
    CMP #$01
    BNE @NextTileObj            ; If not in state 1 (sliding), go check the next object.

    ; If X distance between Link and the object >= $10, skip it.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Math_Abs
    CMP #$10
    BCS @NextTileObj

    ; If Y distance between Link's (Y + 3) and the object >= $10, skip it.
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    JSR Math_Abs
    CMP #$10
    BCS @NextTileObj

    ; Link is near the object. Reset movement direction.
    LDA #$00
    STA $0F

@NextTileObj:
    DEX
    BNE @LoopTileObj
    RTS

; ----------------------------------------------------------------------------
; Runs the Triforce of Power fanfare: flashes the palettes, then gives the item and frees Link.
; IN:   IsTriforceFanfareActive; ObjTimer (Link's timer, $C0 at the start)
; OUT:  while the timer runs: TileBufSelector = $18 or $78; at 0: InvPowerTriforce = 1,
;       ObjState = 0, IsTriforceFanfareActive = 0, SongRequest = $20
; NOTE: Called every frame of play in a dungeon. Link stays halted for the whole $C0 frames.
; ----------------------------------------------------------------------------
Item_UpdatePowerTriforceFanfare:
    ; If the fanfare is not active, then return.
    LDA IsTriforceFanfareActive
    BEQ @Exit

    ; When Link's timer expires, go finish up the fanfare.
    LDA ObjTimer
    BEQ @EndFanfare

    ; While Link's timer is counting down, switch every 4 frames between the level's
    ; palettes (transfer buffer selector $18) and the white palette (selector $78):
    ; white while (timer AND 7) >= 4.
    LDY #$18

    ; Redundant check of Link's timer: it was just found nonzero, so this never branches.
    LDA ObjTimer
    BEQ @FillHearts
    AND #$07
    CMP #$04
    BCC :+
    LDY #$78
:
    STY TileBufSelector
    RTS

@FillHearts:
    ; Unreachable (see the redundant check above): would start a heart refill
    ; (World_IsFillingHearts = 2) and advance the submode.
    LDA #$02
    STA World_IsFillingHearts
    INC GameSubmode
    RTS

@EndFanfare:
    ; The fanfare is done.
    JSR Ganon_AppendAshesPaletteRow_Bank1  ; Cue sprite palette row 7 with the ashes colors again.
    LDA #$20                    ; Play the song for level 9 again.
    STA SongRequest
    LDA #$01
    STA InvPowerTriforce        ; Link now has the Triforce of Power.
    LSR
    STA ObjState                ; Make Link idle and not halted.
    STA IsTriforceFanfareActive   ; Flag the fanfare not active.

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Starts the Triforce of Power fanfare: halts Link for $C0 frames.
; OUT:  IsTriforceFanfareActive incremented; ObjTimer = $C0; ObjState = $40 (Link halted)
; NOTE: Reached from the item pickup code for item slot $1B in a dungeon. The inventory byte is
;       not set here but when the fanfare ends (Item_UpdatePowerTriforceFanfare).
; ----------------------------------------------------------------------------
Item_BeginPowerTriforceFanfare:
    INC IsTriforceFanfareActive

    ; Halt Link for $C0 frames.
    LDA #$C0
    STA ObjTimer
    LDA #$40
    STA ObjState
    RTS

; DATA: PPU transfer record template for sprite palette row 7: address $3F1C, 4 bytes
;       ($0F, $07, $17, $27), end marker $FF. The last 3 colors are replaced from Ganon_ColorTriples_Bank1.
Ganon_PaletteRow7TransferRecord_Bank1:
    .BYTE $3F, $1C, $04, $0F, $07, $17, $27, $FF

; TABLE: 3 color triples for palette row 7; index = end offset of the triple (2, 5, 8);
;        value = NES colors. $07, $17, $30: Ganon brown. $16, $2C, $3C: Ganon blue.
;        $27, $06, $16: ashes and Triforce of Power.
Ganon_ColorTriples_Bank1:
    .BYTE $07, $17, $30, $16, $2C, $3C, $27, $06
    .BYTE $16

; ----------------------------------------------------------------------------
; Never referenced: cues palette row 7 with Ganon's brown colors (see Ganon_AppendAshesPaletteRow_Bank1).
; ----------------------------------------------------------------------------
Unused_ReplaceGanonBrownPaletteRow_Bank1:
    LDY #$02
    BNE :+

; Never referenced: cues palette row 7 with Ganon's blue colors (see Ganon_AppendAshesPaletteRow_Bank1).
Unused_ReplaceGanonBluePaletteRow_Bank1:
    LDY #$05
    BNE :+

; ----------------------------------------------------------------------------
; Cues a transfer of sprite palette row 7 with the colors of the ashes and the Triforce of Power.
; IN:   DynTileBufLen (the color patch below is correct only when it is 0)
; OUT:  8-byte record appended to DynTileBuf; DynTileBufLen += 8; X = CurObjIndex
; NOTE: Same code as Ganon_AppendPaletteRowTransferRecord_Triforce in bank 4. The record is
;       appended at DynTileBufLen, but the 3 colors are patched at the fixed offsets 4-6 of the
;       buffer, and the new length points past the record's $FF end marker instead of at it.
; ----------------------------------------------------------------------------
Ganon_AppendAshesPaletteRow_Bank1:
    LDY #$08
:
    ; Save Y, the offset in Ganon_ColorTriples_Bank1 of the last color of the triple.
    TYA
    PHA

    ; Get the dynamic transfer buf length, so we write from where we last wrote.
    LDX DynTileBufLen

    ; Copy the palette row 7 transfer record to dynamic transfer buf (8 bytes).
    LDY #$00

@CopyPaletteRecord:
    LDA Ganon_PaletteRow7TransferRecord_Bank1, Y
    STA DynTileBuf, X
    INX
    INY
    CPY #$08
    BNE @CopyPaletteRecord

    ; Update the dynamic transfer buf length.
    STX DynTileBufLen
    PLA
    TAY

    ; Replace the last 3 color bytes of the record (the template's brown shades)
    ; with the chosen triple, last color first.
    ;
    ; Note that here the dynamic transfer buf is being written with
    ; absolute offsets (4 to 6), instead of the relative ones used in copying
    ; the template record.
    LDX #$02

@ReplaceColors:
    LDA Ganon_ColorTriples_Bank1, Y
    STA DynTileBuf+4, X
    DEY
    DEX
    BPL @ReplaceColors

    ; Switch back to the current object slot.
    LDX CurObjIndex
    RTS

; ----------------------------------------------------------------------------
; Never referenced: draws the pile of ashes (frame image $0B) at object slot X's position.
; NOTE: Same code as Ending_DrawAshPile in bank 2 and Ganon_DrawAshes in bank 4.
; ----------------------------------------------------------------------------
Unused_DrawAshes_Bank1:
    JSR Anim_FetchObjPosForSpriteDescriptor
    LDA #$0B
    JMP Anim_DrawObjectNotMirrored

; ----------------------------------------------------------------------------
; Never referenced: makes the room item appear (state 0) unless it was already taken.
; NOTE: Same code as Ganon_ActivateRoomItem in bank 4.
; ----------------------------------------------------------------------------
Unused_ActivateRoomItem_Bank1:
    LDA ObjState+19             ; Room item state (slot $13); 0 = already active.
    BEQ :+
    JSR RoomFlags_GetItemTaken  ; Nonzero = the room's item was taken before.
    BNE :+
    LDA #$00
    STA ObjState+19
    LDA #$02                    ; Request square-1 tune bit $02.
    STA Tune1Request
:
    RTS
