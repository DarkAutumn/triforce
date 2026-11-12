; ============================================================================
; FILE:     bank2/demo-story-scroll.asm  BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 540-1001
; SUBJECTS: Demo, Mode0Demo, Unused
; TOPIC:    Animation subphases of the demo: the title wait (phase 0 subphase 0) and the whole story
;           phase: scroll the story screen in, wait, scroll the text and the item parade one 8-pixel
;           line at a time, wait, then restart the title.
; SEE ALSO: bank2/mode0-demo.asm (Mode0Demo_Animate dispatches here); bank2/demo-tables-init.asm (Demo_LineAttrs,
;           item lists, text, Mode0Demo_IncSubphase); bank2/title-animation.asm (title sprites, phase 0 subphase
;           1); bank7/room-item-draw.asm (Item_DrawObject draws the parade items);
;           bank1/item-sprite-draw.asm (Anim_WriteItemSprites).
; KEY FACTS:
;   - Title wait: DemoTimer counts every other frame and the subphase ends when it wraps, 512 frames.
;     Mode0Demo_Animate_Phase0_Sub0.
;   - Story subphase 0 scrolls 1 pixel every other frame from CurVScroll $10 through the name table
;     wrap at $F0 to 8: 232 pixels, 464 frames. Mode0Demo_Animate_Phase1_Sub0.
;   - Story subphase 1 waits 256 frames and sets the first text line's PPU addresses (tiles $2900,
;     attributes $2BE0). Mode0Demo_Animate_Phase1_Sub1.
;   - Story subphase 2 scrolls 1 pixel every other frame for 5 screens of 256 lines plus $80 lines
;     (1408 pixels, 2816 frames); every 8 lines it queues one 32-tile line (blank tile $24, or the
;     next text field) and handles that line's Demo_LineAttrs bits. Mode0Demo_Animate_Phase1_Sub2.
;   - Parade items live in object slots $01-$0A: ObjState 0 = shown, $FF = free; they move up 1 pixel
;     with each scrolled line and are freed at Y = $F0. Demo_SpawnLineItems, Demo_FreeScrolledOffObjects.
;   - Demo_SpawnLineItems takes the highest free slot for the left item and the slot below it for the
;     right item without testing that one; the search loop has no lower bound.
;   - Item id $23 (fairy) is drawn with a 2-frame wing animation that changes every 4 frames; ids
;     >= $30 are drawn by Demo_DrawStoryFinalItem; all others by Item_DrawObject.
;     Demo_AnimateObjects.
;   - Story subphase 3 waits 256 frames, subphase 4 ends on its 57th frame (DemoTimer = $39) and
;     clears DemoPhase, DemoSubphase and IsUpdatingMode, so the title is initialized again.
;     Mode0Demo_Animate_Phase1_Sub3, Mode0Demo_Animate_Phase1_Sub4.
;   - Unused_DemoDecSpriteYs is not referenced by any code.
; CONTENTS: (generated; L = line in this file)
;   L0053 Mode0Demo_Animate_Phase0_Sub0         code  Title subphase 0: animates the title for 512 frames, then advances the subphase.
;   L0070 Mode0Demo_Animate_Phase1_Sub0         code  Story subphase 0: scrolls the story screen up 1 pixel every other frame until it is fully shown.
;   L0104 Mode0Demo_Animate_Phase1_Sub1         code  Story subphase 1: waits 256 frames and sets the PPU addresses of the first scrolled text line.
;   L0125 Mode0Demo_Animate_Phase1_Sub2         code  Story subphase 2: scrolls text and item parade; every 8 pixel lines queues the next line of tiles.
;   L0261 Demo_QueueLineAttrs                   code  If the current line has flag $40, appends a record that zeroes one row of 8 attribute bytes.
;   L0311 Demo_FreeScrolledOffObjects           code  Frees (ObjState = $FF) every parade object in slots $01-$0A whose Y has reached $F0.
;   L0337 Demo_SpawnLineItems                   code  If the current line has flag $20, spawns the next item parade row at Y = $EF in free object slots.
;   L0400 Demo_AnimateObjects                   code  Draws every shown parade object in slots $0A down to $01 according to its DemoItemIds value.
;   L0441 Unused_DemoDecSpriteYs                code  Dead code (no reference to it): decrements bytes of Sprites+80 indexed by 4 or $C, then A-4 down.
;   L0460 Demo_DrawFairy                        code  Draws the parade fairy (item id $23) in place, switching between 2 frames every 4 frames.
;   L0480 Demo_DrawStoryFinalItem               code  Draws one row (up to 6 sprites, 8 pixels apart) of the closing picture for item ids $30-$33.
;   L0532 Mode0Demo_Animate_Phase1_Sub3         code  Story subphase 3: keeps drawing the parade objects for 256 frames, then advances the subphase.
;   L0542 Mode0Demo_Animate_Phase1_Sub4         code  Story subphase 4: draws the objects for 56 more frames, then restarts the demo at phase 0.
;   L0555 Mode0Demo_Animate_Phase1_DrawObjects  code  Shared tail of story subphases 3 and 4: hides all sprites, then draws the parade objects.
; ============================================================================

; ----------------------------------------------------------------------------
; Title subphase 0: animates the title for 512 frames, then advances the subphase.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase0_Sub0:
    ; Animate while we wait about 512 frames.
    LDA FrameCounter
    AND #$01                    ; The timer is incremented every other frame.
    BEQ @SkipTimer
    INC DemoTimer
    LDA DemoTimer
    BNE @SkipTimer
    JMP Mode0Demo_IncSubphase             ; Go advance the demo subphase and return.

@SkipTimer:
    JSR Title_AnimateSpritesAndGlow
    RTS

; ----------------------------------------------------------------------------
; Story subphase 0: scrolls the story screen up 1 pixel every other frame until it is fully shown.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase1_Sub0:
    LDA FrameCounter            ; Increase CurVScroll every odd frame.
    AND #$01
    BEQ @CheckVScroll
    INC CurVScroll
    LDA CurVScroll              ; Have we scrolled to the bottom of nametable 2?
    CMP #$F0
    BNE @CheckVScroll
    INC DemoScrolledScreenCount

    ; The bottom of NT 2 is also the top of NT 0.
    ; So, reset CurVScroll and the base NT.
    LDA #$00
    STA CurVScroll
    INC SwitchNameTablesReq

@CheckVScroll:
    ; Scrolling hasn't ended until we've scrolled to the bottom,
    ; wrapped around, and scrolled 8 more lines.
    LDA CurVScroll
    CMP #$08
    BNE @Exit
    LDA DemoScrolledScreenCount
    BEQ @Exit
    LDA #$00
    STA DemoScrolledScreenCount
    INC DemoSubphase

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Story subphase 1: waits 256 frames and sets the PPU addresses of the first scrolled text line.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase1_Sub1:
    INC DemoTimer
    LDA DemoTimer
    BNE :+
    INC DemoSubphase            ; Go to the next subphase.
:
    LDA #$29
    STA DemoLineTileVramAddrHi
    LDA #$00
    STA DemoLineTileVramAddrLo
    LDA #$2B
    STA DemoLineAttrVramAddrHi
    LDA #$E0
    STA DemoLineAttrVramAddrLo
    RTS

; ----------------------------------------------------------------------------
; Story subphase 2: scrolls text and item parade; every 8 pixel lines queues the next line of tiles.
; NOTE: Ends when the line count reaches 1408 (2816 frames); one Demo_LineAttrs entry is handled
;       every 8 scrolled lines (16 frames), 175 entries in all.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase1_Sub2:
    JSR Sprite_HideAll
    JSR Demo_FreeScrolledOffObjects
    JSR Demo_AnimateObjects
    LDA FrameCounter
    AND #$01
    BEQ @Exit                   ; Every even frame, just return.
    LDX #$0A                    ; Decrement the Y coordinate of every object.

@DecObjYs:
    DEC ObjY, X
    DEX
    BNE @DecObjYs
    INC DemoScrolledLineCount       ; We scrolled one more line.
    LDA DemoScrolledLineCount
    BNE :+                      ; Once we've scrolled a whole screen,
    INC DemoScrolledScreenCount     ; Increment the screen count.
:
    LDA DemoScrolledScreenCount     ; Have we scrolled 5 screens?
    CMP #$05
    BNE @Scroll                 ; Scroll the nametable.
    LDA DemoScrolledLineCount       ; Scroll half a screen more.
    CMP #$80
    BNE @Scroll                 ; Scroll the nametable.
    INC DemoSubphase            ; Go to the next subphase.

@Exit:
    RTS

@Scroll:
    INC CurVScroll              ; Update vertical scrolling.
    LDA CurVScroll
    CMP #$F0
    BNE @CheckLine              ; If we reached the bottom of the screen,
    INC SwitchNameTablesReq     ; Switch the base nametable and reset scroll.
    LDA #$00
    STA CurVScroll

@CheckLine:
    LDA DemoScrolledLineCount       ; 7/8 of the lines,
    AND #$07
    BNE @Exit                   ; just return.
    JSR Demo_SpawnLineItems    ; But every 8 lines, we have to check for new text and objects.

    ; Append a request to transfer a line.
    ; It's blank by default. Change it later.
    LDX #$20
    LDA #$FF
    STA DynTileBuf+3, X
    DEX

@ClearLine:
    LDA #$24
    STA DynTileBuf+3, X
    DEX
    BPL @ClearLine
    LDA #$20
    STA DynTileBuf+2
    LDA DemoLineTileVramAddrHi
    STA DynTileBuf
    LDA DemoLineTileVramAddrLo
    STA DynTileBuf+1
    CLC                         ; The next line is 32 bytes farther.
    ADC #$20
    STA DemoLineTileVramAddrLo
    BNE @CheckNTEnd             ; If crossed a page,
    INC DemoLineTileVramAddrHi  ; then increment high address byte,
    JMP @CheckText              ; and do the next task.

@CheckNTEnd:
    ; Check if we reached the end of a nametable.
    ; If we did, then set the address to the top of the other one.
    ; $2BC0 -> $2000
    ; $23C0 -> $2800
    CMP #$C0
    BNE @CheckText
    LDA DemoLineTileVramAddrHi
    CMP #$2B
    BNE @CheckNT0
    LDA #$20
    STA DemoLineTileVramAddrHi
    JMP @ResetVramLo

@CheckNT0:
    CMP #$23
    BNE @CheckText
    LDA #$28
    STA DemoLineTileVramAddrHi

@ResetVramLo:
    LDA #$00
    STA DemoLineTileVramAddrLo

@CheckText:
    ; Check text and NT attributes.
    LDX DemoLineIndex
    LDA Demo_LineAttrs, X
    AND #$80
    BEQ @ProcessAttrs           ; If attribute $80 isn't set, then leave the line blank.
    LDA DemoLineTextIndex
    ASL
    TAX
    LDY #$00
    LDA Demo_LineTextAddrs, X    ; Get the address of the current text field.
    STA $00
    LDA Demo_LineTextAddrs+1, X
    STA $01
    LDA ($00), Y                ; The first byte of text field is the offset into the line.
    TAX

@CopyLine:
    ; X is an offset into the destination line.
    ; Y is an offset into the source tiles of the current record.
    ;
    ; Get the next source tile.
    INY
    LDA ($00), Y
    CMP #$FF
    BEQ @EndLine                ; When you reach the end marker, quit.
    STA DynTileBuf+3, X         ; Copy to the tile buf.
    INX
    JMP @CopyLine

@EndLine:
    INC DemoLineTextIndex

@ProcessAttrs:
    ; Finished processing attribute $80.
    JSR Demo_QueueLineAttrs
    INC DemoLineIndex           ; Advance to the next line.
    RTS

; ----------------------------------------------------------------------------
; If the current line has flag $40, appends a record that zeroes one row of 8 attribute bytes.
; IN:   DemoLineIndex; DynTileBuf already holds the 35-byte tile line record.
; ----------------------------------------------------------------------------
Demo_QueueLineAttrs:
    LDX DemoLineIndex
    LDA Demo_LineAttrs, X
    AND #$40
    BEQ @Exit                   ; If attribute $40 is missing, then return.

    ; Append a second transfer record.
    ; This one is for nametable attributes.
    LDA DemoLineAttrVramAddrHi
    STA DynTileBuf+35
    LDA DemoLineAttrVramAddrLo
    STA DynTileBuf+36
    LDA #$48                    ; 8 zeroes in VRAM take up 1 byte in record.
    STA DynTileBuf+37
    LDA #$00
    STA DynTileBuf+38
    LDA #$FF
    STA DynTileBuf+39
    INC $0416                   ; $0416 is written here only; no code in this bank reads it.
    LDA DemoLineAttrVramAddrLo  ; The next attributes go 8 bytes farther.
    CLC
    ADC #$08
    STA DemoLineAttrVramAddrLo

    ; Check if we reached the end of nametable attributes.
    ; If we did, then set the address to the top of the other one.
    ; $2C00 -> $23C0
    ; $2400 -> $2BC0
    BNE @Exit
    LDA DemoLineAttrVramAddrHi
    CMP #$23
    BNE @SetNT0
    LDA #$2B
    STA DemoLineAttrVramAddrHi
    JMP @SetVramLo

@SetNT0:
    LDA #$23
    STA DemoLineAttrVramAddrHi

@SetVramLo:
    LDA #$C0
    STA DemoLineAttrVramAddrLo

@Exit:
    RTS

; ----------------------------------------------------------------------------
; Frees (ObjState = $FF) every parade object in slots $01-$0A whose Y has reached $F0.
; ----------------------------------------------------------------------------
Demo_FreeScrolledOffObjects:
    ; For objects 1..10, indexed by X:
    LDX #$0A

@Loop:
    ; Once an object has fallen off the top of the screen,
    ; disable it.
    LDA ObjY, X

    ; The Y coordinate where we consider an object
    ; completely off the screen is $F0.
    CMP #$F0
    BNE @Next
    LDA #$FF                    ; Set the corresponding state to disabled.
    STA ObjState, X

@Next:
    DEX
    BNE @Loop
    RTS

; ----------------------------------------------------------------------------
; If the current line has flag $20, spawns the next item parade row at Y = $EF in free object slots.
; IN:   DemoLineIndex, DemoItemRow.
; OUT:  DemoItemIds, ObjX, ObjY, ObjState of the slots used; DemoItemRow incremented.
; ----------------------------------------------------------------------------
Demo_SpawnLineItems:
    LDY DemoLineIndex
    LDA Demo_LineAttrs, Y
    AND #$20                    ; If attribute $20 is present, then instantiate a new object.
    BNE @MakeObject
    RTS

@MakeObject:
    LDX #$0A                    ; Look for the first disabled object.

@FindDisabled:
    LDA ObjState, X
    BNE @SetUpObject
    DEX
    JMP @FindDisabled

@SetUpObject:
    LDY DemoItemRow
    LDA Demo_LeftItemIds, Y      ; Allocate an object for the item on the left.
    STA DemoItemIds, X
    LDA #$EF                    ; Start at the bottom of the screen.
    STA ObjY, X
    LDA Demo_LeftItemColumnX
    STA ObjX, X
    LDA #$00
    STA ObjState, X
    LDA DemoItemIds, X
    CMP #$30                    ; Link gets a special item ID.
    BCS @CenterLink             ; Go center the object horizontally, if it is Link.
    DEX                         ; Allocate another object for the item on the right.
    LDA Demo_RightItemIds, Y
    STA DemoItemIds, X
    LDA #$EF
    STA ObjY, X
    LDA Demo_RightItemColumnX
    STA ObjX, X
    LDA #$00
    STA ObjState, X
    LDA Demo_LeftItemIds, Y
    CMP #$1B
    BNE @IncRow

    ; Special case: The Triforce piece must be centered.
    ; But it shows up in both column lists.
    ; Two objects were instantiated, but they'll overlap.
    LDA #$78
    STA ObjX, X
    STA ObjX+1, X
    LDA #$00
    STA $0430                   ; $0430 is written here only; no code in this bank reads it.

@IncRow:
    INC DemoItemRow
    RTS

@CenterLink:
    LDA #$68
    STA ObjX, X
    JMP @IncRow

; ----------------------------------------------------------------------------
; Draws every shown parade object in slots $0A down to $01 according to its DemoItemIds value.
; ----------------------------------------------------------------------------
Demo_AnimateObjects:
    ; For each object 1..10, indexed by X:
    LDX #$0A

@LoopObject:
    LDA ObjState, X
    BNE @NextObject             ; If the object is disabled, then skip it.
    TXA                         ; Save the index.
    PHA

    ; Animate the fairy specially.
    LDA DemoItemIds, X
    CMP #$23
    BNE @AnimateNormal
    JSR Demo_DrawFairy
    JMP @PopAndNextObject

@AnimateNormal:
    ; Animate normal items (type < $30).
    CMP #$30
    BCS @AnimateFinal
    JSR Item_DrawObject
    JMP @PopAndNextObject

@AnimateFinal:
    ; At the end of the list of items are special items for Link
    ; and the sheet of paper.
    JSR Demo_DrawStoryFinalItem

@PopAndNextObject:
    PLA                         ; Restore the index.
    TAX

@NextObject:
    DEX
    BNE @LoopObject
    RTS

; ----------------------------------------------------------------------------
; Dead code (no reference to it): decrements bytes of Sprites+80 indexed by 4 or $C, then A-4 down.
; ----------------------------------------------------------------------------
Unused_DemoDecSpriteYs:
    LDX #$04
    CMP #$14
    BNE :+
    LDX #$0C
:
    DEC Sprites+80, X
    TAX
    SEC
    SBC #$04
    TAX
    BPL :-
    RTS


; ----------------------------------------------------------------------------
; Draws the parade fairy (item id $23) in place, switching between 2 frames every 4 frames.
; IN:   X = object slot.
; ----------------------------------------------------------------------------
Demo_DrawFairy:
    JSR Anim_FetchObjPosForSpriteDescriptor
    JSR Anim_SetSpriteDescriptorRedPaletteRow

    ; Rely on the fact that 2 represents normal sprite
    ; attributes with palette 6.
    ; Shift the value to make it 4.
    ASL
    AND FrameCounter            ; Every 4 frames, switch between 2 animation frames.
    LSR
    LSR
    STA $0C                     ; Put frame in [$0C].
    LDY #$14
    JMP Anim_WriteItemSprites

; ----------------------------------------------------------------------------
; Draws one row (up to 6 sprites, 8 pixels apart) of the closing picture for item ids $30-$33.
; IN:   X = object slot; DemoItemIds,X low nibble = row 0-3.
; NOTE: Writes sprite records starting at Sprites + row * 24; does not preserve X.
; ----------------------------------------------------------------------------
Demo_DrawStoryFinalItem:
    ; The demo story ending objects use special item types.
    ; The bottom nibble of the item type is an index into two
    ; tables. Each row has 6 bytes, one for each sprite of the
    ; item type.
    LDA DemoItemIds, X
    AND #$0F
    ASL
    STA $00
    ASL
    CLC
    ADC $00
    TAY                         ; Y gets offset of each row: (low_nibble & $0F) * 6
    LDA ObjY, X                 ; Put ObjY,ObjX in [$00, $01].
    STA $00
    LDA ObjX, X
    STA $01
    LDA #$05
    STA $02                     ; For 6 sprites, counted by [$02]:
    TYA
    ASL
    ASL
    TAX                         ; X gets the offset to the sprite (index * 4).

@LoopSprite:
    LDA Demo_StoryFinalSpriteTiles, Y    ; From this table, get the tile.
    BEQ @SkipSprite             ; If it's zero, skip the sprite.
    STA Sprites+1, X
    LDA $00                     ; Write sprite Y.
    STA Sprites, X              ; From this table, get the attributes.
    LDA Demo_StoryFinalSpriteAttrs, Y
    STA Sprites+2, X
    LDA $01                     ; Write sprite X.
    STA Sprites+3, X
    INX                         ; Point to the next sprite.
    INX
    INX
    INX

@SkipSprite:
    LDA $01                     ; Add 8 to X coordinate.
    CLC
    ADC #$08
    STA $01
    INY                         ; Increment the sprite index.
    DEC $02                     ; Decrement count.
    BPL @LoopSprite
    RTS

; ----------------------------------------------------------------------------
; Story subphase 3: keeps drawing the parade objects for 256 frames, then advances the subphase.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase1_Sub3:
    INC DemoTimer               ; Delay about 256 frames.
    LDA DemoTimer
    BNE Mode0Demo_Animate_Phase1_DrawObjects
    INC DemoSubphase
    RTS

; ----------------------------------------------------------------------------
; Story subphase 4: draws the objects for 56 more frames, then restarts the demo at phase 0.
; ----------------------------------------------------------------------------
Mode0Demo_Animate_Phase1_Sub4:
    INC DemoTimer               ; Ends on the 57th frame, when DemoTimer = $39.
    LDA DemoTimer
    CMP #$39
    BNE Mode0Demo_Animate_Phase1_DrawObjects
    LDA #$00                    ; Go to phase 0 again, and initialize it.
    STA IsUpdatingMode
    STA DemoTimer
    STA DemoPhase
    STA DemoSubphase
    RTS

; Shared tail of story subphases 3 and 4: hides all sprites, then draws the parade objects.
Mode0Demo_Animate_Phase1_DrawObjects:
    JSR Sprite_HideAll
    JSR Demo_AnimateObjects
    RTS
