; ============================================================================
; FILE:     bank2/demo-story-scroll.asm  BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 540-1001
; SUBJECTS: (none)
; TOPIC:    Title wait, story scroll, item parade
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0025 AnimateDemoPhase0Subphase0           code
;   L0039 AnimateDemoPhase1Subphase0           code
;   L0070 AnimateDemoPhase1Subphase1           code
;   L0086 AnimateDemoPhase1Subphase2           code
;   L0218 ProcessDemoLineAttrs                 code
;   L0265 DisableFallenObjects                 code
;   L0286 ProcessDemoLineItems                 code
;   L0346 Demo_AnimateObjects                  code
;   L0384 Unused_97C5                          code
;   L0400 AnimateStationaryFairy               code
;   L0415 AnimateDemoStoryFinalItems           code
;   L0464 AnimateDemoPhase1Subphase3           code
;   L0471 AnimateDemoPhase1Subphase4           code
;   L0483 AnimateDemoPhase1End_AnimateObjects  code
; ============================================================================

AnimateDemoPhase0Subphase0:
    ; Animate while we wait about 512 frames.
    LDA FrameCounter
    AND #$01                    ; The timer is incremented every other frame.
    BEQ @SkipTimer
    INC DemoTimer
    LDA DemoTimer
    BNE @SkipTimer
    JMP IncSubphase             ; Go advance the demo subphase and return.

@SkipTimer:
    JSR AnimateDemoPhase0Subphase0Artifacts
    RTS

AnimateDemoPhase1Subphase0:
    LDA FrameCounter            ; Increase CurVScroll every odd frame.
    AND #$01
    BEQ @CheckVScroll
    INC CurVScroll
    LDA CurVScroll              ; Have we scrolled to the bottom of nametable 2?
    CMP #$F0
    BNE @CheckVScroll
    INC ScrolledScreenCount

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
    LDA ScrolledScreenCount
    BEQ @Exit
    LDA #$00
    STA ScrolledScreenCount
    INC DemoSubphase

@Exit:
    RTS

AnimateDemoPhase1Subphase1:
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

AnimateDemoPhase1Subphase2:
    JSR HideAllSprites
    JSR DisableFallenObjects
    JSR Demo_AnimateObjects
    LDA FrameCounter
    AND #$01
    BEQ @Exit                   ; Every even frame, just return.
    LDX #$0A                    ; Decrement the Y coordinate of every object.

@DecObjYs:
    DEC ObjY, X
    DEX
    BNE @DecObjYs
    INC ScrolledLineCount       ; We scrolled one more line.
    LDA ScrolledLineCount
    BNE :+                      ; Once we've scrolled a whole screen,
    INC ScrolledScreenCount     ; Increment the screen count.
:
    LDA ScrolledScreenCount     ; Have we scrolled 5 screens?
    CMP #$05
    BNE @Scroll                 ; Scroll the nametable.
    LDA ScrolledLineCount       ; Scroll half a screen more.
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
    LDA ScrolledLineCount       ; 7/8 of the lines,
    AND #$07
    BNE @Exit                   ; just return.
    JSR ProcessDemoLineItems    ; But every 8 lines, we have to check for new text and objects.

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
    LDA DemoLineAttrs, X
    AND #$80
    BEQ @ProcessAttrs           ; If attribute $80 isn't set, then leave the line blank.
    LDA DemoLineTextIndex
    ASL
    TAX
    LDY #$00
    LDA DemoLineTextAddrs, X    ; Get the address of the current text field.
    STA $00
    LDA DemoLineTextAddrs+1, X
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
    JSR ProcessDemoLineAttrs
    INC DemoLineIndex           ; Advance to the next line.
    RTS

ProcessDemoLineAttrs:
    LDX DemoLineIndex
    LDA DemoLineAttrs, X
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
    INC $0416                   ; UNKNOWN: It doesn't seem to be used.
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

DisableFallenObjects:
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

ProcessDemoLineItems:
    LDY DemoLineIndex
    LDA DemoLineAttrs, Y
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
    LDA DemoLeftItemIds, Y      ; Allocate an object for the item on the left.
    STA DemoItemIds, X
    LDA #$EF                    ; Start at the bottom of the screen.
    STA ObjY, X
    LDA DemoItemColumnX1
    STA ObjX, X
    LDA #$00
    STA ObjState, X
    LDA DemoItemIds, X
    CMP #$30                    ; Link gets a special item ID.
    BCS @CenterLink             ; Go center the object horizontally, if it is Link.
    DEX                         ; Allocate another object for the item on the right.
    LDA DemoRightItemIds, Y
    STA DemoItemIds, X
    LDA #$EF
    STA ObjY, X
    LDA DemoItemColumnX2
    STA ObjX, X
    LDA #$00
    STA ObjState, X
    LDA DemoLeftItemIds, Y
    CMP #$1B
    BNE @IncRow

    ; Special case: The triforce must be centered.
    ; But it shows up in both column lists.
    ; Two objects were instantiated, but they'll overlap.
    LDA #$78
    STA ObjX, X
    STA ObjX+1, X
    LDA #$00
    STA $0430                   ; UNKNOWN: It doesn't seem to be used.

@IncRow:
    INC DemoItemRow
    RTS

@CenterLink:
    LDA #$68
    STA ObjX, X
    JMP @IncRow

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
    JSR AnimateStationaryFairy
    JMP @PopAndNextObject

@AnimateNormal:
    ; Animate normal items (type < $30).
    CMP #$30
    BCS @AnimateFinal
    JSR AnimateItemObject
    JMP @PopAndNextObject

@AnimateFinal:
    ; At the end of the list of items are special items for Link
    ; and the sheet of paper.
    JSR AnimateDemoStoryFinalItems

@PopAndNextObject:
    PLA                         ; Restore the index.
    TAX

@NextObject:
    DEX
    BNE @LoopObject
    RTS

Unused_97C5:
    ; Begin unverified code 97C5
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
    ; End unverified code

AnimateStationaryFairy:
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

AnimateDemoStoryFinalItems:
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
    LDA DemoStoryFinalSpriteTiles, Y    ; From this table, get the tile.
    BEQ @SkipSprite             ; If it's zero, skip the sprite.
    STA Sprites+1, X
    LDA $00                     ; Write sprite Y.
    STA Sprites, X              ; From this table, get the attributes.
    LDA DemoStoryFinalSpriteAttrs, Y
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

AnimateDemoPhase1Subphase3:
    INC DemoTimer               ; Delay about 256 frames.
    LDA DemoTimer
    BNE AnimateDemoPhase1End_AnimateObjects
    INC DemoSubphase
    RTS

AnimateDemoPhase1Subphase4:
    INC DemoTimer               ; Delay 56 frames.
    LDA DemoTimer
    CMP #$39
    BNE AnimateDemoPhase1End_AnimateObjects
    LDA #$00                    ; Go to phase 0 again, and initialize it.
    STA IsUpdatingMode
    STA DemoTimer
    STA DemoPhase
    STA DemoSubphase
    RTS

AnimateDemoPhase1End_AnimateObjects:
    JSR HideAllSprites
    JSR Demo_AnimateObjects
    RTS
