; ============================================================================
; FILE:     bank7/status-bar-lift-item.asm BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 974-1128
; SUBJECTS: (none)
; TOPIC:    Status bar B item and sword sprites, with the repair of SelectedItemSlot when the selected item
;           is gone; Link holding an item above his head (halt, timer, drawing).
; SEE ALSO: bank7/room-item-draw.asm (DrawItemInInventory, AnimateItemObject); bank5/submenu-items.asm
;           (FindAndSelectOccupiedItemSlot, DrawItemInInventoryWithX); bank1/take-item.asm (TakeItem starts
;           a lift); bank5/mode12-lay-out-room.asm (Triforce piece lift in mode $12);
;           bank7/mode5-play-update.asm (calls CheckLiftItem and the status bar routine every frame).
; KEY FACTS:
;   - SelectedItemSlot 0 means "boomerang": the magic boomerang (slot $1E) is shown if owned, else the
;     wooden one (slot $1D) (DrawStatusBarBoomerang).
;   - If the selected slot is empty (for example the last bomb was used), the selection moves to the
;     nearest occupied selectable slot searching downward from it, and the B item is not drawn that
;     frame (CheckMissingItem, EnsureSelectedItem).
;   - Potion (slot 7) and letter (slot $F) share one selection: an empty potion slot falls back to the
;     letter if Link has it, and a selected letter switches to the potion as soon as InvPotion is
;     nonzero (CheckMissingItem, DrawStatusBarItemsAndEnsureItemSelected, DrawStatusBarPotion).
;   - Status bar sprite positions: B item at X $7C, Y $1F; sword at X $94, Y $1F (DrawStatusBarItemB,
;     DrawStatusBarSword).
;   - While ItemTypeToLift is nonzero, CheckLiftItem sets Link's state to $40 (halted) every frame and
;     counts ItemLiftTimer down; TakeItem starts it at $80 when an item is taken outside mode 5 (the
;     play update that calls CheckLiftItem also serves caves and cellars), so Link is frozen for $7F
;     frames and released on the $80th (CheckLiftItem, EndLinkLiftingItem).
;   - Item id 0 (bomb) cannot be lifted: ItemTypeToLift = 0 means "none" (CheckLiftItem).
;   - The lifted item is drawn with object slot $13 (the room item's slot), placed at Link's X and
;     $10 pixels above his Y (SetUpAndDrawLinkLiftingItem).
;   - When the lift ends in a dungeon the level's song is requested again; in the OW (caves) no song
;     is requested here (EndLinkLiftingItem).
; CONTENTS: (generated; L = line in this file)
;   L0051 DrawStatusBarPotion                      code  Switches the selection from the letter to the potion (item slot 7) and draws the potion as B item.
;   L0064 DrawStatusBarItemsAndEnsureItemSelected  code  Draws the B item and the sword in the status bar, first repairing SelectedItemSlot if needed.
;   L0081 DrawStatusBarItemB                       code  Draws the item of item slot X at the B position of the status bar, then goes on to the sword.
;   L0095 DrawStatusBarBoomerang                   code  Selection 0: draws the magic boomerang if owned, else the wooden one; with neither, reselects.
;   L0108 FindItemOrDrawSword                      code  Draws only the sword if slot X holds an item (the B item follows next frame); else falls into the search.
;   L0119 EnsureSelectedItem                       code  Sets SelectedItemSlot to an occupied selectable item slot, searching backward from slot X.
;   L0130 DrawStatusBarSword                       code  Draws Link's sword at the A position of the status bar, if he has one.
;   L0146 CheckMissingItem                         code  Handles an empty selected slot: the potion slot falls back to the letter, any other slot is reselected.
;   L0162 CheckLiftItem                            code  While Link holds up an item: keeps him halted, counts the lift timer down and draws the pose.
;   L0176 SetUpAndDrawLinkLiftingItem              code  Places the lifted item (object slot $13) at Link's X, $10 pixels above him, then draws both.
;   L0192 DrawLinkLiftingItem                      code  Draws Link in the item-lifting pose and the lifted item at the position of object slot $13.
;   L0217 L1E847_Exit                              code  Shared RTS of DrawLinkLiftingItem and DrawStatusBarSword.
;   L0224 EndLinkLiftingItem                       code  Ends the lift: Link's state back to 0 (idle), ItemTypeToLift cleared, dungeon song restarted.
;   L0234 L1E859_Exit                              code  Shared RTS of EndLinkLiftingItem and CheckLiftItem.
; ============================================================================

; ----------------------------------------------------------------------------
; Switches the selection from the letter to the potion (item slot 7) and draws the potion as B item.
; ----------------------------------------------------------------------------
DrawStatusBarPotion:
    LDX #$07                    ; Potion item slot.
    STX SelectedItemSlot
    BNE DrawStatusBarItemB      ; Always taken.

; ----------------------------------------------------------------------------
; Draws the B item and the sword in the status bar, first repairing SelectedItemSlot if needed.
; NOTE: SelectedItemSlot 0 is the boomerang pseudo slot (InvItems+0 is the sword). A selection whose
;       slot is empty is replaced: potion slot 7 by the letter slot $F if Link has the letter, anything
;       else by a search for an occupied slot (FindAndSelectOccupiedItemSlot, backward). On a frame
;       where the selection changes this way, only the sword is drawn.
; NOTE: Leaves bank 5 switched in whenever something is drawn or searched.
; ----------------------------------------------------------------------------
DrawStatusBarItemsAndEnsureItemSelected:
    LDX SelectedItemSlot
    BEQ DrawStatusBarBoomerang  ; Pseudo slot 0: boomerangs.
    LDA InvItems, X
    BEQ CheckMissingItem        ; Selected slot is empty.
    CPX #$0F                    ; Letter slot?
    BNE DrawStatusBarItemB      ; No: draw the item of slot X.
    LDY InvPotion               ; Letter selected: if Link now has a potion,
    BNE DrawStatusBarPotion     ; select and draw the potion instead.
    LSR                         ; Letter value 1 or 2 -> 0 or 1,
    ORA #$01                    ; -> 1. (Overwritten: DrawItemInInventory reloads [04].)

; ----------------------------------------------------------------------------
; Draws the item of item slot X at the B position of the status bar, then goes on to the sword.
; IN:   X = item slot (1-8, $F, or $1D/$1E for a boomerang).
; NOTE: The [04] written here is replaced by InvItems,X in DrawItemInInventory.
; ----------------------------------------------------------------------------
DrawStatusBarItemB:
    STA $04
    LDA #$1F                    ; B item position: X $7C, Y $1F.
    STA $01
    LDA #$7C
    STA $00
    LDA #$05
    JSR SwitchBank
    JSR DrawItemInInventoryWithX
    JMP DrawStatusBarSword

; ----------------------------------------------------------------------------
; Selection 0: draws the magic boomerang if owned, else the wooden one; with neither, reselects.
; ----------------------------------------------------------------------------
DrawStatusBarBoomerang:
    LDX #$1E                    ; Magic boomerang slot first, then $1D (wooden).

@LoopBoomerang:
    LDA InvItems, X
    BNE DrawStatusBarItemB
    DEX
    CPX #$1C
    BNE @LoopBoomerang
    LDX #$00
    JMP EnsureSelectedItem      ; No boomerang: search for an occupied slot, starting from slot 0.

; Draws only the sword if slot X holds an item (the B item follows next frame); else falls into the search.
FindItemOrDrawSword:
    LDA InvItems, X
    BNE DrawStatusBarSword

; ----------------------------------------------------------------------------
; Sets SelectedItemSlot to an occupied selectable item slot, searching backward from slot X.
; IN:   X = slot to start from.
; NOTE: The search steps downward through slots 8..0 (wrapping), skips the bow slot 3, treats slot 0
;       as the boomerangs, slot 7 as potion-or-letter, accepts arrows only with the bow, and selects
;       0 when nothing is found (FindAndSelectOccupiedItemSlot with A = 2).
; ----------------------------------------------------------------------------
EnsureSelectedItem:
    TXA
    TAY
    LDA #$05
    JSR SwitchBank
    LDA #$02                    ; Search direction: backward.
    JSR FindAndSelectOccupiedItemSlot

; ----------------------------------------------------------------------------
; Draws Link's sword at the A position of the status bar, if he has one.
; ----------------------------------------------------------------------------
DrawStatusBarSword:
    LDX #$00                    ; Item slot 0 = sword.
    LDA InvItems, X
    BEQ L1E847_Exit             ; No sword: return.
    LDA #$1F                    ; Sword position: X $94, Y $1F.
    STA $01
    LDA #$94
    STA $00
    LDA #$05
    JSR SwitchBank
    JMP DrawItemInInventoryWithX

; ----------------------------------------------------------------------------
; Handles an empty selected slot: the potion slot falls back to the letter, any other slot is reselected.
; IN:   X = SelectedItemSlot (nonzero), whose inventory value is 0.
; ----------------------------------------------------------------------------
CheckMissingItem:
    CPX #$07
    BNE FindItemOrDrawSword     ; Not the potion slot: reselect (InvItems,X is 0 there).
    LDA InvLetter               ; Potion slot without potion:
    BEQ EnsureSelectedItem      ; no letter either: reselect.
    LDX #$0F                    ; Letter: select its slot;
    STX SelectedItemSlot
    BNE FindItemOrDrawSword     ; always taken; draws the sword only this frame.

; ----------------------------------------------------------------------------
; While Link holds up an item: keeps him halted, counts the lift timer down and draws the pose.
; NOTE: Does nothing when ItemTypeToLift = 0 (so item id 0, the bomb, is never lifted). Each call
;       decrements ItemLiftTimer first; at 0 the lift ends (EndLinkLiftingItem), otherwise Link's
;       state is forced to $40 (halted: no input, no movement). With the $80 set by TakeItem that is
;       $7F halted frames.
; ----------------------------------------------------------------------------
CheckLiftItem:
    LDA ItemTypeToLift
    BEQ L1E859_Exit
    DEC ItemLiftTimer
    BEQ EndLinkLiftingItem
    LDA #$40                    ; Link's state: halted.
    STA ObjState

; ----------------------------------------------------------------------------
; Places the lifted item (object slot $13) at Link's X, $10 pixels above him, then draws both.
; IN:   ItemTypeToLift = item id.
; NOTE: Overwrites ObjX+$13/ObjY+$13, the room item's position. Also called by the mode $12 init with
;       item id $1B (Triforce piece).
; ----------------------------------------------------------------------------
SetUpAndDrawLinkLiftingItem:
    LDA ObjX
    STA ObjX+19
    LDA ObjY
    SEC
    SBC #$10
    STA ObjY+19

; ----------------------------------------------------------------------------
; Draws Link in the item-lifting pose and the lifted item at the position of object slot $13.
; IN:   ItemTypeToLift = item id; ObjX+$13/ObjY+$13 = item position.
; NOTE: Link is drawn with the image of pseudo item slot $21 in his own sprite records (offsets $48,
;       $4C), attributes 0. A half-width item is drawn at the left (not centered) and Link's right half
;       then uses tile $08, a tile Link's facing-down walk frame also uses (what it shows here:
;       unverified).
; ----------------------------------------------------------------------------
DrawLinkLiftingItem:
    LDX #$00                    ; Link's object slot.
    JSR Anim_FetchObjPosForSpriteDescriptor      ; Returns A = 0.
    JSR Anim_SetSpriteDescriptorAttributes       ; Sprite attributes 0 for both halves.
    STA $0C                     ; Frame 0.
    LDA #$48                    ; Link's two sprite records.
    STA LeftSpriteOffset
    LDA #$4C
    STA RightSpriteOffset
    LDY #$21                    ; Pseudo item slot $21: Link's lifting image.
    JSR Anim_WriteSpecificItemSprites
    INC LeftAlignHalfWidthObj   ; While the item is drawn: do not center a half-width item.
    LDA ItemTypeToLift
    LDX #$13                    ; Draw the item at the room item slot's position.
    JSR AnimateItemObject
    DEC LeftAlignHalfWidthObj
    LDA IsDrawnItemNarrow
    BEQ L1E847_Exit             ; Full-width item: done.

    ; Half-width item: change the tile of Link's right sprite record to $08 (said upstream to be a
    ; tile without the raised arm; the image itself is unverified).
    LDA #$08
    STA Sprites+77

; Shared RTS of DrawLinkLiftingItem and DrawStatusBarSword.
L1E847_Exit:
    RTS

; ----------------------------------------------------------------------------
; Ends the lift: Link's state back to 0 (idle), ItemTypeToLift cleared, dungeon song restarted.
; NOTE: In a dungeon (CurLevel not 0) the level's song is requested again; in the OW nothing is requested.
; ----------------------------------------------------------------------------
EndLinkLiftingItem:
    LDA #$00
    STA ObjState
    STA ItemTypeToLift
    LDY CurLevel
    BEQ L1E859_Exit             ; OW: return.
    LDA LevelSongIds, Y
    STA SongRequest

; Shared RTS of EndLinkLiftingItem and CheckLiftItem.
L1E859_Exit:
    RTS
