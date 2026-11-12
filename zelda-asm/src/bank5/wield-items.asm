; ============================================================================
; FILE:     bank5/wield-items.asm        BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2866-3057
; SUBJECTS: (none)
; TOPIC:    Starting a weapon when A or B is pressed: the sword swing, and the B item dispatch with the
;           boomerang, arrow, food, potion and rod (bomb, candle and flute are started elsewhere).
; SEE ALSO: bank5/link-handle-input.asm (Link_HandleInput: when A and B reach these routines);
;           bank5/border-input-mask.asm (buttons discarded near screen edges);
;           bank1/bomb-candle-wield.asm (WieldBomb, WieldCandle, PlaceWeaponForPlayerState and variants);
;           bank7/flute.asm (WieldFlute); bank7/sword-and-rod.asm (UpdateSwordOrRod, UpdateRodOrArrow);
;           bank7/boomerang-arrow-food.asm (UpdateBoomerangOrFood, UpdateArrowOrBoomerang);
;           bank5/cave-cellar-enter.asm (World_FillHearts: the potion's refill).
; KEY FACTS:
;   - Link_HandleInput calls these only while Link's state (ObjState) is 0, and tests that once: A and B
;     newly pressed in the same frame start the sword and the B item together.
;   - Every weapon started here except the potion sets Link's state to $10 (PlaceWeaponForPlayerState),
;     which is what stops further A/B use until the state returns to 0.
;   - Sword: object slot $0D, state 1, timer 5. Refused if InvSword = 0 or the slot's state <> 0
;     (WieldSword). No cost.
;   - Boomerang: slot $0F, state $10, q-speed $C0 (3 px/frame), ObjMovingLimit $31 wooden or $FF magic
;     (BoomerangLimits; $FF is never reached, see the table). Refused if neither boomerang is owned or
;     a boomerang is already out; food lying in the slot (state bit 7) is replaced (WieldBoomerang).
;     Aims along the held directions, so it can fly diagonally.
;   - Arrow: slot $12, state $10, q-speed $C0. Refused if InvBow = 0, the slot is busy (arrow in flight
;     or rod swing), or InvRupees = 0. Costs 1 rupee (InvRupeesToSubtract). InvArrow is not tested here
;     (WieldArrow).
;   - Food: slot $0F, state $80, timer $FF. Refused if the slot's state <> 0 (boomerang out or food
;     already down). InvFood is neither tested nor consumed here (WieldFood).
;   - Potion: no object. Refused only if InvPotion = 0; takes 1 dose, starts the heart refill and pauses
;     the game (PauseState 2), even at full hearts (WieldPotion).
;   - Rod: slot $12, state $31, timer 5. Refused if the slot's state <> 0 (rod swing or arrow in flight).
;     No cost; InvRod is not tested here (WieldRod).
;   - Slot 3 (bow) does nothing; slot $0F (letter) returns before the dispatch (WieldItem).
; CONTENTS: (generated; L = line in this file)
;   L0059 WieldSword           code  Starts a sword swing in object slot $0D when A is pressed.
;   L0085 BoomerangLimits      data  TABLE: boomerang turn-back distance; index = InvMagicBoomerang (0 wooden, 1 magic); value = pixels,
;   L0095 WieldItem            code  Starts the item selected for the B button by jumping through WieldItem_JumpTable.
;   L0103 WieldItem_JumpTable  data  TABLE: B item handlers; index = SelectedItemSlot (0-8); value = routine address.
;   L0124 WieldBoomerang       code  Throws the boomerang from object slot $0F.
;   L0175 L14E71_Exit          code  Shared exit of WieldItem and WieldBoomerang.
;   L0187 WieldArrow           code  Shoots an arrow from object slot $12 for 1 rupee.
;   L0219 WieldWeapon          code  Activates the weapon in slot X in front of Link: used for the sword, the arrow and the rod.
;   L0237 WieldNothing         code  Handler of B item slot 3 (bow): does nothing. Also the shared exit of WieldArrow and WieldWeapon.
;   L0247 WieldFood            code  Puts the food down in object slot $0F, $10 pixels ahead of Link.
;   L0269 WieldPotion          code  Drinks one potion dose: starts the heart refill and pauses the game.
;   L0284 L14EC6_Exit          code  Shared exit of WieldFood, WieldPotion, WieldRod and CheckSubroom.
;   L0294 WieldRod             code  Starts a rod swing in object slot $12.
; ============================================================================

; ----------------------------------------------------------------------------
; Starts a sword swing in object slot $0D when A is pressed.
; IN:   called from Link_HandleInput with Link's state 0, the sword not blocked and A newly pressed.
; OUT:  slot $0D: state 1, ObjAnimCounter 5, q-speed $C0, placed $10 pixels ahead of Link;
;       Link's state = $10, his ObjAnimCounter = 1; EffectRequest bit $01 (sword noise).
; NOTE: Refused when InvSword = 0 or slot $0D's state <> 0. UpdateSwordOrRod decrements the counter
;       once a frame and steps the state 1..5 each time it reaches 0, freeing the slot on reaching 6:
;       the counter is loaded with 5 for state 1 (here), 8 for state 2, 1 each for 3, 4 and 5.
; ----------------------------------------------------------------------------
WieldSword:
    ; If there's no sword, return.
    LDA InvSword
    BEQ L14DFF_Exit

    ; Switch to the sword slot.
    LDX #$0D

    ; If its state <> 0, a swing is in progress. Return.
    LDA ObjState, X
    BNE L14DFF_Exit

    ; Timer of the first state: 5.
    LDA #$05
    STA ObjAnimCounter, X

    ; The initial state is 1.
    LDA #$01
    JSR WieldWeapon
    LDA #$01                    ; EffectRequest bit $01: sword noise.
    JMP PlayEffect

; TABLE: boomerang turn-back distance; index = InvMagicBoomerang (0 wooden, 1 magic); value = pixels,
;   stored in ObjMovingLimit and compared by UpdateArrowOrBoomerang with the absolute value of the
;   boomerang's signed ObjGridOffset. $31 is reached after 17 frames at 3 px/frame. $FF cannot be
;   reached (the absolute value is at most $80), so the magic boomerang flies until it is blocked.
BoomerangLimits:
    .BYTE $31, $FF

; ----------------------------------------------------------------------------
; Starts the item selected for the B button by jumping through WieldItem_JumpTable.
; IN:   SelectedItemSlot; called from Link_HandleInput with Link's state 0 and B newly pressed.
; NOTE: Slot $0F (letter) does nothing here. Of the handlers in this file only the boomerang (either
;       kind owned), the arrow (bow, rupees) and the potion (doses) test the inventory; the food and
;       rod handlers do not. The table has 9 entries and the index is not range-checked.
; ----------------------------------------------------------------------------
WieldItem:
    ; If the letter slot is selected, return.
    LDA SelectedItemSlot
    CMP #$0F
    BEQ L14E71_Exit
    JSR TableJump

; TABLE: B item handlers; index = SelectedItemSlot (0-8); value = routine address.
WieldItem_JumpTable:
    .ADDR WieldBoomerang        ; 0 boomerang (wooden or magic)
    .ADDR WieldBomb             ; 1 bomb
    .ADDR WieldArrow            ; 2 arrow
    .ADDR WieldNothing          ; 3 bow: nothing
    .ADDR WieldCandle           ; 4 candle
    .ADDR WieldFlute            ; 5 flute
    .ADDR WieldFood             ; 6 food
    .ADDR WieldPotion           ; 7 potion
    .ADDR WieldRod              ; 8 rod

; ----------------------------------------------------------------------------
; Throws the boomerang from object slot $0F.
; OUT:  slot $0F: state $10, ObjMovingLimit = $31 (wooden) or $FF (magic; never reached), q-speed $C0
;       (3 px/frame), ObjAnimCounter 3, placed $10 pixels ahead of Link; Link's state = $10, his
;       ObjAnimCounter = 1.
; NOTE: Refused when neither boomerang is owned, or slot $0F holds a boomerang (state <> 0 with bit 7
;       clear). Food in the slot (state $8x) does not refuse the throw: the boomerang replaces it.
;       Direction = ObjInputDir if any direction is held (two bits when diagonal), else Link's facing.
;       No cost.
; ----------------------------------------------------------------------------
WieldBoomerang:
    ; If missing wooden boomerang and magic boomerang, return.
    LDA InvBoomerang
    ORA InvMagicBoomerang
    BEQ L14E71_Exit

    ; Switch to the boomerang slot.
    LDX #$0F

    ; If the slot's state <> 0 and its high bit is clear (a boomerang is out), return.
    ; State 0 (empty) and states with the high bit set (food) go on.
    LDA ObjState, X
    BEQ :+
    ASL
    BCC L14E71_Exit
:
    ; Set state to $10 for boomerang.
    LDA #$10
    STA ObjState, X

    ; Set the farthest distance the boomerang can fly based on
    ; the type of boomerang.
    LDY InvMagicBoomerang
    LDA BoomerangLimits, Y
    STA ObjMovingLimit, X

    ; Set up the boomerang: Link's state $10, placed $10 pixels ahead of him,
    ; q-speed $C0 (3 pixels a frame).
    ;
    ; The first turning animation frame gets a counter of 3.
    JSR PlaceWeaponForPlayerState
    LDA #$C0
    STA ObjQSpeedFrac, X
    LDA #$03
    STA ObjAnimCounter, X

    ; See PlaceWeaponForPlayerStateAndAnim for the reason
    ; that Link's animation counter is set to 1.
    LDA #$01
    STA ObjAnimCounter

    ; If there's an input direction, then use it as the weapon's direction.
    ; It might be diagonal, with horizontal and vertical components.
    ; Else use Link's facing direction.
    LDA ObjInputDir
    BNE :+
    LDA ObjDir
:
    STA ObjDir, X

; Shared exit of WieldItem and WieldBoomerang.
L14E71_Exit:
    RTS

; ----------------------------------------------------------------------------
; Shoots an arrow from object slot $12 for 1 rupee.
; OUT:  slot $12: state $10, then as WieldWeapon; InvRupeesToSubtract + 1; EffectRequest bit $02.
; NOTE: Refused when InvBow = 0, when slot $12's state <> 0 with bit 7 clear (an arrow in flight, or
;       a rod swing: the rod shares the slot), or when InvRupees = 0. InvArrow is not tested here.
;       No state with bit 7 set is written to slot $12: it holds 0, an arrow ($10 flying, $20/$28
;       spark, $40 only inside the call that frees it) or a rod ($31-$35). So the bit 7 exemption,
;       the same code shape as the boomerang/food test of slot $0F, never applies here.
; ----------------------------------------------------------------------------
WieldArrow:
    ; If there's no bow, return.
    LDA InvBow
    BEQ WieldNothing

    ; Switch to the arrow slot.
    LDX #$12

    ; If state <> 0 and high bit is clear, return.
    LDA ObjState, X
    BEQ :+
    ASL
    BCC WieldNothing
:
    ; If there are no rupees, return.
    LDA InvRupees
    BEQ WieldNothing
    LDA #$02                    ; EffectRequest bit $02: boomerang/arrow noise.
    JSR PlayEffect

    ; Post a rupee to subtract.
    INC InvRupeesToSubtract

    ; Arrows start in state $10.
    LDA #$10

; ----------------------------------------------------------------------------
; Activates the weapon in slot X in front of Link: used for the sword, the arrow and the rod.
; IN:   A = initial state; X = weapon object slot.
; OUT:  ObjState,X = A; q-speed $C0 (3 px/frame); weapon faces as Link and sits $10 pixels ahead of
;       him, moved 3 pixels right when he faces up or down; Link's state = $10, his ObjAnimCounter = 1.
; ----------------------------------------------------------------------------
WieldWeapon:
    STA ObjState, X

    ; Set q-speed $C0 (3 pixels a frame).
    LDA #$C0
    STA ObjQSpeedFrac, X
    JSR PlaceWeaponForPlayerStateAndAnim

    ; If the direction is vertical, move the object right 3 pixels.
    LDA ObjDir, X
    AND #$0C
    BEQ WieldNothing
    LDA ObjX, X
    CLC
    ADC #$03
    STA ObjX, X

; Handler of B item slot 3 (bow): does nothing. Also the shared exit of WieldArrow and WieldWeapon.
WieldNothing:
    RTS

; ----------------------------------------------------------------------------
; Puts the food down in object slot $0F, $10 pixels ahead of Link.
; OUT:  slot $0F: state $80, ObjTimer $FF; Link's state = $10, his ObjAnimCounter = 1.
; NOTE: Refused when slot $0F's state <> 0: a boomerang is out or food is already down. InvFood is
;       not tested and not consumed. UpdateBoomerangOrFood keeps the food through states $80-$82 with
;       a $FF-frame timer each and removes it on reaching $83.
; ----------------------------------------------------------------------------
WieldFood:
    ; Switch to the food slot.
    LDX #$0F

    ; If there's an item already active in the slot (state <> 0), return.
    LDA ObjState, X
    BNE L14EC6_Exit

    ; The timer of the first state of food: $FF frames.
    LDA #$FF
    STA ObjTimer, X

    ; Set up the food in state $80.
    LDA #$80
    JMP PlaceWeaponForPlayerStateAndAnimAndWeaponState

; ----------------------------------------------------------------------------
; Drinks one potion dose: starts the heart refill and pauses the game.
; OUT:  InvPotion - 1; World_IsFillingHearts = 1; PauseState = 2.
; NOTE: Refused only when InvPotion = 0; current hearts are not tested, so a dose is spent even at
;       full hearts. Uses no object slot and leaves Link's state at 0.
; ----------------------------------------------------------------------------
WieldPotion:
    ; If there's no potion dose left, return.
    LDA InvPotion
    BEQ L14EC6_Exit

    ; Use one dose (2 red -> 1 blue -> 0).
    DEC InvPotion

    ; Flag that we're filling hearts and involuntarily paused.
    LDA #$01
    STA World_IsFillingHearts
    LDA #$02
    STA PauseState

; Shared exit of WieldFood, WieldPotion, WieldRod and CheckSubroom.
L14EC6_Exit:
    RTS

; ----------------------------------------------------------------------------
; Starts a rod swing in object slot $12.
; OUT:  slot $12: state $31, ObjAnimCounter 5, then as WieldWeapon.
; NOTE: Refused when slot $12's state <> 0 (a rod swing, or an arrow in flight: they share the slot).
;       InvRod is not tested here. No cost. UpdateRodOrArrow passes states $30 and up to
;       UpdateSwordOrRod, which steps the low nibble 1..5 with the sword's timing.
; ----------------------------------------------------------------------------
WieldRod:
    ; Switch to the rod slot.
    ; If it's already in use, then return.
    LDX #$12
    LDA ObjState, X
    BNE L14EC6_Exit

    ; Timer of the first state: 5.
    LDA #$05
    STA ObjAnimCounter, X

    ; The initial state is $31.
    LDA #$31
    JMP WieldWeapon
