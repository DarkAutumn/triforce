; ============================================================================
; FILE:     bank5/wield-items.asm        BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2866-3057
; SUBJECTS: Arrow, Boomerang, Food, Link, Potion, Rod, Sword
; TOPIC:    Starting a weapon when A or B is pressed: the sword swing, and the B item dispatch with the
;           boomerang, arrow, food, potion and rod (bomb, candle and flute are started elsewhere).
; SEE ALSO: bank5/link-handle-input.asm (Link_HandleInput: when A and B reach these routines);
;           bank5/border-input-mask.asm (buttons discarded near screen edges);
;           bank1/bomb-candle-wield.asm (Bomb_Wield, Candle_Wield, Link_UseItem_SetState10 and variants);
;           bank7/flute.asm (Flute_Wield); bank7/sword-and-rod.asm (Sword_UpdateSwordOrRod, Rod_UpdateRodOrArrow);
;           bank7/boomerang-arrow-food.asm (Boomerang_UpdateOrFood, Boomerang_UpdateOrArrow);
;           bank5/cave-cellar-enter.asm (World_FillHearts: the potion's refill).
; KEY FACTS:
;   - Link_HandleInput calls these only while Link's state (ObjState) is 0, and tests that once: A and B
;     newly pressed in the same frame start the sword and the B item together.
;   - Every weapon started here except the potion sets Link's state to $10 (Link_UseItem_SetState10),
;     which is what stops further A/B use until the state returns to 0.
;   - Sword: object slot $0D, state 1, timer 5. Refused if InvSword = 0 or the slot's state <> 0
;     (Sword_Wield). No cost.
;   - Boomerang: slot $0F, state $10, q-speed $C0 (3 px/frame), ObjMovingLimit $31 wooden or $FF magic
;     (Boomerang_RangeLimits; $FF is never reached, see the table). Refused if neither boomerang is owned or
;     a boomerang is already out; food lying in the slot (state bit 7) is replaced (Boomerang_Wield).
;     Aims along the held directions, so it can fly diagonally.
;   - Arrow: slot $12, state $10, q-speed $C0. Refused if InvBow = 0, the slot is busy (arrow in flight
;     or rod swing), or InvRupees = 0. Costs 1 rupee (InvRupeesToSubtract). InvArrow is not tested here
;     (Arrow_Wield).
;   - Food: slot $0F, state $80, timer $FF. Refused if the slot's state <> 0 (boomerang out or food
;     already down). InvFood is neither tested nor consumed here (Food_Wield).
;   - Potion: no object. Refused only if InvPotion = 0; takes 1 dose, starts the heart refill and pauses
;     the game (PauseState 2), even at full hearts (Potion_Wield).
;   - Rod: slot $12, state $31, timer 5. Refused if the slot's state <> 0 (rod swing or arrow in flight).
;     No cost; InvRod is not tested here (Rod_Wield).
;   - Slot 3 (bow) does nothing; slot $0F (letter) returns before the dispatch (Link_WieldItem).
; CONTENTS: (generated; L = line in this file)
;   L0059 Sword_Wield               code  Starts a sword swing in object slot $0D when A is pressed.
;   L0085 Boomerang_RangeLimits     data  TABLE: boomerang turn-back distance; index = InvMagicBoomerang (0 wooden, 1 magic); value = pixels,
;   L0095 Link_WieldItem            code  Starts the item selected for the B button by jumping through Link_WieldItem_JumpTable.
;   L0103 Link_WieldItem_JumpTable  data  TABLE: B item handlers; index = SelectedItemSlot (0-8); value = routine address.
;   L0124 Boomerang_Wield           code  Throws the boomerang from object slot $0F.
;   L0175 Boomerang_Wield_Return    code  Shared exit of Link_WieldItem and Boomerang_Wield.
;   L0187 Arrow_Wield               code  Shoots an arrow from object slot $12 for 1 rupee.
;   L0219 Link_WieldWeapon          code  Activates the weapon in slot X in front of Link: used for the sword, the arrow and the rod.
;   L0237 Link_WieldNothing         code  Handler of B item slot 3 (bow): does nothing. Also the shared exit of Arrow_Wield and Link_WieldWeapon.
;   L0247 Food_Wield                code  Puts the food down in object slot $0F, $10 pixels ahead of Link.
;   L0269 Potion_Wield              code  Drinks one potion dose: starts the heart refill and pauses the game.
;   L0284 Potion_Wield_Return       code  Shared exit of Food_Wield, Potion_Wield, Rod_Wield and Warp_CheckSubroomExit.
;   L0294 Rod_Wield                 code  Starts a rod swing in object slot $12.
; ============================================================================

; ----------------------------------------------------------------------------
; Starts a sword swing in object slot $0D when A is pressed.
; IN:   called from Link_HandleInput with Link's state 0, the sword not blocked and A newly pressed.
; OUT:  slot $0D: state 1, ObjAnimCounter 5, q-speed $C0, placed $10 pixels ahead of Link;
;       Link's state = $10, his ObjAnimCounter = 1; EffectRequest bit $01 (sword noise).
; NOTE: Refused when InvSword = 0 or slot $0D's state <> 0. Sword_UpdateSwordOrRod decrements the counter
;       once a frame and steps the state 1..5 each time it reaches 0, freeing the slot on reaching 6:
;       the counter is loaded with 5 for state 1 (here), 8 for state 2, 1 each for 3, 4 and 5.
; ----------------------------------------------------------------------------
Sword_Wield:
    ; If there's no sword, return.
    LDA InvSword
    BEQ Link_MaskInputInBorder_Return

    ; Switch to the sword slot.
    LDX #$0D

    ; If its state <> 0, a swing is in progress. Return.
    LDA ObjState, X
    BNE Link_MaskInputInBorder_Return

    ; Timer of the first state: 5.
    LDA #$05
    STA ObjAnimCounter, X

    ; The initial state is 1.
    LDA #$01
    JSR Link_WieldWeapon
    LDA #$01                    ; EffectRequest bit $01: sword noise.
    JMP Sfx_Play

; TABLE: boomerang turn-back distance; index = InvMagicBoomerang (0 wooden, 1 magic); value = pixels,
;   stored in ObjMovingLimit and compared by Boomerang_UpdateOrArrow with the absolute value of the
;   boomerang's signed ObjGridOffset. $31 is reached after 17 frames at 3 px/frame. $FF cannot be
;   reached (the absolute value is at most $80), so the magic boomerang flies until it is blocked.
Boomerang_RangeLimits:
    .BYTE $31, $FF

; ----------------------------------------------------------------------------
; Starts the item selected for the B button by jumping through Link_WieldItem_JumpTable.
; IN:   SelectedItemSlot; called from Link_HandleInput with Link's state 0 and B newly pressed.
; NOTE: Slot $0F (letter) does nothing here. Of the handlers in this file only the boomerang (either
;       kind owned), the arrow (bow, rupees) and the potion (doses) test the inventory; the food and
;       rod handlers do not. The table has 9 entries and the index is not range-checked.
; ----------------------------------------------------------------------------
Link_WieldItem:
    ; If the letter slot is selected, return.
    LDA SelectedItemSlot
    CMP #$0F
    BEQ Boomerang_Wield_Return
    JSR Util_TableJump

; TABLE: B item handlers; index = SelectedItemSlot (0-8); value = routine address.
Link_WieldItem_JumpTable:
    .ADDR Boomerang_Wield        ; 0 boomerang (wooden or magic)
    .ADDR Bomb_Wield             ; 1 bomb
    .ADDR Arrow_Wield            ; 2 arrow
    .ADDR Link_WieldNothing          ; 3 bow: nothing
    .ADDR Candle_Wield           ; 4 candle
    .ADDR Flute_Wield            ; 5 flute
    .ADDR Food_Wield             ; 6 food
    .ADDR Potion_Wield           ; 7 potion
    .ADDR Rod_Wield              ; 8 rod

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
Boomerang_Wield:
    ; If missing wooden boomerang and magic boomerang, return.
    LDA InvBoomerang
    ORA InvMagicBoomerang
    BEQ Boomerang_Wield_Return

    ; Switch to the boomerang slot.
    LDX #$0F

    ; If the slot's state <> 0 and its high bit is clear (a boomerang is out), return.
    ; State 0 (empty) and states with the high bit set (food) go on.
    LDA ObjState, X
    BEQ :+
    ASL
    BCC Boomerang_Wield_Return
:
    ; Set state to $10 for boomerang.
    LDA #$10
    STA ObjState, X

    ; Set the farthest distance the boomerang can fly based on
    ; the type of boomerang.
    LDY InvMagicBoomerang
    LDA Boomerang_RangeLimits, Y
    STA ObjMovingLimit, X

    ; Set up the boomerang: Link's state $10, placed $10 pixels ahead of him,
    ; q-speed $C0 (3 pixels a frame).
    ;
    ; The first turning animation frame gets a counter of 3.
    JSR Link_UseItem_SetState10
    LDA #$C0
    STA ObjQSpeedFrac, X
    LDA #$03
    STA ObjAnimCounter, X

    ; See Link_UseItem_SetAnimCounter1 for the reason
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

; Shared exit of Link_WieldItem and Boomerang_Wield.
Boomerang_Wield_Return:
    RTS

; ----------------------------------------------------------------------------
; Shoots an arrow from object slot $12 for 1 rupee.
; OUT:  slot $12: state $10, then as Link_WieldWeapon; InvRupeesToSubtract + 1; EffectRequest bit $02.
; NOTE: Refused when InvBow = 0, when slot $12's state <> 0 with bit 7 clear (an arrow in flight, or
;       a rod swing: the rod shares the slot), or when InvRupees = 0. InvArrow is not tested here.
;       No state with bit 7 set is written to slot $12: it holds 0, an arrow ($10 flying, $20/$28
;       spark, $40 only inside the call that frees it) or a rod ($31-$35). So the bit 7 exemption,
;       the same code shape as the boomerang/food test of slot $0F, never applies here.
; ----------------------------------------------------------------------------
Arrow_Wield:
    ; If there's no bow, return.
    LDA InvBow
    BEQ Link_WieldNothing

    ; Switch to the arrow slot.
    LDX #$12

    ; If state <> 0 and high bit is clear, return.
    LDA ObjState, X
    BEQ :+
    ASL
    BCC Link_WieldNothing
:
    ; If there are no rupees, return.
    LDA InvRupees
    BEQ Link_WieldNothing
    LDA #$02                    ; EffectRequest bit $02: boomerang/arrow noise.
    JSR Sfx_Play

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
Link_WieldWeapon:
    STA ObjState, X

    ; Set q-speed $C0 (3 pixels a frame).
    LDA #$C0
    STA ObjQSpeedFrac, X
    JSR Link_UseItem_SetAnimCounter1

    ; If the direction is vertical, move the object right 3 pixels.
    LDA ObjDir, X
    AND #$0C
    BEQ Link_WieldNothing
    LDA ObjX, X
    CLC
    ADC #$03
    STA ObjX, X

; Handler of B item slot 3 (bow): does nothing. Also the shared exit of Arrow_Wield and Link_WieldWeapon.
Link_WieldNothing:
    RTS

; ----------------------------------------------------------------------------
; Puts the food down in object slot $0F, $10 pixels ahead of Link.
; OUT:  slot $0F: state $80, ObjTimer $FF; Link's state = $10, his ObjAnimCounter = 1.
; NOTE: Refused when slot $0F's state <> 0: a boomerang is out or food is already down. InvFood is
;       not tested and not consumed. Boomerang_UpdateOrFood keeps the food through states $80-$82 with
;       a $FF-frame timer each and removes it on reaching $83.
; ----------------------------------------------------------------------------
Food_Wield:
    ; Switch to the food slot.
    LDX #$0F

    ; If there's an item already active in the slot (state <> 0), return.
    LDA ObjState, X
    BNE Potion_Wield_Return

    ; The timer of the first state of food: $FF frames.
    LDA #$FF
    STA ObjTimer, X

    ; Set up the food in state $80.
    LDA #$80
    JMP Link_UseItem_SetWeaponState

; ----------------------------------------------------------------------------
; Drinks one potion dose: starts the heart refill and pauses the game.
; OUT:  InvPotion - 1; World_IsFillingHearts = 1; PauseState = 2.
; NOTE: Refused only when InvPotion = 0; current hearts are not tested, so a dose is spent even at
;       full hearts. Uses no object slot and leaves Link's state at 0.
; ----------------------------------------------------------------------------
Potion_Wield:
    ; If there's no potion dose left, return.
    LDA InvPotion
    BEQ Potion_Wield_Return

    ; Use one dose (2 red -> 1 blue -> 0).
    DEC InvPotion

    ; Flag that we're filling hearts and involuntarily paused.
    LDA #$01
    STA World_IsFillingHearts
    LDA #$02
    STA PauseState

; Shared exit of Food_Wield, Potion_Wield, Rod_Wield and Warp_CheckSubroomExit.
Potion_Wield_Return:
    RTS

; ----------------------------------------------------------------------------
; Starts a rod swing in object slot $12.
; OUT:  slot $12: state $31, ObjAnimCounter 5, then as Link_WieldWeapon.
; NOTE: Refused when slot $12's state <> 0 (a rod swing, or an arrow in flight: they share the slot).
;       InvRod is not tested here. No cost. Rod_UpdateRodOrArrow passes states $30 and up to
;       Sword_UpdateSwordOrRod, which steps the low nibble 1..5 with the sword's timing.
; ----------------------------------------------------------------------------
Rod_Wield:
    ; Switch to the rod slot.
    ; If it's already in use, then return.
    LDX #$12
    LDA ObjState, X
    BNE Potion_Wield_Return

    ; Timer of the first state: 5.
    LDA #$05
    STA ObjAnimCounter, X

    ; The initial state is $31.
    LDA #$31
    JMP Link_WieldWeapon
