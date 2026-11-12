; ============================================================================
; FILE:     bank5/wield-items.asm        BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 2866-3057
; SUBJECTS: (none)
; TOPIC:    Wield sword, boomerang, arrow, food, potion, rod; B item dispatch
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 WieldSword           code
;   L0046 BoomerangLimits      data
;   L0049 WieldItem            code
;   L0056 WieldItem_JumpTable  data
;   L0067 WieldBoomerang       code
;   L0117 L14E71_Exit          code
;   L0120 WieldArrow           code
;   L0149 WieldWeapon          code  Params:
;   L0166 WieldNothing         code
;   L0169 WieldFood            code
;   L0185 WieldPotion          code
;   L0199 L14EC6_Exit          code
;   L0202 WieldRod             code
; ============================================================================

WieldSword:
    ; If there's no sword, return.
    LDA InvSword
    BEQ L14DFF_Exit

    ; Switch to the sword slot.
    LDX #$0D

    ; If state <> 0, then sword or item is in use. So, return.
    LDA ObjState, X
    BNE L14DFF_Exit

    ; The first state lasts 5 frames.
    LDA #$05
    STA ObjAnimCounter, X

    ; The initial state is 1.
    LDA #$01
    JSR WieldWeapon
    LDA #$01                    ; Sword sound effect
    JMP PlayEffect

BoomerangLimits:
    .BYTE $31, $FF

WieldItem:
    ; If the letter slot is selected, return.
    LDA SelectedItemSlot
    CMP #$0F
    BEQ L14E71_Exit
    JSR TableJump

WieldItem_JumpTable:
    .ADDR WieldBoomerang
    .ADDR WieldBomb
    .ADDR WieldArrow
    .ADDR WieldNothing
    .ADDR WieldCandle
    .ADDR WieldFlute
    .ADDR WieldFood
    .ADDR WieldPotion
    .ADDR WieldRod

WieldBoomerang:
    ; If missing wooden boomerang and magic boomerang, return.
    LDA InvBoomerang
    ORA InvMagicBoomerang
    BEQ L14E71_Exit

    ; Switch to the boomerang slot.
    LDX #$0F

    ; If state in object slot <> 0 and high bit is clear, return.
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

    ; Set up the boomerang.
    ; QSpeed = $C0 (3 pixels a frame)
    ;
    ; The first turning animation frame lasts 3 screen frames, instead of 2.
    ; Maybe it's to make up for the fact that the first screen frame isn't seen.
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

L14E71_Exit:
    RTS

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
    LDA #$02                    ; Boomerang/arrow sound effect
    JSR PlayEffect

    ; Post a rupee to subtract.
    INC InvRupeesToSubtract

    ; Arrows start in state $10.
    LDA #$10

; Params:
; A: initial state
;
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

WieldNothing:
    RTS

WieldFood:
    ; Switch to the food slot.
    LDX #$0F

    ; If there's an item already active in the slot (state <> 0), return.
    LDA ObjState, X
    BNE L14EC6_Exit

    ; The first state of food lasts $FF frames.
    LDA #$FF
    STA ObjTimer, X

    ; Set up the food in state $80.
    LDA #$80
    JMP PlaceWeaponForPlayerStateAndAnimAndWeaponState

WieldPotion:
    ; If there's nothing in the item slot, return.
    LDA InvPotion
    BEQ L14EC6_Exit

    ; We're using one potion. So decrement the item value.
    DEC InvPotion

    ; Flag that we're filling hearts and involuntarily paused.
    LDA #$01
    STA World_IsFillingHearts
    LDA #$02
    STA PauseState

L14EC6_Exit:
    RTS

WieldRod:
    ; Switch to the rod slot.
    ; If it's already in use, then return.
    LDX #$12
    LDA ObjState, X
    BNE L14EC6_Exit

    ; The first state lasts 5 frames.
    LDA #$05
    STA ObjAnimCounter, X

    ; The initial state is $31.
    LDA #$31
    JMP WieldWeapon
