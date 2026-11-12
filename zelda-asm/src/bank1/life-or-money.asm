; ============================================================================
; FILE:     bank1/life-or-money.asm      BANK: 1 (switchable, $8000-$BFFF)
; UPSTREAM: Z_01.asm 1328-1480
; SUBJECTS: (none)
; TOPIC:    Life-or-money man
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0018 UpdateUnderworldPersonLifeOrMoney_Full            code
;   L0034 UpdateUnderworldPersonLifeOrMoney_Full_JumpTable  data
;   L0041 LifeOrMoneyItemXs                                 data
;   L0044 LifeOrMoneyItemTypes                              data
;   L0047 DrawLifeOrMoneyItems                              code
;   L0073 UpdateUnderworldPersonLifeOrMoneyState_0          code
;   L0080 UpdateUnderworldPersonLifeOrMoneyState_2          code
; ============================================================================

UpdateUnderworldPersonLifeOrMoney_Full:
    ; In state 4, this person and the items are drawn every other frame.
    LDA ObjState+1
    CMP #$04
    BNE @Draw
    LDA FrameCounter
    AND #$01
    BNE :+

@Draw:
    JSR Person_DrawAndCheckCollisions
    JSR DrawLifeOrMoneyItems
:
    LDA ObjState+1
    JSR TableJump

UpdateUnderworldPersonLifeOrMoney_Full_JumpTable:
    .ADDR UpdateUnderworldPersonLifeOrMoneyState_0
    .ADDR UpdatePersonState_Textbox
    .ADDR UpdateUnderworldPersonLifeOrMoneyState_2
    .ADDR UpdatePersonState_CueTransferBlankPersonWares
    .ADDR UpdateUnderworldPersonComplexState_DelayAndQuit

LifeOrMoneyItemXs:
    .BYTE $58, $98

LifeOrMoneyItemTypes:
    .BYTE $1A, $18

DrawLifeOrMoneyItems:
    ; Loop over each item to draw, from 1 to 0.
    LDX #$01

@LoopWare:
    TXA                         ; Save loop index.
    PHA

    ; Look up and set the X coordinate of the item.
    LDA LifeOrMoneyItemXs, X
    STA ObjX+19

    ; Set Y=$98, and look up the item type.
    LDA #$98
    STA ObjY+19
    LDA LifeOrMoneyItemTypes, X

    ; Switch to the room item object slot.
    LDX #$13
    JSR AnimateItemObject
    PLA                         ; Restore loop index.
    TAX
    DEX
    BPL @LoopWare
    RTS

UpdateUnderworldPersonLifeOrMoneyState_0:
    ; Delay in the next state $A frames before showing the first character.
    LDA #$0A
    STA ObjTimer+1
    LDA #$76                    ; "-1   -50" text selector
    JMP CueTransferBufAndAdvanceState

UpdateUnderworldPersonLifeOrMoneyState_2:
    ; Loop over each item to check, from 1 to 0.
    LDX #$01

@LoopWare:
    ; If Link's X doesn't match the item's, then go loop again.
    LDA ObjX
    CMP LifeOrMoneyItemXs, X
    BNE @NextWare

    ; If the vertical distance between the item and Link < 6, then
    ; go handle paying appropriately.
    LDA ObjY
    SEC
    SBC #$98
    JSR Abs
    CMP #$06
    BCC @Pay

@NextWare:
    DEX
    BPL @LoopWare

@Exit:
    RTS

@Pay:
    ; If the player chose the heart container, go handle it.
    CPX #$00
    BEQ @PayWithHeartContainer

    ; The player chose the money.
    ; But if he doesn't have enough, then return.
    LDA #$32
    CMP InvRupees
    BEQ @PayWithMoney
    BCS @Exit

@PayWithMoney:
    ; Post 50 rupees more to subtract, and go flag this offer taken.
    CLC
    ADC InvRupeesToSubtract
    STA InvRupeesToSubtract
    JMP @EndLifeOrMoney

@PayWithHeartContainer:
    ; The player chose the heart container.
    ;
    ; If the number of heart containers < 4, then
    ; - leave the number of heart containers alone
    ; - set HeartPartial to 0, so Link will die with one more hit
    ;
    ; Then go flag this offer taken.
    LDA HeartValues
    AND #$F0
    CMP #$30
    BCS @Reduce
    STA HeartValues
    LDA #$00
    STA HeartPartial
    JMP @EndLifeOrMoney

@Reduce:
    ; Store (HeartValues - 1 heart container) in [00].
    LDA HeartValues
    PHA
    AND #$F0
    SEC
    SBC #$10
    STA $00

    ; Get (HeartValues - 1 heart)
    PLA
    AND #$0F
    SEC
    SBC #$01

    ; If the result is negative, then make the full hearts 0.
    BPL :+
    LDA #$00
:
    ; Combine the heart containers and full hearts, and store it.
    ORA $00
    STA HeartValues

@EndLifeOrMoney:
    LDA #$08                    ; "key taken" sound effect
    STA Tune0Request
    LDA #$01
    STA ShutterTrigger
    JMP L_Person_FlagItemTakenAndAdvanceState
