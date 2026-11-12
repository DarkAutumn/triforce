; ============================================================================
; FILE:     bank1/take-item.asm          BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4322-4739
; SUBJECTS: (none)
; TOPIC:    Item tables, room item flag, pickup range, TakeItem effects
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0033 ItemIdToSlot                      data
;   L0056 ItemIdToDescriptor                data  Each byte describes an item. The high nibble is the type.
;   L0072 ItemSlotToPaletteOffsetsOrValues  data  Maps an item slot to a value used in calculating the
;   L0078 SetRoomFlagUWItemState            code
;   L0087 GetRoomFlagUWItemState            code  Returns:
;   L0098 LinkColors_CommonCode             data
;   L0101 TryTakeRoomItem                   code
;   L0128 TryTakeItem                       code  Params:
;   L0173 TakeItem                          code  Params:
;   L0231 SetItemValue                      code
;   L0234 L6C28_Exit                        code
;   L0237 Take5Rupees                       code
;   L0246 CheckClass1                       code
;   L0319 TakeClass0Complex                 code
;   L0363 TakeHearts                        code  Params:
;   L0366 TakeHeartsNoSound                 code
;   L0396 CompareHeartsToContainers         code  Returns:
;   L0409 L_TakePowerTriforce               code
;   L0414 HandleClass2                      code
;   L0437 TakeOneRupee                      code
;   L0442 L6D1B_Exit                        code
;   L0445 PlayKeyTakenTune                  code
; ============================================================================

ItemIdToSlot:
    .BYTE $01, $00, $00, $00, $06, $05, $04, $04
    .BYTE $02, $02, $03, $0D, $09, $0C, $1B, $1C
    .BYTE $08, $0A, $0B, $0B, $0E, $0F, $10, $11
    .BYTE $16, $17, $18, $1A, $1F, $1D, $1E, $07
    .BYTE $07, $15, $19, $14

; Each byte describes an item. The high nibble is the type.
; The low nibble is the value. Its meaning depends on the type.
;
; 0x - individuals
; * unique: value is boolean: you have it or you don't; also value indicates palette row sprite attribute.
; * bit mask: value doesn't matter.
;
; 1x - amounts
; Value is an amount to add.
;
; 2x - grades
; Value is a grade and palette row. Picking up an item with a lower grade won't change the inventory.
;
; 3x - error?
; Produces an item value of $FF.
;
ItemIdToDescriptor:
    .BYTE $14, $21, $22, $23, $01, $01, $21, $22
    .BYTE $21, $22, $01, $01, $01, $01, $01, $15
    .BYTE $01, $01, $21, $22, $01, $01, $01, $01
    .BYTE $11, $11, $10, $01, $01, $01, $01, $11
    .BYTE $22, $01, $10, $12

; Maps an item slot to a value used in calculating the
; sprite palette row attribute.
;
; For most items, the value is the sprite palette row attribute
; itself.
;
; Some items will use the value as an amount to add to the
; item value in the slot.
;
ItemSlotToPaletteOffsetsOrValues:
    .BYTE $FF, $01, $FF, $00, $00, $02, $02, $00
    .BYTE $01, $00, $02, $00, $00, $02, $02, $01
    .BYTE $02, $02, $02, $02, $02, $02, $02, $02
    .BYTE $02, $02, $02, $02, $01, $00, $01, $00

SetRoomFlagUWItemState:
    JSR GetRoomFlags
    ORA #$10
    STA ($00), Y
    RTS

; Returns:
; A: $10 if item was taken, else 0
;
GetRoomFlagUWItemState:
    LDA LevelInfo_WorldFlagsAddr
    STA $08
    LDA LevelInfo_WorldFlagsAddr+1
    STA $09
    LDY RoomId
    LDA ($08), Y
    AND #$10
:
    RTS

LinkColors_CommonCode:
    .BYTE $29, $32, $16

TryTakeRoomItem:
    ; If Link is halted, then return.
    LDA ObjState
    AND #$C0
    CMP #$40
    BEQ :-

    ; If player took the room item, then return.
    JSR GetRoomFlagUWItemState
    BNE :-

    ; Switch to room item object slot $13.
    LDX #$13

    ; If room item object wasn't activated, then return.
    LDA ObjState, X
    BMI :-

    ; Pass RoomItemId, also known as [98][$13] and ObjDir[$13],
    ; to try to take the item.
    LDA ObjRoomItemId, X
    STA $04                     ; [04] holds the item type.

; Params:
; X: object index
; [04]: item type
;
TryTakeItem:
    ; If the lifetime timer of the item >= $F0, then return;
    ; so that the player can't pick it up right away.
    ;
    ; [03A8][X] is used to count down the life of the item.
    LDA Item_ObjItemLifetime, X
    CMP #$F0
    BCS L6C28_Exit

    ; If Y distance between Link and the item >= 9, return.
    LDA ObjY
    CLC
    ADC #$03
    SEC
    SBC ObjY, X
    JSR Abs
    CMP #$09
    BCS L6C28_Exit

    ; If X distance between Link and the item >= 9, return.
    LDA ObjX
    SEC
    SBC ObjX, X
    JSR Abs
    CMP #$09
    BCS L6C28_Exit

    ; In case this is the room item, set state to $FF to deactivate it.
    LDA #$FF
    STA ObjState, X
    STA ObjY, X

    ; If it is the room item that is taken, then
    ; mark it taken in room flags.
    CPX #$13
    BNE :+
    JSR SetRoomFlagUWItemState
:
    LDA $04                     ; Get the item type.

; Params:
; A: item ID
;
; By default, we'll play the "item taken" tune.
;
TakeItem:
    LDX #$08
    STX Tune1Request

    ; Unless it's the Triforce of Power; in which case
    ; play the "item appears" tune.
    CMP #$0E
    BNE :+
    LDX #$02
    STX Tune1Request
:
    ; If we're in a cave or cellar, then
    ; prepare to lift the item.
    LDX GameMode
    CPX #$05
    BEQ @SkipLift
    LDX #$80                    ; Lift for $80 frames.
    STX ItemLiftTimer
    LDX #$08                    ; Play the "item" song.
    STX SongRequest
    STA ItemTypeToLift          ; Set the item type for lifting.

@SkipLift:
    ; Look up the item slot and sprite attributes by the item type.
    TAX
    LDA ItemIdToSlot, X
    TAY
    LDA ItemIdToDescriptor, X

    ; [0A] holds the item value from the low nibble of descriptor.
    PHA
    AND #$0F
    STA $0A
    PLA

    ; Check the high nibble which specifies the item's class.
    ;
    ; A: item class
    ; X: item type
    ; Y: item slot
    AND #$F0
    BNE CheckClass1             ; Go check classes 1 and 2.

    ; Class 0. We have an item of a type that's unique or complex.
    ;
    ; First, check the exceptions. If the item is a map, compass, or triforce; then go handle them.
    CPY #$11                    ; Map item slot
    BEQ TakeClass0Complex
    CPY #$10                    ; Compass item slot
    BEQ TakeClass0Complex
    CPY #$1A                    ; Triforce pieces item slot
    BEQ TakeClass0Complex
    CPY #$1B                    ; Triforce of Power item slot
    BEQ TakeClass0Complex

    ; Finally, this is a simple item. Put the item value in the item slot.
    LDA $0A

SetItemValue:
    STA Items, Y

L6C28_Exit:
    RTS

Take5Rupees:
    ; For 5 Rupees, perform the action for 1 rupee 5 times.
    LDY #$04
:
    JSR TakeOneRupee
    DEY
    BPL :-
    RTS

CheckClass1:
    ; A: item class
    ; X: item type
    ; Y: item slot
    CMP #$10
    BNE @CheckClass2            ; Go check class 2 items.

    ; Class 1. We have a type of item with an amount.
    ; Item value is the amount to add.
    CPY #$18                    ; Heart container item slot
    BEQ @TakeHeartContainer
    CPY #$1C                    ; 5 Rupees item slot
    BEQ Take5Rupees
    CPY #$16                    ; Rupee item slot
    BEQ @TakeRupee
    CPY #$19                    ; Heart item slot
    BEQ TakeHearts

    ; For key item slot, play the tune here.
    ; Add to the amount below.
    CPY #$17
    BNE :+
    JSR PlayKeyTakenTune
:
    ; For fairy item slot, go handle it where hearts are increased.
    CPY #$14
    BEQ TakeHeartsNoSound

    ; Keys and other items.
    ; Add item value and value in item slot.
    LDA $0A
    CLC
    ADC Items, Y
    BCC :+                      ; If the sum overflowed,

@SetItemValueFF:
    LDA #$FF                    ; then cap it at $FF.
:
    ; For potion item slot, cap it at 2.
    CPY #$07
    BNE :+
    CMP #$03
    BCC :+
    LDA #$02
:
    ; For bomb item slot, cap it at MaxBombs.
    CPY #$01
    BNE :+
    CMP MaxBombs
    BCC :+
    LDA MaxBombs
:
    JMP SetItemValue            ; Go set the item value and return.

@TakeHeartContainer:
    ; For heart containers, cap at $10.
    ; Add a heart container and a heart.
    LDA Items, Y
    CMP #$F0
    BCS L6C28_Exit              ; If we already have max hearts, return.
    ADC #$11
    JMP SetItemValue            ; Go set the item value and return.

@TakeRupee:
    JMP TakeOneRupee

@CheckClass2:
    ; Make sure this is an item of a type ordered by grade.
    ; Then go handle it.
    CMP #$20
    BNE @SetItemValueFF         ; If the class is $30, go set item value to $FF.
    BEQ HandleClass2

TakeClass0Complex:
    ; We're taking a class 0 complex item found in UW.
    ;
    ; A: item class
    ; X: item type
    ; Y: item slot
    LDA CurLevel
    BEQ @Exit                   ; If in OW, return.
    CPY #$1B                    ; Triforce of Power item slot
    BEQ L_TakePowerTriforce
    CPY #$11                    ; Map item slot
    BNE :+                      ; If it's the map item slot,
    LDX #$01                    ; then signal the main loop to draw the status bar map.
    STX StatusBarMapTrigger
:
    ; Choose the right slot for the level. (map/map9, compass/compass9)
    SEC
    SBC #$01                    ; Subtract one from level number to base it on 0.
    CMP #$08
    BCC :+                      ; If the result >= 8, we're in level 9,
    INY                         ; So add 2 to the item slot to use the one for level 9.
    INY
:
    AND #$07                    ; Make sure the level index is in the range 0 to 8.
    TAX

    ; Combine the mask for this level with the mask in the item slot.
    LDA Items, Y
    ORA LevelMasks, X
    STA Items, Y

    ; If the item taken was a triforce piece, then go to mode $12.
    CPY #$1A
    BNE @Exit                   ; If it's not a triforce piece, then return.
    JSR EndGameMode
    LDA #$12
    STA GameMode

@Exit:
    RTS

; Params:
; [0A]: item value/amount
;
TakeHearts:
    JSR PlayKeyTakenTune

TakeHeartsNoSound:
    LDA $0A                     ; Move item value in [0A] to [01].
    STA $01                     ; [01] holds the number of hearts to add minus 1.

@CompareHearts:
    ; For that many hearts, add them one by one.
    JSR CompareHeartsToContainers
    BNE @AddWholeHeart          ; If there's room to add a whole heart, go add it.

    ; If partial heart is not full, then fill it and return.
    LDX HeartPartial
    INX
    BNE @FillHeartPartial
    RTS

@AddWholeHeart:
    ; Add a whole heart.
    INC HeartValues
    DEC $01
    BPL @CompareHearts          ; If there are more hearts to add, go compare them to heart containers again.
    RTS

@FillHeartPartial:
    LDA #$FF
    STA HeartPartial
    RTS

; Returns:
; Z: 1 if hearts = containers
;
CompareHeartsToContainers:
    LDA HeartValues
    PHA
    AND #$0F
    STA $00
    PLA
    LSR
    LSR
    LSR
    LSR
    CMP $00
    RTS

L_TakePowerTriforce:
    LDA #$01
    JSR SwitchBank_Local5
    JMP TakePowerTriforce

HandleClass2:
    ; Class 2. We have a type of item that is ordered by grade.
    ; Item value is the grade.
    ;
    ; A: item class
    ; X: item type
    ; Y: item slot
    ; [0A]: item value
    LDA $0A
    CMP Items, Y
    BCC L6D1B_Exit              ; If we have a higher grade of this kind of item, return.
    STA Items, Y                ; Set the new item grade.
    CPY #$0B                    ; Ring item slot
    BNE L6D1B_Exit              ; If the item is not a ring, return.

    ; We took a ring. Change Link's color.
    LDX CurSaveSlot
    LDY InvRing                 ; Get ring in inventory.
    LDA LinkColors_CommonCode, Y    ; Get the color for this ring value.
    LDY SaveSlotToPaletteRowOffset, X    ; Get the offset of row 4, 5, or 6, depending on save slot.
    STA MenuPalettesTransferBuf+20, Y    ; Patch the color into the menu palette.
    JMP PatchAndCueLevelPalettesTransferAndAdvanceSubmode    ; Then patch the color into the level's palette.

TakeOneRupee:
    LDA #$01                    ; Play "rupee taken" tune.
    STA Tune1Request
    INC RupeesToAdd

L6D1B_Exit:
    RTS

PlayKeyTakenTune:
    LDA #$00                    ; Stop playing tune 2.
    STA Tune1Request
    LDA #$08                    ; Play "key taken" tune.
    STA Tune0Request
    RTS
