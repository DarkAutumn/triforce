; ============================================================================
; FILE:     bank2/mode0-demo.asm         BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 180-359
; SUBJECTS: (none)
; TOPIC:    Mode 0 demo init/update dispatch, save-slot info load
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 InitDemo_RunTasks                   code
;   L0031 InitDemo_RunTasks_Phase0_JumpTable  data
;   L0036 InitDemo_Phase1                     code
;   L0040 InitDemo_RunTasks_Phase1_JumpTable  data
;   L0045 UpdateMode0Demo                     code
;   L0058 UpdateMode0Demo_JumpTable           data
;   L0063 UpdateMode0Demo_Sub0                code
;   L0079 Mode0Demo_Return                    code
;   L0082 UpdateMode0Demo_Sub2                code
;   L0184 AnimateDemo                         code
;   L0190 AnimateDemo_Phase0_JumpTable        data
;   L0194 AnimateDemo_Phase1                  code
;   L0198 AnimateDemo_Phase1_JumpTable        data
; ============================================================================

InitDemo_RunTasks:
    JSR TurnOffAllVideo
    LDA DemoPhase
    BNE InitDemo_Phase1
    LDA DemoSubphase
    JSR TableJump

InitDemo_RunTasks_Phase0_JumpTable:
    .ADDR InitDemoSubphaseClearArtifacts
    .ADDR InitDemoSubphaseTransferTitlePalette
    .ADDR InitDemoSubphasePlayTitleSong

InitDemo_Phase1:
    LDA DemoSubphase
    JSR TableJump

InitDemo_RunTasks_Phase1_JumpTable:
    .ADDR InitDemoSubphaseClearArtifacts
    .ADDR InitDemoSubphaseTransferStoryPalette
    .ADDR InitDemoSubphaseTransferStoryTiles

UpdateMode0Demo:
    LDA GameSubmode
    BNE @HandleSubmodes         ; Only animate if submode = 0 and SkippedDemo = 0
    LDA SkippedDemo
    BNE @HandleSubmodes
    JSR AnimateDemo
    LDA IsUpdatingMode          ; If no longer updating,
    BEQ Mode0Demo_Return                    ; then return.

@HandleSubmodes:
    LDA GameSubmode
    JSR TableJump

UpdateMode0Demo_JumpTable:
    .ADDR UpdateMode0Demo_Sub0
    .ADDR UpdateMode0Demo_Sub1
    .ADDR UpdateMode0Demo_Sub2

UpdateMode0Demo_Sub0:
    LDA ButtonsPressed          ; If Start is not pressed,
    AND #$10
    BEQ Mode0Demo_Return                    ; then return.
    STA TransferredDemoPatterns ; Else, store $10, because it's convenient and not $A5.
    LDA #$00
    STA SongRequest
    JSR SilenceAllSound
    LDA #$5A
    STA SkippedDemo
    INC GameSubmode             ; Go to next submode.
    JSR TurnOffAllVideo
    JSR HideAllSprites
    LDA #$12                    ; Cue transfer record for menu palettes.
    STA TileBufSelector

Mode0Demo_Return:
    RTS

UpdateMode0Demo_Sub2:
    ; Copy data from each save file A to save slot info.
    ; Most of the game will deal with save slot info.
    ; Format each inactive file A, to make sure it's clear.
    JSR TurnOffAllVideo
    LDA #$00
    STA CurSaveSlot
    JSR FetchFileAAddressSet
    LDY #$02

@LoopFormatSlot:
    LDA ($06), Y
    STA IsSaveSlotActive, Y
    BNE @NextFormatSlot
    TYA                         ; Save the slot.
    PHA
    STY CurSaveSlot             ; Switch to this slot's addresses.
    JSR FetchFileAAddressSet
    JSR FormatFileA
    LDA #$00
    STA CurSaveSlot

    ; Fetch the address set for slot 0 again,
    ; so that we can keep referring to its
    ; IsSaveSlotActive address as a table base.
    ;
    ; After this, the slot is still not active, but it will
    ; definitely be clear.
    JSR FetchFileAAddressSet
    PLA
    TAY                         ; Restore the slot.

@NextFormatSlot:
    LDA ($0A), Y                ; Copy death count from file A to save slot info.
    STA DeathCounts, Y
    LDA ($0C), Y                ; Copy quest number from file A to save slot info.
    STA QuestNumbers, Y
    DEY
    BPL @LoopFormatSlot
    LDY #$18                    ; Point to hearts value
    LDX #$00                    ; 0: process hearts value; 1: process heart partial.

@LoopHeart:
    LDA ($00), Y                ; Load the hearts value or heart partial from Items block in file A.
    PHA                         ; Push either value.
    TXA
    LSR

    ; If X is even, then the value is a hearts value.
    ; So, make the hearts equal the heart containers.
    BCS @StoreValue

    ; Pop what we pushed, because we're going to push
    ; a modification.
    PLA
    AND #$F0
    STA $0C
    LSR
    LSR
    LSR
    LSR
    ORA $0C
    PHA                         ; Push the full hearts value.

@StoreValue:
    PLA                         ; Pop whatever we pushed: heart partial or full hearts value
    STA SaveSlotHearts, X       ; Copy to hearts in save slot info.

    ; Point to the next byte in Items block.
    ; hearts value -> hearts partial
    INY
    INX                         ; Increment the index of the value we check.

    ; There are 6 values total:
    ; (hearts value, hearts partial) * 3 slots.
    CPX #$06
    BEQ @CopyNames              ; Once we finish the last value, quit the loop.
    TXA                         ; If X is odd, then go process heart partial instead of hearts value.
    LSR
    BCS @LoopHeart

    ; The 3 files are consecutive in the set.
    ; Point to the hearts value in the next slot.
    TYA
    ADC #$26
    TAY
    JMP @LoopHeart              ; Go process hearts in the next slot.

@CopyNames:
    LDY #$17                    ; Copy the name from file A to save slot info.

@LoopNameByte:
    LDA ($04), Y
    STA Names, Y
    DEY
    BPL @LoopNameByte
    INC GameMode                ; Go to the next game mode (Menu).
    LDA #$00                    ; We're set to initialize the mode.
    STA IsUpdatingMode
    STA GameSubmode
    RTS

AnimateDemo:
    LDA DemoPhase
    BNE AnimateDemo_Phase1
    LDA DemoSubphase
    JSR TableJump

AnimateDemo_Phase0_JumpTable:
    .ADDR AnimateDemoPhase0Subphase0
    .ADDR AnimateDemoPhase0Subphase1

AnimateDemo_Phase1:
    LDA DemoSubphase
    JSR TableJump

AnimateDemo_Phase1_JumpTable:
    .ADDR AnimateDemoPhase1Subphase0
    .ADDR AnimateDemoPhase1Subphase1
    .ADDR AnimateDemoPhase1Subphase2
    .ADDR AnimateDemoPhase1Subphase3
    .ADDR AnimateDemoPhase1Subphase4
