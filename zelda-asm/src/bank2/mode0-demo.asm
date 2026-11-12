; ============================================================================
; FILE:     bank2/mode0-demo.asm         BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 180-359
; SUBJECTS: Mode0Demo
; TOPIC:    Mode 0 (title screen and story demo): the init dispatcher by demo phase and subphase, the
;           update routine (animate, wait for Start), and the copy of the three save files' summary
;           data into save slot info before the file menu (mode 1).
; SEE ALSO: bank7/init-mode-dispatch.asm (Mode0Demo_InitPatternsOrDemo jumps to Mode0Demo_Init);
;           bank7/update-mode-dispatch.asm (Mode_Update_JumpTable entry 0 is Mode0Demo_Update);
;           bank2/demo-tables-init.asm (the init subphases); bank2/demo-story-scroll.asm and
;           bank2/title-animation.asm (the animation subphases); bank2/mode1-init-save-files.asm
;           (Save_CommitOrRepairFiles: save file validation); bank1/modeb-cave-save-sprites.asm
;           (Save_FetchFileAAddressSet).
; KEY FACTS:
;   - DemoPhase 0 is the title screen (2 animation subphases), DemoPhase 1 the story scroll and item
;     parade (5 animation subphases). Mode0Demo_Animate, Mode0Demo_Animate_Phase0_JumpTable,
;     Mode0Demo_Animate_Phase1_JumpTable.
;   - Each phase is initialized over 3 frames, one init subphase per frame: clear, palette, then
;     song/tiles. Mode0Demo_Init_Phase0_JumpTable, Mode0Demo_Init_Phase1_JumpTable.
;   - Only Start (ButtonsPressed bit $10) leaves the demo; it is tested every frame of submode 0,
;     in both phases. Mode0Demo_Update_Sub0.
;   - Start sets HasSkippedDemo = $5A (animation stops), stores $10 in HasDemoPatterns (not $A5, so
;     the demo patterns are sent again the next time mode 0 is initialized), silences sound and cues
;     the menu palettes (TileBufSelector $12). Mode0Demo_Update_Sub0.
;   - Submode 1 validates the save files (other file); submode 2 formats every inactive file A, copies
;     active flag, death count, quest number, name and hearts of each slot to save slot info, then
;     goes to mode 1. Mode0Demo_Update_Sub2.
;   - The file menu always shows full hearts: the low nibble of the saved HeartValues is replaced by
;     its high nibble (heart containers - 1). Mode0Demo_Update_Sub2@LoopHeart.
; CONTENTS: (generated; L = line in this file)
;   L0051 Mode0Demo_Init                      code  Mode 0 init: turns video off and runs the current init subphase of the current demo phase.
;   L0059 Mode0Demo_Init_Phase0_JumpTable     data  TABLE: init subphase handlers of demo phase 0 (title); index = DemoSubphase.
;   L0065 Mode0Demo_Init_Phase1               code  Second half of Mode0Demo_Init: dispatches the init subphase of demo phase 1 (story).
;   L0070 Mode0Demo_Init_Phase1_JumpTable     data  TABLE: init subphase handlers of demo phase 1 (story); index = DemoSubphase.
;   L0079 Mode0Demo_Update                    code  Mode 0 update: animates the demo while nothing was skipped, then runs the current submode.
;   L0094 Mode0Demo_Update_JumpTable          data  TABLE: mode 0 update submode handlers; index = GameSubmode (0 wait for Start, 1 check save files,
;   L0102 Mode0Demo_Update_Sub0               code  Submode 0: on a Start press ends the demo, blanks the screen and cues the menu palettes.
;   L0119 Mode0Demo_Return                    code  Shared return of Mode0Demo_Update and Mode0Demo_Update_Sub0.
;   L0127 Mode0Demo_Update_Sub2               code  Submode 2: formats inactive file A's, copies save file summaries to save slot info, enters mode 1.
;   L0232 Mode0Demo_Animate                   code  Runs the animation handler of the current demo phase and subphase.
;   L0239 Mode0Demo_Animate_Phase0_JumpTable  data  TABLE: animation handlers of demo phase 0 (title); index = DemoSubphase (0 wait, 1 palette fade).
;   L0244 Mode0Demo_Animate_Phase1            code  Second half of Mode0Demo_Animate: dispatches the animation subphase of demo phase 1 (story).
;   L0250 Mode0Demo_Animate_Phase1_JumpTable  data  TABLE: animation handlers of demo phase 1 (story); index = DemoSubphase (0 scroll in, 1 wait,
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 0 init: turns video off and runs the current init subphase of the current demo phase.
; IN:   DemoPhase (0 title, 1 story); DemoSubphase = init subphase 0-2.
; NOTE: The code after each JSR Util_TableJump is the address table; Util_TableJump does not return here.
; ----------------------------------------------------------------------------
Mode0Demo_Init:
    JSR Ppu_TurnOffAllVideo
    LDA DemoPhase
    BNE Mode0Demo_Init_Phase1
    LDA DemoSubphase
    JSR Util_TableJump

; TABLE: init subphase handlers of demo phase 0 (title); index = DemoSubphase.
Mode0Demo_Init_Phase0_JumpTable:
    .ADDR Mode0Demo_Init_ClearArtifacts
    .ADDR Mode0Demo_Init_TransferTitlePalette
    .ADDR Mode0Demo_Init_PlayTitleSong

; Second half of Mode0Demo_Init: dispatches the init subphase of demo phase 1 (story).
Mode0Demo_Init_Phase1:
    LDA DemoSubphase
    JSR Util_TableJump

; TABLE: init subphase handlers of demo phase 1 (story); index = DemoSubphase.
Mode0Demo_Init_Phase1_JumpTable:
    .ADDR Mode0Demo_Init_ClearArtifacts
    .ADDR Mode0Demo_Init_TransferStoryPalette
    .ADDR Mode0Demo_Init_TransferStoryTiles

; ----------------------------------------------------------------------------
; Mode 0 update: animates the demo while nothing was skipped, then runs the current submode.
; NOTE: Mode0Demo_Animate may clear IsUpdatingMode (a phase ended); the submode is then not run this frame.
; ----------------------------------------------------------------------------
Mode0Demo_Update:
    LDA GameSubmode
    BNE @HandleSubmodes         ; Only animate if submode = 0 and HasSkippedDemo = 0
    LDA HasSkippedDemo
    BNE @HandleSubmodes
    JSR Mode0Demo_Animate
    LDA IsUpdatingMode          ; If no longer updating,
    BEQ Mode0Demo_Return                    ; then return.

@HandleSubmodes:
    LDA GameSubmode
    JSR Util_TableJump

; TABLE: mode 0 update submode handlers; index = GameSubmode (0 wait for Start, 1 check save files,
; 2 load save slot info).
Mode0Demo_Update_JumpTable:
    .ADDR Mode0Demo_Update_Sub0
    .ADDR Save_CommitOrRepairFiles
    .ADDR Mode0Demo_Update_Sub2

; ----------------------------------------------------------------------------
; Submode 0: on a Start press ends the demo, blanks the screen and cues the menu palettes.
; ----------------------------------------------------------------------------
Mode0Demo_Update_Sub0:
    LDA ButtonsPressed          ; If Start is not pressed,
    AND #$10
    BEQ Mode0Demo_Return                    ; then return.
    STA HasDemoPatterns ; Else, store $10, because it's convenient and not $A5.
    LDA #$00
    STA SongRequest
    JSR Audio_SilenceAll
    LDA #$5A
    STA HasSkippedDemo
    INC GameSubmode             ; Go to next submode.
    JSR Ppu_TurnOffAllVideo
    JSR Sprite_HideAll
    LDA #$12                    ; Cue transfer record for menu palettes.
    STA TileBufSelector

; Shared return of Mode0Demo_Update and Mode0Demo_Update_Sub0.
Mode0Demo_Return:
    RTS

; ----------------------------------------------------------------------------
; Submode 2: formats inactive file A's, copies save file summaries to save slot info, enters mode 1.
; OUT:  IsSaveSlotActive, DeathCounts, QuestNumbers, SaveSlotHearts, SaveSlotNames filled for slots 0-2;
;       GameMode = 1 with IsUpdatingMode = 0 and GameSubmode = 0; CurSaveSlot = 0.
; ----------------------------------------------------------------------------
Mode0Demo_Update_Sub2:
    ; Copy data from each save file A to save slot info.
    ; Most of the game will deal with save slot info.
    ; Format each inactive file A, to make sure it's clear.
    JSR Ppu_TurnOffAllVideo
    LDA #$00
    STA CurSaveSlot
    JSR Save_FetchFileAAddressSet
    LDY #$02

@LoopFormatSlot:
    LDA ($06), Y
    STA IsSaveSlotActive, Y
    BNE @NextFormatSlot
    TYA                         ; Save the slot.
    PHA
    STY CurSaveSlot             ; Switch to this slot's addresses.
    JSR Save_FetchFileAAddressSet
    JSR Save_FormatFileA
    LDA #$00
    STA CurSaveSlot

    ; Fetch the address set for slot 0 again,
    ; so that we can keep referring to its
    ; IsSaveSlotActive address as a table base.
    ;
    ; After this, the slot is still not active, but it will
    ; definitely be clear.
    JSR Save_FetchFileAAddressSet
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
    STA SaveSlotNames, Y
    DEY
    BPL @LoopNameByte
    INC GameMode                ; Go to the next game mode (Menu).
    LDA #$00                    ; We're set to initialize the mode.
    STA IsUpdatingMode
    STA GameSubmode
    RTS

; ----------------------------------------------------------------------------
; Runs the animation handler of the current demo phase and subphase.
; ----------------------------------------------------------------------------
Mode0Demo_Animate:
    LDA DemoPhase
    BNE Mode0Demo_Animate_Phase1
    LDA DemoSubphase
    JSR Util_TableJump

; TABLE: animation handlers of demo phase 0 (title); index = DemoSubphase (0 wait, 1 palette fade).
Mode0Demo_Animate_Phase0_JumpTable:
    .ADDR Mode0Demo_Animate_Phase0_Sub0
    .ADDR Mode0Demo_Animate_Phase0_Sub1

; Second half of Mode0Demo_Animate: dispatches the animation subphase of demo phase 1 (story).
Mode0Demo_Animate_Phase1:
    LDA DemoSubphase
    JSR Util_TableJump

; TABLE: animation handlers of demo phase 1 (story); index = DemoSubphase (0 scroll in, 1 wait,
; 2 scroll text and items, 3 wait, 4 short wait then restart at phase 0).
Mode0Demo_Animate_Phase1_JumpTable:
    .ADDR Mode0Demo_Animate_Phase1_Sub0
    .ADDR Mode0Demo_Animate_Phase1_Sub1
    .ADDR Mode0Demo_Animate_Phase1_Sub2
    .ADDR Mode0Demo_Animate_Phase1_Sub3
    .ADDR Mode0Demo_Animate_Phase1_Sub4
