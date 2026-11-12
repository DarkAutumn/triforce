; ============================================================================
; FILE:     bank2/mode0-demo.asm         BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 180-359
; SUBJECTS: Mode0Demo
; TOPIC:    Mode 0 (title screen and story demo): the init dispatcher by demo phase and subphase, the
;           update routine (animate, wait for Start), and the copy of the three save files' summary
;           data into save slot info before the file menu (mode 1).
; SEE ALSO: bank7/init-mode-dispatch.asm (InitMode0 jumps to InitDemo_RunTasks);
;           bank7/update-mode-dispatch.asm (UpdateMode_JumpTable entry 0 is UpdateMode0Demo);
;           bank2/demo-tables-init.asm (the init subphases); bank2/demo-story-scroll.asm and
;           bank2/title-animation.asm (the animation subphases); bank2/mode1-init-save-files.asm
;           (UpdateMode0Demo_Sub1: save file validation); bank1/modeb-cave-save-sprites.asm
;           (FetchFileAAddressSet).
; KEY FACTS:
;   - DemoPhase 0 is the title screen (2 animation subphases), DemoPhase 1 the story scroll and item
;     parade (5 animation subphases). AnimateDemo, AnimateDemo_Phase0_JumpTable,
;     AnimateDemo_Phase1_JumpTable.
;   - Each phase is initialized over 3 frames, one init subphase per frame: clear, palette, then
;     song/tiles. InitDemo_RunTasks_Phase0_JumpTable, InitDemo_RunTasks_Phase1_JumpTable.
;   - Only Start (ButtonsPressed bit $10) leaves the demo; it is tested every frame of submode 0,
;     in both phases. UpdateMode0Demo_Sub0.
;   - Start sets HasSkippedDemo = $5A (animation stops), stores $10 in HasDemoPatterns (not $A5, so
;     the demo patterns are sent again the next time mode 0 is initialized), silences sound and cues
;     the menu palettes (TileBufSelector $12). UpdateMode0Demo_Sub0.
;   - Submode 1 validates the save files (other file); submode 2 formats every inactive file A, copies
;     active flag, death count, quest number, name and hearts of each slot to save slot info, then
;     goes to mode 1. UpdateMode0Demo_Sub2.
;   - The file menu always shows full hearts: the low nibble of the saved HeartValues is replaced by
;     its high nibble (heart containers - 1). UpdateMode0Demo_Sub2@LoopHeart.
; CONTENTS: (generated; L = line in this file)
;   L0051 InitDemo_RunTasks                   code  Mode 0 init: turns video off and runs the current init subphase of the current demo phase.
;   L0059 InitDemo_RunTasks_Phase0_JumpTable  data  TABLE: init subphase handlers of demo phase 0 (title); index = DemoSubphase.
;   L0065 InitDemo_Phase1                     code  Second half of InitDemo_RunTasks: dispatches the init subphase of demo phase 1 (story).
;   L0070 InitDemo_RunTasks_Phase1_JumpTable  data  TABLE: init subphase handlers of demo phase 1 (story); index = DemoSubphase.
;   L0079 UpdateMode0Demo                     code  Mode 0 update: animates the demo while nothing was skipped, then runs the current submode.
;   L0094 UpdateMode0Demo_JumpTable           data  TABLE: mode 0 update submode handlers; index = GameSubmode (0 wait for Start, 1 check save files,
;   L0102 UpdateMode0Demo_Sub0                code  Submode 0: on a Start press ends the demo, blanks the screen and cues the menu palettes.
;   L0119 Mode0Demo_Return                    code  Shared return of UpdateMode0Demo and UpdateMode0Demo_Sub0.
;   L0127 UpdateMode0Demo_Sub2                code  Submode 2: formats inactive file A's, copies save file summaries to save slot info, enters mode 1.
;   L0232 AnimateDemo                         code  Runs the animation handler of the current demo phase and subphase.
;   L0239 AnimateDemo_Phase0_JumpTable        data  TABLE: animation handlers of demo phase 0 (title); index = DemoSubphase (0 wait, 1 palette fade).
;   L0244 AnimateDemo_Phase1                  code  Second half of AnimateDemo: dispatches the animation subphase of demo phase 1 (story).
;   L0250 AnimateDemo_Phase1_JumpTable        data  TABLE: animation handlers of demo phase 1 (story); index = DemoSubphase (0 scroll in, 1 wait,
; ============================================================================

; ----------------------------------------------------------------------------
; Mode 0 init: turns video off and runs the current init subphase of the current demo phase.
; IN:   DemoPhase (0 title, 1 story); DemoSubphase = init subphase 0-2.
; NOTE: The code after each JSR TableJump is the address table; TableJump does not return here.
; ----------------------------------------------------------------------------
InitDemo_RunTasks:
    JSR TurnOffAllVideo
    LDA DemoPhase
    BNE InitDemo_Phase1
    LDA DemoSubphase
    JSR TableJump

; TABLE: init subphase handlers of demo phase 0 (title); index = DemoSubphase.
InitDemo_RunTasks_Phase0_JumpTable:
    .ADDR InitDemoSubphaseClearArtifacts
    .ADDR InitDemoSubphaseTransferTitlePalette
    .ADDR InitDemoSubphasePlayTitleSong

; Second half of InitDemo_RunTasks: dispatches the init subphase of demo phase 1 (story).
InitDemo_Phase1:
    LDA DemoSubphase
    JSR TableJump

; TABLE: init subphase handlers of demo phase 1 (story); index = DemoSubphase.
InitDemo_RunTasks_Phase1_JumpTable:
    .ADDR InitDemoSubphaseClearArtifacts
    .ADDR InitDemoSubphaseTransferStoryPalette
    .ADDR InitDemoSubphaseTransferStoryTiles

; ----------------------------------------------------------------------------
; Mode 0 update: animates the demo while nothing was skipped, then runs the current submode.
; NOTE: AnimateDemo may clear IsUpdatingMode (a phase ended); the submode is then not run this frame.
; ----------------------------------------------------------------------------
UpdateMode0Demo:
    LDA GameSubmode
    BNE @HandleSubmodes         ; Only animate if submode = 0 and HasSkippedDemo = 0
    LDA HasSkippedDemo
    BNE @HandleSubmodes
    JSR AnimateDemo
    LDA IsUpdatingMode          ; If no longer updating,
    BEQ Mode0Demo_Return                    ; then return.

@HandleSubmodes:
    LDA GameSubmode
    JSR TableJump

; TABLE: mode 0 update submode handlers; index = GameSubmode (0 wait for Start, 1 check save files,
; 2 load save slot info).
UpdateMode0Demo_JumpTable:
    .ADDR UpdateMode0Demo_Sub0
    .ADDR UpdateMode0Demo_Sub1
    .ADDR UpdateMode0Demo_Sub2

; ----------------------------------------------------------------------------
; Submode 0: on a Start press ends the demo, blanks the screen and cues the menu palettes.
; ----------------------------------------------------------------------------
UpdateMode0Demo_Sub0:
    LDA ButtonsPressed          ; If Start is not pressed,
    AND #$10
    BEQ Mode0Demo_Return                    ; then return.
    STA HasDemoPatterns ; Else, store $10, because it's convenient and not $A5.
    LDA #$00
    STA SongRequest
    JSR SilenceAllSound
    LDA #$5A
    STA HasSkippedDemo
    INC GameSubmode             ; Go to next submode.
    JSR TurnOffAllVideo
    JSR HideAllSprites
    LDA #$12                    ; Cue transfer record for menu palettes.
    STA TileBufSelector

; Shared return of UpdateMode0Demo and UpdateMode0Demo_Sub0.
Mode0Demo_Return:
    RTS

; ----------------------------------------------------------------------------
; Submode 2: formats inactive file A's, copies save file summaries to save slot info, enters mode 1.
; OUT:  IsSaveSlotActive, DeathCounts, QuestNumbers, SaveSlotHearts, SaveSlotNames filled for slots 0-2;
;       GameMode = 1 with IsUpdatingMode = 0 and GameSubmode = 0; CurSaveSlot = 0.
; ----------------------------------------------------------------------------
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
AnimateDemo:
    LDA DemoPhase
    BNE AnimateDemo_Phase1
    LDA DemoSubphase
    JSR TableJump

; TABLE: animation handlers of demo phase 0 (title); index = DemoSubphase (0 wait, 1 palette fade).
AnimateDemo_Phase0_JumpTable:
    .ADDR AnimateDemoPhase0Subphase0
    .ADDR AnimateDemoPhase0Subphase1

; Second half of AnimateDemo: dispatches the animation subphase of demo phase 1 (story).
AnimateDemo_Phase1:
    LDA DemoSubphase
    JSR TableJump

; TABLE: animation handlers of demo phase 1 (story); index = DemoSubphase (0 scroll in, 1 wait,
; 2 scroll text and items, 3 wait, 4 short wait then restart at phase 0).
AnimateDemo_Phase1_JumpTable:
    .ADDR AnimateDemoPhase1Subphase0
    .ADDR AnimateDemoPhase1Subphase1
    .ADDR AnimateDemoPhase1Subphase2
    .ADDR AnimateDemoPhase1Subphase3
    .ADDR AnimateDemoPhase1Subphase4
