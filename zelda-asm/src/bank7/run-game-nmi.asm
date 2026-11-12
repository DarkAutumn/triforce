; ============================================================================
; FILE:     bank7/run-game-nmi.asm       BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 335-550
; SUBJECTS: (none)
; TOPIC:    DMC sample data, the play area column address table, the boot routine RunGame, and IsrNmi:
;           the NMI handler that runs one whole game frame (PPU, timers, input, RNG, audio, mode code).
; SEE ALSO: bank7/reset-mmc1-vectors.asm (IsrReset jumps to RunGame; the NMI vector);
;           bank7/init-mode-dispatch.asm (InitializeGameOrMode); bank7/update-mode-dispatch.asm
;           (UpdateMode); bank7/mode5-play-update.asm (Select pause, submenu);
;           bank7/ppu-tablejump-input.asm (ReadInputs, ClearNameTable);
;           bank6/transfer-buf-dispatch.asm (TransferCurTileBuf); bank5/sprite0-split-scroll.asm (the split);
;           bank5/save-ram-clear-ram.asm (ClearRam seeds Random); bank0/drive-audio-tune0.asm (DriveAudio).
; KEY FACTS:
;   - All game logic runs inside the NMI: RunGame ends in an endless loop (LoopForever) and IsrNmi does
;     one frame per vertical blank, with the NMI disabled until it finishes. A frame that runs past the next
;     vertical blank skips that NMI, so timers, RNG and FrameCounter do not advance on a lag frame (IsrNmi).
;   - Order inside IsrNmi: PPU control/mask, sprite DMA, transfer buffer (bank 6), sprite-0 split wait
;     (bank 5), scroll, timers, controller read, RNG step, audio (bank 0), INC FrameCounter, then the
;     mode's init routine (IsUpdatingMode = 0) or update routine (nonzero).
;   - Frame timers $27-$3C (DoorTimer, ObjTimer[0..$13], FluteTimer) lose 1 per frame down to 0
;     (IsrNmi@DecTimers).
;   - Long timers $3D-$4E (ObjStunTimer[0..$0C], ChaseLongTimer, MonstersFromEdgesLongTimer,
;     SwordBlockedLongTimer, RedLeeverLongTimer, [4E]) lose 1 every 10th frame, when LongTimerCycle
;     wraps from 0 to 9 (IsrNmi@UpdateTimers).
;   - No timer ticks while SubmenuState or PauseState is nonzero (IsrNmi@UpdateTimers); RNG, FrameCounter,
;     audio and the mode code still run every frame.
;   - Random ($18-$24, 13 bytes) steps once per frame, also while paused: new bit = bit 1 of $18 XOR bit 1
;     of $19, rotated right through all 13 bytes (IsrNmi@ScrambleRandom). It does not depend on input.
;   - The controllers are not read while IsSprite0CheckActive is set (room scroll): ButtonsPressed and
;     ButtonsDown keep their last values (IsrNmi@CheckInput).
;   - The timers, the controller read and the RNG step all happen before the mode code of the same frame.
;   - PlayAreaColumnAddrs: column n of the play area tile map is at $6530 + n * $16 (22 rows per column).
;   - Rendering is turned on by IsrNmi itself: every frame with the sprite-0 check active, or with the
;     dynamic transfer buffer selected and [17] = 0. [17] (no symbol) is 1 from InitMode3_Sub0 to
;     InitMode3_Sub8, which keeps the screen off during the mode 3 init (IsrNmi@EnableAllVideo).
; CONTENTS: (generated; L = line in this file)
;   L0048 PcmSamples                     data  DATA: DMC (delta modulation) sample bytes, $2400 bytes at $C000-$E3FF, for the sound driver in bank 0.
;   L0054 PlayAreaColumnAddrs            data  TABLE: address of each tile column of PlayAreaTiles; index = column 0-$1F (times 2); value = address, lo/hi.
;   L0070 RunGame                        code  Boot entry after reset: clears RAM, sound and video, enables the NMI, then idles forever.
;   L0084 LoopForever                    code  The main thread idles here forever; every frame is done by IsrNmi.
;   L0091 ClearAllAudioAndVideo          code  Resets the DMC level, enables the four tone/noise channels, blanks the screen and both name tables.
;   L0109 TurnOffVideoAndClearArtifacts  code  Hides all sprites, resets scroll and PPU control, turns rendering off, blanks both name tables.
;   L0121 ClearNameTableWithHiAddr       code  Fills the name table at address A * $100 with blank tile $24 and attribute byte 0.
;   L0140 IsrNmi                         code  NMI handler: runs one complete game frame, once per vertical blank.
; ============================================================================

; DATA: DMC (delta modulation) sample bytes, $2400 bytes at $C000-$E3FF, for the sound driver in bank 0.
PcmSamples:
    .INCBIN "dat/PcmSamples.dat"

; TABLE: address of each tile column of PlayAreaTiles; index = column 0-$1F (times 2); value = address, lo/hi.
; Entry n = $6530 + n * $16: a column holds 22 tiles (rows top to bottom). Users index it with
; (pixel X AND $F8) / 4.
PlayAreaColumnAddrs:
    .BYTE $30, $65, $46, $65, $5C, $65, $72, $65        ; columns $00-$03
    .BYTE $88, $65, $9E, $65, $B4, $65, $CA, $65        ; columns $04-$07
    .BYTE $E0, $65, $F6, $65, $0C, $66, $22, $66        ; columns $08-$0B
    .BYTE $38, $66, $4E, $66, $64, $66, $7A, $66        ; columns $0C-$0F
    .BYTE $90, $66, $A6, $66, $BC, $66, $D2, $66        ; columns $10-$13
    .BYTE $E8, $66, $FE, $66, $14, $67, $2A, $67        ; columns $14-$17
    .BYTE $40, $67, $56, $67, $6C, $67, $82, $67        ; columns $18-$1B
    .BYTE $98, $67, $AE, $67, $C4, $67, $DA, $67        ; columns $1C-$1F

; ----------------------------------------------------------------------------
; Boot entry after reset: clears RAM, sound and video, enables the NMI, then idles forever.
; NOTE: Clearing IsGameInitialized makes the first NMI copy the common code and data to RAM
;       (InitializeGameOrMode) before any mode runs. ClearRam zeroes GameMode, so the game starts in
;       mode 0 with IsUpdatingMode = 0 (its init routine).
; ----------------------------------------------------------------------------
RunGame:
    LDA #$00
    STA IsGameInitialized
    LDA #$05
    JSR SwitchBank
    JSR InitSaveRam
    JSR ClearRam
    JSR ClearAllAudioAndVideo
    LDA CurPpuControl           ; Enable the NMI (bit 7) and 8x16 sprites (bit 5).
    ORA #$A0
    STA PpuControl
    STA CurPpuControl

; The main thread idles here forever; every frame is done by IsrNmi.
LoopForever:
    JMP LoopForever

; ----------------------------------------------------------------------------
; Resets the DMC level, enables the four tone/noise channels, blanks the screen and both name tables.
; NOTE: Falls into TurnOffVideoAndClearArtifacts.
; ----------------------------------------------------------------------------
ClearAllAudioAndVideo:
    LDA #$00
    STA DmcCounter
    LDA #$0F                    ; Enable the two squares, triangle and noise; DMC off.
    STA ApuStatus

    ; PPU mask $06: background and sprite rendering off; only the two "show the left 8 pixels" bits
    ; are set, which have no effect while rendering is off. CurPpuMask is not updated here;
    ; TurnOffAllVideo below writes 0 to both.
    LDA #$06
    STA PpuMask

; ----------------------------------------------------------------------------
; Hides all sprites, resets scroll and PPU control, turns rendering off, blanks both name tables.
; OUT:  both name tables filled with tile $24 (blank), their attribute tables with 0.
; NOTE: Called by mode init code before a screen is rebuilt. Falls into ClearNameTableWithHiAddr for the
;       second name table.
; ----------------------------------------------------------------------------
TurnOffVideoAndClearArtifacts:
    JSR HideAllSprites
    JSR ResetPpuRegisters
    JSR TurnOffAllVideo
    LDA #$20
    JSR ClearNameTableWithHiAddr
    LDA #$28

; ----------------------------------------------------------------------------
; Fills the name table at address A * $100 with blank tile $24 and attribute byte 0.
; IN:   A = high byte of the name table address ($20 or $28).
; ----------------------------------------------------------------------------
ClearNameTableWithHiAddr:
    LDX #$24
    LDY #$00
    JMP ClearNameTable

; ----------------------------------------------------------------------------
; NMI handler: runs one complete game frame, once per vertical blank.
; NOTE: Steps, in order: (1) PPU control with the NMI disabled, PPU mask; (2) sprite DMA from $0200;
;       (3) send the selected transfer buffer to the PPU (bank 6); (4) with IsSprite0CheckActive, wait
;       for the sprite-0 hit and split the scroll (bank 5); (5) while a mode is updating, write the
;       scroll for modes 0, 5, 9, $B, $C, $13; (6) tick the timers unless paused or in the submenu;
;       (7) read the controllers unless IsSprite0CheckActive; (8) step Random; (9) DriveAudio (bank 0);
;       (10) INC FrameCounter;
;       (11) InitializeGameOrMode when IsUpdatingMode = 0, else UpdateMode; (12) re-enable the NMI.
; NOTE: The NMI stays disabled from step 1 to step 12 and PpuStatus is read before it is re-enabled, so
;       a frame that takes too long simply loses the next NMI (no re-entry).
; NOTE: Uses [00] as scratch for the timer loop and the RNG step. Registers are not saved: the main
;       thread is the empty loop at LoopForever.
; ----------------------------------------------------------------------------
IsrNmi:
    LDA CurPpuControl
    LDX SwitchNameTablesReq
    BEQ :+                      ; If a name table switch was requested,
    EOR #$02                    ; toggle the base name table bit ($2000 <-> $2800).
:
    AND #$7F                    ; Disable the NMI (bit 7) for the rest of the frame.
    STA CurPpuControl
    AND #$7E                    ; Also clear bit 0 in the register only: base name table $2000 or $2800.
    STA PpuControl

    ; PPU mask: start from the current value and turn all rendering on ($1E: background, sprites and
    ; both left-column bits) when the sprite-0 check is active, or when the buffer to transfer is the
    ; dynamic one (TileBufSelector = 0) and [17] = 0. Otherwise the mask stays as it is, so a screen
    ; that a mode turned off stays off while one of the static transfer buffers is being sent.
    ; [17] has no symbol. InitMode3_Sub0 sets it to 1 and InitMode3_Sub8 back to 0 (the only two
    ; writes found; ClearRam zeroes it at reset), so the screen stays off through the nine init steps
    ; of mode 3 even on frames that send the dynamic buffer.
    LDA CurPpuMask
    LDY IsSprite0CheckActive
    BNE @EnableAllVideo
    LDY TileBufSelector
    BNE @SetPpuMask
    LDY $17
    BNE @SetPpuMask

@EnableAllVideo:
    ORA #$1E

@SetPpuMask:
    STA PpuMask
    STA CurPpuMask

    ; Copy the sprite buffer (page $02, Sprites) to the PPU's OAM by DMA.
    LDA #$00
    STA OamAddr
    LDA #$02
    STA SpriteDma

    ; Zero the scroll register, then send this frame's transfer buffer (TileBufSelector) to the PPU.
    ; TransferCurTileBuf also resets TileBufSelector, SwitchNameTablesReq and the dynamic buffer.
    LDA #$00
    STA PpuScroll
    STA PpuScroll
    LDA #$06
    JSR SwitchBank
    JSR TransferCurTileBuf

    ; Park the PPU address: write $3F00, then $0000.
    LDA #$3F
    STA PpuAddr
    LDA #$00
    STA PpuAddr
    STA PpuAddr
    STA PpuAddr

@WaitVBlankEnd:
    ; If the sprite-0 hit flag (bit 6) is still set from the last frame, wait until the PPU clears it
    ; at the end of vertical blank.
    LDA PpuStatus
    AND #$40
    BNE @WaitVBlankEnd
    LDA PpuStatus               ; Reset the PPU address/scroll write latch.

    ; With the sprite-0 check active, wait for this frame's sprite-0 hit (below the status bar) and
    ; set the scroll for the part of the screen under it.
    LDA IsSprite0CheckActive
    BEQ @CheckScroll
    LDA #$05
    JSR SwitchBank
    JSR WaitAndScrollToSplitBottom

@CheckScroll:
    ; Write CurHScroll/CurVScroll and CurPpuControl to the PPU only when a mode is updating (not
    ; initializing) and the mode is 0, 5, 9, $B, $C or $13.
    LDA IsUpdatingMode
    BEQ @UpdateTimers
    LDA GameMode
    BEQ @SetScroll
    CMP #$05
    BEQ @SetScroll
    CMP #$09
    BEQ @SetScroll
    CMP #$0B
    BEQ @SetScroll
    CMP #$0C
    BEQ @SetScroll
    CMP #$13
    BNE @UpdateTimers

@SetScroll:
    LDA PpuStatus               ; Reset the write latch.
    LDA CurHScroll
    STA PpuScroll
    LDA CurVScroll
    STA PpuScroll
    LDA CurPpuControl
    STA PpuControl

@UpdateTimers:
    ; No timer ticks while the submenu is open or the game is paused.
    LDA SubmenuState
    ORA PauseState
    BNE @CheckInput

    ; [00] := $26, the address of LongTimerCycle and the lower bound of the loop below.
    ; A = $3C: last frame timer (FluteTimer). Y = $4E: last long timer.
    ; LongTimerCycle counts 9 down to 0; on the frame it goes below 0 it is reset to 9 and the loop
    ; starts at $4E instead of $3C, so the long timers $3D-$4E tick once every 10 frames.
    LDX #$26
    LDA #$3C
    LDY #$4E
    STX $00
    DEC $00, X                  ; DEC LongTimerCycle ($26).
    BPL @DecTimers
    LDA #$09
    STA $00, X
    TYA

@DecTimers:
    TAX                         ; X = address of the highest timer to tick this frame.

@LoopTimer:
    ; Decrement every nonzero timer from X down to $27 (DoorTimer). A timer at 0 stays at 0.
    LDA $00, X
    BEQ :+
    DEC $00, X
:
    DEX
    CPX $00                     ; Stop at $26.
    BNE @LoopTimer

@CheckInput:
    ; The controllers are not read while the sprite-0 check is active (room scroll and other split
    ; screens): ButtonsPressed and ButtonsDown keep the values of the last read.
    LDA IsSprite0CheckActive
    BNE @ScrambleRandom
    JSR ReadInputs

@ScrambleRandom:
    ; Step the 13-byte random shift register Random ($18-$24): carry := (bit 1 of $18) XOR (bit 1 of
    ; $19); then rotate $18, $19, ... $24 right in turn, the carry entering bit 7 of $18 and each
    ; byte's bit 0 entering bit 7 of the next. Runs every frame, paused or not.
    LDX #$18
    LDY #$0D                    ; 13 bytes.
    LDA $00, X
    AND #$02
    STA $00
    LDA $01, X
    AND #$02
    EOR $00
    CLC
    BEQ @LoopRandom
    SEC

@LoopRandom:
    ROR $00, X
    INX
    DEY
    BNE @LoopRandom
    LDA #$00
    JSR SwitchBank
    JSR DriveAudio
    INC FrameCounter

    ; Run the game mode: its init routine until BeginUpdateMode sets IsUpdatingMode, then its update.
    LDA IsUpdatingMode
    BNE @Update
    JSR InitializeGameOrMode
    JMP @EnableNMI

@Update:
    JSR UpdateMode

@EnableNMI:
    LDA PpuStatus               ; Clear the vertical blank flag so that enabling the NMI cannot fire it.
    LDA CurPpuControl
    ORA #$80
    STA PpuControl
    STA CurPpuControl
    RTI
