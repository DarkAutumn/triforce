; ============================================================================
; FILE:     bank7/run-game-nmi.asm       BANK: 7 (fixed, $C000-$FFFF)
; UPSTREAM: Z_07.asm 335-550
; SUBJECTS: (none)
; TOPIC:    PCM samples, RunGame, NMI frame handler
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0019 PcmSamples                     data
;   L0022 PlayAreaColumnAddrs            data
;   L0032 RunGame                        code
;   L0045 LoopForever                    code
;   L0048 ClearAllAudioAndVideo          code
;   L0059 TurnOffVideoAndClearArtifacts  code
;   L0067 ClearNameTableWithHiAddr       code
;   L0072 IsrNmi                         code
; ============================================================================

PcmSamples:
    .INCBIN "dat/PcmSamples.dat"

PlayAreaColumnAddrs:
    .BYTE $30, $65, $46, $65, $5C, $65, $72, $65
    .BYTE $88, $65, $9E, $65, $B4, $65, $CA, $65
    .BYTE $E0, $65, $F6, $65, $0C, $66, $22, $66
    .BYTE $38, $66, $4E, $66, $64, $66, $7A, $66
    .BYTE $90, $66, $A6, $66, $BC, $66, $D2, $66
    .BYTE $E8, $66, $FE, $66, $14, $67, $2A, $67
    .BYTE $40, $67, $56, $67, $6C, $67, $82, $67
    .BYTE $98, $67, $AE, $67, $C4, $67, $DA, $67

RunGame:
    LDA #$00
    STA InitializedGame
    LDA #$05
    JSR SwitchBank
    JSR InitSaveRam
    JSR ClearRam
    JSR ClearAllAudioAndVideo
    LDA CurPpuControl_2000      ; Enable NMI.
    ORA #$A0
    STA PpuControl_2000
    STA CurPpuControl_2000

LoopForever:
    JMP LoopForever

ClearAllAudioAndVideo:
    LDA #$00
    STA DmcCounter_4011
    LDA #$0F                    ; Turn on audio except DMC.
    STA ApuStatus_4015

    ; Turn off video.
    ; TODO: except clipping is explicitly disabled. Why?
    LDA #$06
    STA PpuMask_2001

TurnOffVideoAndClearArtifacts:
    JSR HideAllSprites
    JSR ResetPpuRegisters
    JSR TurnOffAllVideo
    LDA #$20
    JSR ClearNameTableWithHiAddr
    LDA #$28

ClearNameTableWithHiAddr:
    LDX #$24
    LDY #$00
    JMP ClearNameTable

IsrNmi:
    LDA CurPpuControl_2000
    LDX SwitchNameTablesReq
    BEQ :+                      ; If need to switch name tables,
    EOR #$02                    ; Switch to the other name table ($2000 <-> $2800).
:
    AND #$7F                    ; Disable NMI.
    STA CurPpuControl_2000
    AND #$7E                    ; Make sure table is 0 or 2.
    STA PpuControl_2000

    ; Set PPUMASK.
    ; Start with curren mask value, but ...
    LDA CurPpuMask_2001
    LDY IsSprite0CheckActive    ; If checking for sprite 0,
    BNE @EnableAllVideo
    LDY TileBufSelector         ; TODO: or TileBufSelector = 0 and [$17] = 0,
    BNE @SetPpuMask
    LDY $17
    BNE @SetPpuMask

@EnableAllVideo:
    ORA #$1E                    ; then make sure all video is on.

@SetPpuMask:
    STA PpuMask_2001
    STA CurPpuMask_2001

    ; Copy all our sprites to OAM.
    LDA #$00
    STA OamAddr_2003
    LDA #$02
    STA SpriteDma_4014

    ; Reset scroll register.
    LDA #$00
    STA PpuScroll_2005
    STA PpuScroll_2005
    LDA #$06
    JSR SwitchBank
    JSR TransferCurTileBuf

    ; Reset PPUADDR.
    ; I believe that the write of $3F00 is unneeded.
    LDA #$3F
    STA PpuAddr_2006
    LDA #$00
    STA PpuAddr_2006
    STA PpuAddr_2006
    STA PpuAddr_2006

@WaitVBlankEnd:
    ; If Sprite 0 Hit is set,
    ; then wait for it to be cleared by the ending of VBLANK.
    LDA PpuStatus_2002
    AND #$40
    BNE @WaitVBlankEnd
    LDA PpuStatus_2002          ; Clear shift register.

    ; Wait for Sprite 0 hit, if needed.
    LDA IsSprite0CheckActive
    BEQ @CheckScroll
    LDA #$05
    JSR SwitchBank
    JSR WaitAndScrollToSplitBottom

@CheckScroll:
    ; If updating a game mode instead of initializing one,
    ; then simply scroll for game modes 0, 5, 9, $B, $C, $13.
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
    LDA PpuStatus_2002
    LDA CurHScroll
    STA PpuScroll_2005
    LDA CurVScroll
    STA PpuScroll_2005
    LDA CurPpuControl_2000
    STA PpuControl_2000

@UpdateTimers:
    ; If paused or in menu, then don't decrement timers.
    LDA MenuState
    ORA Paused
    BNE @CheckInput
    LDX #$26                    ; Advance the stun cycle.
    LDA #$3C
    LDY #$4E
    STX $00
    DEC $00, X
    BPL @DecTimers
    LDA #$09
    STA $00, X
    TYA

@DecTimers:
    TAX                         ; Decrement timers (object and optionally stun timers).

@LoopTimer:
    LDA $00, X
    BEQ :+
    DEC $00, X
:
    DEX
    CPX $00
    BNE @LoopTimer

@CheckInput:
    ; Sprite 0 is used for scrolling. Input isn't checked while scrolling.
    LDA IsSprite0CheckActive
    BNE @ScrambleRandom
    JSR ReadInputs

@ScrambleRandom:
    LDX #$18                    ; Scramble the random array.
    LDY #$0D
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
    LDA IsUpdatingMode
    BNE @Update
    JSR InitializeGameOrMode
    JMP @EnableNMI

@Update:
    JSR UpdateMode

@EnableNMI:
    LDA PpuStatus_2002          ; Enable NMI.
    LDA CurPpuControl_2000
    ORA #$80
    STA PpuControl_2000
    STA CurPpuControl_2000
    RTI
