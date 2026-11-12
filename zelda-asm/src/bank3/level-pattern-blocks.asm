; ============================================================================
; FILE:     bank3/level-pattern-blocks.asm BANK: 3 (switchable, $8000-$BFFF)
; UPSTREAM: Z_03.asm 13-264
; SUBJECTS: Chr
; TOPIC:    CHR pattern (tile graphics) blocks that depend on the level, and the routine that copies
;           them into CHR RAM when a level is loaded: overworld background and sprites, or dungeon
;           background, common dungeon sprites, the level's monster sprites and its boss sprites.
;           Ends with the bank's copy of the reset stub and the interrupt vectors.
; SEE ALSO: bank7/init-mode-dispatch.asm (Mode2Load_Init switches to bank 3 and calls
;           Chr_TransferLevelPatterns); bank2/common-patterns.asm (patterns common to all levels);
;           bank1/copy-code-demo-patterns.asm (title/demo patterns); bank7/reset-mmc1-vectors.asm
;           (Reset_Handler, the original of the stub at the end of this file).
; KEY FACTS:
;   - Chr_TransferLevelPatterns is the only entry point (the bank's only export). It turns video off,
;     then copies 2 blocks when CurLevel = 0 (overworld) or 4 blocks otherwise.
;   - Overworld: Chr_PatternsOWBackground ($0820 bytes) to PPU $1700 and Chr_PatternsOWSprites ($0720 bytes) to PPU
;     $08E0 (Chr_LevelPatternAddrsOW, Chr_LevelPatternSizesOW, Chr_LevelPatternPpuAddrs).
;   - Dungeon: Chr_PatternsUWBackground ($0820) to $1700, Chr_PatternsUWSpritesCommon ($0100) to $08E0, the level's monster
;     sprite set ($0220) to $09E0, the level's boss sprite set ($0400) to $0C00
;     (Chr_TransferLevelPatternsUW, Chr_LevelPatternSizesUW, Chr_LevelPatternPpuAddrsExtra).
;   - Monster sprite set by level (Chr_LevelMonsterPatternAddrs): levels 1, 2, 7 share one; 3, 5, 8
;     another; 4, 6, 9 the third.
;   - Boss sprite set by level (Chr_LevelBossPatternAddrs): levels 1, 2, 5, 7 share one; 3, 4, 6, 8
;     another; level 9 has its own.
;   - The sets are chosen by CurLevel only; the quest is not tested anywhere in this file.
;   - PatternBlockIndex ($51D) counts the blocks copied and also indexes the size and PPU address
;     tables; it is 0 on return (Chr_ResetPatternBlockIndex).
;   - The copy is a plain byte loop with a 16-bit count in [02] (high) and [03] (low)
;     (Chr_TransferPatternBlock_Bank3).
; CONTENTS: (generated; L = line in this file)
;   L0062 Chr_LevelMonsterPatternAddrs       data  TABLE: address of the dungeon monster sprite pattern block ($0220 bytes, PPU $09E0) for each level;
;   L0076 Chr_LevelBossPatternAddrs          data  TABLE: address of the boss sprite pattern block ($0400 bytes, PPU $0C00) for each level;
;   L0090 Chr_CommonPatternAddrsUW           data  TABLE: addresses of the two pattern blocks every dungeon uses; index = PatternBlockIndex (0
;   L0096 Chr_LevelPatternAddrsOW            data  TABLE: addresses of the two overworld pattern blocks; index = PatternBlockIndex (0 background,
;   L0102 Chr_LevelPatternPpuAddrs           data  TABLE: destination PPU address of each pattern block; index = PatternBlockIndex; value = PPU address,
;   L0108 Chr_LevelPatternPpuAddrsExtra      data  TABLE: continuation of Chr_LevelPatternPpuAddrs for PatternBlockIndex 2 (level monster sprites) and 3
;   L0114 Chr_LevelPatternSizesOW            data  TABLE: sizes of the two overworld pattern blocks; index = PatternBlockIndex; value = byte count,
;   L0120 Chr_LevelPatternSizesUW            data  TABLE: sizes of the four dungeon pattern blocks; index = PatternBlockIndex (0 background, 1 common
;   L0132 Chr_TransferLevelPatterns          code  Copies the current level's pattern blocks into CHR RAM: 2 on the overworld, 4 in a dungeon.
;   L0147 Chr_ResetPatternBlockIndex         code  Sets PatternBlockIndex to 0; also the end of the overworld loop above and of the dungeon path below.
;   L0155 Chr_TransferLevelPatternsUW        code  Dungeon path: copies the two common dungeon blocks, then the level's monster and boss sprite blocks.
;   L0174 Chr_FetchCommonPatternAddrUW       code  Fetches the address of common dungeon pattern block PatternBlockIndex (0 or 1).
;   L0189 Chr_FetchPatternAddrAndSizeOW      code  Fetches the address and size of overworld pattern block PatternBlockIndex (0 or 1).
;   L0209 Chr_FetchLevelMonsterPatternAddr   code  Fetches the address of the current level's monster sprite pattern block.
;   L0225 Chr_FetchLevelBossPatternAddr      code  Fetches the address of the current level's boss sprite pattern block.
;   L0241 Chr_FetchSizeAndTransferPatternUW  code  Fetches the size of dungeon pattern block PatternBlockIndex, then falls through to copy it.
;   L0257 Chr_TransferPatternBlock_Bank3     code  Copies one pattern block from ROM to the PPU address of entry PatternBlockIndex, then advances it.
;   L0299 Chr_PatternsUWBackground           data  DATA: dungeon background patterns, $0820 bytes (130 tiles of 16 bytes), copied to PPU $1700.
;   L0303 Chr_PatternsOWBackground           data  DATA: overworld background patterns, $0820 bytes (130 tiles), copied to PPU $1700.
;   L0307 Chr_PatternsOWSprites              data  DATA: overworld sprite patterns, $0720 bytes (114 tiles), copied to PPU $08E0.
;   L0311 Chr_PatternsUWMonsters358          data  DATA: dungeon monster sprite patterns for levels 3, 5 and 8, $0220 bytes (34 tiles), copied to $09E0.
;   L0315 Chr_PatternsUWMonsters469          data  DATA: dungeon monster sprite patterns for levels 4, 6 and 9, $0220 bytes (34 tiles), copied to $09E0.
;   L0319 Chr_PatternsUWSpritesCommon        data  DATA: sprite patterns common to all dungeons, $0100 bytes (16 tiles), copied to PPU $08E0.
;   L0323 Chr_PatternsUWMonsters127          data  DATA: dungeon monster sprite patterns for levels 1, 2 and 7, $0220 bytes (34 tiles), copied to $09E0.
;   L0327 Chr_PatternsUWBoss1257             data  DATA: boss sprite patterns for levels 1, 2, 5 and 7, $0400 bytes (64 tiles), copied to PPU $0C00.
;   L0331 Chr_PatternsUWBoss3468             data  DATA: boss sprite patterns for levels 3, 4, 6 and 8, $0400 bytes (64 tiles), copied to PPU $0C00.
;   L0335 Chr_PatternsUWBoss9                data  DATA: boss sprite patterns for level 9, $0400 bytes (64 tiles), copied to PPU $0C00.
; ============================================================================

; TABLE: address of the dungeon monster sprite pattern block ($0220 bytes, PPU $09E0) for each level;
;   index = CurLevel (0-9); value = ROM address. Entry 0 (overworld) is never read.
Chr_LevelMonsterPatternAddrs:
    .ADDR Chr_PatternsUWMonsters127           ; 0 (unused)
    .ADDR Chr_PatternsUWMonsters127           ; level 1
    .ADDR Chr_PatternsUWMonsters127           ; level 2
    .ADDR Chr_PatternsUWMonsters358           ; level 3
    .ADDR Chr_PatternsUWMonsters469           ; level 4
    .ADDR Chr_PatternsUWMonsters358           ; level 5
    .ADDR Chr_PatternsUWMonsters469           ; level 6
    .ADDR Chr_PatternsUWMonsters127           ; level 7
    .ADDR Chr_PatternsUWMonsters358           ; level 8
    .ADDR Chr_PatternsUWMonsters469           ; level 9

; TABLE: address of the boss sprite pattern block ($0400 bytes, PPU $0C00) for each level;
;   index = CurLevel (0-9); value = ROM address. Entry 0 (overworld) is never read.
Chr_LevelBossPatternAddrs:
    .ADDR Chr_PatternsUWBoss1257      ; 0 (unused)
    .ADDR Chr_PatternsUWBoss1257      ; level 1
    .ADDR Chr_PatternsUWBoss1257      ; level 2
    .ADDR Chr_PatternsUWBoss3468      ; level 3
    .ADDR Chr_PatternsUWBoss3468      ; level 4
    .ADDR Chr_PatternsUWBoss1257      ; level 5
    .ADDR Chr_PatternsUWBoss3468      ; level 6
    .ADDR Chr_PatternsUWBoss1257      ; level 7
    .ADDR Chr_PatternsUWBoss3468      ; level 8
    .ADDR Chr_PatternsUWBoss9         ; level 9

; TABLE: addresses of the two pattern blocks every dungeon uses; index = PatternBlockIndex (0
;   background, 1 sprites); value = ROM address.
Chr_CommonPatternAddrsUW:
    .ADDR Chr_PatternsUWBackground
    .ADDR Chr_PatternsUWSpritesCommon

; TABLE: addresses of the two overworld pattern blocks; index = PatternBlockIndex (0 background,
;   1 sprites); value = ROM address.
Chr_LevelPatternAddrsOW:
    .ADDR Chr_PatternsOWBackground
    .ADDR Chr_PatternsOWSprites

; TABLE: destination PPU address of each pattern block; index = PatternBlockIndex; value = PPU address,
;   high byte first. The dungeon's blocks 2 and 3 read on into Chr_LevelPatternPpuAddrsExtra.
Chr_LevelPatternPpuAddrs:
    .DBYT $1700
    .DBYT $08E0

; TABLE: continuation of Chr_LevelPatternPpuAddrs for PatternBlockIndex 2 (level monster sprites) and 3
;   (boss sprites); value = PPU address, high byte first.
Chr_LevelPatternPpuAddrsExtra:
    .DBYT $09E0
    .DBYT $0C00

; TABLE: sizes of the two overworld pattern blocks; index = PatternBlockIndex; value = byte count,
;   high byte first.
Chr_LevelPatternSizesOW:
    .DBYT $0820
    .DBYT $0720

; TABLE: sizes of the four dungeon pattern blocks; index = PatternBlockIndex (0 background, 1 common
;   sprites, 2 level monster sprites, 3 boss sprites); value = byte count, high byte first.
Chr_LevelPatternSizesUW:
    .DBYT $0820
    .DBYT $0100
    .DBYT $0220
    .DBYT $0400

; ----------------------------------------------------------------------------
; Copies the current level's pattern blocks into CHR RAM: 2 on the overworld, 4 in a dungeon.
; IN:   CurLevel (0 = overworld).
; OUT:  PatternBlockIndex = 0; video is off (Ppu_TurnOffAllVideo); uses [00]-[03].
; NOTE: Called only from Mode2Load_Init, when GameSubmode = 0 (the first step of loading a level).
; ----------------------------------------------------------------------------
Chr_TransferLevelPatterns:
    JSR Ppu_TurnOffAllVideo
    LDA PpuStatus
    JSR Chr_ResetPatternBlockIndex
    LDA CurLevel
    BNE Chr_TransferLevelPatternsUW    ; Go handle UW levels.

@LoopBlockOW:
    JSR Chr_FetchPatternAddrAndSizeOW
    JSR Chr_TransferPatternBlock_Bank3
    LDA PatternBlockIndex
    CMP #$02                    ; There are two blocks.
    BNE @LoopBlockOW            ; If we haven't transferred the second, then go do so.

; Sets PatternBlockIndex to 0; also the end of the overworld loop above and of the dungeon path below.
Chr_ResetPatternBlockIndex:
    LDA #$00
    STA PatternBlockIndex
    RTS

; ----------------------------------------------------------------------------
; Dungeon path: copies the two common dungeon blocks, then the level's monster and boss sprite blocks.
; ----------------------------------------------------------------------------
Chr_TransferLevelPatternsUW:
    JSR Chr_FetchCommonPatternAddrUW
    JSR Chr_FetchSizeAndTransferPatternUW
    LDA PatternBlockIndex
    CMP #$02
    BNE Chr_TransferLevelPatternsUW    ; Loop until blocks 0 and 1 are copied.

    ; The two blocks common to all dungeons (background and sprites) are copied.
    ; Now copy the level's monster sprites (block 2) and boss sprites (block 3).
    JSR Chr_FetchLevelMonsterPatternAddr
    JSR Chr_FetchSizeAndTransferPatternUW
    JSR Chr_FetchLevelBossPatternAddr
    JSR Chr_FetchSizeAndTransferPatternUW
    JMP Chr_ResetPatternBlockIndex

; ----------------------------------------------------------------------------
; Fetches the address of common dungeon pattern block PatternBlockIndex (0 or 1).
; OUT:  [00:01] = source address.
; ----------------------------------------------------------------------------
Chr_FetchCommonPatternAddrUW:
    LDA PatternBlockIndex
    ASL
    TAX
    LDA Chr_CommonPatternAddrsUW, X
    STA $00
    INX
    LDA Chr_CommonPatternAddrsUW, X
    STA $01
    RTS

; ----------------------------------------------------------------------------
; Fetches the address and size of overworld pattern block PatternBlockIndex (0 or 1).
; OUT:  [00:01] = source address; [02] = size high byte, [03] = size low byte.
; ----------------------------------------------------------------------------
Chr_FetchPatternAddrAndSizeOW:
    LDA PatternBlockIndex
    ASL
    TAX
    LDA Chr_LevelPatternAddrsOW, X
    STA $00
    LDA Chr_LevelPatternSizesOW, X
    STA $02
    INX
    LDA Chr_LevelPatternAddrsOW, X
    STA $01
    LDA Chr_LevelPatternSizesOW, X
    STA $03
    RTS

; ----------------------------------------------------------------------------
; Fetches the address of the current level's monster sprite pattern block.
; IN:   CurLevel (1-9).
; OUT:  [00:01] = source address.
; ----------------------------------------------------------------------------
Chr_FetchLevelMonsterPatternAddr:
    LDA CurLevel
    ASL
    TAX
    LDA Chr_LevelMonsterPatternAddrs, X
    STA $00
    INX
    LDA Chr_LevelMonsterPatternAddrs, X
    STA $01
    RTS

; ----------------------------------------------------------------------------
; Fetches the address of the current level's boss sprite pattern block.
; IN:   CurLevel (1-9).
; OUT:  [00:01] = source address.
; ----------------------------------------------------------------------------
Chr_FetchLevelBossPatternAddr:
    LDA CurLevel
    ASL
    TAX
    LDA Chr_LevelBossPatternAddrs, X
    STA $00
    INX
    LDA Chr_LevelBossPatternAddrs, X
    STA $01
    RTS

; ----------------------------------------------------------------------------
; Fetches the size of dungeon pattern block PatternBlockIndex, then falls through to copy it.
; IN:   [00:01] = source address.
; OUT:  [02] = size high byte, [03] = size low byte (consumed by the copy below).
; ----------------------------------------------------------------------------
Chr_FetchSizeAndTransferPatternUW:
    LDA PatternBlockIndex
    ASL
    TAX
    LDA Chr_LevelPatternSizesUW, X
    STA $02
    INX
    LDA Chr_LevelPatternSizesUW, X
    STA $03

; ----------------------------------------------------------------------------
; Copies one pattern block from ROM to the PPU address of entry PatternBlockIndex, then advances it.
; IN:   [00:01] = source address; [02] = size high byte, [03] = size low byte.
; OUT:  PatternBlockIndex incremented; [00:01] past the block; [02], [03] = 0.
; NOTE: Writes straight to PpuAddr/PpuData, so video must be off (the entry routine turns it off).
; ----------------------------------------------------------------------------
Chr_TransferPatternBlock_Bank3:
    LDA PatternBlockIndex
    ASL
    TAX
    LDA Chr_LevelPatternPpuAddrs, X
    STA PpuAddr
    INX
    LDA Chr_LevelPatternPpuAddrs, X
    STA PpuAddr
    LDY #$00                    ; Start copying.

@LoopCopy:
    LDA ($00), Y                ; Transfer 1 byte from source pattern block in ROM to PPU.
    STA PpuData

    ; Increment source address.
    LDA $00
    CLC
    ADC #$01
    STA $00
    LDA $01
    ADC #$00
    STA $01

    ; Decrement count.
    LDA $03
    SEC
    SBC #$01
    STA $03
    LDA $02
    SBC #$00
    STA $02

    ; If count is not zero, go copy more.
    LDA $02
    BNE @LoopCopy
    LDA $03
    BNE @LoopCopy
    INC PatternBlockIndex       ; Mark this block finished, and we're ready for the next one.
    RTS

; DATA: dungeon background patterns, $0820 bytes (130 tiles of 16 bytes), copied to PPU $1700.
Chr_PatternsUWBackground:
    .INCBIN "dat/PatternBlockUWBG.dat"

; DATA: overworld background patterns, $0820 bytes (130 tiles), copied to PPU $1700.
Chr_PatternsOWBackground:
    .INCBIN "dat/PatternBlockOWBG.dat"

; DATA: overworld sprite patterns, $0720 bytes (114 tiles), copied to PPU $08E0.
Chr_PatternsOWSprites:
    .INCBIN "dat/PatternBlockOWSP.dat"

; DATA: dungeon monster sprite patterns for levels 3, 5 and 8, $0220 bytes (34 tiles), copied to $09E0.
Chr_PatternsUWMonsters358:
    .INCBIN "dat/PatternBlockUWSP358.dat"

; DATA: dungeon monster sprite patterns for levels 4, 6 and 9, $0220 bytes (34 tiles), copied to $09E0.
Chr_PatternsUWMonsters469:
    .INCBIN "dat/PatternBlockUWSP469.dat"

; DATA: sprite patterns common to all dungeons, $0100 bytes (16 tiles), copied to PPU $08E0.
Chr_PatternsUWSpritesCommon:
    .INCBIN "dat/PatternBlockUWSP.dat"

; DATA: dungeon monster sprite patterns for levels 1, 2 and 7, $0220 bytes (34 tiles), copied to $09E0.
Chr_PatternsUWMonsters127:
    .INCBIN "dat/PatternBlockUWSP127.dat"

; DATA: boss sprite patterns for levels 1, 2, 5 and 7, $0400 bytes (64 tiles), copied to PPU $0C00.
Chr_PatternsUWBoss1257:
    .INCBIN "dat/PatternBlockUWSPBoss1257.dat"

; DATA: boss sprite patterns for levels 3, 4, 6 and 8, $0400 bytes (64 tiles), copied to PPU $0C00.
Chr_PatternsUWBoss3468:
    .INCBIN "dat/PatternBlockUWSPBoss3468.dat"

; DATA: boss sprite patterns for level 9, $0400 bytes (64 tiles), copied to PPU $0C00.
Chr_PatternsUWBoss9:
    .INCBIN "dat/PatternBlockUWSPBoss9.dat"

.SEGMENT "BANK_03_ISR"




; Bank 3's copy of the reset stub, at $BF50 (unlabeled machine code; the same instructions as
; Reset_Handler, Mmc1_SetControl and Mmc1_SwitchBank in bank7/reset-mmc1-vectors.asm, except that its two JSRs
; target this copy's own subroutines at $BF98 and $BFAC instead of $FF98 and $FFAC).
; Whatever bank is mapped at $8000 on reset, it runs the same thing: SEI, CLD, NMI off, stack $01FF,
; wait for two vblanks, reset the MMC1 shift registers, set control $0F, CHR bank 0, switch to
; bank 7, then JMP $E440 in the fixed bank (Reset_RunGame: Reset_Handler's JMP Reset_RunGame has the same bytes).
;   $BF50: reset entry (first 72 bytes).
;   $BF98: writes A to the MMC1 control register at $8000, 5 serial bits (next 20 bytes).
;   $BFAC: writes A to the MMC1 PRG bank register at $E000, 5 serial bits (last 20 bytes).
    .BYTE $78, $D8, $A9, $00, $8D, $00, $20, $A2
    .BYTE $FF, $9A, $AD, $02, $20, $29, $80, $F0
    .BYTE $F9, $AD, $02, $20, $29, $80, $F0, $F9
    .BYTE $09, $FF, $8D, $00, $80, $8D, $00, $A0
    .BYTE $8D, $00, $C0, $8D, $00, $E0, $A9, $0F
    .BYTE $20, $98, $BF, $A9, $00, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $4A, $8D, $00, $A0, $4A, $8D, $00, $A0
    .BYTE $A9, $07, $20, $AC, $BF, $4C, $40, $E4
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $4A, $8D, $00, $80, $4A
    .BYTE $8D, $00, $80, $60, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $4A
    .BYTE $8D, $00, $E0, $4A, $8D, $00, $E0, $60

.SEGMENT "BANK_03_VEC"




; Interrupt vectors at $BFFA: NMI = $E484 (Nmi_Isr; the fixed bank's own vectors hold the same
; address), reset = $BF50 (the stub above), IRQ = $BFF0 ($FF filler between the stub and the
; vectors; the fixed bank's IRQ vector is $FFF0).
    .BYTE $84, $E4, $50, $BF, $F0, $BF
