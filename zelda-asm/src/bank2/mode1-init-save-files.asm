; ============================================================================
; FILE:     bank2/mode1-init-save-files.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2224-2554
; SUBJECTS: (none)
; TOPIC:    Mode 1 init; save file validate, checksum, format
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0024 Mode1SlotLineTransferBuf                   data
;   L0030 Mode1DeathCountsTransferBuf                data
;   L0035 LinkColors                                 data
;   L0038 InitMode1_Full                             code
;   L0043 InitMode1_Full_JumpTable                   data
;   L0052 UpdateMode0Demo_Sub1                       code
;   L0121 CalculateFileAChecksum                     code  Returns:
;   L0165 AddATo0F0E                                 code
;   L0174 FormatFileA                                code
;   L0240 CalculateAndStoreFileBChecksumUncommitted  code
;   L0258 CalculateFileBChecksum                     code  Returns:
;   L0302 AddAToCFCE                                 code
;   L0311 FormatFileB                                code
; ============================================================================

Mode1SlotLineTransferBuf:
    .BYTE $21, $09, $11, $24, $24, $24, $24, $24
    .BYTE $24, $24, $24, $62, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $21, $32, $08, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $FF

Mode1DeathCountsTransferBuf:
    .BYTE $21, $89, $03, $24, $24, $01, $21, $E9
    .BYTE $03, $24, $24, $01, $22, $49, $03, $24
    .BYTE $24, $01, $FF

LinkColors:
    .BYTE $29, $32, $16

InitMode1_Full:
    JSR TurnOffAllVideo
    LDA GameSubmode
    JSR TableJump

InitMode1_Full_JumpTable:
    .ADDR UpdateMode0Demo_Sub1
    .ADDR InitMode1_Sub1
    .ADDR InitMode1_Sub2
    .ADDR InitMode1_FillAndTransferSlotTiles
    .ADDR InitMode1_FillAndTransferSlotTiles
    .ADDR InitMode1_FillAndTransferSlotTiles
    .ADDR InitMode1_Sub6

UpdateMode0Demo_Sub1:
    ; For each save slot:
    ;   If file B is uncommitted and valid, then
    ;     Copy file B to file A (also marks file B committed)
    ;   If file A's static markers are wrong or file is invalid, then
    ;     Format file A
    ;
    ; This is also mode 1 submode 0 init.
    JSR TurnOffAllVideo
    LDA #$00                    ; Reset CurSaveSlot.
    STA CurSaveSlot             ; For every save file B (3):

@LoopSlot:
    LDY CurSaveSlot
    LDA IsSaveFileBCommitted, Y
    BNE @CheckFileA
    JSR FetchFileBAddressSet
    JSR CalculateFileBChecksum
    LDA CurSaveSlot
    ASL
    TAY
    LDA FileBChecksums, Y       ; Does the checksum match?
    CMP $CE
    BNE @CheckFileA
    INY
    LDA FileBChecksums, Y
    CMP $CF
    BNE @CheckFileA             ; If not, then go check save file A.
    JSR CopyFileBToFileA
    JMP @NextSlot               ; Go process the next slot.

@CheckFileA:
    ; Calculate the checksum of the save file.
    ; It's only a sum.
    JSR FetchFileAAddressSet
    JSR CalculateFileAChecksum
    LDY CurSaveSlot
    LDA SaveFileOpenMarkers, Y  ; Are the static markers intact?
    CMP #$5A
    BNE @FormatA
    LDA SaveFileCloseMarkers, Y
    CMP #$A5
    BNE @FormatA                ; If not, then go format the file.
    LDA CurSaveSlot
    ASL
    TAY
    LDA FileAChecksums, Y       ; Does the checksum match?
    CMP $0E
    BNE @FormatA
    INY
    LDA FileAChecksums, Y
    CMP $0F
    BEQ @NextSlot               ; If not, then format the file.

@FormatA:
    JSR FetchFileAAddressSet
    JSR FormatFileA

@NextSlot:
    INC CurSaveSlot             ; Advance to the next save slot.
    LDA CurSaveSlot
    CMP #$03
    BNE @LoopSlot
    INC GameSubmode             ; After checking every file, go to the next submode.
    RTS

; Returns:
; [0F:0E]: checksum
;
CalculateFileAChecksum:
    LDA #$00                    ; Reset the sum.
    STA $0E
    STA $0F
    LDY #$07                    ; Sum the name (8 bytes).

@SumName:
    LDA ($04), Y
    JSR AddATo0F0E
    DEY
    BPL @SumName
    LDY #$27                    ; Add the Items block ($28 bytes) to [$0F:0E].

@SumItems:
    LDA ($00), Y
    JSR AddATo0F0E
    DEY
    BPL @SumItems
    LDA #$80                    ; Will count $180 with [01:00].
    STA $01
    LDA #$01
    STA $00
    LDY #$00                    ; Add World Flags ($180 bytes) to [$0F:0E].

@SumWorldFlags:
    LDA ($02), Y
    JSR AddATo0F0E
    INC $02
    BNE :+
    INC $03
:
    DEC $01
    BNE @SumWorldFlags
    DEC $00
    LDA $00
    BPL @SumWorldFlags
    LDA ($06), Y                ; IsSaveSlotActive
    JSR AddATo0F0E
    LDA ($08), Y                ; TODO: Add byte at [[$08:09]] to [$0F:0E].
    JSR AddATo0F0E
    LDA ($0A), Y                ; DeathCount
    JSR AddATo0F0E
    LDA ($0C), Y                ; QuestNumber

AddATo0F0E:
    CLC
    ADC $0F
    STA $0F
    LDA $0E
    ADC #$00
    STA $0E
    RTS

FormatFileA:
    LDY #$07
    LDA #$24                    ; Make the name 8 spaces.

@ClearName:
    STA ($04), Y
    DEY
    BPL @ClearName
    LDY #$27                    ; Reset the file's Items block.
    LDA #$00

@ClearItems:
    STA ($00), Y
    DEY
    BPL @ClearItems
    LDA #$80                    ; Will count $180 with [01:00].
    STA $01
    LDA #$01
    STA $00
    LDY #$00                    ; Reset World Flags ($180 bytes) at [$02:03].

@ClearWorldFlags:
    LDA #$00
    STA ($02), Y
    INC $02
    BNE :+
    INC $03
:
    DEC $01
    BNE @ClearWorldFlags
    DEC $00
    LDA $00
    BPL @ClearWorldFlags
    LDA #$00
    STA ($06), Y                ; Reset the file's IsSaveSlotActive.
    STA ($08), Y                ; TODO: Reset byte at [[$08:09]].
    STA ($0A), Y                ; Reset the file's DeathCount.
    STA ($0C), Y                ; Reset the file's QuestNumber.
    JSR FetchFileAAddressSet
    JSR CalculateFileAChecksum
    LDY CurSaveSlot

    ; Be proactive and reset these values in save slot info.
    LDA #$00
    STA IsSaveSlotActive, Y
    STA QuestNumbers, Y
    STA DeathCounts, Y

    ; Since file A is in a good state, we don't care about file B.
    ; Treat it as committed.
    LDA #$FF
    STA IsSaveFileBCommitted, Y
    LDA #$5A                    ; Store the static markers in the save file.
    STA SaveFileOpenMarkers, Y
    LDA #$A5
    STA SaveFileCloseMarkers, Y
    TYA                         ; Store the checksum in the save file.
    ASL
    TAY
    LDA $0E
    STA FileAChecksums, Y
    INY
    LDA $0F
    STA FileAChecksums, Y
    RTS

CalculateAndStoreFileBChecksumUncommitted:
    JSR CalculateFileBChecksum
    LDY CurSaveSlot
    LDA #$00                    ; Now file B is valid, but uncommitted.
    STA IsSaveFileBCommitted, Y
    TYA
    ASL
    TAY
    LDA $CE
    STA FileBChecksums, Y
    INY
    LDA $CF
    STA FileBChecksums, Y
    RTS

; Returns:
; [CF:CE]: checksum
;
CalculateFileBChecksum:
    LDA #$00                    ; Reset the sum.
    STA $CE
    STA $CF
    LDY #$07                    ; Sum the name (8 bytes).

@SumName:
    LDA ($C4), Y
    JSR AddAToCFCE
    DEY
    BPL @SumName
    LDY #$27                    ; Sum the $28 bytes of the file's Items block with [$CF:CE].

@SumItems:
    LDA ($C0), Y
    JSR AddAToCFCE
    DEY
    BPL @SumItems
    LDA #$80                    ; Will count $180 with [C1:C0].
    STA $C1
    LDA #$01
    STA $C0
    LDY #$00                    ; Add World Flags ($180 bytes) to [CF:CE].

@SumWorldFlags:
    LDA ($C2), Y
    JSR AddAToCFCE
    INC $C2
    BNE :+
    INC $C3
:
    DEC $C1
    BNE @SumWorldFlags
    DEC $C0
    LDA $C0
    BPL @SumWorldFlags
    LDA ($C6), Y                ; Add IsSaveSlotActive byte to [$CF:CE].
    JSR AddAToCFCE
    LDA ($C8), Y                ; TODO: Add byte at [[$C8:C9]] to [$CF:CE].
    JSR AddAToCFCE
    LDA ($CA), Y                ; Add DeathCount byte to [$CF:CE].
    JSR AddAToCFCE
    LDA ($CC), Y                ; Add byte QuestNumber to [$CF:CE].

AddAToCFCE:
    CLC
    ADC $CF
    STA $CF
    LDA $CE
    ADC #$00
    STA $CE
    RTS

FormatFileB:
    LDY #$07                    ; Clear the name (to all spaces).
    LDA #$24

@ClearName:
    STA ($C4), Y
    DEY
    BPL @ClearName
    LDY #$27                    ; Clear $28 bytes of the Items block in file.
    LDA #$00

@ClearItems:
    STA ($C0), Y
    DEY
    BPL @ClearItems
    LDA #$80                    ; Will count $180 with [C1:C0].
    STA $C1
    LDA #$01
    STA $C0
    LDY #$00                    ; Clear $180 bytes of World flags.

@ClearWorldFlags:
    LDA #$00
    STA ($C2), Y
    INC $C2
    BNE :+
    INC $C3
:
    DEC $C1
    BNE @ClearWorldFlags
    DEC $C0
    LDA $C0
    BPL @ClearWorldFlags
    LDA #$00                    ; Clear individual bytes.
    STA ($C6), Y                ; IsSaveSlotActive
    STA ($C8), Y                ; TODO: ?
    STA ($CA), Y                ; DeathCount
    STA ($CC), Y                ; QuestNumber
    JSR FetchFileBAddressSet
    JSR CalculateFileBChecksum  ; Leave the checksum at [$CF:CE].
    LDA #$FF
    LDY CurSaveSlot
    STA IsSaveFileBCommitted, Y ; Mark this file B committed.
    RTS
