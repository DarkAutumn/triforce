; ============================================================================
; FILE:     bank2/mode1-init-save-files.asm BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2224-2554
; SUBJECTS: (none)
; TOPIC:    Mode 1 (file select) init dispatcher and its templates; the save-file pass that commits a
;           pending file B or reformats a damaged file A; checksum and format routines of files A and B.
; SEE ALSO: bank2/mode1-slot-tiles.asm (the other mode 1 init submodes); bank2/moded-save.asm
;           (CopyFileBToFileA, the save sequence); bank2/register-eliminate-init.asm
;           (FetchFileBAddressSet); bank1/modeb-cave-save-sprites.asm (FetchFileAAddressSet);
;           bank2/mode0-demo.asm (mode 0 submodes around UpdateMode0Demo_Sub1).
; KEY FACTS:
;   - A save file is: name (8 bytes), items block ($28 bytes, the image of InvItems), world flags
;     ($180 bytes), active flag, one unused byte, death count, quest number (CalculateFileAChecksum).
;   - The checksum is the 16-bit sum of all those bytes, kept high byte first ([0E] high, [0F] low for
;     file A; [CE], [CF] for file B) (AddATo0F0E, AddAToCFCE).
;   - Saving is two-phase: file B is the staging copy, file A the committed one. For each slot a file B
;     that is uncommitted (IsSaveFileBCommitted = 0) with a matching checksum is copied over file A
;     (UpdateMode0Demo_Sub1).
;   - File A is reformatted (slot erased) when its markers are not $5A / $A5 or its checksum does not
;     match (UpdateMode0Demo_Sub1, FormatFileA).
;   - That pass runs as mode 0 update submode 1 (after Start on the title, after saving, after
;     registering) and again as mode 1 init submode 0 (InitMode1_Full_JumpTable).
;   - Both checksum routines destroy the items pointer and advance the world flags pointer, so callers
;     fetch the address set again afterwards (CalculateFileAChecksum, CalculateFileBChecksum).
;   - LinkColors: the file menu colours each slot's Link by its ring: $29 none, $32 blue, $16 red.
; CONTENTS: (generated; L = line in this file)
;   L0044 Mode1SlotLineTransferBuf                   data  DATA: transfer buffer template for one save slot's menu line: 8 name tiles, tile $62 and 8 heart
;   L0051 Mode1DeathCountsTransferBuf                data  DATA: transfer buffer template for the three death counts: 3 tiles each at $2189, $21E9, $2249.
;   L0057 LinkColors                                 data  TABLE: Link's tunic colour on the file menu; index = InvRing value (0 none, 1 blue, 2 red).
;   L0063 InitMode1_Full                             code  Mode 1 (file select) init: turns video off and runs the init step selected by GameSubmode.
;   L0069 InitMode1_Full_JumpTable                   data  TABLE: mode 1 init handlers; index = GameSubmode (3-5 build the lines of save slots 0-2).
;   L0082 UpdateMode0Demo_Sub1                       code  For each save slot: commits a valid pending file B to file A, else reformats file A if damaged.
;   L0153 CalculateFileAChecksum                     code  Sums every byte of the current slot's file A into a 16-bit checksum.
;   L0198 AddATo0F0E                                 code  Adds A to the 16-bit sum at [0E] (high) and [0F] (low); also the tail of the routine above.
;   L0212 FormatFileA                                code  Erases the current slot's file A and its slot info, then stores fresh markers and checksum.
;   L0281 CalculateAndStoreFileBChecksumUncommitted  code  Checksums the current slot's file B, stores the sum in FileBChecksums and marks file B uncommitted.
;   L0301 CalculateFileBChecksum                     code  Sums every byte of the current slot's file B into a 16-bit checksum.
;   L0346 AddAToCFCE                                 code  Adds A to the 16-bit sum at [CE] (high) and [CF] (low); also the tail of the routine above.
;   L0359 FormatFileB                                code  Erases the current slot's file B (name to spaces, the rest to 0) and marks it committed.
; ============================================================================

; DATA: transfer buffer template for one save slot's menu line: 8 name tiles, tile $62 and 8 heart
; tiles at $2109, a second row of 8 heart tiles at $2132; the code adds $60 per slot to both addresses.
Mode1SlotLineTransferBuf:
    .BYTE $21, $09, $11, $24, $24, $24, $24, $24
    .BYTE $24, $24, $24, $62, $00, $00, $00, $00
    .BYTE $00, $00, $00, $00, $21, $32, $08, $00
    .BYTE $00, $00, $00, $00, $00, $00, $00, $FF

; DATA: transfer buffer template for the three death counts: 3 tiles each at $2189, $21E9, $2249.
Mode1DeathCountsTransferBuf:
    .BYTE $21, $89, $03, $24, $24, $01, $21, $E9
    .BYTE $03, $24, $24, $01, $22, $49, $03, $24
    .BYTE $24, $01, $FF

; TABLE: Link's tunic colour on the file menu; index = InvRing value (0 none, 1 blue, 2 red).
LinkColors:
    .BYTE $29, $32, $16

; ----------------------------------------------------------------------------
; Mode 1 (file select) init: turns video off and runs the init step selected by GameSubmode.
; ----------------------------------------------------------------------------
InitMode1_Full:
    JSR TurnOffAllVideo
    LDA GameSubmode
    JSR TableJump

; TABLE: mode 1 init handlers; index = GameSubmode (3-5 build the lines of save slots 0-2).
InitMode1_Full_JumpTable:
    .ADDR UpdateMode0Demo_Sub1
    .ADDR InitMode1_Sub1
    .ADDR InitMode1_Sub2
    .ADDR InitMode1_FillAndTransferSlotTiles
    .ADDR InitMode1_FillAndTransferSlotTiles
    .ADDR InitMode1_FillAndTransferSlotTiles
    .ADDR InitMode1_Sub6

; ----------------------------------------------------------------------------
; For each save slot: commits a valid pending file B to file A, else reformats file A if damaged.
; NOTE: Mode 0 update submode 1 and mode 1 init submode 0. Ends by advancing GameSubmode.
; ----------------------------------------------------------------------------
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
    LDA SaveFileAChecksums, Y       ; Does the checksum match?
    CMP $0E
    BNE @FormatA
    INY
    LDA SaveFileAChecksums, Y
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

; ----------------------------------------------------------------------------
; Sums every byte of the current slot's file A into a 16-bit checksum.
; IN:   the file A address set at [00]-[0D] (FetchFileAAddressSet).
; OUT:  [0E] = checksum high byte, [0F] = low byte. [00:01] is destroyed, [02:03] advanced by $180.
; ----------------------------------------------------------------------------
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
    LDA ($08), Y                ; The unused per-slot byte ($6515 + slot) is summed too.
    JSR AddATo0F0E
    LDA ($0A), Y                ; DeathCount
    JSR AddATo0F0E
    LDA ($0C), Y                ; QuestNumber

; Adds A to the 16-bit sum at [0E] (high) and [0F] (low); also the tail of the routine above.
AddATo0F0E:
    CLC
    ADC $0F
    STA $0F
    LDA $0E
    ADC #$00
    STA $0E
    RTS

; ----------------------------------------------------------------------------
; Erases the current slot's file A and its slot info, then stores fresh markers and checksum.
; IN:   the file A address set at [00]-[0D]; CurSaveSlot.
; NOTE: Also marks the slot's file B committed, so a pending save of that slot is dropped.
; ----------------------------------------------------------------------------
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
    STA ($08), Y                ; Reset the unused per-slot byte ($6515 + slot).
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
    STA SaveFileAChecksums, Y
    INY
    LDA $0F
    STA SaveFileAChecksums, Y
    RTS

; ----------------------------------------------------------------------------
; Checksums the current slot's file B, stores the sum in FileBChecksums and marks file B uncommitted.
; ----------------------------------------------------------------------------
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

; ----------------------------------------------------------------------------
; Sums every byte of the current slot's file B into a 16-bit checksum.
; IN:   the file B address set at [C0]-[CD] (FetchFileBAddressSet).
; OUT:  [CE] = checksum high byte, [CF] = low byte. [C0:C1] is destroyed, [C2:C3] advanced by $180.
; ----------------------------------------------------------------------------
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
    LDA ($C8), Y                ; The unused per-slot byte ($6D93 + slot) is summed too.
    JSR AddAToCFCE
    LDA ($CA), Y                ; Add DeathCount byte to [$CF:CE].
    JSR AddAToCFCE
    LDA ($CC), Y                ; Add byte QuestNumber to [$CF:CE].

; Adds A to the 16-bit sum at [CE] (high) and [CF] (low); also the tail of the routine above.
AddAToCFCE:
    CLC
    ADC $CF
    STA $CF
    LDA $CE
    ADC #$00
    STA $CE
    RTS

; ----------------------------------------------------------------------------
; Erases the current slot's file B (name to spaces, the rest to 0) and marks it committed.
; OUT:  [CE:CF] = checksum of the empty file (not stored in FileBChecksums).
; ----------------------------------------------------------------------------
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
    STA ($C8), Y                ; The unused per-slot byte ($6D93 + slot).
    STA ($CA), Y                ; DeathCount
    STA ($CC), Y                ; QuestNumber
    JSR FetchFileBAddressSet
    JSR CalculateFileBChecksum  ; Leave the checksum at [$CF:CE].
    LDA #$FF
    LDY CurSaveSlot
    STA IsSaveFileBCommitted, Y ; Mark this file B committed.
    RTS
