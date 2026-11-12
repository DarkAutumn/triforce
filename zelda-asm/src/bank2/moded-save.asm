; ============================================================================
; FILE:     bank2/moded-save.asm         BANK: 2 (switchable, $8000-$BFFF)
; UPSTREAM: Z_02.asm 2915-3201
; SUBJECTS: (none)
; TOPIC:    Mode D save
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 SaveSlotHeartsAddrsLo      data
;   L0028 SaveSlotHeartsAddrsHi      data
;   L0033 ProfileNameAddrsLo         data
;   L0038 ProfileNameAddrsHi         data
;   L0043 UpdateModeDSave            code
;   L0047 UpdateModeDSave_JumpTable  data
;   L0052 UpdateModeDSave_Sub0       code
;   L0124 UpdateModeDSave_Sub1       code
;   L0156 CopyFileBToFileA           code
;   L0250 UpdateModeDSave_Sub2       code
;   L0260 FetchProfileNameAddress    code  Returns:
;   L0268 StoreSaveSlotHearts        code
; ============================================================================

SaveSlotHeartsAddrsLo:
    .LOBYTES SaveSlotHearts+0
    .LOBYTES SaveSlotHearts+2
    .LOBYTES SaveSlotHearts+4

SaveSlotHeartsAddrsHi:
    .HIBYTES SaveSlotHearts+0
    .HIBYTES SaveSlotHearts+2
    .HIBYTES SaveSlotHearts+4

ProfileNameAddrsLo:
    .LOBYTES SaveSlotNames+0
    .LOBYTES SaveSlotNames+8
    .LOBYTES SaveSlotNames+16

ProfileNameAddrsHi:
    .HIBYTES SaveSlotNames+0
    .HIBYTES SaveSlotNames+8
    .HIBYTES SaveSlotNames+16

UpdateModeDSave:
    LDA GameSubmode
    JSR TableJump

UpdateModeDSave_JumpTable:
    .ADDR UpdateModeDSave_Sub0
    .ADDR UpdateModeDSave_Sub1
    .ADDR UpdateModeDSave_Sub2

UpdateModeDSave_Sub0:
    ; Initialize file B, and copy profile to it.
    ; Calculate and store file B checksum.
    ; Mark file B uncommitted.
    JSR FetchFileBAddressSet
    JSR FormatFileB
    JSR FetchFileBAddressSet
    JSR FetchFileAAddressSet    ; This seems to be called only for $067F put in [$0E:0F].
    LDY #$27                    ; Copy Items block ($28 bytes) from profile to file B.

@CopyItems:
    LDA InvItems, Y
    STA ($C0), Y
    DEY
    BPL @CopyItems
    LDY CurSaveSlot
    LDA DeathCounts, Y          ; Copy death count from profile to file B.
    LDY #$00
    STA ($CA), Y
    LDA #$01                    ; We're saving, so make sure the current slot is active.
    STA ($C6), Y
    LDY CurSaveSlot
    STA IsSaveSlotActive, Y     ; Also set the slot active in the save slot info.
    LDA QuestNumbers, Y         ; Copy quest number from profile to file B.
    LDY #$00
    STA ($CC), Y
    JSR FetchProfileNameAddress
    LDY #$07                    ; Copy name from save slot info to file B.

@CopyName:
    LDA ($0C), Y
    STA ($C4), Y
    DEY
    BPL @CopyName
    LDA HeartValues             ; Put in [$0A] a full hearts value for the profile's heart containers.
    AND #$F0
    PHA
    LSR
    LSR
    LSR
    LSR
    STA $0A
    PLA
    ORA $0A
    STA HeartValues
    LDA #$FF                    ; Completely fill the hearts.
    STA HeartPartial
    JSR StoreSaveSlotHearts
    LDY #$00                    ; Copy World Flags ($180 bytes) from profile to file B.

@CopyWorldFlags:
    LDA ($0E), Y
    STA ($C2), Y
    INC $C2
    BNE :+
    INC $C3
:
    INC $0E
    BNE :+
    INC $0F
:
    LDA $0E
    CMP #$FF
    BNE @CopyWorldFlags
    LDA $0F
    CMP #$07
    BNE @CopyWorldFlags
    JSR FetchFileBAddressSet
    JSR CalculateAndStoreFileBChecksumUncommitted
    INC GameSubmode             ; Go to the next submode.
    RTS

UpdateModeDSave_Sub1:
    LDY CurSaveSlot
    LDA IsSaveFileBCommitted, Y
    BNE @IncSubmode
    JSR FetchFileBAddressSet
    JSR CalculateFileBChecksum
    LDA CurSaveSlot
    ASL
    TAY
    LDA FileBChecksums, Y
    CMP $CE
    BNE @DiscardFileB
    INY
    LDA FileBChecksums, Y
    CMP $CF
    BNE @DiscardFileB
    JSR CopyFileBToFileA        ; The checksum matches, so commit and copy file B to A.

@IncSubmode:
    INC GameSubmode
    RTS

@DiscardFileB:
    ; Discard file B, because it couldn't be validated.
    ; Begin unverified code A75A
    LDY CurSaveSlot
    LDA #$FF
    STA IsSaveFileBCommitted, Y
    INC GameSubmode
    RTS
    ; End unverified code

CopyFileBToFileA:
    LDY CurSaveSlot
    LDA #$00                    ; Reset the save file markers.
    STA SaveFileOpenMarkers, Y
    STA SaveFileCloseMarkers, Y
    TYA                         ; Reset the checksum.
    ASL
    TAY
    LDA #$00
    STA SaveFileAChecksums, Y
    INY
    STA SaveFileAChecksums, Y
    JSR FetchFileBAddressSet
    JSR FetchFileAAddressSet    ; This puts $067F in [$0E:0F].
    LDY #$27                    ; Copy Items block ($28 bytes) from file B to file A.

@CopyItems:
    LDA ($C0), Y
    STA ($00), Y
    DEY
    BPL @CopyItems
    LDY #$00                    ; Copy these individual bytes from file B to file A.
    LDA ($C6), Y                ; IsSaveSlotActive
    STA ($06), Y
    LDA ($C8), Y                ; TODO: ?
    STA ($08), Y
    LDA ($CA), Y                ; DeathCount
    STA ($0A), Y
    LDA ($CC), Y                ; QuestNumber
    STA ($0C), Y
    LDA ($06), Y                ; Push IsSaveSlotActive from file A to help copy it to save slot info.
    PHA
    LDA ($0A), Y                ; Push death count from file A to help copy it to save slot info.
    PHA
    LDA ($0C), Y                ; Push quest number from file A to help copy it to save slot info.
    PHA
    LDY CurSaveSlot
    PLA                         ; Finish copying quest number from file A to save slot info.
    STA QuestNumbers, Y
    PLA                         ; Finish copying death count from file A to save slot info.
    STA DeathCounts, Y
    PLA                         ; Finish copying IsSaveSlotActive  from file A to save slot info.
    STA IsSaveSlotActive, Y
    LDY #$07                    ; Copy the name from file B to file A.

@CopyName:
    LDA ($C4), Y
    STA ($04), Y
    DEY
    BPL @CopyName

    ; Copy World Flags ($180 bytes) file B to file A.
    ; It counts up from what's in [$0E:0F] ($067F) to $07FF.
    LDY #$00

@CopyWorldFlags:
    LDA ($C2), Y
    STA ($02), Y
    INC $02
    BNE :+
    INC $03
:
    INC $C2
    BNE :+
    INC $C3
:
    INC $0E
    BNE :+
    INC $0F
:
    LDA $0E
    CMP #$FF
    BNE @CopyWorldFlags
    LDA $0F
    CMP #$07
    BNE @CopyWorldFlags
    LDY CurSaveSlot
    LDA #$5A                    ; Write the save file markers.
    STA SaveFileOpenMarkers, Y
    LDA #$A5
    STA SaveFileCloseMarkers, Y
    TYA                         ; Copy the checksum from file B to file A.
    ASL
    TAY
    LDA FileBChecksums, Y
    STA SaveFileAChecksums, Y
    INY
    LDA FileBChecksums, Y
    STA SaveFileAChecksums, Y
    LDY CurSaveSlot
    LDA #$FF                    ; File B has been committed.
    STA IsSaveFileBCommitted, Y
    RTS

UpdateModeDSave_Sub2:
    LDA #$00                    ; Go to mode 0 submode 1. Keep it updating.
    STA GameMode
    LDA #$01
    STA GameSubmode
    RTS

; Returns:
; [0C:0D]: save slot name pointer
;
FetchProfileNameAddress:
    LDY CurSaveSlot
    LDA ProfileNameAddrsLo, Y
    STA $0C
    LDA ProfileNameAddrsHi, Y
    STA $0D
    RTS

StoreSaveSlotHearts:
    LDY CurSaveSlot
    LDA SaveSlotHeartsAddrsLo, Y
    STA $0C
    LDA SaveSlotHeartsAddrsHi, Y
    STA $0D
    LDY #$01                    ; Copy HeartsValue and HeartsPartial to set B.

@CopyHearts:
    LDA HeartValues, Y
    STA ($0C), Y
    DEY
    BPL @CopyHearts
    RTS

; Unknown block
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
