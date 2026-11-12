; ============================================================================
; FILE:     bank1/modeb-cave-save-sprites.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 2978-3151
; SUBJECTS: (none)
; TOPIC:    Sound requests, mode B cave entry, save-slot addresses, sprite cycling, person blocking
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0027 PlaySample                 code
;   L0031 PlayEffect                 code
;   L0038 InitModeB_EnterCave_Bank5  code
;   L0052 FileBChecksums             code
;   L0078 ResetRoomTileObjInfo       code  Returns:
;   L0085 ReverseDirections          data
;   L0088 SaveFileAAddressSets       data
;   L0092 SaveFileAAddressSet1       data
;   L0096 SaveFileAAddressSet2       data
;   L0112 FetchFileAAddressSet       code  TODO: [08:09]
;   L0143 PlayBoomerangSfx           code  Params:
;   L0153 HideObjectSprites          code
;   L0171 CycleCurSpriteIndex        code
;   L0174 CycleSpriteIndexInA        code
;   L0180 ResetCurSpriteIndex        code
;   L0193 CheckPersonBlocking        code  Params:
; ============================================================================

PlaySample:
    LDY #$01
    BNE :+

PlayEffect:
    LDY #$03
:
    ORA SongRequest, Y
    STA SongRequest, Y
    RTS

InitModeB_EnterCave_Bank5:
    ; Save the submode.
    LDA GameSubmode
    PHA
    JSR InitMode_EnterRoom
    JSR ResetInvObjState

    ; Put Link at the cave entrance ($70, $DD), and facing up.
    LDA #$70
    STA ObjX
    LDA #$DD
    STA ObjY
    LDA #$08

FileBChecksums:
    STA ObjDir
    JSR Link_EndMoveAndAnimate
    JSR RunCrossRoomTasksAndBeginUpdateMode_PlayModesNoCellar

    ; Restore the submode.
    PLA
    STA GameSubmode

    ; Still initializing. A routine called here changed it.
    LDA #$00
    STA IsUpdatingMode
    INC GameSubmode

    ; Get Link ready to move automatically this number of pixels.
    LDA #$30
    STA ObjGridOffset

    ; When coming out of the cave eventually, this has to be set.
    LDA #$01
    STA UndergroundExitType
    RTS

; Returns:
; A: 0
;
ResetRoomTileObjInfo:
    LDA #$00
    STA RoomTileObjType
    STA RoomTileObjX
    STA RoomTileObjY
    RTS

ReverseDirections:
    .BYTE $08, $04, $02, $01

SaveFileAAddressSets:
    .BYTE $1A, $60, $92, $60, $02, $60, $12, $65
    .BYTE $15, $65, $18, $65, $1B, $65

SaveFileAAddressSet1:
    .BYTE $42, $60, $12, $62, $0A, $60, $13, $65
    .BYTE $16, $65, $19, $65, $1C, $65

SaveFileAAddressSet2:
    .BYTE $6A, $60, $92, $63, $12, $60, $14, $65
    .BYTE $17, $65, $1A, $65, $1D, $65

; TODO: [08:09]
;
; Returns:
; [00:01]: items pointer
; [02:03]: world flags pointer
; [04:05]: name pointer
; [06:07]: IsSaveSlotActive pointer
; [08:09]:
; [0A:0B]: death count pointer
; [0C:0D]: quest pointer
; [0E:0F]: profile/save-slot world flags pointer
;
FetchFileAAddressSet:
    LDA #$FF                    ; Calculate the end of the address set for current file.
    LDY CurSaveSlot
:
    CLC
    ADC #$0E
    DEY
    BPL :-
    TAY

    ; Copy the save file address set (14 bytes) for the current
    ; save file to [$00] to make it easier to work with.
    LDX #$0D
:
    LDA SaveFileAAddressSets, Y
    STA $00, X
    DEY
    DEX
    BPL :-

    ; Put the address of the profile's whole
    ; WorldFlags block in [$0E:0F].
    LDA #<WorldFlags
    STA $0E
    LDA #>WorldFlags
    STA $0F
    RTS

; Params:
; Y: 2
;
PlayBoomerangSfx:
    LDA ObjTimer+19
    BNE :+
    TYA
    JSR PlayEffect
    LDA #$0A
    STA ObjTimer+19
:
    RTS

HideObjectSprites:
    ; Hide all sprites $18 and on.
    ; Cycle the starting sprite index.
    LDX #$60
    LDA #$F8
:
    STA Sprites, X
    INX
    INX
    INX
    INX
    CPX #$00
    BNE :-
    LDA FirstSpriteIndex
    JSR CycleSpriteIndexInA
    STA FirstSpriteIndex
    RTS

CycleCurSpriteIndex:
    LDA RollingSpriteIndex      ; From 0 to $27.

CycleSpriteIndexInA:
    CLC
    ADC #$01
    CMP #$28
    BNE :+

ResetCurSpriteIndex:
    LDA #$00
:
    STA RollingSpriteIndex
:
    RTS

; Params:
; [0F]: movement direction
;
; Returns:
; [0F]: untouched, or 0
;
CheckPersonBlocking:
    LDA ObjY
    CMP #$8E
    BCS :-                      ; Return if Y coordinate >= $8E.
    LDA $0F
    AND #$08
    BEQ :-                      ; If not moving up, then return.
    JMP ResetMovingDir
