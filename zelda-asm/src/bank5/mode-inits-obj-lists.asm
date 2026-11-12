; ============================================================================
; FILE:     bank5/mode-inits-obj-lists.asm BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 1369-1448
; SUBJECTS: (none)
; TOPIC:    Mode 8/D/10 init stubs; spawn position lists; ObjLists
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0023 InitMode8                      code
;   L0036 InitModeD                      code
;   L0048 InitMode10                     code
;   L0067 SpawnPosListAddrsLo            data
;   L0073 SpawnPosListAddrsHi            data
;   L0079 SpawnPosList0                  data
;   L0083 SpawnPosList1                  data
;   L0087 SpawnPosList2                  data
;   L0091 SpawnPosList3                  data
;   L0095 EnteringRoomRelativePositions  data
;   L0098 ObjLists                       data
;   L0101 ObjListAddrs                   data
; ============================================================================

InitMode8:
    JSR TurnOffAllVideo
    LDA GameSubmode
    BNE :+
    STA a:UndergroundExitType   ; Reset.
    JSR TurnOffVideoAndClearArtifacts
    JSR PatchAndCueLevelPalettesTransferAndAdvanceSubmode
    JMP ClearRoomHistory
:
    LDA #$04                    ; Cue the transfer of text and attributes for mode 8.
    STA TileBufSelector
    JMP BeginUpdateMode

InitModeD:
    JSR TurnOffVideoAndClearArtifacts
    JSR InitSaveRam

    ; If initialized save RAM, then it wasn't previously
    ; initialized or something went wrong.
    ; So, go reset the game.
    BCS :+
    JMP BeginUpdateMode
:
    JMP IsrReset

InitMode10:
    LDX #$00                    ; Get the tile the player is standing on (on the hotspot).
    JSR GetCollidableTileStill
    CMP #$24
    BNE :+                      ; If player touched any stairs tile instead of a cave or dungeon entrance, go start updating.
    LDA #$00                    ; TODO: ?
    STA SongEnvelopeSelector
    LDA #$08                    ; Stairs effect
    STA EffectRequest

    ; Set the target Y coordinate $10 pixels below current one.
    LDA ObjY
    CLC
    ADC #$10
    STA StairsTargetY
:
    INC IsUpdatingMode
    RTS

SpawnPosListAddrsLo:
    .LOBYTES SpawnPosList0
    .LOBYTES SpawnPosList1
    .LOBYTES SpawnPosList2
    .LOBYTES SpawnPosList3

SpawnPosListAddrsHi:
    .HIBYTES SpawnPosList0
    .HIBYTES SpawnPosList1
    .HIBYTES SpawnPosList2
    .HIBYTES SpawnPosList3

SpawnPosList0:
    .BYTE $55, $B5, $78, $98, $7A, $9A, $6C, $AC
    .BYTE $8D

SpawnPosList1:
    .BYTE $82, $63, $A3, $75, $95, $77, $97, $5A
    .BYTE $BA

SpawnPosList2:
    .BYTE $A3, $75, $B5, $96, $87, $99, $7A, $BA
    .BYTE $AC

SpawnPosList3:
    .BYTE $63, $55, $95, $76, $88, $79, $5A, $9A
    .BYTE $6C

EnteringRoomRelativePositions:
    .BYTE $18, $E8, $28, $D8

ObjLists:
    .INCBIN "dat/ObjLists.dat"

ObjListAddrs:
    .INCLUDE "dat/ObjListAddrs.inc"
