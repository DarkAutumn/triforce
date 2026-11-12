; ============================================================================
; FILE:     bank5/next-room-calc.asm     BANK: 5 (switchable, $8000-$BFFF)
; UPSTREAM: Z_05.asm 7482-7565
; SUBJECTS: (none)
; TOPIC:    Next room by direction; invalid room reloads overworld
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0017 NextRoomIdOffsets         data
;   L0020 CalculateNoNextRoom       code
;   L0034 CalculateNextRoomForDoor  code  Params:
;   L0062 EndGameMode12             code
;   L0072 MaskCurPpuMaskGrayscale   code
;   L0084 CalcNextRoomByDir         code  Params:
; ============================================================================

NextRoomIdOffsets:
    .BYTE $F0, $10, $FF, $01

CalculateNoNextRoom:
    ; Begin unverified code 17517
    LDA #$00
    STA $E7                     ; UNKNOWN: Write 0? But CalculateNextRoomForDoor reads [E7] instead.
    RTS
    ; End unverified code
:
    ASL $00                     ; Shift single-bit mask.
    DEX
    JMP :+                      ; Go test next bit.

; Params:
; [E7]: direction
;
CalculateNextRoomForDoor:
    ; Map door bit/direction to an index as follows:
    ; 1: 3
    ; 2: 2
    ; 4: 1
    ; 8: 0
    ;
    ; Note that this is the opposite of the usual mapping.
    LDA #$01
    STA $00                     ; [00] holds a single-bit mask to compare
    LDX #$03                    ; Index
:
    LDA $E7                     ; Compare direction argument to single-bit mask.
    BIT $00
    BEQ :--                     ; If they don't match, go try the next one.
    JSR GetUniqueRoomId
    STA $04E4                   ; UNKNOWN: [$04E4] holds room layout. But it's unused.
    LDA NextRoomIdOffsets, X    ; Look up offset used to calculate next room ID from current one.
    CLC
    ADC RoomId
    STA NextRoomId              ; Adding the offset to RoomId yields the room ID in the desired direction.
    LDA CurLevel
    BNE :+
    JSR CheckMazes
:
    LDA NextRoomId
    BPL MaskCurPpuMaskGrayscale ; If the next room ID is invalid, then fall thru, and reload OW.

EndGameMode12:
    JSR EndGameMode
    STA $E7                     ; UNKNOWN: Mode 2 loads a level. But it doesn't depend on [E7].
    STA CurLevel                ; Set OW (level 0).
    LDA #$02
    STA GameMode
    STA UndergroundExitType     ; Set to type 2: dungeon level.
    LDA #$80                    ; Silence the song.
    STA Tune0Request

MaskCurPpuMaskGrayscale:
    LDA CurPpuMask_2001
    AND #$FE
    STA CurPpuMask_2001
    RTS

; Params:
; A: direction
;
; Returns:
; A: next room ID
;
CalcNextRoomByDir:
    LDX #$01
    STX $00                     ; Set up the single-bit mask.

    ; For each direction in [00]  indexed by X:
    LDX #$03
:
    BIT $00                     ; Compare the direction argument with the single-bit mask [00].
    BNE :+                      ; If they match, go use this index.
    ASL $00
    DEX
    JMP :-                      ; Go check the next direction.
:
    LDA NextRoomIdOffsets, X
    CLC                         ; Adding offset and room ID yields next room ID.
    ADC RoomId
    RTS
