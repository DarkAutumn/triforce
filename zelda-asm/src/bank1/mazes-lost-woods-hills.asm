; ============================================================================
; FILE:     bank1/mazes-lost-woods-hills.asm BANK: 1 (stored at $A500-$B76F; copied to RAM and run at $6C90-$7EFF)
; UPSTREAM: Z_01.asm 4832-4935
; SUBJECTS: (none)
; TOPIC:    Lost Woods and Lost Hills maze steps; filler
; SEE ALSO:
; KEY FACTS:
; CONTENTS: (generated; L = line in this file)
;   L0016 ForestMazeDirs         data
;   L0019 MountainMazeDirs       data
;   L0025 CheckMazes             code  Params:
;   L0089 MapScreenPosToPpuAddr  code  Params:
;   L0117 Filler_6DFA            data
; ============================================================================

ForestMazeDirs:
    .BYTE $08, $02, $04, $02

MountainMazeDirs:
    .BYTE $08, $08, $08, $08

; Params:
; Y: room ID
;
CheckMazes:
    LDX MazeStep
    LDA ObjDir
    LDY RoomId
    CPY #$61
    BNE @CheckMountainMaze      ; If not in forest maze, go check mountain maze.
    CMP ForestMazeDirs, X
    BNE @Mismatch               ; If direction doesn't match current step, go handle it.

    ; The direction matches the step.
    CPX #$03
    BEQ @PlaySecretTune         ; If the last step matches, go let the player pass.

@AdvanceMaze:
    ; Haven't passed the last step. So increment the step,
    ; but repeat this room.
    INC MazeStep

@SetNextRoom:
    STY NextRoomId

@Exit:
    RTS

@ResetMazeStep:
    LDA #$00
    STA MazeStep
    RTS

@Mismatch:
    CMP #$01
    BEQ @Exit                   ; If going right in the forest, allow exit.

@Reset:
    JSR @ResetMazeStep          ; Else you have to start over.
    BEQ @SetNextRoom            ; Go set the next room to the current one, and return.

@CheckMountainMaze:
    CPY #$1B
    BNE @ResetMazeStep          ; If not in the mountain maze either, go reset the maze step and leave room alone.
    CMP MountainMazeDirs, X
    BEQ @Match                  ; If the direction matches the current step, go see if it's the last one.
    CMP #$02
    BNE @Reset                  ; If not going left in the mountain, go start over.
    RTS                         ; Else allow exit.

@Match:
    CPX #$03
    BNE @AdvanceMaze            ; If this isn't the last step, then go advance the step, but repeat the room.

@PlaySecretTune:
    ; Play secret tune, and let the player leave the maze.
    LDA #$04
    STA Tune1Request
    RTS

; Params:
; [02]: Y
; [03]: X
;
; Returns:
; [00]: high byte of PPU address
; [01]: low byte of PPU address
;
MapScreenPosToPpuAddr:
    ; Turn the 16-bit value 08YY into the PPU address
    ; ($2000 + (YY * 4)) by multiplying it by 4.
    ;
    ; Each row of tiles is 32 tiles (and bytes) long, and 8 pixels tall.
    ; This is why the Y coordinate is multiplied by 4 (= 32 / 8).
    LDA #$08
    STA $00
    LDA $02
    ASL
    ROL $00
    ASL
    ROL $00
    AND #$E0
    STA $01

    ; Now divide X coordinate by 8, because each tile is 8 pixels wide.
    LDA $03
    LSR
    LSR
    LSR

    ; Combine the low address of the row with the column to get
    ; the final low byte of PPU address.
    ORA $01
    STA $01
    RTS

Filler_6DFA:
    .BYTE $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    .BYTE $FF, $FF
