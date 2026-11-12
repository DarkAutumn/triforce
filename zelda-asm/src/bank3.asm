.INCLUDE "Variables.inc"

.SEGMENT "BANK_03_00"


; Imports from program bank 07

.IMPORT Ppu_TurnOffAllVideo

.EXPORT Chr_TransferLevelPatterns

.INCLUDE "bank3/level-pattern-blocks.asm"
