.INCLUDE "Variables.inc"

.SEGMENT "BANK_03_00"


; Imports from program bank 07

.IMPORT TurnOffAllVideo

.EXPORT TransferLevelPatternBlocks

.INCLUDE "bank3/level-pattern-blocks.asm"
