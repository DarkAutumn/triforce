.INCLUDE "Variables.inc"

.SEGMENT "BANK_06_00"


; Imports from program bank 07

.IMPORT Util_TableJump

.EXPORT Bank_CopyCommonDataToRam
.EXPORT Mode2Load_Init_Submodes
.EXPORT Mode2Load_Update_Full

.INCLUDE "bank6/level-load-mode2.asm"
.INCLUDE "bank6/quest2-replacements.asm"
.INCLUDE "bank6/level-blocks-common-data.asm"
.INCLUDE "bank6/transfer-buf-dispatch.asm"
.INCLUDE "bank6/transfer-bufs-static.asm"
