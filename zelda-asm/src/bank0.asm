.INCLUDE "Variables.inc"

.SEGMENT "BANK_00_00"


.EXPORT Audio_Drive

.INCLUDE "bank0/song-data.asm"
.INCLUDE "bank0/drive-audio-tune0.asm"
.INCLUDE "bank0/sound-effects.asm"
.INCLUDE "bank0/tune1-and-samples.asm"
.INCLUDE "bank0/note-emit-vibrato.asm"
.INCLUDE "bank0/drive-song.asm"
.INCLUDE "bank0/song-note-tables.asm"
