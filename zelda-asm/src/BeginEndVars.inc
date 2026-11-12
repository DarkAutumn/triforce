; Variables of the title/demo (mode 0), name registration (mode $E) and ending (mode $13) code in bank 2.
; They reuse bytes that hold object data during play.

; ---- Mode 0: title screen and story demo ----
TriforceGlowTimer := $412           ; scalar; title: frames until the Triforce's next glow colour.
;   Reloaded with 6, or $10 at the end of the 8-step cycle (TriforceGlowCycle).
IsWaterfallInitialized := $41F      ; scalar; title: 0 until the waterfall wave Ys have been set.
TitleWaveYs := $420                 ; [6] title: Y of the waterfall wave sprites.
;   Entries 0-2 move down 2 pixels per frame and wrap from $E3 to $B2; entries 3-5 are only initialized.
DemoPhase := $42C                   ; scalar; mode 0: 0 = title screen, 1 = story scroll and item parade.
DemoPhase0Subphase1Cycle := $437    ; scalar; title: step 0-$E of the palette fade; at $E DemoPhase advances.
;   Selects the 32-byte palette in DemoPhase0Subphase1Palettes and the next delay.
DemoItemIds := $444                 ; per object slot; item parade: item id shown; $23 fairy, >= $30 Link.

; ---- Mode $E: register a name ----
CharBoardIndex := $41F              ; scalar; index $00-$2B of the highlighted character in ModeE_CharMap.
;   4 rows of 11: left/right change it by 1, up/down by $B.
IsNameFieldInitialized := $420      ; scalar; 0 until the name field of the chosen save slot is set up.
NameCharOffset := $421              ; scalar; offset in SaveSlotNames of the name character the cursor is on.
NameInputCharBuf := $422            ; [3] transfer record header for the next name character.
;   Bytes 0-1: nametable address high, low; byte 2 is the third header byte (unverified: length). The low
;   address byte moves with the cursor and wraps after 8 characters.

; ---- Mode $13: ending ----
EndingPeaceLongTimer := $4D         ; scalar; ending: long timer for the "peace returns" text submode.
;   Set to $40 and decremented once every 10 frames ($280 frames). Link and Zelda are drawn until it is
;   below 4; the submode ends at 0. Same address as RedLeeverLongTimer.
PeaceCharDelayCounter := $412       ; scalar; ending: +1 per frame; a character is written at (n AND 7) = 4.
PeaceCharIndex := $413              ; scalar; ending: index of the next character in PeaceText ($FF = end).
TextboxCharIndex := $416            ; scalar; ending: index of the next character of ThanksText.
TextboxCharVramLo := $45F           ; scalar; ending: low byte of the nametable address of the next character.
;   Starts at $A4; line starts from ThanksTextboxLineAddrsLo.
