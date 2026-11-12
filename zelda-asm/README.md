# The Legend of Zelda (NES), annotated disassembly

A ca65 disassembly of The Legend of Zelda, NES, USA, revision PRG0 (SHA-1 of the image without its 16-byte
header: `a12d74c73a0481599a5d832361d168f4737bbcf6`), split into small topical files, renamed to one scheme and
commented so that a question about the game can be answered from here without reading 40,000 lines. It is
written for agents: search first, read a file header, then read the routine.

Source: the disassembly by Aldo Núñez, <https://github.com/aldonunez/zelda1-disassembly>. The code is his,
unchanged; the file split, the names, the comments, the documents, the data listings and the tools were made
here. Every original name and line can still be found (see "Finding things").

## Start here

| You want | Go to |
|---|---|
| A monster, boss or hazard: HP, damage, weapons, behaviour, drops | [enemies.md](enemies.md), heading `# <Name>` |
| An item or weapon: slots, states, frame counts, limits | [items.md](items.md), heading `# <Name>` |
| How the game works: modes, movement, collision, screen edges and the screen wrap, doors, secrets, spawning, combat, saved state, second quest | [mechanics.md](mechanics.md) |
| What is in a room, a level, a monster list, a text | [data/README.md](data/README.md) and the listings beside it |
| A RAM address | [index/ram.md](index/ram.md) (by address), `src/*.inc` (by topic) |
| A label or symbol by name or prefix | `grep -P "^Dodongo_" index/symbols.tsv` |
| Which file holds a topic | [index/files.md](index/files.md); each file's header lists its labels with line numbers |
| A name or line of the original disassembly | `tools/asm.py where UpdateDodongo`, `tools/asm.py where Z_04:5855` |

The three documents start with a generated table of contents (`- L<line> <heading>`), so a section can be read
by line range. Every bullet in them ends with a link to the label that proves it, for example
[`Link_CheckScreenEdge`](src/bank7/walker-collision-screen-edge.asm#L243); the links carry current line numbers.

## Layout

```
README.md  enemies.md  items.md  mechanics.md      the documents
src/bankN.asm                                      bank header (imports, exports) + .INCLUDE lines in ROM order
src/bankN/<topic>.asm                              the code and tables, 70-700 lines per file
src/Variables.inc ObjVars.inc CommonVars.inc CaveVars.inc BeginEndVars.inc    RAM symbols (Name := $addr)
src/dat/*.inc                                      address tables included by the code
src/Z.cfg  src/bins.xml                            linker memory map; where each .INCBIN blob sits in the ROM
data/                                              level blocks, level info, monster lists, layouts, texts, decoded
index/symbols.tsv  files.md  ram.md  renames.tsv   generated indexes; upstream name -> current name
index/upstream/                                    frozen canonical form of the original (never edited)
tools/asm.py  tools/romdata.py  tools/subjects.txt checker and generator; data decoder; label subjects
build/                                             the original build script and iNES header
```

## Banks

| Bank | Mapped at | Holds | Files |
|---|---|---|---|
| 0 | `$8000-$BFFF` | sound driver, songs, effects | [bank 0](index/files.md#bank-0-srcbank0asm) |
| 1 | `$8000-$BFFF`; its second half is copied to RAM and runs at `$6C90-$7EFF` | caves and persons, whirlwind, traps; in RAM: status bar, items and pickups, mazes, drawing, collision, damage, knockback | [bank 1](index/files.md#bank-1-srcbank1asm) |
| 2 | `$8000-$BFFF` | title, story, file menu, name entry, save, ending, credits | [bank 2](index/files.md#bank-2-srcbank2asm) |
| 3 | `$8000-$BFFF` | pattern (CHR) data and its transfer | [bank 3](index/files.md#bank-3-srcbank3asm) |
| 4 | `$8000-$BFFF` | monsters, bosses, tile objects (rocks, docks, trees), drops | [bank 4](index/files.md#bank-4-srcbank4asm) |
| 5 | `$8000-$BFFF` | room entry and layout, doors, scrolling, submenu, Link's input, warps, mode code | [bank 5](index/files.md#bank-5-srcbank5asm) |
| 6 | `$8000-$BFFF` | level data, second-quest patches, PPU transfer buffers | [bank 6](index/files.md#bank-6-srcbank6asm) |
| 7 | `$C000-$FFFF` (fixed) | reset, NMI, mode dispatch, play update, Link, weapons, object tables | [bank 7](index/files.md#bank-7-srcbank7asm) |

## Reading the source

- **File header.** Every `src/bankN/*.asm` begins with `FILE`, `UPSTREAM` (the original file and lines),
  `SUBJECTS`, `TOPIC`, `SEE ALSO`, `KEY FACTS` (what an investigator needs first, each naming its label) and
  `CONTENTS` (each global label with its line, kind and one-line summary).
- **Labels.** Global labels are `<Subject>_<Rest>`; the subject is the monster, item, mode or system
  (`tools/subjects.txt` lists them), so `grep '^Zora_'` finds everything about the Zora. Routines are
  verb-first (`Dodongo_TryEatBomb`), state handlers `State<N><Name>`, mode code `Mode7Scroll_Init_Sub0`, data
  noun phrases (`Dodongo_BloatedWaitTimes`). Suffixes: `_JumpTable` (addresses for a table jump), `_Full` (the
  far routine behind a fixed-bank trampoline of the same base name), `_Bank<N>` (a per-bank copy), `Lo`/`Hi`
  (split address tables), `_Return` (a shared `RTS`). `Unused_`, `Filler_`: dead code, padding. Cheap labels
  are `@PascalCase`.
- **RAM symbols.** PascalCase; arrays indexed by object slot start with `Obj`; an alias of a slot array used
  by one object type is `<Subject>_Obj<Field>`; flags `Is`/`Has`; timers `...Timer`; counts `...Count`.
  Several names on one address are aliases for different object types or modes.
- **Routine header.** The comment block directly above a label: one sentence saying what it does (the index
  summary), then `IN:`, `OUT:`, `NOTE:` lines when they carry information. Data labels start `; TABLE:` or
  `; DATA:`.
- **Vocabulary.** "Link" is object slot 0; "monster" a hostile object; "object slot" with a hex index
  (`$00` Link, `$01-$0B` monsters and room objects, `$0D` sword, `$0E` sword shot or magic shot, `$0F`
  boomerang or food, `$10/$11` bombs or fires, `$12` arrow or rod, `$13` room item); "room" one screen;
  `Q1`/`Q2` the quests; `[00]`..`[0F]` the zero-page temporaries; time in frames (60 per second); direction
  bits `1` right, `2` left, `4` down, `8` up; q-speed `$40` = 1 pixel per frame.
- **`(unverified)`** marks a statement that was not confirmed by reading the code. `(trace)` in the documents
  marks a result observed in the running game.

## Finding things with the tool

```
tools/asm.py where Dodongo_Update     # current name -> file:line and the original name
tools/asm.py where UpdateDodongo      # original name -> current name, file:line
tools/asm.py where Z_07:3025          # original file:line -> current file:line, label
tools/asm.py check                    # code unchanged, links and indexes current
tools/asm.py check --final            # also naming, headers and summaries, self-containment
tools/asm.py gen                      # regenerate indexes, file headers, contents, link targets
tools/asm.py rename map.tsv           # rename labels/symbols everywhere (rows: Old<TAB>New)
tools/romdata.py --rom <PRG0 rom>     # regenerate data/ from the ROM
```

## Rules for editing

1. **Never change the code.** Instructions, operands, directives, data rows and their order are fixed.
   `tools/asm.py check` reduces every statement to a canonical form (identifiers resolved to values and
   positions) and compares it with the frozen original in `index/upstream/`. Allowed in `src/`: comments and
   blank lines; turning an unnamed label into a cheap label; writing a raw address as a symbol of the same
   value.
2. **Rename only through `tools/asm.py rename`.** It rewrites every use, the comments, the documents and
   `index/renames.tsv`, then runs the check; it changes nothing if the check fails.
3. **After any edit run `tools/asm.py gen` and `tools/asm.py check --final`.** `gen` refreshes the file
   headers' contents, the indexes, the documents' tables of contents and every link's path and line.
4. **Stay self-contained.** Nothing in this tree refers to another project, tool or emulator; a fact is stated
   with the label that proves it.
5. **Documents.** Each bullet in `enemies.md`, `items.md` and `mechanics.md` carries at least one link written
   like the example under "Start here" (label name in backticks inside the brackets, then the file path and
   `#L` line); write any line number, `gen` corrects path and line from the label name.

## Building (not set up here)

The source is kept in a form ca65/ld65 can assemble; no ROM is built here. `src/bins.xml` lists the blobs the source
pulls in with `.INCBIN "dat/..."` (offset in the PRG image and length); they are cut from a PRG0 ROM, not
stored here. `src/Z.cfg` is the linker configuration, `build/` holds the original build script (written for
the unsplit layout: it expects `src/`, `ext/`, `obj/`, `bin/` beside it) and the iNES header. Chunk files use
include paths relative to `src/`, so the assembler needs `src/` as an include directory.
