# Where this source came from

The files under `src/` are a disassembly of The Legend of Zelda (NES, USA, PRG0) written for the ca65
assembler by **Aldo Núñez** (aldonunez1@gmail.com), project `zelda1-disassembly`.

- Imported from: `git@github.com:DarkAutumn/zelda1-disassembly.git`, commit `3952c35` (2026-02-25), a fork of
  Aldo Núñez's repository. Aldo's last commit in that history is `50a1c86` (2021-02-19, "Format: Giant update
  for the preferred format."). The three fork commits after it (`af1073e`, `a6f60e9`, `3952c35`) add 62 comment
  lines and change no code.
- The first commit that adds `zelda-asm/` to this repository holds the import byte for byte: `src/` as
  upstream's `src/`; upstream's `build.ps1`, `util.ps1` and `OriginalNesHeader.bin` under `build/`. Everything
  after that commit (splitting, renaming, comments, documents, tools) was done here.
- Upstream has no license file.
- `index/upstream/` holds a frozen canonical form of the imported code (`tools/asm.py freeze-baseline`) and
  `index/renames.tsv` maps every upstream name to its current name. `tools/asm.py where Z_07:1888` or
  `tools/asm.py where OldName` translates an upstream line or name to the current file, line and name.

## Upstream README (verbatim)

> Introduction
> ------------
>
> Years ago, I reverse engineered The Legend of Zelda ad hoc for the purpose of remaking it in
> a high-level language. I remade it as complete as I could. But, I always felt that the
> disassembly was a bit wasted, because - despite being necessary for the remake - was incomplete,
> poorly documented, and generally unfit for sharing.
>
> The desire to start this complete disassembly project came after taking inspiration from
> the work of Disch on the Final Fantasy disassembly, and Doppelganger's Super Mario Bros.
> disassembly.
>
>
> Development
> -----------
>
> I started this project in October 2020. Although I used and enjoyed FCEUX for the original
> ad hoc reverse engineering project in 2013; I found the Mesen emulator to be more helpful this
> time around, because of several features, including:
>
> - adding and changing labels on the fly
> - adding comments
> - syntax coloring
> - the event viewer (useful in figuring out scrolling)
>
> The disassembly listing in this project was produced by a tool I wrote that processes Mesen's
> .MLB file and the ROM. Among other features, it figures out:
>
> - how to generate ca65 cheap and unnamed labels
> - what procedures are mapped from different banks at each call site
> - replaces memory instruction operands with custom expressions
> - disassembles machine code
>
>
> Building a ROM
> --------------
>
> The source files can be assembled with ca65. To build a ROM, first make an ext folder next to
> src and put ca65 and ld65 inside. Then run build.ps1. The output will appear in a bin folder.
>
> The build script expects to find Original.nes in the ext folder to compare the output to.
> Original.nes must be the "(U) (PRG0) [!]" version of the ROM.
>
> To skip the verification, pass the option --NoVerify to build.ps1.
>
>
> Contact
> -------
>
> Aldo Núñez
>
> aldonunez1@gmail.com

## Building (later step; nothing here builds a ROM yet)

The source is kept assemblable: instructions, operands, directives and their order are never changed
(`tools/asm.py check` proves it against the frozen baseline). To build it one needs:

- `ca65` and `ld65` (cc65 suite). Upstream's `build/build.ps1` expects them in an `ext/` folder and expects to
  sit beside `src/` (it uses the relative paths `src\`, `ext\`, `obj\`, `bin\`).
- A PRG0 ROM ("(U) (PRG0) [!]"; SHA-1 of the image without its 16-byte header:
  `a12d74c73a0481599a5d832361d168f4737bbcf6`). The 48 binary blobs that the source pulls in with `.INCBIN`
  (`dat/*.dat`) are not in git: `src/bins.xml` lists each blob's `FileName`, `Offset` (offset into the PRG
  image; add 16 for the offset in the `.nes` file) and `Length`, and `build.ps1` (`ExtractBins`) cuts them out
  of the ROM into `bin/dat/`, which ca65 then searches (`--bin-include-dir bin`).
- The linker configuration `src/Z.cfg` (one 16 KiB memory area per bank; `BANK_01_CODE` runs from RAM at
  `$6C90`, `BANK_06_DATA` from RAM at `$67F0`).
- One object file per bank file, linked in bank order; the 16-byte `build/OriginalNesHeader.bin` is put in
  front of the linked image.
- After the split into chunk files, each bank file `src/bankN.asm` includes its chunks as
  `.INCLUDE "bankN/<file>.asm"`, and chunk files contain `.INCLUDE "dat/..."` and `.INCLUDE "Variables.inc"`
  style paths relative to `src/`. ca65 resolves an include relative to the including file first, so the build
  needs `-I zelda-asm/src` (and `build.ps1`'s file list must become `src\bank*.asm`).
