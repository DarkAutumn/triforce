# FCEUmm quirks (FQ-*)

zcore reproduces FCEUmm (libretro `fceumm`, commit `e0f3086a8104f8fbbe49a2615b10401504a9533f`, the
emulator the promoted models were trained on through stable-retro) instead of the hardware wherever the
two differ and the difference shows in the corpus. Every such behaviour is a numbered quirk:

- it was added behind `Config::fceumm_compat` (TetaNES branch `bugs`); zcore hard-wired it on at ladder
  stage Z3 and deleted the hardware paths, until step 18 of the port plan restores them;
- its ID appears in a code comment at every place it is implemented, in the commit message that adds it,
  and here;
- it is never a `fix:` commit (`fix:` means TetaNES was wrong about the hardware).

Step 18 of the port plan ("Retire the FCEUmm oracle and FCEUmm quirks") restores the hardware behaviour
for each entry, one at a time, and keeps it unless the promoted models get worse. FCEUmm file/line
citations are for the pinned commit (upstream numbering, before `oracle/fceumm/patches/`). Implementation
sites are the named functions under `crates/zcore/src`; the tests are `src/fceumm_compat_tests.rs`, and
every test ROM the quirks break is `#[ignore]`d with its FQ ID in `src/common.rs`.

| ID | Quirk | Affects |
|---|---|---|
| FQ-01 | Two dead PPU frames after reset | boot only |
| FQ-02 | No $2002 read-before-VBlank suppression race | per-frame timing, RAM |
| FQ-03 | PPUSTATUS flags visible only at instruction boundaries | per-frame timing, RAM |
| FQ-04 | A $2002 read never cancels an NMI; enabling NMI during VBlank gives a delayed NMI | per-frame timing, RAM |
| FQ-05 | NMI requested 12 dots after VBlank, taken at the next instruction start | per-frame timing, RAM |
| FQ-06 | 0-cycle CPU reset | boot only |
| FQ-07 | PPU starts at scanline 240, one dot ahead | boot, then CPU/PPU phase for the whole run |
| FQ-08 | Frame boundary at dot 1 of line 240 | frame boundary (when inputs and pokes apply) |
| FQ-09 | Odd-frame dot skip even when rendering is off | per-frame timing |
| FQ-10 | 512-cycle OAM DMA, run inside `STA $4014` | per-frame timing |
| FQ-11 | 4-read DMC DMA | per-frame timing |
| FQ-12 | DMC fetch timed by FCEUmm's per-instruction sound hook | per-frame timing |

## FQ-01 — Two dead PPU frames after reset

- **Hardware:** VBlank is raised from the first frame after power-on/reset; only some register writes are
  ignored during the ~29,658-cycle warm-up. nesdev "PPU power up state"; blargg `ppu_vbl_nmi`.
- **FCEUmm:** `FCEUPPU_Reset` sets `ppudead = 2` (`src/ppu.c:1163-1171`); while `ppudead` is non-zero,
  `FCEUPPU_Loop` runs a whole frame of CPU time with no VBlank, no NMI and no rendering
  (`src/ppu.c:1204-1209`).
- **Affects:** boot only (Zelda's reset code waits for two VBlanks; with FQ-01 it starts two frames later).
  Changes RAM at the end of `power_on_to_start`'s registration, so every based movie's starting state.
- **Implemented:** `Ppu::dead_frames` (field), `Ppu::reset` (set to 2), `Ppu::start_vblank` (no VBlank
  while non-zero), `Bus::fceumm_events` (count down at the frame boundary), `Bus::ppu_clock` (FQ-09 skip
  gated on it).
- **Removal:** delete `dead_frames` and its uses; VBlank rises in frame 1. Regenerate the base state
  (`power_on_to_start` RAM changes). TetaNES `ppu_vbl_nmi` ROMs already pass without it.

## FQ-02 — No $2002 read-before-VBlank suppression race

- **Hardware:** reading $2002 one PPU clock before VBlank sets returns clear and suppresses both the flag
  and the NMI for that frame. nesdev "PPU frame timing" ("VBL flag timing"); `ppu_vbl_nmi`
  `06-suppression`.
- **FCEUmm:** `A2002` (`src/ppu.c:189-203`) returns and clears `PPU_status`; VBlank is set between CPU
  run chunks (`src/ppu.c:1211-1212`), so no read can land on the set edge.
- **Affects:** per-frame timing and RAM: Zelda's boot VBlank wait loop hit the race in stock TetaNES and
  ran a frame late.
- **Implemented:** `Ppu::read_status` (FCEUmm's read; the hardware race check was deleted at Z3).
- **Removal:** drop the compat branch so the race check in `Ppu::read_status` runs again; `06-suppression`
  passes again.

## FQ-03 — PPUSTATUS flags visible only at instruction boundaries

- **Hardware:** VBlank rises at dot 1 of line 241; VBlank, sprite 0 hit and overflow fall at dot 1 of
  line 261; a $2002 read sees each change if its read cycle comes after the edge, even part way through
  an instruction. nesdev "PPU frame timing"; `ppu_vbl_nmi` `02-vbl_set_time`, `03-vbl_clear_time`;
  `sprite_hit_tests` `09-timing`.
- **FCEUmm:** the flags are set (`PPU_status |= 0x80`) and cleared (`PPU_status &= 0x1f`) between
  `X6502_Run` chunks (`src/ppu.c:1211-1212`, `1236`); a chunk always finishes the instruction it is in
  (`src/x6502.c:402-470`), so a read sees those changes as they stood when the reading instruction
  started. Sprite 0 hit and overflow raised while rendering go through `FCEUPPU_LineUpdate` inside the
  read and are seen at once.
- **Affects:** per-frame timing and RAM (boot VBlank waits; the sprite 0 "wait for clear" loop at the
  end of VBlank).
- **Implemented:** `Ppu::status_at_instr` (field, bits 5-7), `Bus::clock_instr` (snapshot at
  instruction start), `Ppu::read_status` (VBlank from the snapshot, cleared only if seen; bits 5/6 as
  live | snapshot), `Bus::write_ctrl` (FQ-04 uses the snapshot).
- **Removal:** delete `status_at_instr` and read the live flags in `Ppu::read_status`; `02-vbl_set_time`,
  `03-vbl_clear_time` and `09-timing` pass again.

## FQ-04 — A $2002 read never cancels an NMI; enabling NMI during VBlank gives a delayed NMI

- **Hardware:** reading $2002 at or just after the VBlank edge suppresses that frame's NMI; setting
  $2000 bit 7 while the flag is up raises an NMI after the next instruction boundary; clearing bit 7 near
  the edge cancels a pending NMI. nesdev "NMI"; `ppu_vbl_nmi` `04-nmi_control`, `06-suppression`,
  `07-nmi_on_timing`, `08-nmi_off_timing`.
- **FCEUmm:** `A2002` never touches the NMI (`src/ppu.c:189-203`); `B2000` raises `TriggerNMI2` only on
  a 0→1 enable while `PPU_status` bit 7 is set and never cancels (`src/ppu.c:270-280`); `FCEU_IQNMI2`
  becomes `FCEU_IQNMI` one instruction later (`src/x6502.c:348-350`, `413-415`).
- **Affects:** per-frame timing and RAM (when the NMI handler starts).
- **Implemented:** `Ppu::fceu_nmi` (value 2), `Bus::write_ctrl`, `Ppu::read_status` (no NMI cancel),
  `Bus::fceumm_nmi`.
- **Removal:** delete the compat branches; `nmi_pending` and TetaNES' edge detector handle NMI again;
  `04`, `06`, `07`, `08` pass again.

## FQ-05 — NMI requested 12 dots after VBlank, taken at the next instruction start

- **Hardware:** /NMI goes low with the VBlank flag at dot 1 of line 241; the CPU takes it after the
  instruction in progress, polling on the second-to-last cycle. nesdev "NMI", "CPU interrupts";
  `ppu_vbl_nmi` `05-nmi_timing`.
- **FCEUmm:** after setting the flag it runs 12 more dots, then `TriggerNMI` (`src/ppu.c:1211-1227`,
  `src/x6502.c:344-346`); the run loop takes the NMI before the next instruction, 7 cycles
  (`src/x6502.c:406-437`).
- **Affects:** per-frame timing and RAM (NMI entry is 2-6 cycles later than hardware, so every
  NMI-relative cycle count shifts: sprite-0 polls, frame-boundary placement of main-loop code).
- **Implemented:** `Ppu::start_vblank` (no `nmi_pending` in compat), `Bus::fceumm_events` (request at
  line 241 dot 13), `Ppu::fceu_nmi` (value 1), `Bus::clock_instr` / `Bus::fceumm_nmi` (take it before
  the instruction).
- **Removal:** delete the compat request and service path; `05-nmi_timing` passes again.

## FQ-06 — 0-cycle CPU reset

- **Hardware:** reset/power-on takes 7 cycles (an interrupt sequence with the writes suppressed) before
  the first instruction. nesdev "CPU interrupts", "CPU power up state"; `cpu_reset`.
- **FCEUmm:** `X6502_Reset` only sets `FCEU_IQRESET` (`src/x6502.c:352-354`); the run loop loads the
  vector without `ADDCYC` (`src/x6502.c:407-412`).
- **Affects:** boot only (shifts the CPU against the PPU by 7 cycles; together with FQ-07 sets the whole
  run's CPU/PPU phase).
- **Implemented:** `Bus::reset` (fetches the reset vector without spending any cycles; the 7-cycle
  sequence was deleted at Z3).
- **Removal:** always run the 7 reset cycles; `cpu_reset` passes; re-measure FQ-07.

## FQ-07 — PPU starts at scanline 240, one dot ahead

- **Hardware:** the PPU starts at the top of the frame and the CPU/PPU clock alignment is one of four
  phases chosen at power-on. nesdev "PPU power up state", "CPU ALL"; TetaNES starts at line 0 dot 0.
- **FCEUmm:** each frame begins with line 240 (`src/ppu.c:1204-1211`: `X6502_Run(256 + 85)` then VBlank),
  so after power-on the CPU runs from the post-render line, a whole frame before the PPU completes its
  first; measured against the corpus, the PPU is one dot further on (dot 2) than the CPU cycle count puts
  it.
- **Affects:** boot, then the CPU/PPU phase for the whole run (which instruction straddles each frame
  boundary and each VBlank edge).
- **Implemented:** `Ppu::reset` (compat branch: scanline 240, cycle 2, scanline flags).
- **Removal:** start at line 0 dot 0 again; regenerate the base state.

## FQ-08 — Frame boundary at dot 1 of line 240

- **Hardware:** no frame boundary exists; it is the emulator's convention for "one frame". TetaNES ends a
  frame when the PPU enters line 240 (dot 0).
- **FCEUmm:** a frame ends after `DoLine` for line 239 (`src/ppu.c:1278-1288`, `687-786`), 341 dots before
  VBlank rises; relative to TetaNES' dot numbering that is dot 1 of line 240.
- **Affects:** frame boundary only: which instructions count toward which frame, and therefore when
  movie inputs and pokes apply. Matters while Zelda's main loop runs outside NMI (boot, loading).
- **Implemented:** `Bus::end_scanline` (no increment in compat), `Bus::fceumm_events` (increment at
  line 240 dot 1), `Bus::ppu_clock` (calls it).
- **Removal:** increment on entering line 240 again. No test ROM; corpus ladder self-diffs re-baselined.

## FQ-09 — Odd-frame dot skip even when rendering is off

- **Hardware:** on NTSC the pre-render line's last dot is skipped on odd frames only while rendering is
  enabled. nesdev "PPU frame timing" ("even/odd frames"); `ppu_vbl_nmi` `09-even_odd_frames`,
  `10-even_odd_timing`.
- **FCEUmm:** `X6502_Run(16 - kook); kook ^= 1;` every non-dead frame, not gated on `ScreenON`/`SpriteON`
  (`src/ppu.c:1262-1263`).
- **Affects:** per-frame timing (frames are one dot shorter every other frame during boot and screen
  transitions with rendering off).
- **Implemented:** `Bus::ppu_clock` (the branch when rendering is off; parity from
  `Ppu::frame_is_odd` matches FCEUmm's `kook` given FQ-01).
- **Removal:** delete the branch; `09` and `10` pass again.

## FQ-10 — 512-cycle OAM DMA, run inside `STA $4014`

- **Hardware:** OAM DMA halts the CPU for 513 cycles, 514 if it starts on an odd (put) cycle, starting
  at the next read cycle after the write. nesdev "PPU registers" (OAMDMA), "DMA"; `sprdma_and_dmc_dma`,
  `cpu_interrupts_v2`.
- **FCEUmm:** `B4014` performs 256 `X6502_DMR`/`X6502_DMW` pairs of one cycle each inside the write
  (`src/ppu.c:374-380`, `src/x6502.c:74-82`): 512 cycles, no halt or alignment cycle, counted toward the
  `STA` itself (so they reach FQ-12's DMC timer at the next instruction's hook).
- **Affects:** per-frame timing (every frame's NMI handler: 1-2 cycles, and DMC fetch placement).
- **Implemented:** `Bus::clock_instr` (runs the DMA after the `STA`), `Bus::read` (a halted read runs
  it), `Bus::handle_dma_fceumm`.
- **Removal:** delete the compat paths; `sprdma_and_dmc_dma` passes again.

## FQ-11 — 4-read DMC DMA

- **Hardware:** a DMC sample fetch steals 1-4 cycles depending on what the CPU is doing and on alignment
  with OAM DMA. nesdev "APU DMC", "DMA"; `dmc_dma_during_read4`, `sprdma_and_dmc_dma`.
- **FCEUmm:** `DMCDMA` performs four `X6502_DMR` reads of the sample address (`src/sound.c:502-508`) from
  the per-instruction sound hook.
- **Affects:** per-frame timing while samples play.
- **Implemented:** `Bus::fceumm_dmc_fetch` (four read cycles after the instruction whose hook asked).
- **Removal:** delete with FQ-12; `dmc_dma_during_read4` passes again.

## FQ-12 — DMC fetch timed by FCEUmm's per-instruction sound hook

- **Hardware:** the DMC timer counts CPU cycles exactly; the reader requests a fetch as soon as the sample
  buffer empties, and the CPU services it at its next read cycle (with halt/alignment). nesdev "APU DMC";
  `dmc_dma_during_read4`, `dmc_basics`.
- **FCEUmm:** `FCEU_SoundCPUHook` (`src/sound.c:525-555`) runs once per instruction, after the opcode's
  base cycles are counted: if the buffer is empty and bytes remain it fetches at once (`DMCDMA`,
  `src/sound.c:503-523`), then counts the cycles since the last hook (the opcode's base cycles plus any
  extra cycles of the previous instruction, DMA included) down `DMCacc`, emptying the buffer every eight
  bits. `$4010/$4012/$4013/$4015` per `Write_DMCRegs`/`StatusWrite` (`src/sound.c:303-370`).
- **Affects:** per-frame timing while samples play (Zelda plays DMC sound effects).
- **Implemented:** `FceuDmc` in `apu/dmc.rs` (state, `write`, `hook`), `Apu::write` (register writes),
  `Bus::clock_instr` (hook before the opcode fetch). The hardware sample reader was deleted with audio
  at Z4 (it fed only sound and the DMA requests FQ-12 dropped).
- **Removal:** delete `FceuDmc`; restore the hardware reader (`Dmc` timer, buffer, `load_buffer`, DMA
  requests and IRQ, from Z3's `apu/dmc.rs`) and the CPU's DMC DMA; `dmc_dma_during_read4` and
  `dmc_basics` pass again.

## Not quirks

- **OAM digest masking (comparison tool only).** FCEUmm stores all 8 bits of each sprite's attribute byte;
  the PPU has no bits 2-4 there and TetaNES stores them masked (`& 0xE3`). `tf`'s FCEUmm backend reads
  OAM as $2004 would return it (`crates/tools/src/oracle.rs`, `Domain::Oam`). No emulator behaviour is
  involved; it disappears with the oracle in step 18c.
- **Zero-filled battery PRG-RAM at power-on.** Part of the TFMOVIE v1 power-on spec (an empty battery
  file), applied by each backend's harness, not by emulation code.
