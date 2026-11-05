# zcore upstream

zcore is TetaNES's `tetanes-core` (MIT OR Apache-2.0, <https://github.com/lukexor/tetanes>; licence
files alongside) cut down to what *The Legend of Zelda* needs, one verified subtraction at a time.

## Base

- `../tetanes` commit `a0a6b17f` ("Merge pull request #480 ... release-plz-2026-09-07"), `tetanes-core/`.
- Imported (Z0) from branch `bugs` at `98542624`, which adds on top of the base:

| Commit | Kind | Summary |
|---|---|---|
| `955220a1` | fceumm-compat | `Config::fceumm_compat` with quirks FQ-01..FQ-11 |
| `0eab30a5` | fceumm-compat | FQ-04/FQ-05 NMI serviced after the instruction; tests for FQ-01..FQ-11 |
| `e820ad48` | fceumm-compat | FQ-12 FCEUmm DMC fetch timing; FQ-10 OAM DMA inside `STA $4014` |
| `c719d7b8` | fix | No sprites from line 239's evaluation on line 0 |
| `81b32c70` | fix | Draw sprite pixels at x = 255 |
| `98542624` | fceumm-compat | FQ-03 covers the end-of-VBlank clear of sprite 0 hit and overflow |

The two `fix:` commits are hardware fixes suitable for upstream TetaNES; the `fceumm-compat` commits are
off by default there. The quirks are documented in `FCEUMM_QUIRKS.md`.

## Ladder

Each stage is checked with `tf ladder-diff target/ladder/s1-tetanes --backend zcore` (all 11 digest
domains, cycles included, over the whole corpus, against the goldens of the S1 TetaNES backend; since every
stage must be identical, that one golden set is the baseline for all of them) plus `cargo test -p zcore`.

| Stage | Change |
|---|---|
| Z0 | Verbatim import of `tetanes-core` from `bugs` `98542624`; crate renamed `zcore`; `cdylib` and wasm dependencies dropped. Checked against the `tetanes` backend. |
| Z1 | Removed run-ahead, frame speed and `Clocked`, `clock_seconds`/`clock_scanline`, the deprecated shims and crate-wide `Error`, Game Genie (`genie.rs`), patches (`patch.rs`), the debugger (`debug.rs`, `debug/expr.rs`, breakpoints, PC history, code map, call stack, `Disasm`), `action.rs` (the ROM-test harness keeps its own action enum), save-state files and SRAM files/directories (`fs.rs`, `sys/`, `time.rs`), and the `dirs`, `flate2`, `cfg-if` dependencies (`bincode` is test-only now). The game database went with `fs.rs`, one stage early; Zelda's entry matched its header. The TetaNES README/CHANGELOG and multi-ROM bench were replaced by a zcore README and a Zelda `frames` bench. |
| Z2 | Production boards reduced to `Sxrom`/MMC1 (MMC1B/C only: the MMC1A revision, SOROM/SXROM 32K PRG-RAM, 512K PRG and SUROM submapper paths went, and PRG-RAM is the 8K Zelda maps); `Nrom` is `#[cfg(test)]`. Deleted every other board, mapper revisions, the per-board hooks (`MapperOps`: IRQ, DMA, audio, PPU-bus watching, board-served reads, battery staging, registers), ExRAM and `BatteryExt`, the mapper-number board-name table, the `m004`/`m005` ROM groups and the three CPU/PPU ROMs on other boards (`dummy_reads`, `read_buffer`: CNROM; `_240pee`: UxROM). |
| Z3 | NTSC only (`NesRegion`, PAL/Dendy tables and PAL sprite evaluation gone), two raw-byte controllers (no four-player adapter, Zapper, turbo or D-pad rule), no PPU warm-up, no `RamState` (power-on RAM is `Bus::POWER_ON_RAM` = $FF), and `fceumm_compat` hard-wired: every FQ path is unconditional and the hardware alternatives (7-cycle reset, hardware OAM/DMC DMA, $2002 race, edge-detected NMI, frame increment at line 240) are deleted (restore them from Z2 for plan step 18). Dropped the PAL APU and Zapper ROM groups. 51 test ROMs/unit tests that assert hardware timing are `#[ignore]`d with the FQ IDs they trip; the attribution of the boot-alignment group (FQ-01/06/07) was not bisected. |
| Z4 | No audio: deleted the band-limited synthesiser, filter chain, mixer tables, samples, sample rates, speed/rate control, channel muting and `HeadlessMode::NO_AUDIO`, and with them the pulse/triangle/noise timers, sequencers, envelopes, sweeps, the linear counter and the noise shift register, and the hardware DMC timer/reader/output (FQ-12 already dropped its DMA requests, so it never fetched, counted down or raised its IRQ). Kept: every APU register write's frame-counter catch-up, the four length counters (`$4015` bits 0-3, halt/reload timing), the frame counter and its IRQ, the DMC's `$4015` bit and the "clock every cycle while a sample plays" rule, and `FceuDmc` (FQ-11/FQ-12). The eight ROMs judged by sound (`apu_env`, `dmc_pitch`, `lin_ctr`, `noise_pitch`, `square_pitch`, `sweep_cutoff`, `sweep_sub`, `triangle_pitch`) and the audio-hash/profile harness went; the `dmc_tests` tone counters read pulse 1's length counter as before. The ladder's APU domain writes the DMC IRQ byte as a constant 0. |
| Z5 | Flattened the emulated state into `bus::State: Copy` (22,680 bytes; `MachineState` = `State` + frame count): work RAM, OAM, palette and sprite tables inline, and `Memory` a fixed 19 KiB arena (unmapped zero page, 8K PRG-RAM, 8K CHR, 2K CIRAM) addressed by `u16`-offset page tables, plus `u32` PRG-ROM page offsets. The PRG-ROM image (mirrored to 128 KiB) is an `Arc<PrgRom>` outside the state, held by the running `Bus` (= `State` + ROM + frame buffer + render flag, which derefs to its `State`); `Rom` shares one image with every `ZCore`. The frame buffer moved out of `Ppu` into `Bus` and is lent to the pixel path only when rendering; `skip_rendering`, `Frame`/`Buffer`, ROM CRC/`is_same_cart`/`restore_rom_from`, `StateMismatch`, `Bus::load_state`/`swap_state`, `MemoryLayout`, four-screen mirroring, carts larger than 128K PRG/8K PRG-RAM/8K CHR, and every `#[serde(skip)]` derived field (all state is now serialized, so a decoded state needs no rebuild) are gone. `snapshot`/`restore` are plain copies; `ZCore` runs a `Bus` directly (`ControlDeck` is the test harness's). Test changes: memory/mapper/cart tests rewritten for the fixed layout; `loading_a_state_refreshes_the_frame` became `the_frame_after_a_restore_is_the_restored_consoles` (the frame buffer is no longer state); new `machine_state_is_small_and_copy`, `snapshot_restore_replays_identically`. |
| Z6 | Final API (plan step 11). Dropped `tracing`, `thiserror`, `bincode`, `tracing-subscriber` and the unused `rand`: `RomError`, `StateFormatError`, `cart::Error`, `mapper::Error` and the harness's `control_deck::Error` have hand-written `Display`/`Error`; the PPU's three "unexpected PPU memory access" logs became silent (the PPU bus is 14 bits, so they never fire). The `trace` feature and its `println!` are gone: `ZCore::set_trace(Option<TraceHook>)` installs a per-instruction callback handed a `TraceRow` (PC, opcode, A/X/Y/P/SP, CPU cycle, scanline, dot) before the opcode fetch; unset, it costs one branch per instruction. `ControlDeck` and `video` (RGBA/NTSC filters) are `#[cfg(test)]`, kept only for the test-ROM harness (unused methods, `ParseVideoFilterError`, `apply_filter_into` deleted); the `prelude` is gone and the emulator modules are `#[doc(hidden)]`. `ZCore::cycles()` is a `u64` accumulated per frame (carried in `MachineState`), no longer the CPU's wrapping 32-bit counter. `tf`'s zcore backend drives `ZCore` (internals through the hidden `ZCore::bus()`, traces through `set_trace`), and `tf bench` times render on and off. New tests: render on/off over `chain-milestone1-111` (whole `MachineState` per frame; the corpus has no `transition_under_fire`), `title_idle_3600` frame 600 RGB against the oracle golden's index digest with a non-backdrop pixel in row 0 or 239, trace hook observes without changing state, BLAKE3 pin of the encoded power-on `TFZSTATE` (`state_file::VERSION` stays 1). |

## Performance

After Z6 the core ran ~620/700 frames/s per core (render on/off) against FCEUmm's ~3,070. The changes
below keep the emulation bit-identical: each commit passes `tf ladder-diff target/ladder/s1-tetanes`
(all 11 domains), `tf diff --all` against the oracle goldens, `cargo test -p zcore` and the zelda
`zcore_session` test, and the whole `MachineState` (postcard, BLAKE3 per frame) of every corpus movie
matches the Z6 build `99b97b4`, render on and off. Numbers are `tf bench` best of 3 (render on / off).

| Commit | Change | frames/s |
|---|---|---|
| `24fe4b3` | Lazy PPU: the CPU catches it up only before a `$2000-$3FFF` access, a board register write, a trace row or a frame-boundary/NMI-request dot falling due (`Bus::ppu_sync`, `Ppu::next_event_clock`); FQ-03's `status_at_instr` is taken at an instruction's first catch-up, as of its start. Catch-ups run scanline spans (`ppu/span.rs`): rendering lines a tile at a time with sprite evaluation as its own pass, idle lines in O(1); headless lines skip the BG fetches the last three tiles overwrite. `ppu_clock` stays the general dot for settling `$2001`/`$2006` updates. | 623/705 → 1148/2095 |
| `ea6509f` | APU deadline (`Apu::next_clock`; a playing DMC sample no longer forces per-cycle clocking, which changes nothing without the Z4 sample reader); bit-level `handle_interrupts`; sprite evaluation on locals; whole-tile BG-only pixels; live sprite slots and last garbage fetches only; whole-tile prefetch. | → 1727/3155 |
| `657c009` | One per-cycle deadline for the MMC1 write countdown and the APU (`Bus::tick_deadline`); interrupt polling skipped while every input and flag is low (`Bus::irq_quiet`). | → 1849/3631 |
| `c825138` | Sprite evaluation as (odd, even) pairs, skipping runs of out-of-range sprites; render-on colour table for BG-only tiles. | → 2295/4143 |
| (last) | Out-of-range runs found by a direct scan; the evaluation's "done" tail in closed form. | → 2415/4298 |

Tried and dropped: a 256-arm `match` opcode dispatch instead of the function-pointer table (+5%
instructions, +4% cycles), fat LTO (no gain), the colour table on tiles with sprites (+3% cycles).

## After the ladder

| Change | Kind | Check |
|---|---|---|
| The PPU's internal OAMADDR reset during sprite fetches no longer clears the I/O latch (it went through `write_oamaddr`); a $2002 read drives its status bits into the latch instead of OR-ing them (upstream TetaNES bugs, also in `a0a6b17f`). FQ-13: the latch never decays (FCEUmm `PPUGenLatch`). | fix + FQ-13 | `tf trace-diff` agrees where it diverged (`sword_cave_in_out` frame 300, `LDA $2002` at `$E573`); `tf diff --all` 0 of 63; the ladder baseline moves to `target/ladder/z6-fq13`: against `s1-tetanes` only the PPU domain (the latch byte, 1,351,805 frame records) and the CPU domain (2 records: a register loaded from the latch at a frame boundary) changed; the `open_bus` test ROM is `#[ignore]`d under FQ-13 (it passes with the decay restored). |

## Plan step 18a: hardware behaviour restored

Each FCEUmm quirk's hardware behaviour back, one commit per quirk, gated on the promoted models
(`scripts/port_gate.sh`, `docs/experiments/round9.md`). `tf bench` render on / off after each.

| Commit | Quirk | Change | frames/s |
|---|---|---|---|
| `zcore: restore hardware PPU I/O latch decay (FQ-13 removed)` | FQ-13 | Per-bit I/O latch decay after 36 frames without a refresh (`Ppu::open_bus_now`, `drive_open_bus`, `write_open_bus`, `open_bus_refreshed`); `ppu_bus_read`/`ppu_bus_write` no longer touch the latch (the register paths drive it). `TFZSTATE` v2, v1 decoded through `state_file::v1`. `open_bus` passes. | 2,441 / 4,697 |
| `zcore: restore hardware odd-frame skip only while rendering (FQ-09 removed)` | FQ-09 | The rendering-off odd-frame skip in `Bus::ppu_clock` / `Bus::idle_span` deleted. `vbl_nmi_basics`, `vbl_nmi_clear_timing`, `vbl_nmi_frame_basics` pass. | 2,419 / 4,595 |
