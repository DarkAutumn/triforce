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
