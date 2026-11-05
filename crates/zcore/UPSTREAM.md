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
