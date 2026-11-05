# Round 9: hardware behaviour in zcore, one FCEUmm quirk at a time (plan step 18a)

zcore reproduces FCEUmm, the emulator the promoted models were trained on, through 13 labelled quirks
(`crates/zcore/FCEUMM_QUIRKS.md`). Step 18a restores the hardware behaviour for each, one commit per quirk (or
per entangled pair), and keeps it unless the promoted models get worse.

## Setup

- **Gate:** `scripts/port_gate.sh LABEL [SEED]` on the rebuilt module (`maturin develop --release`): every
  promoted specialist evaluated at 20 episodes per start state (sampled, CPU, 16 workers), and the five chain
  segments at 40 runs each (full `milestone1`, full `milestone1-fairy`, post-D1 main and fairy from `0_37c_2`,
  D2 from `2_7ds_2`). ~11-15 minutes on 20 cores. Outputs `target/gate/LABEL/` (gitignored).
- **Baselines** (the FCEUmm-faithful build, `fe6e5af`): `baseline` (seed 0) and `baseline-s1` (seed 1, the
  noise run). Chain totals 32 and 24 (spread 8), so the chain floor is 24 − 8 − 4 = **12**.
- **Decision rule:** keep the hardware behaviour unless, confirmed by a second gate run at seed 1, a
  specialist's success falls more than 15 points below its old `models/<s>/<run>/eval.json`, or it is clearly
  worse than both baselines beyond their spread (success and `mean_health_change`), or the chain total falls
  below the floor.
- **State files:** the start states are the step-17 `TFZSTATE` v1 files. A restore that changes
  `MachineState`'s layout bumps `state_file::VERSION`; v1 bodies are decoded through frozen v1 copies of the
  changed types (`state_file::v1`, deleted in 18b with the regenerated states).
- Cells are success / `mean_health_change`; "old" is the promoted run's `eval.json` (round 7, old stack).

## FQ-13: the PPU I/O latch decays

**Change.** The PPU's CPU-facing I/O latch ("open bus") decays per bit: each bit refreshed by a register access
(writes drive all eight; a $2002 read drives bits 5-7, $2004 and non-palette $2007 reads all eight, palette
$2007 reads bits 0-5) holds for 36 frames (~600 ms, blargg `ppu_open_bus`) and then reads 0
(`Ppu::open_bus_now`, `Ppu::drive_open_bus`, per-bit `Ppu::open_bus_refreshed`). This models the hardware
decay instead of TetaNES' old approximation (clearing the whole latch at the end of every VBlank). FCEUmm's
latch never decayed. Loaded v1 states count every bit as refreshed at the save. `state_file::VERSION` 1 → 2.
Tests: the `open_bus` test ROM passes and is un-ignored; all zcore, zelda, env and pool tests pass (no movie
desynced: Zelda masks the latch bits). `tf bench` 2,337-2,439 / 4,401-4,441 before, 2,419-2,441 /
4,623-4,697 after (render on/off).

| Specialist | Old | Baseline s0 | Baseline s1 | FQ-13 s0 |
|---|---:|---:|---:|---:|
| boss-aquamentus | 100.0% | 100.0% / +0.592 | 98.3% / +0.575 | 100.0% / +0.583 |
| d2-rope | 95.2% | 94.0% / −0.254 | 93.6% / −0.231 | 93.6% / −0.251 |
| goriya | 76.7% | 78.3% / −0.758 | 77.9% / −0.725 | 80.4% / −0.758 |
| ow-leever-peahat | 100.0% | 97.1% / −0.325 | 96.2% / −0.296 | 97.9% / −0.281 |
| ow-moblin | 97.1% | 95.3% / −0.215 | 96.4% / −0.236 | 95.8% / −0.231 |
| ow-octorok | 99.0% | 98.3% / −0.073 | 98.2% / −0.066 | 98.0% / −0.065 |
| ow-start | 98.5% | 100.0% / +0.278 | 100.0% / +0.287 | 100.0% / +0.278 |
| ow-zora | 97.4% | 95.4% / −0.191 | 95.4% / −0.206 | 95.6% / −0.222 |
| stalfos-keese-gel | — | 96.8% / −0.035 | 95.7% / −0.041 | 96.0% / −0.047 |
| wallmaster | 87.5% | 98.3% / +0.167 | 100.0% / +0.208 | 100.0% / +0.175 |

| Segment | Baseline s0 | Baseline s1 | FQ-13 s0 |
|---|---:|---:|---:|
| Full milestone1 | 1/40 | 1/40 | 0/40 |
| Full milestone1-fairy | 3/40 | 2/40 | 2/40 |
| Post-D1 main | 7/40 | 3/40 | 1/40 |
| Post-D1 fairy | 6/40 | 6/40 | 7/40 |
| D2 | 15/40 | 12/40 | 16/40 |
| **Total** | 32 | 24 | 26 |

**Decision: keep.** Every specialist is within the baselines' range or above it; the chain total (26) is between
the two baselines. No seed-1 confirmation needed.

## FQ-09: the odd-frame dot skip only while rendering

**Change.** The pre-render line's last dot is skipped on odd frames only with rendering enabled (nesdev "PPU
frame timing"); FCEUmm skipped it every other frame regardless. Deleted the rendering-off branches in
`Bus::ppu_clock` and `Bus::idle_span`; the FQ-09 compat test became the hardware test
`odd_frames_keep_every_dot_with_rendering_off`. No state change. Tests: `vbl_nmi_basics`,
`vbl_nmi_clear_timing` and `vbl_nmi_frame_basics` pass and are un-ignored; every zcore, zelda, env and pool
test passes (no movie desynced). Gate build: FQ-13 + FQ-09. `tf bench` 2,396-2,419 / 4,526-4,595.

| Specialist | Old | Baseline s0 | Baseline s1 | FQ-09 s0 |
|---|---:|---:|---:|---:|
| boss-aquamentus | 100.0% | 100.0% / +0.592 | 98.3% / +0.575 | 100.0% / +0.517 |
| d2-rope | 95.2% | 94.0% / −0.254 | 93.6% / −0.231 | 95.2% / −0.199 |
| goriya | 76.7% | 78.3% / −0.758 | 77.9% / −0.725 | 75.4% / −0.706 |
| ow-leever-peahat | 100.0% | 97.1% / −0.325 | 96.2% / −0.296 | 97.1% / −0.302 |
| ow-moblin | 97.1% | 95.3% / −0.215 | 96.4% / −0.236 | 96.4% / −0.196 |
| ow-octorok | 99.0% | 98.3% / −0.073 | 98.2% / −0.066 | 98.0% / −0.057 |
| ow-start | 98.5% | 100.0% / +0.278 | 100.0% / +0.287 | 100.0% / +0.278 |
| ow-zora | 97.4% | 95.4% / −0.191 | 95.4% / −0.206 | 96.1% / −0.202 |
| stalfos-keese-gel | — | 96.8% / −0.035 | 95.7% / −0.041 | 96.5% / −0.044 |
| wallmaster | 87.5% | 98.3% / +0.167 | 100.0% / +0.208 | 100.0% / +0.292 |

| Segment | Baseline s0 | Baseline s1 | FQ-09 s0 |
|---|---:|---:|---:|
| Full milestone1 | 1/40 | 1/40 | 3/40 |
| Full milestone1-fairy | 3/40 | 2/40 | 5/40 |
| Post-D1 main | 7/40 | 3/40 | 1/40 |
| Post-D1 fairy | 6/40 | 6/40 | 6/40 |
| D2 | 15/40 | 12/40 | 12/40 |
| **Total** | 32 | 24 | 27 |

**Decision: keep.** goriya's success (75.4%) is under both baselines (78.3/77.9), but its health change is
better than both (−0.706 vs −0.758/−0.725) and it is 1.3 points from its old eval; d2-rope, ow-zora and
wallmaster are above both baselines. The chain total (27) is between the baselines.

## FQ-10: OAM DMA halts the CPU for 513/514 cycles

**Change.** A `$4014` write halts the CPU at its next read (the next instruction's opcode fetch, or an
interrupt's dummy read): the halted read is repeated, one alignment cycle follows when the halt lands on a put
(odd) cycle, then 256 get/put pairs, so 513 or 514 cycles (nesdev "DMA"; TetaNES' `handle_dma` without the DMC
half, which comes with FQ-11/12). FCEUmm ran 512 cycles inside the `STA`. `Bus::handle_dma` replaces
`handle_dma_fceumm`; the FQ-10 compat test became `oam_dma_takes_513_or_514_cycles_from_the_next_read`.
FQ-03's instruction-start snapshot is re-taken after the DMA, since FCEUmm's instruction after the `STA`
started after it. No state change. Tests: `int_irq_and_dma` passes and is un-ignored; `spr_hit_timing_order`
(passing before on FCEUmm timing) fails on hardware DMA with FCEUmm's VBlank/NMI timing and is ignored under
FQ-03..FQ-05 until those are restored; every zelda, env and pool test passes (no movie desynced). Gate build:
FQ-13 + FQ-09 + FQ-10. `tf bench` 2,452-2,456 / 4,384-4,577.

| Specialist | Old | Baseline s0 | Baseline s1 | FQ-10 s0 |
|---|---:|---:|---:|---:|
| boss-aquamentus | 100.0% | 100.0% / +0.592 | 98.3% / +0.575 | 100.0% / +0.650 |
| d2-rope | 95.2% | 94.0% / −0.254 | 93.6% / −0.231 | 94.3% / −0.218 |
| goriya | 76.7% | 78.3% / −0.758 | 77.9% / −0.725 | 77.5% / −0.675 |
| ow-leever-peahat | 100.0% | 97.1% / −0.325 | 96.2% / −0.296 | 96.7% / −0.308 |
| ow-moblin | 97.1% | 95.3% / −0.215 | 96.4% / −0.236 | 95.6% / −0.203 |
| ow-octorok | 99.0% | 98.3% / −0.073 | 98.2% / −0.066 | 97.9% / −0.074 |
| ow-start | 98.5% | 100.0% / +0.278 | 100.0% / +0.287 | 100.0% / +0.278 |
| ow-zora | 97.4% | 95.4% / −0.191 | 95.4% / −0.206 | 95.9% / −0.226 |
| stalfos-keese-gel | — | 96.8% / −0.035 | 95.7% / −0.041 | 96.0% / −0.063 |
| wallmaster | 87.5% | 98.3% / +0.167 | 100.0% / +0.208 | 98.3% / +0.217 |

| Segment | Baseline s0 | Baseline s1 | FQ-10 s0 |
|---|---:|---:|---:|
| Full milestone1 | 1/40 | 1/40 | 0/40 |
| Full milestone1-fairy | 3/40 | 2/40 | 2/40 |
| Post-D1 main | 7/40 | 3/40 | 2/40 |
| Post-D1 fairy | 6/40 | 6/40 | 8/40 |
| D2 | 15/40 | 12/40 | 16/40 |
| **Total** | 32 | 24 | 28 |

**Decision: keep.** Every specialist is within or near the baselines' range: ow-octorok (97.9%, −0.074) and
stalfos-keese-gel's health (−0.063) are a hair outside, well inside episode noise (±0.6 points at 660
episodes); goriya's health is the best of the three runs. Chain total 28 (floor 12).

## FQ-11 + FQ-12: the hardware DMC reader and its DMA

**Change (one step: FQ-11's 4-read fetch only exists inside FQ-12's hook).** The DMC's timer counts CPU cycles
whether or not a sample plays; each expiry shifts a bit, every eighth starts an output cycle that empties the
buffer and requests the next byte, and a `$4015` enable with an empty buffer requests one 2-3 cycles later
(`apu/dmc.rs` `Dmc`, from TetaNES' reader minus the output unit). The request halts the CPU at its next read:
halt, dummy, alignment and fetch, 3-4 cycles alone, fewer inside an OAM DMA (`Bus::handle_dma`, TetaNES' whole
loop now). The last byte raises the DMC IRQ (`$4015` bit 7) when enabled; `$4015` bit 4 counts down. The DMC
stays off the per-cycle path: `Apu::next_clock` includes the cycle of its next fetch request
(`Dmc::next_event`), and `Bus::cpu_clock_due` turns a request into the CPU's DMA flags. FCEUmm's
per-instruction hook (`FceuDmc`, `fceumm_dmc_fetch`) is deleted. `TFZSTATE` v3; v1 DMC state converts from
FCEUmm's fetcher (registers, address, bytes left, buffer, bit position, cycles to the next bit). The compat
tests became `dmc_fetches_halt_the_cpu_once_per_byte`. Tests: `dmc_basics`, `dmc_rates`, `dmc_status`,
`dmc_dma_2007_read`, `dmc_dma_2007_write`, `dmc_dma_4016_read`, `sprdma_and_dmc_dma`,
`sprdma_and_dmc_dma_512` and `reset_works_immediately` pass and are un-ignored (`dpcmletterbox` and
`instr_timing` still fail, now attributed to the remaining timing quirks); every zelda, env and pool test
passes (no movie desynced). Gate build: FQ-13, 09, 10, 11, 12. `tf bench` 2,531-2,584 / 4,800-4,908 (faster:
no per-instruction hook).

| Specialist | Old | Baseline s0 | Baseline s1 | FQ-11-12 s0 |
|---|---:|---:|---:|---:|
| boss-aquamentus | 100.0% | 100.0% / +0.592 | 98.3% / +0.575 | 100.0% / +0.575 |
| d2-rope | 95.2% | 94.0% / −0.254 | 93.6% / −0.231 | 93.1% / −0.249 |
| goriya | 76.7% | 78.3% / −0.758 | 77.9% / −0.725 | 75.8% / −0.683 |
| ow-leever-peahat | 100.0% | 97.1% / −0.325 | 96.2% / −0.296 | 96.2% / −0.344 |
| ow-moblin | 97.1% | 95.3% / −0.215 | 96.4% / −0.236 | 96.4% / −0.201 |
| ow-octorok | 99.0% | 98.3% / −0.073 | 98.2% / −0.066 | 98.0% / −0.083 |
| ow-start | 98.5% | 100.0% / +0.278 | 100.0% / +0.287 | 100.0% / +0.278 |
| ow-zora | 97.4% | 95.4% / −0.191 | 95.4% / −0.206 | 95.7% / −0.203 |
| stalfos-keese-gel | — | 96.8% / −0.035 | 95.7% / −0.041 | 95.7% / −0.037 |
| wallmaster | 87.5% | 98.3% / +0.167 | 100.0% / +0.208 | 100.0% / +0.192 |

| Segment | Baseline s0 | Baseline s1 | FQ-11-12 s0 |
|---|---:|---:|---:|
| Full milestone1 | 1/40 | 1/40 | 0/40 |
| Full milestone1-fairy | 3/40 | 2/40 | 3/40 |
| Post-D1 main | 7/40 | 3/40 | 3/40 |
| Post-D1 fairy | 6/40 | 6/40 | 9/40 |
| D2 | 15/40 | 12/40 | 17/40 |
| **Total** | 32 | 24 | 32 |

**Decision: keep.** Chain total 32, equal to the better baseline. goriya's success (75.8%) is again under both
baselines with better health (−0.683); every other specialist is within or at the edge of the baselines.

## FQ-02..FQ-05: the VBlank flag and NMI timing

**Change (one step).** `$2002` reads see and clear the live flags at their PPU dot (FQ-03: no instruction-start
snapshot), a read at line 241 dot 0 keeps VBlank from rising that frame (FQ-02), /NMI is VBlank AND the `$2000`
enable bit: raised with VBlank at line 241 dot 1, dropped by a `$2002` read or by clearing the bit, raised again
by setting it during VBlank (FQ-04), and the CPU's edge detector takes it after the instruction whose
second-to-last cycle saw it (FQ-05; FCEUmm requested it 12 dots late and took it at the next instruction start).
The PPU stays lazy: `Ppu::next_event_clock` names both VBlank edges (and FQ-08's frame boundary), and
`Bus::end_cycle` catches the PPU up when one falls due, so /NMI is polled on the cycle per-cycle clocking would
have seen it. `fceu_nmi`, `fceumm_nmi`, `status_at_instr` and the snapshot's catch-up stop are deleted;
`TFZSTATE` v4 (v1 drops the two fields; `state_file::v1`'s macro now takes `#[dropped]` fields anywhere).

**Why together:** FQ-02/03 alone (hardware `$2002` reads with FCEUmm's NMI) failed five ROMs that pass on both
the FCEUmm build and hardware (`instr_brk`, `instr_zp_xy`, `flag_concurrency`, `test_2`, `test_6`); FQ-04's
delayed NMI was defined on FQ-03's snapshot. With all four, three of those still fail: their snapshot frames
line up only without FQ-01's dead frames (checked by removing FQ-01 too: they pass), and `test_2`/`test_6` need
FQ-06/07 as well. They are ignored under FQ-01 and FQ-01/06/07 until those are restored.

**Tests:** un-ignored and passing: `vbl_nmi_set_time`, `vbl_nmi_disable`, `vbl_nmi_on_timing`,
`vbl_nmi_off_timing`, `vbl_nmi_timing`, `vbl_nmi_even_odd_timing`, `int_nmi_and_brk`, `int_nmi_and_irq`,
`scanline`, `spr_hit_timing_basics`, `spr_hit_timing_order` (re-checked: passes again), `spr_overflow_timing`,
`dpcmletterbox`, and the unit tests `read_write_ppu`, `read_status_resets_vblank`. Still failing (boot quirks):
`vbl_nmi_suppression`, `vbl_nmi_even_odd_frames`, `vbl_timing`, `overclock`. The FCEUmm compat tests for FQ-02..05
are deleted. **Movie replay:** `zelda` `session_matches_python_fixtures` (the FCEUmm-recorded parity fixtures)
desyncs in one fixture (`0_39_fairy` step 35, RAM) and is ignored; every other zelda, env and pool test passes.
Gate build: FQ-13, 09..12, 02..05. `tf bench` 2,436-2,592 / 4,738-4,943 (before 2,566 / 5,089: the per-cycle
deadline compare in `end_cycle` costs ~2-4%).

| Specialist | Old | Baseline s0 | Baseline s1 | FQ-02-05 s0 | FQ-02-05 s1 |
|---|---:|---:|---:|---:|---:|
| boss-aquamentus | 100.0% | 100.0% / +0.592 | 98.3% / +0.575 | 100.0% / +0.600 | 100.0% / +0.592 |
| d2-rope | 95.2% | 94.0% / −0.254 | 93.6% / −0.231 | 93.1% / −0.257 | 94.0% / −0.279 |
| goriya | 76.7% | 78.3% / −0.758 | 77.9% / −0.725 | 80.0% / −0.704 | 76.2% / −0.767 |
| ow-leever-peahat | 100.0% | 97.1% / −0.325 | 96.2% / −0.296 | 97.5% / −0.319 | 97.1% / −0.304 |
| ow-moblin | 97.1% | 95.3% / −0.215 | 96.4% / −0.236 | 97.2% / −0.192 | 96.1% / −0.239 |
| ow-octorok | 99.0% | 98.3% / −0.073 | 98.2% / −0.066 | 97.9% / −0.075 | 98.3% / −0.077 |
| ow-start | 98.5% | 100.0% / +0.278 | 100.0% / +0.287 | 100.0% / +0.278 | 99.5% / +0.287 |
| ow-zora | 97.4% | 95.4% / −0.191 | 95.4% / −0.206 | 94.8% / −0.220 | 96.5% / −0.184 |
| stalfos-keese-gel | — | 96.8% / −0.035 | 95.7% / −0.041 | 96.4% / −0.035 | 96.1% / −0.030 |
| wallmaster | 87.5% | 98.3% / +0.167 | 100.0% / +0.208 | 100.0% / +0.258 | 100.0% / +0.258 |

| Segment | Baseline s0 | Baseline s1 | FQ-02-05 s0 | FQ-02-05 s1 |
|---|---:|---:|---:|---:|
| Full milestone1 | 1/40 | 1/40 | 2/40 | 1/40 |
| Full milestone1-fairy | 3/40 | 2/40 | 3/40 | 3/40 |
| Post-D1 main | 7/40 | 3/40 | 3/40 | 2/40 |
| Post-D1 fairy | 6/40 | 6/40 | 5/40 | 10/40 |
| D2 | 15/40 | 12/40 | 17/40 | 19/40 |
| **Total** | 32 | 24 | 30 | 35 |

**Decision: keep.** At seed 0 ow-zora (94.8%, −0.220) and d2-rope (93.1%, −0.257) were a hair below both baselines
in success and health, so the seed-1 run was made: ow-zora 96.5% / −0.184 and d2-rope 94.0% / −0.279 there, i.e.
within the baselines' range in success; nothing is worse than both baselines in both runs. Chain totals 30 and
35 (baselines 32/24, floor 12).

## FQ-01, FQ-06, FQ-07 + FQ-08: hardware boot timing; the frame ends when VBlank starts

**Change (one step).** Boot: `Bus::reset` spends the CPU's 7 reset cycles (FQ-06), the PPU starts at line 0
dot 0 (FQ-07) and VBlank rises from the first frame (FQ-01: `dead_frames` deleted, `TFZSTATE` v5). Frame
boundary (FQ-08): the frame counter advances in `Ppu::start_vblank`, at **the start of VBlank (line 241, dot
1)**, the hardware-natural point: `run_frame` returns with every line drawn and VBlank and /NMI just raised, and
the NMI handler (joypad reads, OAM DMA, game logic) runs in the next frame, after that frame's inputs and pokes
are applied. TetaNES ended a frame on entering line 240, FCEUmm one dot later; VBlank start is a hardware event
(the NMI), and between it and line 240 the CPU sees only the idle post-render line. With the boundary on a VBlank
edge the lazy PPU has two events left (`Ppu::next_event_clock`), and `fceumm_events` is gone.

**Why together:** FQ-07 placed the PPU at FCEUmm's frame boundary; moving the boundary alone (tried first) made
the first frame 340 dots long and failed six more snapshot ROMs (`dummy_writes_ppumem`, `instr_ind_y`,
`regs_after_reset`, `scanline`, `spr_overflow_obscure`, `vbl_nmi_clear_timing`) that pass with all four. The
gate's start states are mid-game, so the boot quirks cannot show in it: this gate measures FQ-08; FQ-01/06/07's
real test is step 18b.

**Tests:** un-ignored and passing: `flag_concurrency`, `instr_brk`, `instr_zp_xy`, `overclock`,
`vbl_nmi_even_odd_frames`, `vbl_nmi_suppression`, `vbl_timing`, `test_2`, `test_6`, `reset_4017_timing`,
`reset_4017_written`, and the unit tests `cycle_timing` and
`stopping_and_resuming_leaves_the_same_console_as_running_through`; a new `frame_ends_when_vblank_starts`. The
FCEUmm compat tests are all gone (`fceumm_compat_tests.rs` deleted). **No `FQ-` ignore is left.** Twelve ROMs
stay ignored for another reason: `instr_timing` and eleven blargg 2005 APU ROMs (`len_ctr`, `len_table`,
`irq_flag`, `clock_jitter`, `len_timing_mode0/1`, `irq_flag_timing`, `irq_timing`, `reset_timing`,
`len_halt_timing`, `len_reload_timing`) were snapshotted by TetaNES with zero-filled power-on RAM; with every
quirk gone all twelve pass with `State::POWER_ON_RAM = $00` and differ with zcore's $FF (FCEUmm's fill, which the
corpus needs), so they are labelled power-on RAM (`FCEUMM_QUIRKS.md` "Not quirks"). **Movie replay:** ignored as
FCEUmm-recorded desyncs: `zcore` `title_rgb_frame_is_full_and_matches_the_oracle` (frame 600 vs the oracle
golden), `zelda` `zcore_matches_oracle_over_random_play` (base-state RAM after `power_on_to_start` differs), `env`
`env_matches_python_fixtures` (`0_38` step 2 takes a different path). Every other zcore, zelda, env and pool test
passes. Gate build: every quirk restored. `tf bench` 2,459-2,564 / 4,683-4,976.

| Specialist | Old | Baseline s0 | Baseline s1 | Boot+FQ-08 s0 | Boot+FQ-08 s1 |
|---|---:|---:|---:|---:|---:|
| boss-aquamentus | 100.0% | 100.0% / +0.592 | 98.3% / +0.575 | 100.0% / +0.650 | 98.3% / +0.400 |
| d2-rope | 95.2% | 94.0% / −0.254 | 93.6% / −0.231 | 94.0% / −0.260 | 94.8% / −0.217 |
| goriya | 76.7% | 78.3% / −0.758 | 77.9% / −0.725 | 78.8% / −0.738 | 77.1% / −0.688 |
| ow-leever-peahat | 100.0% | 97.1% / −0.325 | 96.2% / −0.296 | 96.7% / −0.354 | 95.8% / −0.304 |
| ow-moblin | 97.1% | 95.3% / −0.215 | 96.4% / −0.236 | 95.3% / −0.237 | 96.4% / −0.207 |
| ow-octorok | 99.0% | 98.3% / −0.073 | 98.2% / −0.066 | 98.2% / −0.066 | 98.2% / −0.070 |
| ow-start | 98.5% | 100.0% / +0.278 | 100.0% / +0.287 | 100.0% / +0.278 | 99.5% / +0.287 |
| ow-zora | 97.4% | 95.4% / −0.191 | 95.4% / −0.206 | 95.9% / −0.211 | 96.3% / −0.210 |
| stalfos-keese-gel | — | 96.8% / −0.035 | 95.7% / −0.041 | 95.8% / −0.051 | 96.4% / −0.044 |
| wallmaster | 87.5% | 98.3% / +0.167 | 100.0% / +0.208 | 100.0% / +0.233 | 100.0% / +0.192 |

| Segment | Baseline s0 | Baseline s1 | Boot+FQ-08 s0 | Boot+FQ-08 s1 |
|---|---:|---:|---:|---:|
| Full milestone1 | 1/40 | 1/40 | 1/40 | 0/40 |
| Full milestone1-fairy | 3/40 | 2/40 | 2/40 | 4/40 |
| Post-D1 main | 7/40 | 3/40 | 4/40 | 3/40 |
| Post-D1 fairy | 6/40 | 6/40 | 6/40 | 8/40 |
| D2 | 15/40 | 12/40 | 17/40 | 16/40 |
| **Total** | 32 | 24 | 30 | 31 |

**Decision: keep.** At seed 0 ow-leever-peahat (−0.354) and stalfos-keese-gel (−0.051) took more damage than
both baselines with success inside their range; at seed 1 both are back in range (−0.304, −0.044) and
boss-aquamentus' health dips instead (+0.400 over 60 episodes, +0.650 at seed 0). Nothing is worse than both
baselines in both runs; chain totals 30 and 31 (baselines 32/24, floor 12).

## Step 18b: regenerated base movie, start states and trajectories on the hardware core

- **Movies.** `tf movie script` and `tf movie random` record on zcore. `power_on_to_start` re-records with
  identical inputs. `sword_cave_in_out` and `die_continue` now wait on game state (`until` directives): on
  hardware timing the cave text ends, and Link dies, at different frames.
- **States** (`target/step18/build_states.sh`, logs in `target/step18/logs/`): `start`, `build --chains 0,1,2`,
  the D1 bow rooms, `milestone1-fairy` from `0_37c_<c>` and the 0x6d off-route states, into a fresh directory.
  Three on-route crossings failed at default budgets and built at 12,000 iterations / patience 500
  (`2_2es_0`, `2_0es_1`, `2_0es_2`, plus `2_1es_0`/`2_0es_0` after it); the chain-1 and chain-2 fairy routes
  needed one `path` backtrack each at 0x5c. Same 181 files as step 17; `entry_states audit`: **0 problems**;
  all 180 manifest entries record their command. The v1 state conversion of 18a is deleted (`TFZSTATE` v5).
- **Trajectories**: all six curriculum specialists re-explored, 84/84 states solved (five needed 12,000 / 500;
  `goriya` `2_1es_2` needed explore seed 2). Each training pool loads one track per start state.
- `triforce/chain_hp.json` rebuilt from the 400 gate chains below.

| Specialist | Old eval.json | FCEUmm baseline s0 / s1 | Hardware + new states s0 | s1 |
|---|---:|---:|---:|---:|
| boss-aquamentus | 100.0% / +0.692 | 100.0 / 98.3 | 100.0% / +0.625 | 100.0% / +0.658 |
| d2-rope | 95.2% / −0.205 | 94.0 / 93.6 | 94.0% / −0.267 | 93.8% / −0.218 |
| goriya | 76.7% / −0.769 | 78.3 / 77.9 | 79.6% / −0.760 | 77.1% / −0.729 |
| ow-leever-peahat | 100.0% / −0.163 | 97.1 / 96.2 | 96.2% / −0.369 | 97.1% / −0.315 |
| ow-moblin | 97.1% / −0.195 | 95.3 / 96.4 | 96.4% / −0.199 | 96.1% / −0.218 |
| ow-octorok | 99.0% / −0.064 | 98.3 / 98.2 | 98.3% / −0.071 | 97.9% / −0.087 |
| ow-start | 98.5% / +0.278 | 100.0 / 100.0 | 99.0% / +0.278 | 100.0% / +0.287 |
| ow-zora | 97.4% / −0.147 | 95.4 / 95.4 | 96.1% / −0.157 | 95.9% / −0.158 |
| stalfos-keese-gel | — | 96.8 / 95.7 | 96.7% / −0.038 | 96.8% / −0.026 |
| wallmaster | 87.5% / −0.417 | 98.3 / 100.0 | 100.0% / +0.292 | 100.0% / +0.192 |

| Segment | FCEUmm baseline s0 / s1 | Hardware, old states (boot+FQ-08 gate) s0 / s1 | Hardware + new states s0 / s1 |
|---|---:|---:|---:|
| Full milestone1 | 1 / 1 | 1 / 0 | 0 / 0 |
| Full milestone1-fairy | 3 / 2 | 2 / 4 | 1 / 1 |
| Post-D1 main | 7 / 3 | 4 / 3 | 5 / 0 |
| Post-D1 fairy | 6 / 6 | 6 / 8 | 6 / 4 |
| D2 | 15 / 12 | 17 / 16 | 16 / 9 |
| **Total (of 200)** | 32 / 24 | 30 / 31 | 28 / 14 |

Every specialist stays within the plan's 15-point gate and inside or near the FCEUmm baselines' spread. The
chains are lower (42/400 vs 56/400) but the hardware core with the *old* states scored 61/400, so the dip
comes with the regenerated start states (each `--from` segment starts from a different `0_37c_2`/`2_7ds_2`,
and the chain-hp profile changed) or chance at 40 runs per segment. Not blocking; recorded as an open
experiment in `todo.md` (larger chain samples on the new states, then fine-tune the team on hardware states).

## Status

Plan step 18 complete except Lee's sign-off items: no quirk was kept; the one open choice is the power-on RAM
fill ($FF, FCEUmm's; 12 test ROMs pass only with $00) — see `docs/decisions.md`.
