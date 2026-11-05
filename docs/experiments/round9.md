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
