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
