# Triforce v2

Specialist PPO agents that play *The Legend of Zelda* (NES) from structured game state read out of RAM.
No pixels reach a network. Each specialist owns a family of rooms; a runner picks the specialist for each
room and plays the route end to end.

- Working on the repo (people or agents): `AGENTS.md` (standing rules, workflow, how to resume), then the newest
  `docs/experiments/round*.md` and `docs/experiments/todo.md`.
- Design and status: `docs/architecture.md` (modules and contracts), `docs/decisions.md` (departures from
  the design), `docs/progress.md` (phase status), `docs/performance.md` (training throughput: benchmarks,
  profiles, `scripts/train_bench.sh` / `scripts/profile_train.sh` / `scripts/rollout_timeline.py`).
- Game knowledge: `docs/ram-catalog.md` (RAM features), `docs/route.md` (milestone route),
  `docs/specialists.md` (room families), `triforce/game.yaml` (room graph with `specialist:` tags),
  `triforce/routes.yaml` (route data).
- Data: start states in `triforce/states/` (`TFZSTATE` files, `$TRIFORCE_STATES_DIR` overrides), curriculum
  trajectories in `trajectories/<specialist>/<state>/`, checkpoints in `models/` (`$TRIFORCE_MODELS_DIR`), emulator
  test movies in `corpus/`.

## Setup

The emulator, game model, environments and env pool are Rust (`crates/`); Python keeps PPO, evaluation and the
CLIs and reaches the game through the extension module `triforce._native`. Needs Python 3.12 and stable Rust
(`rust-toolchain.toml`).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'             # builds triforce._native through maturin
.venv/bin/maturin develop --release                              # rebuild the module after Rust changes
```

The ROM isn't checked in. Put an uncompressed *Legend of Zelda (U)* ROM at `roms/zelda.nes` or point
`$TRIFORCE_ROM` at it. The ROM payload (after the 16-byte iNES header) must have SHA-1
`a12d74c73a0481599a5d832361d168f4737bbcf6`. ROM-dependent tests skip without it.

## Use

```bash
cargo test --release --workspace --features zelda/zcore        # Rust tests (~1 min; see "Testing")
.venv/bin/python -m pytest -q                                   # Python tests (~1 min)
.venv/bin/python -m pytest -q -m e2e                           # only the end-to-end tests (untrained nets, seconds)
.venv/bin/python -m triforce.play --state 1_44w_0               # load a savestate and look at the room
.venv/bin/python -m triforce.play "MOVE E 16" "SWORD E"         # act; the session persists between calls
.venv/bin/python -m triforce.train --specialist ow-start --parallel 8 --iterations 100
.venv/bin/python -m triforce.evaluate --specialist ow-start      # promoted model on its own rooms
.venv/bin/python -m triforce.explore --specialist ow-zora --parallel 16   # Go-Explore: trajectories/ow-zora
.venv/bin/python -m triforce.train --specialist ow-zora --parallel 16   # curriculum: on in ow-zora.yaml
.venv/bin/python scripts/train_round.py --round round1          # train + evaluate every specialist
uv run training-gui.py                                          # live training monitor (newest logs/ folder)
.venv/bin/python play-zelda.py 4x3                               # watch 12 games played by the promoted models (--cpu, --scale N; keys u/+/-/p)
.venv/bin/python -m triforce.run --route milestone1 --seed 1   # --render (window) or --video out.mp4 (ffmpeg)
.venv/bin/python -m triforce.entry_states build --chains 0,1,2  # regenerate start states (start/build/path/audit)
triforce promote <specialist> <run-id>
triforce eval-chain --route milestone1 --runs 20 --parallel 20
cargo run --release -p triforce-tools --bin tf -- --help        # movies, zcore goldens/self-diff, trace, corpus, bench
```

Training GUI charts use the plotted history (respecting **skip startup**). Success ranges use 10-percentage-point
bands and ticks, enclosing every displayed success series and the early-stop/eval reference lines; any value below 25%
restores the full 0–100% range. Health change encloses both real and total series with ±0.2 hearts of padding;
the "no change" reference appears only when zero is within that range.

`triforce.play` is the debugging tool: it drives the game through the same Rust code path as training
(option executor, interrupts, masks, look-ahead) and prints the room, entities, legal actions and an ASCII
tile map. `--screenshot`, `--ram`, `--set`, `--save-state` and `--capture` help build and inspect savestates.

## Emulator regression suite

zcore is checked against its own per-frame digests (`TFGOLDEN`, all 11 domains: RAM, PRG-RAM, CHR, nametables,
palette, OAM, pixels, CPU, PPU, mapper and APU state) over the corpus movies. Around any zcore change:

```bash
alias tf='cargo run --release -q -p triforce-tools --bin tf --'
tf ladder-golden -o target/ladder/hardware      # before the change: goldens of every corpus movie (~1.2M frames)
# ... change zcore ...
tf ladder-diff target/ladder/hardware           # after: "0 of N movies diverge", or each movie's first bad frame
tf ladder-diff                                  # the committed goldens of the seven short movies (corpus/golden/)
```

A change meant to be invisible (refactor, speed-up) must leave `ladder-diff` at 0. A deliberate behaviour change
will diverge: check that the divergence is the intended one (`tf trace MOVIE --frame K` before and after, then
`diff` the outputs), then regenerate the goldens, including the committed ones
(`tf ladder-golden --short -o corpus/golden`), and commit those with the change. A re-recorded movie makes its
golden stale (`ladder-diff` says so); regenerate it the same way.

## Testing

Two suites, both about a minute: `cargo test --release --workspace --features zelda/zcore` and
`.venv/bin/python -m pytest -q`. Nothing trains a model from scratch; pipeline tests use untrained networks.

| Layer | Where | What it guards |
|---|---|---|
| Emulator | `crates/zcore` unit tests, the TetaNES test ROMs (`src/common.rs`, `test_roms/`), `tests/machine.rs` | CPU (official and unofficial opcodes), PPU/APU timing, MMC1, iNES parsing, the `ZCore` API (render toggle, snapshot/restore, state files, the pinned power-on state), shipped states load |
| Emulator regression | `tf ladder-diff` (above), `crates/tools/tests/cli.rs` | every corpus frame unchanged; the `tf` CLI end to end |
| Formats | `crates/movie` | `TFMOVIE`/`TFGOLDEN` round trips and refusals |
| Game model | `crates/zelda` (unit tests, `tests/game.rs`, `game_view.rs`, `objectives.rs`) | RAM decoding, masks, the option executor and its interrupts, look-ahead credit, room entry, objectives, hearts, linger |
| Env | `crates/env` (unit tests, `tests/*.rs`) | configs, every reward/penalty term, end-condition precedence, observations, curriculum, chains, explorer |
| Pool | `crates/pool` | lockstep determinism across thread counts, auto-reset, render toggle, pacing, failures, eval/chain adapters |
| Python | `tests/` (`tests/native` for `triforce._native`) | PPO/buffer/metrics, CLIs (train, evaluate, eval-chain, run, play, entry_states, explore), watch grid, registry, the native wrapper's errors and pickling; `-m e2e` runs the end-to-end pipelines |

Coverage (`cargo llvm-cov --release --workspace --features zelda/zcore --summary-only`,
`pytest --cov=triforce`) guides where behaviour is untested; it is not a target.
