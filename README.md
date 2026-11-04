# Triforce v2

Specialist PPO agents that play *The Legend of Zelda* (NES) from structured game state read out of RAM.
No pixels reach a network. Each specialist owns a family of rooms; a runner picks the specialist for each
room and plays the route end to end.

- Working on the repo (people or agents): `AGENTS.md` (standing rules, workflow, how to resume), then the newest
  `docs/experiments/round*.md` and `docs/experiments/todo.md`.
- Design and status: `docs/architecture.md` (modules and contracts), `docs/decisions.md` (departures from
  the design), `docs/progress.md` (phase status).
- Game knowledge: `docs/ram-catalog.md` (RAM features), `docs/route.md` (milestone route),
  `docs/specialists.md` (room families), `triforce/game.yaml` (room graph with `specialist:` tags),
  `triforce/routes.yaml` (route data).

## Setup

Python 3.12 is required (stable-retro has no 3.13+ wheels).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

The ROM isn't checked in. Put an uncompressed *Legend of Zelda (U)* ROM at
`triforce/custom_integrations/Zelda-NES/rom.nes` (SHA-1 in `rom.sha`).

## Use

```bash
.venv/bin/python -m pytest -q                                   # fast tests
.venv/bin/python -m triforce.play --state 1_44w                 # load a savestate and look at the room
.venv/bin/python -m triforce.play "MOVE E 16" "SWORD E"         # act; the session persists between calls
.venv/bin/python -m triforce.train --specialist ow-start --parallel 8 --iterations 100
.venv/bin/python -m triforce.evaluate --specialist ow-start      # promoted model on its own rooms
.venv/bin/python -m triforce.explore --specialist ow-zora --parallel 16   # Go-Explore: trajectories/ow-zora
.venv/bin/python -m triforce.train --specialist ow-zora --parallel 16   # curriculum: on in ow-zora.yaml
.venv/bin/python scripts/train_round.py --round round1          # train + evaluate every specialist
uv run training-gui.py                                          # live training monitor (newest logs/ folder)
.venv/bin/python play-zelda.py 4x3                               # watch 12 games played by the promoted models (--cpu, --scale N)
.venv/bin/python -m triforce.run --route milestone1 --seed 1
triforce promote <specialist> <run-id>
triforce eval-chain --route milestone1 --runs 20
```

Training GUI charts use the plotted history (respecting **skip startup**). Success ranges use 10-percentage-point
bands and ticks, enclosing every displayed success series and the early-stop/eval reference lines; any value below 25%
restores the full 0–100% range. Health change encloses both real and total series with ±0.2 hearts of padding;
the "no change" reference appears only when zero is within that range.

`triforce.play` is the debugging tool: it drives the game through the same code path as training
(frame skipping, interrupts, masks, look-ahead) and prints the room, entities, legal actions and an ASCII
tile map. `--screenshot`, `--ram`, `--set`, `--save-state` and `--capture` help build and inspect savestates.
