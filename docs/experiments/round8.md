# Round 8: Rust port, data regeneration (plan step 17)

Step 17 of the Rust port (the regenerated data and the promoted team's results on the new stack). Paused on
Nov 5 to merge the `rust` branch into main and finished in main. Raw logs and per-run outputs are under
`target/step17/` and `target/l6/` (gitignored).

## Done

- `start.state` from `power_on_to_start`; `entry_states build --chains 0,1,2` (chain 0/2 Dodongo entries
  `2_0es_*` needed a retry at 12,000 iterations / patience 500); D1 bow rooms (`path --source 1_43w_<c>
  --rooms 1:0x33,1:0x23,1:0x22`); 0x6d off-route (`path --source 2_7ds_<c> --rooms 2:0x6d --route
  milestone1`); `milestone1-fairy` from `0_37c_<c>` for chains 0 and 1.
- Builder fixes: the room-key guard now applies to dungeon rooms only (game.yaml lists a 'key' treasure for
  overworld 0x5a, which stopped every fairy build there); concurrent builders lock the manifest; new entries
  record the exact `entry_states` command.
- Trajectories (default budgets, then 12,000/500 for the four unsolved states): boss-aquamentus 3/3,
  wallmaster 3/3, goriya 12/12, d2-rope 21/21, ow-zora 27/27, ow-moblin 18/18. Each curriculum specialist's
  training pool loads exactly one track per configured start state (checked through `curriculum.json`).
- `triforce/chain_hp.json` rebuilt from the five chain segments below.
- `entry_states audit`: **0 problems over all 181 states**. The audit now counts a post-entry save as a problem
  only in rooms that spawn monsters (as the test always did), so the game-start base state isn't flagged.
- All 180 manifest entries record the exact `entry_states` command that wrote them (`start.state` is written
  by `entry_states start`); off-route `path` entries record `from_step`.
- **Chain 2's fairy route** first entered 0x5c through the bottom-left lane, from which no exploration
  (up to 12,000 iterations / patience 500, three seeds) reaches the east exit. `path` now backtracks: when a
  crossing fails it re-crosses the previous room with a new explore seed (up to three times). Re-crossing 0x5b
  with seed 1002 entered 0x5c higher up and the rest of the route built in under a minute per room.

## Promoted models on the new stack (20 episodes per start state, sampled, seed 0)

| Specialist | Old success | New success | Δ pts | Old health change | New health change |
|---|---:|---:|---:|---:|---:|
| boss-aquamentus | 100.0% (60) | 100.0% (60) | +0.0 | +0.692 | +0.583 |
| wallmaster | 87.5% (120) | 100.0% (60) | +12.5 | −0.417 | +0.200 |
| goriya | 76.7% (240) | 76.7% (240) | +0.0 | −0.769 | −0.767 |
| d2-rope | 95.2% (420) | 92.6% (420) | −2.6 | −0.205 | −0.238 |
| ow-leever-peahat | 100.0% (80) | 96.7% (240) | −3.3 | −0.163 | −0.265 |
| ow-start | 98.5% (200) | 99.5% (200) | +1.0 | +0.278 | +0.278 |
| ow-zora | 97.4% (460) | 95.2% (540) | −2.2 | −0.147 | −0.226 |
| ow-octorok | 99.0% (620) | 98.0% (660) | −1.0 | −0.064 | −0.073 |
| ow-moblin | 97.1% (280) | 96.4% (360) | −0.8 | −0.195 | −0.204 |
| stalfos-keese-gel | no eval.json | 96.4% (720) | | | −0.057 |

All within the 15-point gate (the largest drop is ow-leever-peahat, −3.3). ow-octorok and ow-moblin were
evaluated after the chain-2 fairy states were rebuilt.

## Chains (40 runs, seeds 0–39, promoted team) vs round 7 §19 (r35 requested-key arrivals)

| Segment | Round 7 | New stack |
|---|---:|---:|
| Full milestone1 | 1/40 | 1/40 |
| Full milestone1-fairy | 3/40 | 4/40 |
| Post-D1 main (`--from 0_37c_2`) | 4/40 | 3/40 |
| Post-D1 fairy (`--from 0_37c_2`) | 7/40 | 8/40 |
| D2 (`--from 2_7ds_2`) | 19/40 | 18/40 |

## L6 training parity (ow-start, 120 iterations × 4,096 steps, 8 envs, seed 1)

Old stack final `balanced_success_rate` 1.000 (210 steps/s, idle machine); new stack 1.000 (154 steps/s while
chain evaluations and state builds shared the 20 cores, so not a speed comparison). Gate (≥ old − 0.05) passes.
Curves: `target/l6/old-metrics.jsonl`, `target/l6/new-metrics.jsonl`.

## Status

Plan step 17 is complete. Next: plan step 18 (retire the FCEUmm oracle and quirks), then the test-coverage and
performance steps (`todo.md` "Next up").
