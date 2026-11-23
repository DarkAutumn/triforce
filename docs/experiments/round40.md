# Round 40

## zol from scratch: the repeatability check and the timing (branch `r40-fresh`)

The owner, Nov 23: the next generation (2.0.1.0) is trained from scratch, "breaking the ship of Theseus", and "this
also helps us understand if our approach is repeatable or we are building the ship of Theseus" (decisions
2026-11-23). The backlog's recipe for it (the lead's) starts by timing one family from scratch on today's code, which
is also the repeatability check. This section is that check for zol on `d3first` (the network `promoted.d3first`
names; it plays Dungeon 3's 0x7c, 0x7b, 0x6b, 0x4b, 0x49 and 0x3d, Dungeon 4's 0x40 and Dungeon 5's 0x57). Its
three runs are also the baseline arm of the beams experiment (backlog "Beams at range, not melee": from scratch
without the firing-line targets), so they stay in the worktree's `models/zol/`.

### The recipe and what the lineage carried

`promoted.d3first` is `r38head-fam-w4`: the policy `r34d3-zol-s341` with zol's own hit head at weight 4 (2.0.0.8,
round38.md "The hit head across families"; a supervised network trained on recordings of that policy, played as a
`hitshift`). The policy's lineage is complete in `models/zol` (every `run.json` names its parent by path, `source`):

| run | code (`version`) | steps (run / total) | start states | extras |
|---|---|---:|---|---|
| `r71-e` | 1.4.1.0, Nov 13 | 0.63M / 0.63M | `d2first`'s 3_7cs, 3_7be, 3_6bs, 3_4bs, 3_49s, 3_3ds, 4_40s | from scratch; early stop at 99% (153 iterations) |
| `r71-zol-cont` | 1.4.2.0 | 0.24M / 0.87M | the same | converted (obs schema 5: the recorder, food, candle inputs added) |
| `r74-d5cont` | 1.5.0.0, Nov 14 | 0.24M / 1.11M | + 5_57w (Dungeon 5) | |
| `r75-f-b` | 1.6.0.0, Nov 15 | 1.00M / 2.11M | the same | converted (obs schema 6: contact columns, `in_line`, phases, sword blocked) |
| `r77-b` | 1.7.0.0, Nov 16 | 1.00M / 3.10M | + 6_68s (Dungeon 6) | |
| `r79b-zol` | Nov 17 | 1.00M / 4.10M | + 9_63c (Dungeon 9) | `--load zol` (the promoted run then; the path is in `source`) |
| `r32d3-zol` | 2.0.0.0, Nov 19 | 2.00M / 6.10M | `d3first`'s 3_7cs, 3_7be, 3_6bs, 3_4bs, 3_49e, 3_3ds, 4_40s only | harvested handoffs (946) at 25% |
| `r34d3-zol-s341` | 2.0.0.0, Nov 21 | 1.00M / **7.10M** | `specialists/d3first/zol.yaml` (both routes' tracks) | planner demonstrations `zol-plan.npz` at 0.05 (21,810 rows; file gone with its worktree), harvested 2,293 |

Every run: PPO lr 1e-4, entropy 0.01, 4096 steps an update, SIL 0.1, best-of-N cloning 0.2 every 20 iterations,
`--parallel 8`, the family's 866,084-parameter network (`model_config` unchanged throughout).

What the lineage carried that no file of today's recipe records:

1. **Five weeks of other code.** Its first 4.1M steps were played under 1.4-1.9's rewards, route data and engine
   (each a different version), on `d2first`'s room entries; only the last 3.0M saw `d3first`.
2. **Two schema widenings.** The network learned without the recorder/food/candle inputs (obs schema 4) and then
   without the contact columns, `in_line` and the monsters' phases (schema 5) for its first 2.1M steps; `convert-model`
   added them as zero weights, so what it learned before reads the room without them.
3. **Rooms dropped from the task.** Dungeon 6's 0x68 and Dungeon 9's 0x63 (bare states) were in its training from
   r77/r79 on; today's yaml draws only `d3first`'s states (`state_routes: [d3first]`). The bare curricula tracks
   are listed in every run's `curriculum_tracks` but its start states are the route's.
4. **Demonstrations that are gone.** `r34d3-zol-s341` cloned 21,810 planner rows (from 3_6bs, 3_49e, 3_4bs, 3_7be,
   20 a state, and 120 handoffs into 0x49 and 0x4b) at 0.05; the file lived in the round's worktree. Today's recipe
   has no demonstrations unless someone records them again (`r27_plan_demos.py`).
5. **The hit head.** Trained on recordings of the parent policy's own play (round 38); a fresh policy needs its
   own recordings for a head of its own.
6. **Not carried, new in today's recipe:** the beam terms (`reward-beam-kill`, `reward-beam-timed`,
   `penalty-beam-unfired`, round 38, on by default), the Bubble's sword-blocked penalty, rupees once a pickup, and
   3,396 harvested handoffs (the lineage's last run saw 2,293). The fresh runs train with them; the lineage never
   did.

### The runs (stopped by the queue's full stop; to be resumed)

Today's recipe, from scratch (no `--load`, no demonstrations, no hit head), three seeds:
`triforce train --specialist specialists/d3first/zol.yaml --parallel 8 --iterations 1954 --checkpoint-every 122
--seed 40N --stop-success 1.01` (the lineage's PPO settings; learning rate and entropy are constant, so a run stopped
at iteration N is the first N iterations of a longer one). Started 08:31 EST Nov 23, two at once; at load 40 they ran
at 600 decisions a second, falling to 320 (about 1.2M steps an hour each), so the budget was cut to 4M steps
(iteration 977; the third seed re-queued as `--iterations 977 --checkpoint-every 61`). All three were stopped at
14:43 UTC by the lead's queue sweep and then the owner's full stop of the queue (decisions: the queue is replaced by
one scheduler), with their checkpoints saved:

| run | iterations | steps | wall | decisions a second | training success (balanced), by 0.25M |
|---|---:|---:|---:|---:|---|
| `r40-fresh-zol-s401` | 347 | 1.42M | 4,422 s | 321 | 0.50, 0.84, 0.92, 0.95, 0.94, **0.96** at 1.42M |
| `r40-fresh-zol-s402` | 354 | 1.45M | 4,408 s | 329 | 0.54, 0.87, 0.94, 0.94, 0.95, **0.95** at 1.45M |
| `r40-fresh-zol-s403` | 0 | 0 | | | queued, never started |

Against the lineage's own training numbers (the same statistic, start-state balanced, the last 20 iterations):
`r71-e` from scratch on `d2first`'s rooms reached 0.996 at 0.63M; on `d3first`'s states `r32d3-zol` ended at 0.971
and `r34d3-zol-s341` at 0.982. The promoted networks' single rooms on today's yaml (`triforce evaluate`, 100
episodes a state, 30 states, the act mode): `r34d3-zol-s341` **98.5%**, `r38head-fam-w4` **99.1%**.

Not yet measured (the queue's full stop): the checkpoints' single rooms, the steps to 98.5%, and the `d3` chains.
The resume: each seed from its own `ckpt.pt` (`--load` of its own run dir restores the optimizer, the counters and
the curriculum; same task, so still from scratch) to 4M steps, the third seed from nothing; then
`scripts/experiments/r40_fresh.py` on the chains (the segment paired by seed, and in zol's rooms hits, hearts lost,
frames, far kills per sword kill and unfired lined-up decisions).
