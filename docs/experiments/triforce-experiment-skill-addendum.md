# Triforce experiment skill addendum

This repo-local addendum captures Triforce experiment-memory protocol that should be folded into the `triforce-experiment` skill when skill files are editable.

## Before starting any Triforce training experiment

Read:

1. `docs/experiments/experiment-memory.md`
2. The most relevant prior `docs/experiments/<experiment-id>-summary.md`
3. Any referenced eval `.md`/`.json` files for the baseline checkpoint being compared against, if available locally

Then state in `training/experiments/<experiment-id>/journal.md`:

- baseline checkpoint/eval path
- what metric must improve
- whether the experiment starts from scratch or a checkpoint
- final eval scenario and episode count

## After finishing any Triforce training experiment

Append a compact record to `docs/experiments/experiment-memory.md` with:

- experiment id and run path
- final checkpoint
- final eval command and artifacts
- success rate and progress distribution
- what improved or regressed versus the prior baseline
- current bottleneck
- next recommended experiment

Do not paste the full journal. Link full artifacts instead. Keep large artifacts under gitignored `training/experiments/`.

## Current baseline pointer

Current baseline record:

`docs/experiments/experiment-memory.md#baseline-all-items-circuit-2026-06`

Key baseline to beat:

- Eval scenario: `full-game-all-items-finite`
- Eval episodes: `100`
- Success: `0/100`
- Median progress: `13/17`
- Main bottleneck: late Dungeon 1, especially `1_43 -> 1_44 -> 1_45 -> 1_35 -> 1_36`
