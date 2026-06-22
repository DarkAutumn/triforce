---
name: triforce-experiment
description: Use when running autonomous Triforce training experiments through OMP: choose experiment scope, launch train.py, handle milestone wake reports, decide continue/stop-edit-restart/finish, compare to baselines, and write final summaries.
---

# Triforce Experiment Skill

Use this skill when the user asks to run or manage a Triforce training experiment, tune hyperparameters, tune rewards, modify training circuits, compare training runs, or debug reward hacking through training.

## Setup discipline

Before implementation work or experiment code changes, start from the repository root and run:

```bash
git checkout main
git pull origin main
git checkout -b <experiment-branch>
```

Treat untracked savestates under `triforce/custom_integrations/Zelda-NES/` as user work. Do not edit the OMP extension during an autonomous experiment; record extension pain points in `summary.md` instead.

## New experiment scoping

Ask only for preferences not inferable from files:

- Experiment goal.
- Scenario or training circuit.
- Success criteria.
- Baseline model or `.eval.json` path.
- Maximum wall-clock when not the default 7 days.
- Whether code edits are allowed before the first run.

Defaults when the user does not override:

- `scenario = all-items-circuit` for broad training improvement.
- `action_space = all-items`.
- `model_kind = impala-multihead`.
- `parallel = 16`.
- Wake cadence = 1,000,000 steps.
- Dedup window = 100,000 steps.
- Anomaly check interval = 100,000 steps.
- Wall-clock failsafe = 7 days.

Before calling `triforce_experiment_start`, write or update `training/experiments/<experiment-id>/journal.md` with the hypothesis, baseline, intended code/config changes, run command parameters, and success criteria.

## Milestone decision loop

On every milestone wake, read the milestone report and `journal.md`, then make two decisions: exactly one run action, and a wake-tuning decision for every metric that caused the wake.

For anomaly wakes, append a `Wake tuning decision` block to `journal.md` before calling a control/restart/finish tool:

```markdown
### Wake tuning decision

| Metric | Decision | Evidence | tuning.json change |
|---|---|---|---|
| <metric> | keep waking / loosen / disable | <observed trend and why> | <exact change or none> |
```

Use `keep waking` when the metric is a real training-health risk. Use `loosen` when the metric is noisy or acceptable for this phase but still useful at a wider bound. Use `disable` only when the metric is not actionable for the current experiment. If the decision is `loosen` or `disable`, edit the active `tuning.json` before continuing or restarting.

### Continue

Use when training is healthy enough or intentionally being observed through a danger zone.

1. Decide and document whether each wake-causing metric should keep waking, be loosened, or be disabled.
2. Edit the active `tuning.json` when the wake-tuning decision is `loosen` or `disable`.
3. Append a journal entry with the milestone, checkpoint path, observed metrics, run decision, wake-tuning decision, and evidence.
4. Call `triforce_experiment_control` with `{ "command": "continue" }`.
5. Stop; let the extension wake the agent at the next milestone.

### Stop, edit, restart

Use when training is reward hacking, diverging, stuck, or the experiment hypothesis requires code/config correction.

1. Call `triforce_experiment_control` with `{ "command": "stop" }`.
2. Wait for stopped status or process-failure wake.
3. Edit Triforce Python/YAML only; do not edit `.omp/extensions/` mid-experiment.
4. Run targeted tests covering the edit.
5. Call `triforce_experiment_restart` from the latest checkpoint path in the milestone report or `status.json`. Keep optimizer state preserved by using the saved `.pt` checkpoint.
6. Append a journal entry with the edit, test result, checkpoint path, and restart parameters.

### Finish

Use when the experiment succeeded, failed after exhausting high-value ideas, hit a guardrail, or needs human review.

1. Use the evaluation plugin for final evaluation and comparison:

   - Call `triforce_evaluation_start` with the final model path, final evaluation scenario, and the evaluation plugin default of 50 episodes unless the journal records a different user-approved episode count.
   - Stop until the evaluation plugin sends a completion or failure wake.
   - When comparing against a baseline `.eval.json`, call `triforce_evaluation_compare` and stop until its completion or failure wake.
   - If the evaluation plugin fails, record the failure in `summary.md` and ask the user to fix the plugin/environment. Do not bypass the plugin by calling `evaluate.py` directly.

Evaluation is part of finish/reporting only. It is not a gate for training continue/restart decisions.

2. Write `training/experiments/<experiment-id>/summary.md` with what was tried, code/config changes, outcomes, comparison results, and final decision.
3. End `summary.md` with this exact heading and subsections:

```markdown
## Improvements for the next tooling pass

### Extension

- ...

### Triforce library

- ...

### Reporting/prompting

- ...
```

4. Call `triforce_experiment_finish` with the summary path.

## Process failure wakes

If the extension wakes with `Triforce training process failure`, inspect the included stdout/stderr tails and referenced log files before deciding. Prefer restart from the latest checkpoint when one exists and the failure is code/config related; finish with a summary when the failure is environmental and cannot be resolved from repo context.
