---
name: triforce-evaluation
description: Use when running, comparing, or inspecting long Triforce model evaluations through OMP. Requires the evaluation plugin instead of direct evaluate.py execution.
---

# Triforce Evaluation Skill

Use this skill when the user asks to evaluate a Triforce model, compare eval JSON files, run final evaluation, inspect a `.eval.json` or `.eval.md`, or evaluate a checkpoint/run directory.

Rule: Use the triforce_evaluation_* tools. Do not call evaluate.py directly. If the plugin or Python environment fails, ask the user to fix that problem instead of bypassing the plugin.

Evaluation is optional and not a training gate. The training skill may recommend evaluation during finish/reporting, but evaluation is never required before training continue/restart decisions.

## Defaults

- `episodes = 50`.
- `reprocess = false` unless the user explicitly asks to overwrite prior evaluation or the prior eval has fewer than requested episodes.
- `frame_stack = 3`.
- `limit = -1`.

## Running an evaluation

1. Confirm model path and scenario from user input or current training milestone/status (`final_model_path`, `latest_checkpoint_path`, or `final_eval_scenario`). Do not ask when those paths are present in the report/status.
2. Call `triforce_evaluation_start` with episodes default 50.
3. End the turn immediately after the tool returns. Do not sleep, poll, wait, or call status in a loop; the plugin will wake the agent on completion or failure.
4. On the completion wake, read the result paths from the wake/status, then summarize results in chat or append to experiment `summary.md` when this is part of a training experiment.
5. On failure, ask the user to fix the plugin/environment issue. Do not run evaluate.py directly.

## Comparing evaluations

1. Use `triforce_evaluation_compare` for `evaluate.py --compare` behavior.
2. End the turn immediately after the tool returns. Do not sleep, poll, wait, or call status in a loop; the plugin will wake the agent on completion or failure.
3. On the completion wake, summarize `compare.md`. Do not run compare directly in bash.

## Failure handling

If the plugin reports a missing `.venv/bin/python`, missing model path, missing eval JSON path, failed process, or `complete_no_results`, report the plugin/environment issue and ask the user how they want it fixed. Do not bypass the plugin by running `evaluate.py` directly.
