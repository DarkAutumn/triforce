# Triforce harness monitoring follow-up

Audience: next agent working on the Oh My Pi Triforce experiment harness.

Question answered: how can the next run be more successful from a monitoring/supervision standpoint?

## Baseline evidence

Run: `training/experiments/baseline/runs/all-items-circuit/0`

The baseline completed end-to-end:

- Final state in `status.json`: `complete`
- Final checkpoint: `training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_all-items-polish_10719232.pt`
- Final eval: `0/100` successes, median progress `13/17`
- Eval outputs are local under the final checkpoint path.

Harness verdict: pass overall. The issues below are monitoring/reporting improvements, not blockers observed during training.

## Fixes needed before future runs

### Suppress `process_disappeared` after normal completion

A post-completion wake reported `process_disappeared`, but `status.json` already had `state: complete` and stdout showed normal final output. If the pid disappears and latest status is complete, emit a terminal completion wake or no failure. Do not classify normal process exit as failure.

### Add terminal `complete` wake

Completion should be distinct from `leg_end` and should include final model path, final checkpoint path, run dir, final status, final eval command, and baseline comparison config.

### Make final progress and ETA unambiguous

When `state: complete`, status should report `pct: 100.0` and `eta_seconds: 0` or `null`. Keep planned-vs-actual counters separately if useful.

### Record leg completion reason as structured data

Each leg should state `completion_reason`, `exit_metric_value`, `exit_metric_threshold`, and `exit_metric_met`. Leg-end wakes should say whether the leg ended by exit criterion or budget.

### Add final evaluation metadata to circuits

Circuit config or run metadata should expose `final_eval_scenario`, `final_eval_episodes`, and optional `baseline_eval_json`. Completion wake should print the exact final eval command.

### Reduce anomaly wake noise with trend-aware grouping

Repeated mild health misses should be grouped as `persistent_anomaly` unless a severity/trend threshold changes. Keep hard wakes for process, checkpoint, log, status, or control failures.

### Separate harness health from training health

Every wake should show process/state/checkpoint/control/log freshness separately from PPO/model metrics so agents can distinguish harness failures from training-health issues quickly.

### Link logs robustly

Only link logs that exist. If stderr was never created, say so explicitly. Include final stdout line in process-exit wakes.

### Mark stale/queued wakes

Every wake should include monotonic `milestone_id`, `status_updated_at`, and `status_hash`. Stale queued wakes should say they are stale and point at newest status.

### Compact metric reporting

Top-level wake should show key exit-owner metric, progress, reward, endings, and health anomalies. Full nested scenario metrics should be linked as JSON instead of flooding the prompt.

## Monitoring success criteria

A better next run from a monitoring standpoint satisfies:

1. No false process-failure wake after normal completion.
2. Clear terminal completion wake with final checkpoint and eval instructions.
3. Final status shows `state: complete`, `pct: 100`, and no nonzero ETA.
4. Every leg-end milestone states whether it ended by metric or budget.
5. Final eval scenario and command are discoverable.
6. Repeated model-health threshold misses are grouped or trend-gated.
7. Harness-health failures remain immediate and actionable.
8. Wakes never link missing log files.
9. Queued/stale wake status is explicit.
10. Metric-heavy reports link full detail instead of flooding the prompt.
