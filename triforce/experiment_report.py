"""Deterministic reporting helpers for agent-controlled training runs."""

import json
import os
from copy import deepcopy
from numbers import Number

DEFAULT_HEALTH_RANGES = {
    "charts/SPS": (50, None),
    "losses/value_loss": (0.01, 1.0),
    "losses/policy_loss": (-0.05, 0.05),
    "losses/entropy": (0.5, 2.0),
    "losses/approx_kl": (0.001, 0.03),
    "losses/clipfrac": (0.05, 0.25),
    "losses/explained_variance": (0.3, 0.9),
    "losses/attention/entropy": (2.0, 6.5),
    "losses/attention/top1_weight": (None, 0.10),
    "losses/cross_attention/entropy": (2.0, 6.5),
    "losses/cross_attention/top1_weight": (None, 0.10),
    "losses/entropy/action_type": (0.3, 1.5),
    "losses/entropy/direction": (0.5, 1.4),
}

DEFAULT_TUNING = {
    "schema_version": 1,
    "wake_interval_steps": 1_000_000,
    "wake_dedup_steps": 100_000,
    "anomaly_check_interval_steps": 100_000,
    "wall_clock_limit_seconds": 604_800,
    "health_ranges": deepcopy(DEFAULT_HEALTH_RANGES),
    "reward_hacking": {
        "enabled": True,
        "window": 3,
        "min_reward_average_delta": 0.10,
        "max_success_rate_delta": 0.02,
    },
}


def flatten_metrics(metrics: dict) -> dict[str, float]:
    """Flatten normal or weighted metric dictionaries for display/reporting."""
    flat_metrics = {}
    first_scenario_metrics = None
    for key, value in metrics.items():
        if isinstance(value, dict):
            if first_scenario_metrics is None:
                first_scenario_metrics = value
            for metric_name, metric_value in value.items():
                flat_metrics[f"{key}/{metric_name}"] = metric_value
        else:
            flat_metrics[key] = value

    if first_scenario_metrics is not None:
        for metric_name, metric_value in first_scenario_metrics.items():
            flat_metrics.setdefault(metric_name, metric_value)

    return flat_metrics


def is_metric_healthy(key: str, value: float, health_ranges: dict | None = None) -> bool:
    """Return whether a metric value falls inside the configured healthy range."""
    if not isinstance(value, Number):
        return True

    normalized_key = key
    if "/head_" in normalized_key:
        if normalized_key.endswith("/entropy"):
            normalized_key = "losses/attention/entropy"
        elif normalized_key.endswith("/top1_weight") or normalized_key.endswith("/top1"):
            normalized_key = "losses/attention/top1_weight"

    ranges = health_ranges or DEFAULT_HEALTH_RANGES
    if normalized_key not in ranges:
        return True

    low, high = ranges[normalized_key]
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def _range_for_key(key: str, health_ranges: dict) -> tuple[float | None, float | None] | None:
    normalized_key = key
    if "/head_" in normalized_key:
        if normalized_key.endswith("/entropy"):
            normalized_key = "losses/attention/entropy"
        elif normalized_key.endswith("/top1_weight") or normalized_key.endswith("/top1"):
            normalized_key = "losses/attention/top1_weight"
    value = health_ranges.get(normalized_key)
    if value is None:
        return None
    return value[0], value[1]


def detect_anomalies(metric_history: list[dict], latest_metrics: dict, latest_stats: dict, tuning: dict) -> list[dict]:
    """Detect health and reward-hacking anomalies using live tuning configuration."""
    anomalies = []
    health_ranges = tuning.get("health_ranges") or DEFAULT_HEALTH_RANGES
    combined = dict(latest_metrics or {})
    combined.update(latest_stats or {})

    for key in sorted(combined):
        value = combined[key]
        expected = _range_for_key(key, health_ranges)
        if expected is None or not isinstance(value, Number):
            continue
        if not is_metric_healthy(key, value, health_ranges):
            anomalies.append({
                "kind": "health",
                "metric": key,
                "value": float(value),
                "expected": [expected[0], expected[1]],
                "message": f"{key}={value:.6g} outside expected range {expected}",
            })

    reward_cfg = tuning.get("reward_hacking") or {}
    if not reward_cfg.get("enabled", True):
        return anomalies

    window = int(reward_cfg.get("window", 3))
    if window < 2 or len(metric_history) < window:
        return anomalies

    recent = metric_history[-window:]
    first = recent[0]
    last = recent[-1]
    if "reward-average" not in first or "reward-average" not in last:
        return anomalies
    if "success-rate" not in first or "success-rate" not in last:
        return anomalies

    reward_delta = last["reward-average"] - first["reward-average"]
    success_delta = last["success-rate"] - first["success-rate"]
    min_reward_delta = reward_cfg.get("min_reward_average_delta", 0.10)
    max_success_delta = reward_cfg.get("max_success_rate_delta", 0.02)
    if reward_delta >= min_reward_delta and success_delta <= max_success_delta:
        anomalies.append({
            "kind": "reward_hacking",
            "metric": "reward-average",
            "value": float(last["reward-average"]),
            "expected": [None, None],
            "message": (
                "reward-average improved by "
                f"{reward_delta:.6g} while success-rate improved by only {success_delta:.6g}"
            ),
        })

    return anomalies


def atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON atomically so readers never observe a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(path: str, payload: dict) -> None:
    """Append one JSON object to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def render_milestone_report(snapshot: dict, tuning: dict, baseline: dict | None = None) -> str:
    """Render a deterministic markdown report for an agent wake milestone."""
    lines = ["# Triforce Training Milestone", ""]
    _render_run(lines, snapshot)
    _render_harness_health(lines, snapshot)
    _render_circuit(lines, snapshot)
    _render_health(lines, snapshot, tuning)
    _render_scenario_metrics(lines, snapshot)
    _render_reward_breakdown(lines, snapshot)
    _render_anomalies(lines, snapshot)
    _render_baseline(lines, baseline)
    _render_decision_prompt(lines, snapshot)
    return "\n".join(lines).rstrip() + "\n"


def _render_run(lines: list[str], snapshot: dict) -> None:
    status = snapshot.get("status", {})
    lines.extend([
        "## Run",
        "",
        f"- State: {status.get('state', 'unknown')}",
        f"- Scenario: {status.get('scenario', 'unknown')}",
        f"- Action space: {status.get('action_space', 'unknown')}",
        f"- Model kind: {status.get('model_kind', 'unknown')}",
        f"- Run dir: {status.get('run_dir', 'unknown')}",
        f"- Experiment dir: {status.get('experiment_dir', 'unknown')}",
        f"- Latest checkpoint: {status.get('latest_checkpoint_path') or 'none'}",
        f"- Final model: {status.get('final_model_path') or 'none'}",
        f"- Full metrics: {snapshot.get('full_metrics_path') or 'not written'}",
        f"- Status updated at: {status.get('updated_at', 'unknown')}",
        f"- Status hash: {snapshot.get('status_hash', 'unknown')}",
        "",
    ])


def _render_circuit(lines: list[str], snapshot: dict) -> None:
    status = snapshot.get("status", {})
    lines.extend([
        "## Circuit Progress",
        "",
        "| Leg | Status | Reason | Steps | Total | Exit metric | Value | Threshold | Met |",
        "|---|---|---|---:|---:|---|---:|---:|---|",
    ])
    circuit = status.get("circuit") or []
    if not circuit:
        lines.append("| none | n/a | 0 | 0 | n/a |")
    for leg in circuit:
        exit_criteria = leg.get("exit_criteria") or {}
        exit_metric = exit_criteria.get("metric") if isinstance(exit_criteria, dict) else "n/a"
        lines.append(
            f"| {leg.get('name')} | {leg.get('status')} | {leg.get('completion_reason') or 'n/a'} | "
            f"{leg.get('steps', 0)} | {leg.get('total_steps', 0)} | {exit_metric or 'n/a'} | "
            f"{_format_value(leg.get('exit_metric_value'))} | {_format_value(leg.get('exit_metric_threshold'))} | "
            f"{_format_value(leg.get('exit_metric_met'))} |"
        )
    overall = status.get("overall", {})
    lines.extend([
        "",
        f"Overall: {overall.get('steps', 0)} / {overall.get('total_steps', 0)} "
        f"steps ({overall.get('pct', 0.0):.2f}%), SPS={overall.get('sps')}, "
        f"ETA={overall.get('eta_seconds')}",
        "",
    ])


def _render_health(lines: list[str], snapshot: dict, tuning: dict) -> None:
    status = snapshot.get("status", {})
    combined = dict(status.get("latest_metrics") or {})
    combined.update(status.get("latest_stats") or {})
    health_ranges = tuning.get("health_ranges") or DEFAULT_HEALTH_RANGES
    keys = [
        "losses/entropy",
        "losses/approx_kl",
        "losses/clipfrac",
        "losses/explained_variance",
        "losses/value_loss",
        "losses/attention/entropy",
        "losses/attention/top1_weight",
        "charts/SPS",
    ]
    lines.extend([
        "## Training Health",
        "",
        "| Metric | Value | Status |",
        "|---|---:|---|",
    ])
    rendered = False
    for key in keys:
        if key not in combined:
            continue
        value = combined[key]
        status_text = "OK" if is_metric_healthy(key, value, health_ranges) else "WARN"
        lines.append(f"| {key} | {_format_value(value)} | {status_text} |")
        rendered = True
    if not rendered:
        lines.append("| none | n/a | n/a |")
    lines.append("")


def _render_scenario_metrics(lines: list[str], snapshot: dict) -> None:
    metrics = snapshot.get("status", {}).get("latest_metrics") or {}
    keys = _summary_metric_keys(metrics)
    lines.extend([
        "## Scenario Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ])
    if not keys:
        lines.append("| none | n/a |")
    for key in keys:
        lines.append(f"| {key} | {_format_value(metrics[key])} |")
    omitted = max(0, len(metrics) - len(keys))
    if omitted:
        lines.append(f"| omitted detailed metrics | {omitted} linked in full metrics JSON |")
    lines.append("")


def _render_reward_breakdown(lines: list[str], snapshot: dict) -> None:
    metrics = snapshot.get("status", {}).get("latest_metrics") or {}
    lines.extend(["## Reward Breakdown", ""])
    for prefix, count_prefix, title in (
        ("rewards/", "rewards-count/", "Rewards"),
        ("punishments/", "punishments-count/", "Punishments"),
    ):
        lines.extend([f"### {title}", "", "| Name | Total | Count |", "|---|---:|---:|"])
        entries = []
        for key, value in metrics.items():
            if key.startswith(prefix):
                name = key[len(prefix):]
                entries.append((name, value, metrics.get(f"{count_prefix}{name}", 0)))
        entries.sort(key=lambda item: (-abs(item[1]), item[0]))
        if not entries:
            lines.append("| none | 0 | 0 |")
        for name, total, count in entries[:10]:
            lines.append(f"| {name} | {_format_value(total)} | {_format_value(count)} |")
        lines.append("")


def _render_anomalies(lines: list[str], snapshot: dict) -> None:
    anomalies = snapshot.get("anomalies") or []
    lines.extend(["## Anomalies", ""])
    if not anomalies:
        lines.extend(["None detected.", ""])
        return
    for anomaly in anomalies:
        category = anomaly.get("category") or "uncategorized"
        delta = anomaly.get("delta")
        suffix = f" delta={_format_value(delta)}" if delta is not None else ""
        lines.append(f"- {category} {anomaly.get('kind')}: {anomaly.get('message')}{suffix}")
    lines.append("")


def _render_baseline(lines: list[str], baseline: dict | None) -> None:
    lines.extend(["## Baseline Comparison", ""])
    if baseline is None:
        lines.extend(["No baseline supplied.", ""])
        return
    metrics = baseline.get("metrics") or {}
    lines.extend([
        f"- Episodes: {baseline.get('episodes', 'unknown')}",
        f"- Scenario: {baseline.get('scenario', 'unknown')}",
    ])
    for key in sorted(metrics):
        lines.append(f"- {key}: {_format_value(metrics[key])}")
    lines.append("")


def _render_harness_health(lines: list[str], snapshot: dict) -> None:
    status = snapshot.get("status", {})
    harness = snapshot.get("harness_health") or {}
    lines.extend([
        "## Harness Health",
        "",
        f"- Process: {harness.get('process', 'unknown')}",
        f"- Status freshness: {harness.get('status_freshness', 'unknown')}",
        f"- Checkpoint: {status.get('latest_checkpoint_path') or 'none'}",
        f"- Control file: {harness.get('control_file', 'unknown')}",
        f"- Logs: {harness.get('logs', 'unknown')}",
        f"- Final eval command: {status.get('final_eval_command') or 'not configured'}",
        "",
    ])


def _summary_metric_keys(metrics: dict) -> list[str]:
    preferred = [
        "success-rate",
        "reward-average",
        "rewards",
        "progress/max",
        "progress/success",
        "room-result/correct-exit",
        "endings/success-exit",
        "endings/success-entered-dungeon",
        "endings/failure-stuck",
        "endings/failure-terminated-death",
    ]
    keys = [key for key in preferred if key in metrics]
    if len(keys) >= 12:
        return keys[:12]
    for key in sorted(metrics):
        if len(keys) >= 12:
            break
        if key in keys or key.startswith("rewards/") or key.startswith("rewards-count/"):
            continue
        if key.startswith("punishments/") or key.startswith("punishments-count/"):
            continue
        if "/" in key and not key.startswith("endings/"):
            continue
        keys.append(key)
    return keys


def _render_decision_prompt(lines: list[str], snapshot: dict) -> None:
    journal_path = snapshot.get("journal_path")
    status = snapshot.get("status", {})
    if status.get("state") == "complete" or snapshot.get("reason") == "complete":
        lines.extend([
            "## Decision Prompt",
            "",
            f"Journal: {journal_path if journal_path else 'Journal not found.'}",
            "This is a terminal completion wake. Do not continue training.",
            "Run or inspect the recommended final evaluation command, write summary.md, "
            "then call triforce_experiment_finish.",
            "",
        ])
        return
    lines.extend([
        "## Decision Prompt",
        "",
        f"Journal: {journal_path if journal_path else 'Journal not found.'}",
        "Decide exactly one run action: continue, stop/edit/restart, or finish.",
        "Also decide whether each wake-causing metric should keep waking, be loosened, or be disabled.",
        "Cite evidence for that wake-tuning decision.",
        "If changing wake sensitivity, edit tuning.json before calling triforce_experiment_control or restart.",
        "If continuing, call triforce_experiment_control with command continue and then yield.",
        "",
    ])


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
