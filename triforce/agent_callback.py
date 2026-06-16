"""File-based training callback used by the OMP Triforce experiment extension."""

import json
import os
import time
from copy import deepcopy
from glob import glob

from torch.utils.tensorboard import SummaryWriter

from .experiment_report import DEFAULT_TUNING, append_jsonl, atomic_write_json, detect_anomalies, \
    flatten_metrics, render_milestone_report
from .ml_ppo_callback import TrainingCallback

_RUNNING = "running"
_PAUSED = "paused"
_STOPPING = "stopping"
_COMPLETE = "complete"

_ALLOWED_COMMANDS = {"continue", "pause", "stop"}
_ALLOWED_MILESTONE_REASONS = {"cadence", "anomaly", "leg_end", "complete", "wall_clock_limit"}


class AgentTrainingCallback(TrainingCallback):
    """Training callback that emits deterministic files for OMP agent supervision."""

    def __init__(self, run_dir: str, experiment_dir: str, log_dir: str, *, scenario: str,
                 action_space: str, model_kind: str, baseline_eval_json: str | None = None) -> None:
        self.run_dir = run_dir
        self.experiment_dir = experiment_dir
        self.log_dir = log_dir
        self.scenario = scenario
        self.action_space = action_space
        self.model_kind = model_kind
        self.baseline_eval_json = baseline_eval_json

        self.events_path = os.path.join(run_dir, "events.jsonl")
        self.status_path = os.path.join(run_dir, "status.json")
        self.control_path = os.path.join(run_dir, "control.json")
        self.tuning_path = os.path.join(run_dir, "tuning.json")
        self.journal_path = os.path.join(experiment_dir, "journal.md")

        self._tensorboard = None
        self._scenarios = []
        self._active_index = -1
        self._current_name = None
        self._current_steps = 0
        self._current_total = 0
        self._total_spent = 0
        self._total_budget = 0
        self._completion_info = {}
        self._metric_history = []
        self._latest_metrics = {}
        self._latest_stats = {}
        self._latest_checkpoint_path = None
        self._last_milestone_step = None
        self._last_milestone_reason = None
        self._last_anomaly_check_step = None
        self._milestone_id = 0
        self._kl_rollback_count = 0
        self._started_at = time.time()
        self._last_progress_time = None
        self._last_progress_steps = 0
        self._sps = None
        self._last_valid_tuning = deepcopy(DEFAULT_TUNING)
        self._baseline = self._load_baseline(baseline_eval_json)

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self._ensure_control_file()
        self._ensure_tuning_file()
        self._write_status("starting")

    def on_circuit_start(self, scenarios):
        self._scenarios = [{
            "name": name,
            "total_steps": iters,
            "steps": 0,
            "status": "pending",
            "exit_criteria": None,
        } for name, iters in scenarios]
        self._total_budget = sum(iters for _, iters in scenarios)
        self._total_spent = 0
        self._write_status(_RUNNING)
        self._append_event("run_started")

    def on_progress(self, steps, total_steps):
        self._current_steps += steps
        self._current_total = total_steps
        if self._active_index >= 0:
            self._scenarios[self._active_index]["steps"] = self._current_steps
            self._scenarios[self._active_index]["total_steps"] = total_steps

        now = time.time()
        if self._last_progress_time is not None:
            elapsed = now - self._last_progress_time
            if elapsed > 0:
                self._sps = steps / elapsed
        self._last_progress_time = now
        self._last_progress_steps += steps
        self._write_status(_RUNNING)

    def on_scenario_start(self, scenario_name, iterations, exit_criteria=None,
                          exit_threshold=None, exit_criteria_scenario=None):
        self._current_name = scenario_name
        self._active_index = next(
            (i for i, leg in enumerate(self._scenarios) if leg["name"] == scenario_name), -1)
        self._current_steps = 0
        self._current_total = iterations
        self._latest_metrics = {}
        self._latest_stats = {}
        if self._active_index >= 0:
            self._scenarios[self._active_index]["status"] = _RUNNING
            self._scenarios[self._active_index]["total_steps"] = iterations
            if exit_criteria:
                self._scenarios[self._active_index]["exit_criteria"] = {
                    "metric": exit_criteria,
                    "threshold": exit_threshold,
                    "scenario": exit_criteria_scenario,
                }

        if self._tensorboard:
            self._tensorboard.close()
        scenario_log_dir = os.path.join(self.log_dir, scenario_name)
        os.makedirs(scenario_log_dir, exist_ok=True)
        self._tensorboard = SummaryWriter(scenario_log_dir)
        self._write_status(_RUNNING)
        self._append_event("scenario_start", scenario=scenario_name)

    def on_metrics(self, metrics, iteration, total_iterations):
        tuning = self._load_tuning()
        flat_metrics = _jsonable_dict(flatten_metrics(metrics))
        self._latest_metrics.update(flat_metrics)
        self._metric_history.append(dict(self._latest_metrics))
        if self._tensorboard:
            timestamp = time.time()
            for name, value in flat_metrics.items():
                scalar_name = name if '/' in name else f"metrics/{name}"
                self._tensorboard.add_scalar(scalar_name, value, iteration, timestamp)
            self._tensorboard.flush()

        self._write_status(_RUNNING)
        self._append_event("metrics", metrics=flat_metrics)
        self._maybe_emit_cadence_or_anomaly(iteration, tuning)

    def on_optimize(self, stats, iteration, total_iterations):
        json_stats = _jsonable_dict(stats)
        self._latest_stats = json_stats
        if json_stats.get("losses/kl_rollback", 0) > 0:
            self._kl_rollback_count += 1
        if self._tensorboard:
            for name, value in stats.items():
                self._tensorboard.add_scalar(name, value, iteration)
            self._tensorboard.flush()
        self._write_status(_RUNNING)

    def on_scenario_end(self, scenario_name, checkpoint_path=None):
        milestone_step = self._current_step()
        if self._active_index >= 0:
            self._scenarios[self._active_index]["status"] = "complete"
            self._scenarios[self._active_index]["steps"] = self._current_steps
        self._total_spent += self._current_steps
        self._completion_info[scenario_name] = self._completion_snapshot(scenario_name)
        self._active_index = -1
        self._latest_checkpoint_path = checkpoint_path or self._latest_checkpoint_path
        self._write_status(_RUNNING)
        self._emit_milestone("leg_end", milestone_step, checkpoint_path=checkpoint_path)

    def on_scenario_resumed(self, scenario_name, history_entry):
        index = next((i for i, leg in enumerate(self._scenarios) if leg["name"] == scenario_name), -1)
        if index >= 0:
            steps = history_entry.get("steps", 0)
            self._scenarios[index]["status"] = "complete"
            self._scenarios[index]["steps"] = steps
            self._total_spent += steps
        self._write_status(_RUNNING)

    def get_completion_info(self, scenario_name):
        return self._completion_info.get(scenario_name)

    def on_training_complete(self):
        if self._tensorboard:
            self._tensorboard.close()
            self._tensorboard = None
        self._write_status(_COMPLETE)
        self._emit_milestone("complete", self._current_step())
        self._append_event("training_complete")

    def check_pause(self):
        tuning = self._load_tuning()
        elapsed = time.time() - self._started_at
        if elapsed > tuning.get("wall_clock_limit_seconds", DEFAULT_TUNING["wall_clock_limit_seconds"]):
            self._write_control("stop", "wall clock limit exceeded")
            self._append_event("wall_clock_limit")
            self._emit_milestone("wall_clock_limit", self._current_step())
            self._write_status(_STOPPING)
            return False

        while True:
            command = self._read_control_command()
            if command == "continue":
                self._write_status(_RUNNING)
                return True
            if command == "stop":
                self._write_status(_STOPPING)
                if self._latest_checkpoint_path is None:
                    self._emit_milestone("anomaly", self._current_step(), anomalies=[{
                        "kind": "control",
                        "metric": "command",
                        "value": 0.0,
                        "expected": [None, None],
                        "message": "training stopped before any checkpoint was recorded",
                    }])
                self._write_status(_STOPPING)
                return False
            if command == "pause":
                self._write_status(_PAUSED)
                time.sleep(1)
                self._load_tuning()
                continue

    def _maybe_emit_cadence_or_anomaly(self, iteration, tuning):
        wake_interval = int(tuning.get("wake_interval_steps", DEFAULT_TUNING["wake_interval_steps"]))
        if wake_interval > 0 and iteration > 0 and iteration % wake_interval == 0:
            self._emit_milestone("cadence", iteration, dedupe=True)

        anomaly_interval = int(tuning.get(
            "anomaly_check_interval_steps", DEFAULT_TUNING["anomaly_check_interval_steps"]))
        if anomaly_interval <= 0 or iteration <= 0:
            return
        if self._last_anomaly_check_step is not None and iteration - self._last_anomaly_check_step < anomaly_interval:
            return
        self._last_anomaly_check_step = iteration
        anomalies = detect_anomalies(self._metric_history, self._latest_metrics, self._latest_stats, tuning)
        if anomalies:
            self._emit_milestone("anomaly", iteration, anomalies=anomalies, dedupe=True)

    def _emit_milestone(self, reason, step, checkpoint_path=None, anomalies=None, dedupe=False):
        if reason not in _ALLOWED_MILESTONE_REASONS:
            raise ValueError(f"Unknown milestone reason: {reason}")
        tuning = self._load_tuning()
        if dedupe and self._last_milestone_step is not None:
            dedup_steps = int(tuning.get("wake_dedup_steps", DEFAULT_TUNING["wake_dedup_steps"]))
            if step - self._last_milestone_step < dedup_steps:
                return None

        self._milestone_id += 1
        self._last_milestone_step = step
        self._last_milestone_reason = reason
        if checkpoint_path:
            self._latest_checkpoint_path = checkpoint_path
        self._write_status(self._status_state())
        snapshot = self._snapshot(anomalies or [])
        report_path = os.path.join(self.run_dir, f"milestone_{self._milestone_id:04d}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(render_milestone_report(snapshot, tuning, self._baseline))
        event = self._base_event("milestone", step=step)
        event.update({
            "id": self._milestone_id,
            "reason": reason,
            "scenario": self._current_name,
            "report_path": report_path,
            "checkpoint_path": checkpoint_path or self._latest_checkpoint_path,
            "anomalies": anomalies or [],
        })
        append_jsonl(self.events_path, event)
        self._write_status(self._status_state())
        return event

    def _completion_snapshot(self, scenario_name):
        metric = None
        threshold = None
        value = None
        if self._active_index >= 0:
            exit_criteria = self._scenarios[self._active_index].get("exit_criteria") or {}
            metric = exit_criteria.get("metric")
            threshold = exit_criteria.get("threshold")
            criteria_scenario = exit_criteria.get("scenario")
            if metric:
                if criteria_scenario:
                    value = self._latest_metrics.get(f"{criteria_scenario}/{metric}")
                if value is None:
                    value = self._latest_metrics.get(metric)
        return {
            "scenario": scenario_name,
            "metric": metric,
            "threshold": threshold,
            "value": value,
            "met": value is not None and threshold is not None and value >= threshold,
            "steps": self._current_steps,
            "total": self._current_total,
        }

    def _snapshot(self, anomalies):
        return {
            "status": self._build_status(self._status_state()),
            "anomalies": anomalies,
            "journal_path": self.journal_path if os.path.exists(self.journal_path) else None,
        }

    def _status_state(self):
        if self._last_milestone_reason == "complete":
            return _COMPLETE
        return _RUNNING

    def _build_status(self, state):
        elapsed = time.time() - self._started_at
        total_done = self._total_spent + (self._current_steps if self._active_index >= 0 else 0)
        pct = total_done / self._total_budget * 100 if self._total_budget else 0.0
        current_pct = self._current_steps / self._current_total * 100 if self._current_total else 0.0
        eta = None
        if self._sps and self._total_budget and total_done < self._total_budget:
            eta = (self._total_budget - total_done) / self._sps
        return {
            "schema_version": 1,
            "state": state,
            "pid": os.getpid(),
            "experiment_dir": self.experiment_dir,
            "run_dir": self.run_dir,
            "scenario": self.scenario,
            "action_space": self.action_space,
            "model_kind": self.model_kind,
            "started_at": self._started_at,
            "updated_at": time.time(),
            "wall_clock_limit_seconds": self._last_valid_tuning.get("wall_clock_limit_seconds", 604_800),
            "circuit": self._scenarios,
            "current": {
                "name": self._current_name,
                "steps": self._current_steps,
                "total_steps": self._current_total,
                "pct": current_pct,
                "exit_metric": self._current_exit_metric(),
            },
            "overall": {
                "steps": total_done,
                "total_steps": self._total_budget,
                "pct": pct,
                "sps": self._sps,
                "eta_seconds": eta,
                "elapsed_seconds": elapsed,
            },
            "latest_metrics": self._latest_metrics,
            "latest_stats": self._latest_stats,
            "latest_checkpoint_path": self._latest_checkpoint_path,
            "last_milestone_step": self._last_milestone_step,
            "last_milestone_reason": self._last_milestone_reason,
        }

    def _current_exit_metric(self):
        if self._active_index < 0:
            return None
        return self._scenarios[self._active_index].get("exit_criteria")

    def _write_status(self, state):
        atomic_write_json(self.status_path, self._build_status(state))

    def _append_event(self, event_type, **extra):
        event = self._base_event(event_type)
        event.update(extra)
        append_jsonl(self.events_path, event)

    def _base_event(self, event_type, step=None):
        return {
            "schema_version": 1,
            "type": event_type,
            "timestamp": time.time(),
            "run_dir": self.run_dir,
            "experiment_dir": self.experiment_dir,
            "step": self._current_step() if step is None else step,
        }

    def _current_step(self):
        if self._active_index >= 0:
            return self._total_spent + self._current_steps
        return self._total_spent

    def _ensure_control_file(self):
        if not os.path.exists(self.control_path):
            self._write_control("continue", "")

    def _write_control(self, command, reason):
        atomic_write_json(self.control_path, {
            "schema_version": 1,
            "command": command,
            "reason": reason,
            "updated_at": time.time(),
        })

    def _read_control_command(self):
        try:
            with open(self.control_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._ensure_control_file()
            return "continue"
        command = payload.get("command", "continue")
        if command not in _ALLOWED_COMMANDS:
            self._append_event("control_error", command=command)
            return "continue"
        return command

    def _ensure_tuning_file(self):
        if not os.path.exists(self.tuning_path):
            atomic_write_json(self.tuning_path, deepcopy(DEFAULT_TUNING))

    def _load_tuning(self):
        self._ensure_tuning_file()
        try:
            with open(self.tuning_path, "r", encoding="utf-8") as f:
                tuning = json.load(f)
        except json.JSONDecodeError as exc:
            self._append_event("tuning_error", error=str(exc))
            return self._last_valid_tuning
        self._last_valid_tuning = tuning
        return tuning

    @staticmethod
    def _load_baseline(path):
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None


def _jsonable_dict(values):
    return {key: _jsonable_value(value) for key, value in values.items()}


def _jsonable_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def newest_active_experiment(root="training/experiments"):
    """Return the newest active status.json path under a training experiments root."""
    candidates = []
    for status_path in glob(os.path.join(root, "*", "runs", "*", "*", "status.json")):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("state") in {_RUNNING, _PAUSED}:
            candidates.append((os.path.getmtime(status_path), status_path))
    if not candidates:
        return None
    return max(candidates)[1]
