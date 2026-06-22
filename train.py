#!/usr/bin/env python3
# pylint: disable=too-many-lines
"""Train models to play The Legend of Zelda (NES)."""

# pylint: disable=duplicate-code

import argparse
import cProfile
import sys
import os
import select
import termios
import threading
import time
import tty
import faulthandler
import traceback
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.table import Table
import torch
from torch.utils.tensorboard import SummaryWriter
from triforce.experiment_report import DEFAULT_HEALTH_RANGES, flatten_metrics, is_metric_healthy
from triforce.demo import collect_demo_batch

from triforce import (ActionSpaceDefinition, ModelKindDefinition, TrainingScenarioDefinition,
                      TrainingCallback, make_zelda_env)
from triforce.ml_ppo import PPO
from triforce.models import Network
from triforce.scenario_wrapper import TrainingCircuitDefinition, TrainingCircuitEntry

BAR_WIDTH = 50
BAR_FILL = "▬"
BAR_EMPTY = " "
SPS_WINDOW = 25_000

# Pause states
_RUNNING = "running"
_REQUESTING = "requesting"
_PAUSED = "paused"


PROFILE_WARMUP_STEPS = 20_000
PROFILE_OUTPUT = "training.prof"


class ProfilingCallback(TrainingCallback):
    """Wraps another callback to collect cProfile data after a warmup period.

    Skips the first PROFILE_WARMUP_STEPS steps, then profiles the next `profile_steps`
    environment steps plus the subsequent model update, saves to PROFILE_OUTPUT, and exits.
    """
    def __init__(self, inner, profile_steps):
        self._inner = inner
        self._profile_steps = profile_steps
        self._total_steps = 0
        self._profiled_steps = 0
        self._profiler = None
        self._collecting = False
        self._ready_to_stop = False
        self._done = False

    def on_progress(self, steps, total_steps):
        self._total_steps += steps
        if self._inner:
            self._inner.on_progress(steps, total_steps)

        if self._done:
            return

        # After warmup, start collecting on the next rollout
        if not self._collecting and self._total_steps >= PROFILE_WARMUP_STEPS:
            self._collecting = True
            self._profiler = cProfile.Profile()
            self._profiler.enable()

        # Count profiled steps
        if self._collecting and not self._ready_to_stop:
            self._profiled_steps += steps
            if self._profiled_steps >= self._profile_steps:
                self._ready_to_stop = True

    def on_metrics(self, metrics, iteration, total_iterations):
        if self._inner:
            self._inner.on_metrics(metrics, iteration, total_iterations)

    def on_optimize(self, stats, iteration, total_iterations):
        if self._inner:
            self._inner.on_optimize(stats, iteration, total_iterations)

        # Stop profiling after the model update that follows our collection window
        if self._ready_to_stop and not self._done:
            self._profiler.disable()
            self._profiler.dump_stats(PROFILE_OUTPUT)
            self._done = True

    def check_pause(self):
        if self._done:
            return False
        if self._inner:
            return self._inner.check_pause()
        return True

    # Delegate circuit/scenario lifecycle methods
    def on_scenario_start(self, scenario_name, iterations, exit_criteria=None,
                          exit_threshold=None, exit_criteria_scenario=None):
        """Delegate scenario start to inner callback."""
        if self._inner and hasattr(self._inner, 'on_scenario_start'):
            self._inner.on_scenario_start(scenario_name, iterations,
                                          exit_criteria=exit_criteria,
                                          exit_threshold=exit_threshold,
                                          exit_criteria_scenario=exit_criteria_scenario)

    def on_scenario_complete(self, scenario_name):
        """Delegate scenario completion to inner callback."""
        if self._inner and hasattr(self._inner, 'on_scenario_complete'):
            self._inner.on_scenario_complete(scenario_name)

    def on_scenario_end(self, scenario_name, checkpoint_path=None):
        """Delegate scenario end to inner callback."""
        if self._inner:
            self._inner.on_scenario_end(scenario_name, checkpoint_path)

    def on_scenario_resumed(self, scenario_name, history_entry):
        """Delegate scenario resumed to inner callback."""
        if self._inner:
            self._inner.on_scenario_resumed(scenario_name, history_entry)

    def get_completion_info(self, scenario_name):
        """Delegate completion info to inner callback."""
        if self._inner:
            return self._inner.get_completion_info(scenario_name)
        return None

    def on_training_complete(self):
        """Delegate training completion to inner callback."""
        if self._inner and hasattr(self._inner, 'on_training_complete'):
            self._inner.on_training_complete()

    def on_circuit_start(self, scenarios):
        """Delegate circuit start to inner callback."""
        if self._inner and hasattr(self._inner, 'on_circuit_start'):
            self._inner.on_circuit_start(scenarios)


class _SubCircuitCallback:
    """Wrapper that forwards training events but suppresses circuit/scenario display events.

    Used when running a sub-circuit so the parent circuit's TUI display is not clobbered."""

    def __init__(self, inner):
        self._inner = inner

    def on_progress(self, steps, total_steps):
        """Forward progress updates to the parent display."""
        self._inner.on_progress(steps, total_steps)

    def on_optimize(self, stats, iteration, total_iterations):
        """Forward optimize stats to the parent display."""
        self._inner.on_optimize(stats, iteration, total_iterations)

    def on_metrics(self, metrics, iteration, total_iterations):
        """Forward metrics to the parent display."""
        self._inner.on_metrics(metrics, iteration, total_iterations)

    def check_pause(self):
        """Delegate pause checks to the parent."""
        return self._inner.check_pause()

    def on_circuit_start(self, scenarios):
        """Suppressed — sub-circuit must not overwrite parent's scenario list."""

    def on_scenario_start(self, scenario_name, iterations, exit_criteria=None,
                          exit_threshold=None, exit_criteria_scenario=None):
        """Suppressed — sub-circuit must not change parent's active scenario."""

    def on_scenario_end(self, scenario_name, checkpoint_path=None):
        """Suppressed — sub-circuit must not mark parent scenarios as complete."""

    def get_completion_info(self, scenario_name):
        """Delegate to parent — completion info is tracked at the display level."""
        return self._inner.get_completion_info(scenario_name)

    def on_scenario_complete(self, scenario_name):
        """Suppressed."""

    def on_training_complete(self):
        """Suppressed."""


class _KeyboardListener:
    """Background thread that reads single keypresses from stdin in raw mode."""

    def __init__(self, on_key):
        self._on_key = on_key
        self._stop = threading.Event()
        self._old_settings = None
        self._thread = None

    def start(self):
        """Start listening for keypresses."""
        if not sys.stdin.isatty():
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening and restore terminal."""
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _loop(self):
        while not self._stop.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch:
                    self._on_key(ch)


class TrainingDisplay(TrainingCallback):
    """Rich-based TUI for training progress and metrics, with tensorboard logging."""

    def __init__(self, live, log_dir):
        self._live = live
        self._log_dir = log_dir
        self._tensorboard = None

        # Circuit-level state
        self._scenarios = []        # list of (name, iterations)
        self._name_width = 0
        self._active_index = -1
        self._scenario_steps = {}   # scenario_name -> steps completed
        self._scenario_total = {}   # scenario_name -> total steps
        self._completed = set()     # scenario names that are done
        self._total_budget = 0
        self._total_spent = 0

        # Current scenario state
        self._current_steps = 0
        self._current_total = 0

        # SPS tracking for ETA — rolling window over last SPS_WINDOW iterations
        self._sps_samples = deque()   # (adjusted_time, cumulative_steps)
        self._cumulative_steps = 0
        self._rolling_sps = None
        self._pause_time_offset = 0.0  # total seconds spent paused, subtracted from monotonic

        # Metrics state
        self._game_metrics = {}
        self._optimize_stats = {}
        self._prev_game_metrics = {}
        self._prev_optimize_stats = {}
        self._kl_rollback_count = 0

        # Exit criteria for current scenario
        self._exit_criteria = None
        self._exit_threshold = None
        self._exit_criteria_scenario = None

        # Per-scenario completion metadata: name -> {metric, value, met, steps, total, duration}
        self._completion_info = {}
        self._scenario_start_time = None

        # Pause state
        self._pause_state = _RUNNING
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._frozen_eta = None     # ETA seconds frozen at pause time
        self._pause_start = None    # monotonic time when pause began

        # Quit state
        self._quit_confirm = False
        self._stop_requested = False
        self._last_refresh_time = 0.0
        self._circuit_start_time = None

        # Keyboard listener
        self._keyboard = _KeyboardListener(self._on_key)
        self._keyboard.start()

    def _active_time(self):
        """Returns time.monotonic() adjusted for pause durations."""
        return time.monotonic() - self._pause_time_offset

    def _on_key(self, ch):
        ch = ch.lower()

        if ch == 'q':
            if self._quit_confirm:
                self._stop_requested = True
                self._quit_confirm = False
                # If paused, resume so check_pause can return False
                if self._pause_state == _PAUSED:
                    self._resume()
                self._refresh(force=True)
            else:
                self._quit_confirm = True
                self._refresh(force=True)
            return

        # Any non-q key cancels quit confirm
        if self._quit_confirm:
            self._quit_confirm = False
            self._refresh(force=True)

        if ch != 'p':
            return

        if self._pause_state == _RUNNING:
            self._pause_state = _REQUESTING
            self._refresh(force=True)
        elif self._pause_state == _REQUESTING:
            # Cancel pause request before it took effect
            self._pause_state = _RUNNING
            self._refresh(force=True)
        elif self._pause_state == _PAUSED:
            self._resume()

    def _resume(self):
        """Resume from paused state."""
        if self._pause_start is not None:
            self._pause_time_offset += time.monotonic() - self._pause_start
            self._pause_start = None
        self._frozen_eta = None
        self._pause_state = _RUNNING
        self._resume_event.set()
        self._refresh(force=True)

    def check_pause(self):
        """Called by PPO after each training iteration. Blocks while paused.
        Returns True to continue training, False to stop early."""
        if self._stop_requested:
            return False

        if self._pause_state == _REQUESTING:
            # Freeze ETA before pausing
            if self._rolling_sps:
                total_done = self._total_spent + self._current_steps
                remaining = self._total_budget - total_done
                self._frozen_eta = remaining / self._rolling_sps
            self._pause_start = time.monotonic()
            self._pause_state = _PAUSED
            self._resume_event.clear()
            self._refresh(force=True)

            # Block until resumed
            self._resume_event.wait()

        return not self._stop_requested

    def on_circuit_start(self, scenarios):
        self._scenarios = scenarios
        self._name_width = max(len(name) for name, _ in scenarios)
        self._name_width = max(self._name_width, len("Total"))
        self._total_budget = sum(iters for _, iters in scenarios)
        self._total_spent = 0
        self._circuit_start_time = time.monotonic()
        for name, iters in scenarios:
            self._scenario_steps[name] = 0
            self._scenario_total[name] = iters
        self._refresh(force=True)

    def on_scenario_start(self, scenario_name, iterations, exit_criteria=None,
                          exit_threshold=None, exit_criteria_scenario=None):
        self._active_index = next(
            i for i, (name, _) in enumerate(self._scenarios) if name == scenario_name)
        self._current_steps = 0
        self._current_total = iterations
        self._scenario_total[scenario_name] = iterations
        self._game_metrics = {}
        self._optimize_stats = {}
        self._prev_game_metrics = {}
        self._prev_optimize_stats = {}
        self._exit_criteria = exit_criteria
        self._exit_threshold = exit_threshold
        self._exit_criteria_scenario = exit_criteria_scenario
        self._scenario_start_time = time.monotonic()

        # Set up tensorboard for this scenario
        if self._tensorboard:
            self._tensorboard.close()
        scenario_log_dir = os.path.join(self._log_dir, scenario_name)
        os.makedirs(scenario_log_dir, exist_ok=True)
        self._tensorboard = SummaryWriter(scenario_log_dir)
        self._refresh(force=True)

    def on_scenario_end(self, scenario_name, checkpoint_path=None):
        self._scenario_steps[scenario_name] = self._current_steps
        self._total_spent += self._current_steps
        self._completed.add(scenario_name)

        # Snapshot completion metadata for display
        duration = time.monotonic() - self._scenario_start_time if self._scenario_start_time else 0

        # For weighted circuits, the exit criteria metric belongs to a specific scenario.
        # Look up the qualified key (e.g. "overworld-skip-sword-all-items/success-rate") first,
        # then fall back to the unqualified key (which may be the first scenario's promoted value).
        metric_value = None
        if self._exit_criteria:
            if self._exit_criteria_scenario:
                qualified = f"{self._exit_criteria_scenario}/{self._exit_criteria}"
                metric_value = self._game_metrics.get(qualified)
            if metric_value is None:
                metric_value = self._game_metrics.get(self._exit_criteria)
        met = (metric_value is not None and self._exit_threshold is not None
               and metric_value >= self._exit_threshold)
        self._completion_info[scenario_name] = {
            'metric': self._exit_criteria,
            'threshold': self._exit_threshold,
            'value': metric_value,
            'met': met,
            'steps': self._current_steps,
            'total': self._current_total,
            'duration': duration,
        }

        self._prev_game_metrics = {}
        self._prev_optimize_stats = {}
        self._refresh(force=True)

    def on_scenario_resumed(self, scenario_name, history_entry):
        """Mark a scenario as completed from loaded training history (for --resume display)."""
        self._active_index = next(
            (i for i, (name, _) in enumerate(self._scenarios) if name == scenario_name), -1)
        self._completed.add(scenario_name)

        steps = history_entry.get('steps', 0)
        self._scenario_steps[scenario_name] = steps

        em = history_entry.get('exit_metric') or {}
        metric_name = em.get('name')
        target = em.get('target')
        actual = em.get('actual')
        met = actual is not None and target is not None and actual >= target

        self._completion_info[scenario_name] = {
            'metric': metric_name,
            'threshold': target,
            'value': actual,
            'met': met,
            'steps': steps,
            'total': self._scenario_total.get(scenario_name, steps),
            'duration': 0,
        }
        self._refresh(force=True)

    def get_completion_info(self, scenario_name):
        return self._completion_info.get(scenario_name)

    def on_progress(self, steps, total_steps):
        self._current_steps += steps
        self._current_total = total_steps
        if self._active_index >= 0:
            name = self._scenarios[self._active_index][0]
            self._scenario_steps[name] = self._current_steps
            self._scenario_total[name] = total_steps

        # Track SPS samples using pause-adjusted time
        self._cumulative_steps += steps
        now = self._active_time()
        self._sps_samples.append((now, self._cumulative_steps))

        # Evict samples older than SPS_WINDOW iterations
        cutoff = self._cumulative_steps - SPS_WINDOW
        while len(self._sps_samples) > 1 and self._sps_samples[0][1] < cutoff:
            self._sps_samples.popleft()

        # Compute rolling SPS once we have enough data
        if self._cumulative_steps >= SPS_WINDOW and len(self._sps_samples) > 1:
            oldest_time, oldest_steps = self._sps_samples[0]
            dt = now - oldest_time
            ds = self._cumulative_steps - oldest_steps
            self._rolling_sps = ds / dt if dt > 0 else None

        self._refresh()

    def on_metrics(self, metrics, iteration, total_iterations):
        self._prev_game_metrics = dict(self._game_metrics)

        flat_metrics = flatten_metrics(metrics)

        self._game_metrics.update(flat_metrics)
        if self._tensorboard:
            timestamp = time.time()
            for name, value in flat_metrics.items():
                if '/' not in name:
                    name = f"metrics/{name}"
                self._tensorboard.add_scalar(name, value, iteration, timestamp)
            self._tensorboard.flush()
        self._refresh(force=True)

    def on_optimize(self, stats, iteration, total_iterations):
        self._prev_optimize_stats = dict(self._optimize_stats)
        self._optimize_stats = dict(stats)
        if stats.get("losses/kl_rollback", 0) > 0:
            self._kl_rollback_count += 1
        if self._tensorboard:
            for name, value in stats.items():
                self._tensorboard.add_scalar(name, value, iteration)
            self._tensorboard.flush()
        self._refresh(force=True)

    def on_training_complete(self):
        if self._tensorboard:
            self._tensorboard.close()
            self._tensorboard = None
        self._keyboard.stop()
        self._refresh(force=True)

    def _refresh(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_refresh_time < 0.5:
            return
        self._last_refresh_time = now
        self._live.update(self._render())

    def _render_completed_scenario(self, name):
        """Render a completed scenario line with checkmark, duration, steps, and metric."""
        info = self._completion_info.get(name, {})
        line = Text("  ✔ ", style="green")
        line.append(name.ljust(self._name_width), style="green")

        dur = info.get('duration', 0)
        if dur == 0 and info.get('steps', 0) > 0:
            line.append("  resumed", style="dim")
        else:
            line.append(f"  {self._format_duration(dur):>7}", style="dim")

        steps = info.get('steps', 0)
        total = info.get('total', 0)
        line.append(f"  {steps:>10,}", style="cyan")
        line.append(" of ", style="white")
        line.append(f"{total:>10,}", style="cyan")
        line.append(" steps", style="white")

        metric_name = info.get('metric')
        metric_val = info.get('value')
        if metric_name is not None and metric_val is not None:
            met = info.get('met', False)
            style = "white" if met else "red"
            line.append(f"  {metric_name}: ", style="dim")
            line.append(f"{metric_val:.4f}", style=style)

        return line

    def _render(self):
        parts = []
        parts.append(Text(""))

        for i, (name, _) in enumerate(self._scenarios):
            if name in self._completed:
                parts.append(self._render_completed_scenario(name))
            elif i == self._active_index:
                steps = self._scenario_steps.get(name, 0)
                total = self._scenario_total.get(name, 1)
                parts.append(self._render_bar(name, steps, total, active=True))
            else:
                total = self._scenario_total.get(name, 0)
                parts.append(self._render_bar(name, 0, total, active=False))

        # Total progress bar with ETA
        total_done = self._total_spent + (self._current_steps if self._active_index >= 0 else 0)
        parts.append(Text(""))
        total_bar = self._render_bar("Total", total_done, self._total_budget, active=True)

        if self._pause_state == _PAUSED and self._frozen_eta is not None:
            total_bar.append(f"  ETA {self._format_duration(self._frozen_eta)}", style="yellow")
        elif self._rolling_sps and total_done < self._total_budget:
            remaining = self._total_budget - total_done
            eta_seconds = remaining / self._rolling_sps
            total_bar.append(f"  ETA {self._format_duration(eta_seconds)}", style="yellow")
        parts.append(total_bar)

        # Pause / quit status messages
        if self._stop_requested:
            parts.append(Text(""))
            parts.append(Text("  ⏹  Stopping after next training point...", style="red"))
        elif self._quit_confirm:
            parts.append(Text(""))
            parts.append(Text("  ⏹  Press q again to end training early.", style="red"))
        elif self._pause_state == _REQUESTING:
            parts.append(Text(""))
            parts.append(Text("  ⏸  Pausing after next training point...", style="yellow"))
        elif self._pause_state == _PAUSED:
            parts.append(Text(""))
            parts.append(Text("  ⏸  Paused (press p to resume)", style="yellow bold"))

        # Metrics
        if self._game_metrics or self._optimize_stats:
            parts.append(Text(""))
            parts.append(self._render_metrics())

        return Group(*parts)

    def _render_bar(self, name, steps, total, active):
        padded = name.ljust(self._name_width)
        pct = (steps / total * 100) if total > 0 else 0
        filled = int(BAR_WIDTH * min(steps, total) / total) if total > 0 else 0
        bar_str = BAR_FILL * filled + BAR_EMPTY * (BAR_WIDTH - filled)

        line = Text(f"  {padded}  [")
        line.append(bar_str[:filled], style="green")
        line.append(bar_str[filled:])
        line.append(f"] {pct:5.1f}%")
        if active and total > 0:
            line.append(f"  ({steps:,} / {total:,})")
        return line

    @staticmethod
    def _format_duration(seconds):
        """Format seconds into a human-readable duration string (no seconds)."""
        if seconds < 60:
            return "<1m"
        if seconds < 3600:
            m = int(seconds) // 60
            return f"{m}m"
        h, remainder = divmod(int(seconds), 3600)
        m = remainder // 60
        return f"{h}h{m:02d}m"

    _HEALTHY_RANGES = DEFAULT_HEALTH_RANGES

    def _is_healthy(self, key, value):
        """Returns True if value is within healthy range for the given metric key."""
        return is_metric_healthy(key, value, self._HEALTHY_RANGES)

    def _get_target_text(self, key):
        """Return a dim-styled Text showing the target/healthy range for a metric."""
        # Check exit criteria first (takes priority for the matched metric)
        if self._exit_criteria and self._exit_criteria == key and self._exit_threshold is not None:
            return Text(f"≥{self._exit_threshold:g}", style="dim")

        # Resolve per-head keys to their aggregate range
        lookup = key
        if "/head_" in key:
            if key.endswith("/entropy"):
                lookup = "losses/attention/entropy"
            elif key.endswith("/top1"):
                lookup = "losses/attention/top1_weight"

        bounds = self._HEALTHY_RANGES.get(lookup)
        if bounds is None:
            return Text("", style="dim")
        lo, hi = bounds
        if lo is not None and hi is not None:
            return Text(f"{lo:g}–{hi:g}", style="dim")
        if lo is not None:
            return Text(f"≥{lo:g}", style="dim")
        if hi is not None:
            return Text(f"≤{hi:g}", style="dim")
        return Text("", style="dim")

    def _add_metric_row(self, table, key, display_name, fmt, value, prev_value):
        """Add a metric row with value coloring and delta column."""
        style = "white" if self._is_healthy(key, value) else "red"
        val_text = Text(f"{value:{fmt}}", style=style)

        if prev_value is not None:
            delta = value - prev_value
            delta_text = Text(f"{delta:+{fmt}}", style="dim")
        else:
            delta_text = Text("", style="dim")

        target_text = self._get_target_text(key)
        table.add_row(display_name, val_text, delta_text, target_text)

    def _render_metrics(self):  # pylint: disable=too-many-statements
        table = Table(show_header=False, show_edge=False, pad_edge=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", min_width=24)
        table.add_column("Value", justify="right", min_width=12)
        table.add_column("Δ", justify="right", min_width=10, style="dim")
        table.add_column("Target", justify="left", min_width=10, style="dim")

        # Total elapsed time — always first
        if self._circuit_start_time is not None:
            elapsed = time.monotonic() - self._circuit_start_time - self._pause_time_offset
            table.add_row("Time", Text(self._format_duration(elapsed), style="yellow"), "", "")

        # SPS — always next, standalone
        sps = self._optimize_stats.get("charts/SPS")
        has_sps = False
        if sps is not None:
            prev_sps = self._prev_optimize_stats.get("charts/SPS")
            self._add_metric_row(table, "charts/SPS", "SPS", ".0f", sps, prev_sps)
            has_sps = True

        # Performance metrics
        perf_metrics = [
            ("room-progress", "progress/avg", ".1f"),
            ("progress/max", "progress/max", ".0f"),
            ("rewards", "rewards", ".2f"),
            ("success-rate", "success-rate", ".4f"),
        ]
        has_perf = False
        for key, display_name, fmt in perf_metrics:
            val = self._game_metrics.get(key)
            if val is not None:
                if not has_perf and has_sps:
                    table.add_row("", "", "", "")
                prev = self._prev_game_metrics.get(key)
                self._add_metric_row(table, key, display_name, fmt, val, prev)
                has_perf = True

        # Top ending — find the highest-percentage endings/* metric.
        # In weighted mode, prefer the exit-criteria scenario's qualified endings
        # (e.g. "scenario/endings/X") over the first-scenario promoted ones.
        endings_prefix = f"{self._exit_criteria_scenario}/endings/" if self._exit_criteria_scenario \
            else "endings/"
        top_ending_key, top_ending_val = None, -1
        for key, val in self._game_metrics.items():
            if key.startswith(endings_prefix) and isinstance(val, (int, float)) \
                    and val > top_ending_val:
                top_ending_key, top_ending_val = key, val
        if top_ending_key is not None:
            if not has_perf and has_sps:
                table.add_row("", "", "", "")
            ending_name = top_ending_key.rsplit("/", 1)[-1]
            prev_val = self._prev_game_metrics.get(top_ending_key)
            self._add_metric_row(table, top_ending_key, ending_name, ".2f",
                                 top_ending_val, prev_val)
            has_perf = True

        # Entropy / attention metrics
        entropy_metrics = [
            ("losses/attention/entropy", "attention/entropy", ".4f"),
            ("losses/attention/top1_weight", "attention/top1_weight", ".4f"),
        ]
        for i in range(4):
            entropy_metrics.append((f"losses/attention/head_{i}/entropy", f"  head_{i}/entropy", ".4f"))
            entropy_metrics.append((f"losses/attention/head_{i}/top1_weight", f"  head_{i}/top1", ".4f"))
        entropy_metrics.extend([
            ("losses/cross_attention/entropy", "cross_attn/entropy", ".4f"),
            ("losses/cross_attention/top1_weight", "cross_attn/top1", ".4f"),
            ("losses/entropy/action_type", "entropy/action_type", ".4f"),
            ("losses/entropy/direction", "entropy/direction", ".4f"),
        ])
        has_entropy = False
        for key, display_name, fmt in entropy_metrics:
            val = self._optimize_stats.get(key)
            if val is not None:
                if not has_entropy and (has_perf or has_sps):
                    table.add_row("", "", "", "")
                prev = self._prev_optimize_stats.get(key)
                self._add_metric_row(table, key, display_name, fmt, val, prev)
                has_entropy = True

        # Loss metrics
        loss_metrics = [
            ("losses/value_loss", "value_loss", ".4f"),
            ("losses/policy_loss", "policy_loss", ".4f"),
            ("losses/entropy", "entropy", ".4f"),
            ("losses/approx_kl", "approx_kl", ".6f"),
            ("losses/clipfrac", "clipfrac", ".4f"),
            ("losses/explained_variance", "explained_var", ".4f"),
        ]
        has_loss = False
        for key, display_name, fmt in loss_metrics:
            val = self._optimize_stats.get(key)
            if val is not None:
                if not has_loss and (has_perf or has_entropy or has_sps):
                    table.add_row("", "", "", "")
                prev = self._prev_optimize_stats.get(key)
                self._add_metric_row(table, key, display_name, fmt, val, prev)
                has_loss = True

        # KL rollback counter — only show when rollbacks have occurred
        if self._kl_rollback_count > 0:
            if has_loss or has_perf or has_entropy or has_sps:
                table.add_row("", "", "", "")
            table.add_row("[bold red]⚠ KL rollbacks[/bold red]",
                          f"[bold red]{self._kl_rollback_count}[/bold red]", "", "")

        return table

def _dump_trace_with_locals(exc_type, exc_value, exc_traceback):
    with open("crash_log.txt", "w", encoding="utf8") as f:
        f.write(f"Unhandled exception: {exc_type.__name__}: {exc_value}\n\n")

        for frame, lineno in traceback.walk_tb(exc_traceback):
            f.write(f"File: {frame.f_code.co_filename}, Line: {lineno}, Function: {frame.f_code.co_name}\n")
            f.write("Locals:\n")
            for var_name, var_value in frame.f_locals.items():
                typename = type(var_value).__name__
                f.write(f"  {typename} {var_name}: {var_value}\n")
            f.write("\n")


def _next_counter(base_dir):
    """Find the next counter directory (0, 1, 2, ...) under base_dir."""
    if not os.path.exists(base_dir):
        return 0
    existing = [int(d) for d in os.listdir(base_dir) if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))]
    return max(existing, default=-1) + 1


def _model_stem(model_kind_name, action_space_name):
    """Returns the base filename stem: {model-kind}_{action-space}."""
    return f"{model_kind_name}_{action_space_name}"


def _get_kwargs_from_args(args, model_kind, action_space_def):
    kwargs = {}
    if args.load is not None:
        obs, act = Network.load_spaces(args.load)
        network = model_kind.network_class(obs, act,
                                           model_kind=model_kind.name,
                                           action_space_name=action_space_def.name)
        network.load(args.load)
        kwargs['model'] = network

        # Restore optimizer state if present in checkpoint
        optimizer_state = Network.load_optimizer_state(args.load)
        if optimizer_state is not None:
            kwargs['optimizer_state'] = optimizer_state

        # Infer obs_kind and frame_stack from saved model when not explicitly set
        if args.obs_kind is None and args.frame_stack is None:
            from triforce.observation_wrapper import infer_obs_kind  # pylint: disable=import-outside-toplevel
            inferred_kind, inferred_stack = infer_obs_kind(obs)
            kwargs['obs_kind'] = inferred_kind
            kwargs['frame_stack'] = inferred_stack

    if args.frame_stack is not None:
        kwargs['frame_stack'] = args.frame_stack

    # Auto-select obs_kind for impala models (they require full-rgb)
    if 'obs_kind' not in kwargs:
        obs_kind = args.obs_kind
        if obs_kind is None and model_kind.name.startswith('impala'):
            obs_kind = 'full-rgb'
        if obs_kind is not None:
            kwargs['obs_kind'] = obs_kind

    # impala models use frame_stack=1 by default with full-rgb
    if kwargs.get('obs_kind') == 'full-rgb' and 'frame_stack' not in kwargs:
        kwargs['frame_stack'] = 1

    if args.render_mode:
        kwargs['render_mode'] = args.render_mode

    if args.ent_coef is not None:
        kwargs['ent_coeff'] = args.ent_coef

    if args.device is not None:
        kwargs['device'] = args.device

    if args.demo_trace is not None:
        if args.load is None:
            print("Error: --demo-trace requires --load")
            sys.exit(1)
        demo_device = torch.device(args.device or 'cpu')
        kwargs["demo_batch"] = collect_demo_batch(args.load, args.demo_scenario, args.demo_trace,
                                                    args.demo_prefix_east, demo_device)
        kwargs["demo_bc_coeff"] = args.demo_bc_coeff
        kwargs["demo_trace"] = args.demo_trace
        kwargs["demo_scenario"] = args.demo_scenario
        kwargs["demo_prefix_east"] = args.demo_prefix_east

    if args.parallel > 1:
        kwargs['envs'] = args.parallel

    circuit_def = TrainingCircuitDefinition.get(args.scenario)
    if circuit_def is None:
        circuit = [TrainingCircuitEntry(scenario=args.scenario)]
    else:
        circuit = circuit_def.scenarios

    return kwargs, circuit, circuit_def

def _get_circuit_exit_criteria(scenario_entry, sub_circuit_def):
    """Get exit criteria for an embedded circuit entry, checking the entry then the sub-circuit.

    Returns (ExitCriteria, scenario_name) where scenario_name is the owning scenario
    (used to look up qualified metric keys in weighted mode).
    """
    ec = scenario_entry.exit_criteria
    if ec is not None:
        return ec, None

    for sub_entry in sub_circuit_def.scenarios:
        if sub_entry.exit_criteria:
            return sub_entry.exit_criteria, sub_entry.scenario
    return None, None


def _build_history_entry(scenario_name, steps, callback=None):
    """Build a training history entry for a completed scenario/circuit leg."""
    entry = {"scenario": scenario_name, "steps": steps}
    if callback:
        info = callback.get_completion_info(scenario_name)
        if info:
            entry["steps"] = info.get("steps", steps)
            if info.get("metric"):
                entry["exit_metric"] = {
                    "name": info["metric"],
                    "target": info.get("threshold"),
                    "actual": info.get("value"),
                }
            entry["exit_metric_met"] = info.get("met")
            entry["completion_reason"] = info.get("completion_reason")
    return entry


def train_once(ppo, scenario_def, model_kind, action_space_def, checkpoint_dir, iterations,
               callback=None, **kwargs):
    """Trains a model with the given scenario.  Returns (model, iterations_used, checkpoint_path)."""
    multihead = getattr(model_kind.network_class, 'is_multihead', False)
    kwargs['multihead'] = multihead

    def create_env():
        return make_zelda_env(scenario_def, action_space_def.actions, **kwargs)

    steps_before = kwargs.get('model', None)
    steps_before = steps_before.steps_trained if steps_before else 0

    # Pass env-creation info for multi-env subprocess spawning
    kwargs['scenario_def'] = scenario_def
    kwargs['action_space_name'] = action_space_def.actions

    # Pass metadata for checkpoint naming
    stem = _model_stem(model_kind.name, action_space_def.name)
    kwargs['model_name'] = stem
    kwargs['model_kind'] = model_kind.name
    kwargs['action_space_name_str'] = action_space_def.name

    model = ppo.train(model_kind.network_class, create_env, iterations, callback,
                      save_path=checkpoint_dir, **kwargs)

    # Save leg checkpoint with training history
    training_history = kwargs.get('training_history')
    checkpoint_path = f"{checkpoint_dir}/{stem}_{scenario_def.name}_{model.steps_trained}.pt"
    model.save(checkpoint_path, optimizer=ppo.optimizer, training_history=training_history)
    return model, model.steps_trained - steps_before, checkpoint_path

def _run_circuit(ppo, circuit, model_kind, action_space_def, checkpoint_dir, kwargs, total_budget,
                 callback=None, circuit_def=None, skip_to=None, outer_exit_criteria=None):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Run training circuit and return (final_model, final_scenario_def).

    For weighted circuits, dispatches to _run_weighted_circuit.
    outer_exit_criteria: ExitCriteria from the parent sequential entry, overrides inner criteria.
    """
    if circuit_def is not None and circuit_def.kind == 'weighted':
        return _run_weighted_circuit(ppo, circuit_def, model_kind, action_space_def,
                                     checkpoint_dir, kwargs, total_budget, callback,
                                     outer_exit_criteria=outer_exit_criteria)

    return _run_sequential_circuit(ppo, circuit, model_kind, action_space_def,
                                    checkpoint_dir, kwargs, total_budget, callback,
                                    skip_to=skip_to)


def _run_sequential_circuit(ppo, circuit, model_kind, action_space_def, checkpoint_dir, kwargs,
                             total_budget, callback=None, skip_to=None):
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-branches,too-many-locals
    """Run training circuit and return (final_model, final_scenario_def)."""
    iterations_spent = 0
    model = None
    scenario_def = None
    skipping = skip_to is not None

    # Initialize training history from kwargs (may be inherited from loaded checkpoint)
    training_history = list(kwargs.get('training_history') or [])

    # Resolve iteration counts for all scenarios upfront so the display can show them
    scenario_plan = []
    for scenario_entry in circuit:
        if scenario_entry.circuit:
            name = f"[circuit] {scenario_entry.circuit}"
            sub_circuit_def = TrainingCircuitDefinition.get(scenario_entry.circuit)
            if sub_circuit_def is None:
                raise ValueError(f"Unknown circuit: {scenario_entry.circuit}")
            iters = scenario_entry.iterations or total_budget or 2000000
        else:
            sdef = TrainingScenarioDefinition.get(scenario_entry.scenario)
            if sdef is None:
                raise ValueError(f"Unknown scenario: {scenario_entry.scenario}")
            name = sdef.name

            if scenario_entry.iterations is not None:
                iters = scenario_entry.iterations
            elif total_budget is not None:
                iters = total_budget
            else:
                iters = sdef.iterations
        scenario_plan.append((name, iters))

    if callback:
        callback.on_circuit_start(scenario_plan)

    # Build lookup from training history for resumed scenario display
    history_by_name = {}
    for entry in training_history:
        history_by_name[entry.get('scenario', '')] = entry

    for scenario_entry in circuit:
        # Determine the name for skip-to matching
        entry_name = (f"[circuit] {scenario_entry.circuit}" if scenario_entry.circuit
                      else scenario_entry.scenario)

        # Skip completed legs when resuming
        if skipping:
            if entry_name in (skip_to, f"[circuit] {skip_to}"):
                skipping = False
            if callback:
                hist = history_by_name.get(entry_name, {})
                callback.on_scenario_resumed(entry_name, hist)
            continue

        if scenario_entry.circuit:
            sub_circuit_def = TrainingCircuitDefinition.get(scenario_entry.circuit)
            sub_budget = scenario_entry.iterations or total_budget

            if total_budget is not None:
                remaining = total_budget - iterations_spent
                sub_budget = min(sub_budget, remaining) if sub_budget else remaining

            if sub_budget is not None and sub_budget <= 0:
                break

            if callback:
                ec, ec_scenario = _get_circuit_exit_criteria(scenario_entry, sub_circuit_def)
                callback.on_scenario_start(f"[circuit] {scenario_entry.circuit}", sub_budget or 0,
                                           exit_criteria=ec.metric if ec else None,
                                           exit_threshold=ec.threshold if ec else None,
                                           exit_criteria_scenario=ec_scenario)

            # Pass current model and history into sub-circuit with suppressed display events
            sub_kwargs = dict(kwargs)
            sub_kwargs['training_history'] = training_history
            sub_callback = _SubCircuitCallback(callback) if callback else None
            model, scenario_def = _run_circuit(ppo, sub_circuit_def.scenarios, model_kind,
                                               action_space_def, checkpoint_dir, sub_kwargs,
                                               sub_budget, sub_callback, sub_circuit_def,
                                               outer_exit_criteria=scenario_entry.exit_criteria)

            stem = _model_stem(model_kind.name, action_space_def.name)
            circuit_name = scenario_entry.circuit
            checkpoint_path = f"{checkpoint_dir}/{stem}_{circuit_name}_{model.steps_trained}.pt"

            if callback:
                callback.on_scenario_end(f"[circuit] {scenario_entry.circuit}", checkpoint_path)

            # Record completed circuit in training history before saving the leg checkpoint.
            circuit_label = f"[circuit] {scenario_entry.circuit}"
            training_history.append(_build_history_entry(circuit_label, sub_budget or 0, callback))
            model.save(checkpoint_path, optimizer=ppo.optimizer, training_history=training_history)

            kwargs['model'] = model
            iterations_spent += sub_budget or 0
            continue

        scenario_def = TrainingScenarioDefinition.get(scenario_entry.scenario)

        if scenario_entry.iterations is not None:
            iterations = scenario_entry.iterations
        elif total_budget is not None:
            iterations = total_budget - iterations_spent
        else:
            iterations = scenario_def.iterations

        # Cap to remaining budget if a total budget was specified
        if total_budget is not None:
            iterations = min(iterations, total_budget - iterations_spent)

        if iterations <= 0:
            break

        if scenario_entry.exit_criteria:
            kwargs['exit_criteria'] = scenario_entry.exit_criteria.metric
            kwargs['exit_threshold'] = scenario_entry.exit_criteria.threshold
        elif 'exit_criteria' in kwargs:
            del kwargs['exit_criteria']
            del kwargs['exit_threshold']

        if callback:
            ec = scenario_entry.exit_criteria
            callback.on_scenario_start(scenario_def.name, iterations,
                                       exit_criteria=ec.metric if ec else None,
                                       exit_threshold=ec.threshold if ec else None)

        # Thread training history through so periodic saves include it
        kwargs['training_history'] = training_history
        model, used, checkpoint_path = train_once(ppo, scenario_def, model_kind, action_space_def,
                                                 checkpoint_dir, iterations, callback, **kwargs)

        if callback:
            callback.on_scenario_end(scenario_def.name, checkpoint_path)

        # Record completed scenario in training history and resave checkpoint metadata.
        training_history.append(_build_history_entry(scenario_def.name, used, callback))
        model.save(checkpoint_path, optimizer=ppo.optimizer, training_history=training_history)

        kwargs['model'] = model
        iterations_spent += used

    return model, scenario_def


def _run_weighted_circuit(ppo, circuit_def, model_kind, action_space_def, checkpoint_dir, kwargs,
                           total_budget, callback=None, outer_exit_criteria=None):
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Run a weighted training circuit.

    All scenarios run concurrently with step-count proportional to their weights.
    The WeightedScenarioSelector (via RPC) decides which scenario each worker runs on reset.
    outer_exit_criteria: ExitCriteria from the parent sequential entry. When provided, overrides
    inner entry exit criteria — applied to the first (primary) scenario.
    """
    # Resolve scenario definitions and weights
    scenario_defs = []
    weights = []
    exit_criteria_map = {}

    for entry in circuit_def.scenarios:
        sdef = TrainingScenarioDefinition.get(entry.scenario)
        if sdef is None:
            raise ValueError(f"Unknown scenario: {entry.scenario}")
        scenario_defs.append(sdef)
        weights.append(entry.weight)
        if entry.exit_criteria:
            exit_criteria_map[entry.scenario] = entry.exit_criteria

    # Outer exit criteria overrides inner — applied to the first (primary) scenario
    if outer_exit_criteria and scenario_defs:
        exit_criteria_map = {scenario_defs[0].name: outer_exit_criteria}

    # Determine iteration budget
    if total_budget is not None:
        iterations = total_budget
    else:
        # Use max iterations from any scenario, or a sensible default
        iterations = max((s.iterations for s in scenario_defs), default=2_000_000)

    multihead = getattr(model_kind.network_class, 'is_multihead', False)
    kwargs['multihead'] = multihead

    def create_env():
        from triforce.zelda_env import make_weighted_zelda_env  # pylint: disable=import-outside-toplevel
        from triforce.scenario_wrapper import WeightedScenarioSelector  # pylint: disable=import-outside-toplevel
        dummy_selector = WeightedScenarioSelector([s.name for s in scenario_defs], weights)
        return make_weighted_zelda_env(scenario_defs, action_space_def.actions,
                                       dummy_selector, **kwargs)

    # Pass metadata for checkpoint naming
    stem = _model_stem(model_kind.name, action_space_def.name)
    kwargs['model_name'] = stem
    kwargs['model_kind'] = model_kind.name
    kwargs['action_space_name_str'] = action_space_def.name
    kwargs['network_class'] = model_kind.network_class

    weighted_label = f"weighted[{len(scenario_defs)}]"
    if callback:
        callback.on_circuit_start([(weighted_label, iterations)])

    if callback:
        # Find the scenario that has exit criteria for display purposes
        ec_scenario_name = None
        ec = None
        for sdef in scenario_defs:
            ec = exit_criteria_map.get(sdef.name)
            if ec is not None:
                ec_scenario_name = sdef.name
                break
        callback.on_scenario_start(weighted_label, iterations,
                                   exit_criteria=ec.metric if ec else None,
                                   exit_threshold=ec.threshold if ec else None,
                                   exit_criteria_scenario=ec_scenario_name)

    model = ppo.train_weighted(
        model_kind.network_class, create_env, scenario_defs, weights,
        action_space_def.actions, iterations, exit_criteria_map, callback,
        save_path=checkpoint_dir,
        **{k: v for k, v in kwargs.items() if k != 'network_class'})
    # Save final checkpoint with training history after emitting the leg-end callback.
    training_history = kwargs.get('training_history')
    checkpoint_path = f"{checkpoint_dir}/{stem}_weighted_{model.steps_trained}.pt"

    if callback:
        label = f"weighted[{len(scenario_defs)}]"
        callback.on_scenario_end(label, checkpoint_path)
        if training_history is not None:
            training_history.append(_build_history_entry(label, iterations, callback))
    model.save(checkpoint_path, optimizer=ppo.optimizer, training_history=training_history)
    return model, scenario_defs[0]


def main():
    # pylint: disable=too-many-statements
    """Main entry point."""
    args = parse_args()

    if args.hook_exceptions:
        faulthandler.enable()
        sys.excepthook = _dump_trace_with_locals

    if args.profile:
        if args.parallel not in (1, 6):
            print("Error: --profile requires single-env mode. Do not set --parallel with --profile.")
            sys.exit(1)
        args.parallel = 1

    # Resolve model kind and action space (use defaults if not specified)
    action_space_def = ActionSpaceDefinition.get(args.action_space) if args.action_space \
        else ActionSpaceDefinition.get_default()
    model_kind = ModelKindDefinition.get(args.model_kind) if args.model_kind \
        else ModelKindDefinition.get_default()

    # Build directory structure: output/scenario/counter/
    output_base = args.output or 'training'
    scenario_dir = os.path.join(output_base, args.scenario)
    counter = _next_counter(scenario_dir)
    run_dir = os.path.join(scenario_dir, str(counter))
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    console = Console()
    experiment_dir = args.experiment_dir or run_dir
    console.print(f"Output: {run_dir}")
    if args.headless_agent:
        console.print(f"Experiment: {experiment_dir}")
        console.print("Mode: headless-agent")
    console.print(f"Model kind: {model_kind.name}, Action space: {action_space_def.name}")

    kwargs, circuit, circuit_def = _get_kwargs_from_args(args, model_kind, action_space_def)

    # Resolve --resume / --skip-to
    skip_to = args.skip_to
    training_history = None
    if args.resume and not skip_to:
        if args.load is None:
            console.print("[red]--resume requires --load[/red]")
            sys.exit(1)
        training_history = Network.load_training_history(args.load)
        if training_history:
            last_scenario = training_history[-1].get("scenario", "")
            skip_to = last_scenario
            console.print(f"Resuming after: {last_scenario} ({len(training_history)} legs in history)")
        else:
            console.print("[yellow]Warning: --resume but checkpoint has no training history[/yellow]")

    # Seed training history into kwargs so circuit runners pick it up
    if training_history:
        kwargs['training_history'] = training_history

    ppo = PPO(**kwargs)

    if args.headless_agent:
        from triforce.agent_callback import AgentTrainingCallback  # pylint: disable=import-outside-toplevel
        callback = AgentTrainingCallback(run_dir, experiment_dir, log_dir,
                                         scenario=args.scenario,
                                         action_space=action_space_def.name,
                                         model_kind=model_kind.name,
                                         reporting={
                                             'baseline_eval_json': args.baseline_eval_json,
                                             'final_eval_scenario': getattr(circuit_def, 'final_eval_scenario', None),
                                             'final_eval_episodes': getattr(circuit_def, 'final_eval_episodes', 100),
                                         })
        model, scenario_def = _run_circuit(ppo, circuit, model_kind, action_space_def,
                                           checkpoint_dir, kwargs, args.iterations,
                                           callback=callback, circuit_def=circuit_def,
                                           skip_to=skip_to)
    else:
        with Live(console=console, refresh_per_second=4) as live:
            display = TrainingDisplay(live, log_dir)
            callback = display
            if args.profile:
                callback = ProfilingCallback(display, args.profile)
                console.print(f"Profiling: {args.profile} steps after {PROFILE_WARMUP_STEPS} warmup → {PROFILE_OUTPUT}")
            model, scenario_def = _run_circuit(ppo, circuit, model_kind, action_space_def,
                                               checkpoint_dir, kwargs, args.iterations,
                                               callback=callback, circuit_def=circuit_def,
                                               skip_to=skip_to)
            display.on_training_complete()

    # Save final result in the run directory (not checkpoints)
    stem = _model_stem(model_kind.name, action_space_def.name)
    final_path = f"{run_dir}/{stem}.pt"
    final_history = kwargs.get('training_history')
    model.save(final_path, optimizer=ppo.optimizer, training_history=final_history)
    console.print(f"\nFinal model: {final_path}")
    if args.headless_agent:
        callback.set_final_model_path(final_path)
        callback.on_training_complete()

    if args.evaluate:
        _run_post_training_eval(model, action_space_def, model_kind, scenario_def,
                                args.evaluate, console, **kwargs)


def _run_post_training_eval(model, action_space_def, model_kind, scenario_def, episodes,
                            console=None, **kwargs):
    """Runs evaluation episodes after training and prints a progress report."""
    # pylint: disable=import-outside-toplevel
    from rich.progress import Progress
    from evaluate import evaluate_one_model, print_progress_report

    if console is None:
        console = Console()
    console.print(f"\nRunning {episodes} evaluation episodes...")

    multihead = getattr(model_kind.network_class, 'is_multihead', False)
    kwargs['multihead'] = multihead

    def create_eval_env():
        return make_zelda_env(scenario_def, action_space_def.actions, **kwargs)

    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating...", total=episodes)
        def update():
            progress.advance(task)
        _, progress_values, max_progress = evaluate_one_model(
            create_eval_env, model, episodes, update)

    if progress_values is not None:
        print_progress_report(progress_values, max_progress, episodes, scenario_def.name, metrics=None)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="train - Train Zelda ML models")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--frame-stack", type=int, default=None, help="The number of frames to stack.")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default=None, help="The device to use.")
    parser.add_argument("--render-mode", type=str, default=None, help="The render mode to use.")
    parser.add_argument("--obs-kind", type=str, default=None,
                        choices=['viewport', 'gameplay', 'full-rgb'],
                        help="Observation kind. Auto-selected for impala models if not set.")

    parser.add_argument('scenario', type=str, help='The scenario or circuit to train on.')
    parser.add_argument('action_space', type=str, nargs='?', default='all-items',
                        help='Action space name (default: all-items).')
    parser.add_argument('model_kind', type=str, nargs='?', default='impala-multihead',
                        help='Model kind name (default: impala-multihead).')
    parser.add_argument("--output", type=str, help="Location to write to.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--parallel", type=int, default=16, help="Number of parallel environments to run.")
    parser.add_argument("--load", type=str, help="Load a model to continue training.")
    parser.add_argument("--resume", action='store_true',
                        help="Resume circuit from saved position in --load checkpoint.")
    parser.add_argument("--skip-to", type=str, default=None, metavar="SCENARIO",
                        help="Skip circuit legs until reaching SCENARIO (overrides --resume).")
    parser.add_argument("--evaluate", type=int, default=None, metavar="N",
                        help="Run N evaluation episodes after training and print a progress report.")
    parser.add_argument("--hook-exceptions", action='store_true', help="Dump tracebacks on unhandled exceptions.")
    parser.add_argument("--profile", type=int, default=None, metavar="N",
                        help="Profile N environment steps (after 20K warmup), save to training.prof, then exit.")
    parser.add_argument("--headless-agent", action="store_true",
                        help="Use file-based OMP agent callback instead of Rich TUI.")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Experiment directory for journal and summary files.")
    parser.add_argument("--baseline-eval-json", type=str, default=None,
                        help="Optional baseline .eval.json for milestone reports.")
    parser.add_argument("--demo-trace", type=str, default=None,
                        help="Expert movement trace to regularize PPO updates.")
    parser.add_argument("--demo-scenario", type=str, default="dungeon1-wallmaster-north-exit",
                        help="Scenario used to collect expert demo observations.")
    parser.add_argument("--demo-prefix-east", type=int, default=0,
                        help="Number of MOVE E actions to prepend before the demo trace.")
    parser.add_argument("--demo-bc-coeff", type=float, default=0.0,
                        help="Behavior-cloning loss coefficient for demo-regularized PPO.")

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
