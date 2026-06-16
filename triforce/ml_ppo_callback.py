"""Callback protocol for PPO training progress and metrics reporting."""


class TrainingCallback:
    """Abstract callback for training progress reporting.

    Implement this to receive training updates from PPO. The triforce library
    calls these methods during training — the library itself has no display or
    logging dependencies.
    """

    def on_circuit_start(self, scenarios):
        """Called at circuit start with list of (scenario_name, iterations) for all legs."""

    def on_progress(self, steps, total_steps):
        """Called as env steps are collected. Used for progress bars."""

    def on_scenario_start(self, scenario_name, iterations, exit_criteria=None,
                          exit_threshold=None, exit_criteria_scenario=None):
        """Called when a scenario begins training."""

    def on_scenario_end(self, scenario_name, checkpoint_path=None):
        """Called when a scenario finishes; checkpoint_path is the leg checkpoint if available."""

    def on_scenario_resumed(self, scenario_name, history_entry):
        """Called for skipped scenarios during --resume to populate display from history."""

    def get_completion_info(self, scenario_name):  # pylint: disable=unused-argument
        """Return completion metadata for a finished scenario.

        Returns dict with keys: metric, threshold, value, met, steps, total.
        Returns None if scenario not found or callback doesn't track this.
        """
        return None

    def on_metrics(self, metrics, iteration, total_iterations):
        """Called at each metric reporting interval with collected metrics."""

    def on_optimize(self, stats, iteration, total_iterations):
        """Called after each optimization step with loss/training stats."""

    def on_training_complete(self):
        """Called when all training is done."""

    def check_pause(self):
        """Called after each training iteration. May block if training is paused.
        Returns True to continue training, False to stop early."""
        return True


class NullCallback(TrainingCallback):
    """No-op callback used when no display is needed."""
