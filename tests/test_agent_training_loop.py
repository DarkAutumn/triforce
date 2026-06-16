import json
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

from triforce.agent_callback import AgentTrainingCallback
from triforce.experiment_report import DEFAULT_TUNING, detect_anomalies, flatten_metrics
from triforce.ml_ppo import PPO
from triforce.models import Network


def test_flatten_metrics_promotes_weighted_first_scenario():
    metrics = {
        "scenario-a": {"success-rate": 0.5},
        "scenario-b": {"success-rate": 0.25},
    }

    flat = flatten_metrics(metrics)

    assert flat["scenario-a/success-rate"] == 0.5
    assert flat["scenario-b/success-rate"] == 0.25
    assert flat["success-rate"] == 0.5


def test_detect_anomalies_uses_live_tuning_thresholds():
    strict_tuning = dict(DEFAULT_TUNING)
    strict_tuning["health_ranges"] = {"losses/entropy": [0.5, 2.0]}
    loose_tuning = dict(DEFAULT_TUNING)
    loose_tuning["health_ranges"] = {"losses/entropy": [0.05, 2.0]}
    stats = {"losses/entropy": 0.1}

    strict = detect_anomalies([], {}, stats, strict_tuning)
    loose = detect_anomalies([], {}, stats, loose_tuning)

    assert len(strict) == 1
    assert strict[0]["metric"] == "losses/entropy"
    assert loose == []


def test_reward_hacking_anomaly_requires_reward_gain_without_success_gain():
    tuning = dict(DEFAULT_TUNING)
    history = [
        {"reward-average": 1.00, "success-rate": 0.20},
        {"reward-average": 1.06, "success-rate": 0.21},
        {"reward-average": 1.12, "success-rate": 0.22},
    ]

    anomalies = detect_anomalies(history, history[-1], {}, tuning)

    assert any(anomaly["kind"] == "reward_hacking" for anomaly in anomalies)

    improving_success = [
        {"reward-average": 1.00, "success-rate": 0.20},
        {"reward-average": 1.06, "success-rate": 0.25},
        {"reward-average": 1.12, "success-rate": 0.35},
    ]
    clean = detect_anomalies(improving_success, improving_success[-1], {}, tuning)

    assert not any(anomaly["kind"] == "reward_hacking" for anomaly in clean)


def test_agent_callback_writes_status_events_and_milestone(tmp_path):
    run_dir = tmp_path / "run"
    experiment_dir = tmp_path / "experiment"
    log_dir = run_dir / "logs"
    callback = AgentTrainingCallback(str(run_dir), str(experiment_dir), str(log_dir),
                                     scenario="dummy-scenario", action_space="all-items",
                                     model_kind="impala-multihead")

    callback.on_circuit_start([("dummy", 1_000_000)])
    callback.on_scenario_start("dummy", 1_000_000, exit_criteria="success-rate", exit_threshold=0.8)
    callback.on_progress(1_000_000, 1_000_000)
    callback.on_metrics({"success-rate": 0.5, "reward-average": 1.0}, 1_000_000, 1_000_000)
    checkpoint_path = str(run_dir / "checkpoints" / "dummy.pt")
    callback.on_scenario_end("dummy", checkpoint_path=checkpoint_path)
    callback.on_training_complete()

    assert (run_dir / "status.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "tuning.json").exists()
    assert list(run_dir.glob("milestone_*.md"))

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    milestones = [event for event in events if event["type"] == "milestone"]
    assert milestones[-2]["reason"] == "leg_end"
    assert milestones[-2]["checkpoint_path"] == checkpoint_path
    assert milestones[-2]["step"] == 1_000_000
    assert milestones[-1]["reason"] == "complete"
    assert milestones[-1]["step"] == 1_000_000


def test_control_stop_returns_false(tmp_path):
    run_dir = tmp_path / "run"
    experiment_dir = tmp_path / "experiment"
    callback = AgentTrainingCallback(str(run_dir), str(experiment_dir), str(run_dir / "logs"),
                                     scenario="dummy-scenario", action_space="all-items",
                                     model_kind="impala-multihead")
    control = {
        "schema_version": 1,
        "command": "stop",
        "reason": "test",
        "updated_at": 0.0,
    }
    (run_dir / "control.json").write_text(json.dumps(control), encoding="utf-8")

    assert callback.check_pause() is False
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "stopping"


class TinyNetwork(Network):
    def __init__(self):
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
        action_space = gym.spaces.Discrete(2)
        super().__init__(nn.Linear(3, 4), obs_space, action_space, base_output_size=4)


def test_optimizer_state_round_trips_through_checkpoint(tmp_path):
    network = TinyNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    obs = torch.zeros(1, 3)
    logits, value = network(obs)
    loss = logits.sum() + value.sum()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    network.save(str(checkpoint_path), optimizer=optimizer,
                 training_history=[{"scenario": "dummy", "steps": 1}])

    loaded_state = Network.load_optimizer_state(str(checkpoint_path))
    assert loaded_state is not None
    assert loaded_state["state"]

    resumed_network = TinyNetwork()
    ppo = PPO(optimizer_state=loaded_state)
    ppo._setup_optimizer(resumed_network)  # pylint: disable=protected-access

    assert ppo.optimizer is not None
    assert ppo.optimizer.state_dict()["state"]
