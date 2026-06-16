import sys

from train import parse_args


def test_parse_args_accepts_headless_agent_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "overworld-room-walk",
        "all-items",
        "impala-multihead",
        "--headless-agent",
        "--experiment-dir",
        "training/experiments/test",
        "--baseline-eval-json",
        "baseline.eval.json",
    ])

    args = parse_args()

    assert args.headless_agent is True
    assert args.experiment_dir == "training/experiments/test"
    assert args.baseline_eval_json == "baseline.eval.json"
