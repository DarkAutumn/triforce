#!/usr/bin/env python3
"""Headless diagnostic tool for analyzing trained model behavior.

Runs N episodes and produces a detailed text report showing:
- Per-episode room trace (sequence of rooms visited)
- Per-episode reward breakdown (which rewards/penalties fire, how often)
- Where and why episodes end (room, reason, progress)
- Stuck detection (rooms visited repeatedly without progress)
- Aggregate statistics across all episodes

Usage:
    python diagnose.py --model sword-and-beams-multihead --scenario skip-sword-to-triforce \\
                       --model-path training/run-006-multihead/sword-and-beams-multihead-*.pt \\
                       --episodes 20

    python diagnose.py --model sword-and-beams-multihead --scenario skip-sword-to-triforce \\
                       --model-path training/run-006-multihead/sword-and-beams-multihead-*.pt \\
                       --episodes 20 --output diag_report.txt
"""

import argparse
import glob as globmod
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import gymnasium as gym

from triforce import ModelDefinition, Network, TrainingScenarioDefinition, make_zelda_env

# ---------------------------------------------------------------------------
# Game progress milestone labels (from metrics.py game-progress map)
# ---------------------------------------------------------------------------
MILESTONE_NAMES = {
    0: "Sword Cave (0,0x77)",
    1: "South of Cave / East (0,0x67/0x78)",
    2: "Overworld (0,0x68)",
    3: "Overworld (0,0x58)",
    4: "Overworld (0,0x48)",
    5: "Overworld (0,0x38)",
    6: "Dungeon 1 Entrance Screen (0,0x37)",
    7: "Dungeon 1 Entry Room (1,0x73)",
    8: "Dungeon 1 West/East (1,0x72/0x74)",
    9: "Dungeon 1 (1,0x63)",
    10: "Dungeon 1 (1,0x53)",
    11: "Dungeon 1 (1,0x52)",
    12: "Dungeon 1 (1,0x42)",
    13: "Dungeon 1 (1,0x43)",
    14: "Dungeon 1 (1,0x44)",
    15: "Dungeon 1 (1,0x45)",
    16: "Dungeon 1 Boss Room (1,0x35)",
    17: "Triforce Room (1,0x36)",
}


@dataclass
class EpisodeRecord:
    """All diagnostic data collected for one episode."""
    episode_idx: int
    room_trace: List[Tuple[int, int, bool]] = field(default_factory=list)  # (level, location, in_cave)
    room_step_counts: Dict[Tuple[int, int, bool], int] = field(default_factory=lambda: Counter())
    reward_totals: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    reward_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_steps: int = 0
    max_progress: int = 0
    ending_reason: Optional[str] = None
    final_room: Optional[Tuple[int, int, bool]] = None
    final_health: int = 0
    final_max_health: int = 0
    total_reward: float = 0.0
    steps_without_progress: int = 0  # consecutive steps at same progress level
    stuck_rooms: List[Tuple[Tuple[int, int, bool], int]] = field(default_factory=list)  # rooms visited > threshold


# Game progress room map (level, location) -> progress value
PROGRESS_MAP = {
    (0, 0x77): 0, (0, 0x67): 1, (0, 0x78): 1, (0, 0x68): 2,
    (0, 0x58): 3, (0, 0x48): 4, (0, 0x38): 5, (0, 0x37): 6,
    (1, 0x73): 7, (1, 0x72): 8, (1, 0x74): 8, (1, 0x63): 9,
    (1, 0x53): 10, (1, 0x52): 11, (1, 0x42): 12, (1, 0x43): 13,
    (1, 0x44): 14, (1, 0x45): 15, (1, 0x35): 16, (1, 0x36): 17,
}

STUCK_THRESHOLD = 200  # steps in one room before flagging as stuck


class DiagnosticWrapper(gym.Wrapper):
    """Wraps ScenarioWrapper to capture raw StateChange/StepRewards while passing through
    the standard Gym interface (float reward, info dict) for the network."""

    def __init__(self, env):
        super().__init__(env)
        self.last_state_change = None
        self.last_rewards = None
        self.last_reset_state = None

    def reset(self, **kwargs):
        obs, state = super().reset(**kwargs)
        self.last_reset_state = state  # ZeldaGame
        self.last_state_change = None
        self.last_rewards = None
        state.deactivate()
        return obs, state.info

    def step(self, action):
        obs, rewards, terminated, truncated, state_change = super().step(action)
        self.last_state_change = state_change
        self.last_rewards = rewards
        state = state_change.state
        state.deactivate()
        return obs, rewards.value, terminated, truncated, state.info


def run_diagnostic(model_name, scenario_name, model_path, episodes, output_path=None):
    """Run diagnostic episodes and produce a report."""
    scenario_def = TrainingScenarioDefinition.get(scenario_name)
    model_def = ModelDefinition.get(model_name)
    multihead = getattr(model_def.neural_net, 'is_multihead', False)

    # Create env WITHOUT GymTranslationWrapper, use our DiagnosticWrapper instead
    env = make_zelda_env(scenario_def, model_def.action_space,
                         render_mode=None, multihead=multihead, translation=False)
    env = DiagnosticWrapper(env)

    # Load model
    obs_space, act_space = Network.load_spaces(model_path)
    network = model_def.neural_net(obs_space, act_space)
    network.load(model_path)
    network.eval()

    records = []
    for ep_idx in range(episodes):
        record = run_one_episode(env, network, ep_idx)
        records.append(record)
        progress_str = f"progress={record.max_progress}"
        ending_str = record.ending_reason or "unknown"
        print(f"  Episode {ep_idx+1}/{episodes}: {progress_str}, ended={ending_str}, "
              f"steps={record.total_steps}, reward={record.total_reward:.2f}")

    env.close()

    report = generate_report(records, model_name, scenario_name, model_path, episodes)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport written to: {output_path}")
    else:
        print(report)

    return records


def run_one_episode(env, network, ep_idx):
    """Run one episode, collecting detailed diagnostics."""
    record = EpisodeRecord(episode_idx=ep_idx)

    obs, info = env.reset()

    # Initial state from DiagnosticWrapper
    state = env.last_reset_state
    loc = (state.level, state.location, state.in_cave)
    record.room_trace.append(loc)
    record.room_step_counts[loc] += 1

    current_progress = PROGRESS_MAP.get((state.level, state.location), 0)
    record.max_progress = current_progress
    steps_at_current = 0

    terminated = truncated = False
    while not terminated and not truncated:
        action_mask = info.get('action_mask', None)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(0) if action_mask.dim() == 1 else action_mask

        action = network.get_action(obs, action_mask)
        action = action.squeeze(0)  # [1, 2] -> [2] for multihead, [1] -> scalar for flat
        obs, reward_value, terminated, truncated, info = env.step(action)

        record.total_steps += 1

        # Access raw data from DiagnosticWrapper
        state_change = env.last_state_change
        rewards = env.last_rewards
        state = state_change.state

        # Track room
        loc = (state.level, state.location, state.in_cave)
        if not record.room_trace or record.room_trace[-1] != loc:
            record.room_trace.append(loc)
        record.room_step_counts[loc] += 1

        # Track progress
        room_progress = PROGRESS_MAP.get((state.level, state.location), 0)
        if room_progress > record.max_progress:
            record.max_progress = room_progress
            steps_at_current = 0
        else:
            steps_at_current += 1

        # Track rewards from raw StepRewards
        record.total_reward += rewards.value
        for outcome in rewards:
            record.reward_totals[outcome.name] += outcome.value
            record.reward_counts[outcome.name] += outcome.count

        # Track ending
        if rewards.ending is not None:
            record.ending_reason = str(rewards.ending)

    # Final state
    record.final_room = loc
    record.final_health = state.link.health
    record.final_max_health = state.link.max_health
    record.steps_without_progress = steps_at_current

    # Flag stuck rooms
    for room, count in record.room_step_counts.items():
        if count >= STUCK_THRESHOLD:
            record.stuck_rooms.append((room, count))

    return record


def generate_report(records, model_name, scenario_name, model_path, episodes):
    """Generate a text report from episode records."""
    lines = []
    w = lines.append

    w("=" * 72)
    w(f"DIAGNOSTIC REPORT: {model_name}")
    w(f"Scenario: {scenario_name}")
    w(f"Model: {model_path}")
    w(f"Episodes: {episodes}")
    w("=" * 72)
    w("")

    # ---- Overall Summary ----
    progress_values = [r.max_progress for r in records]
    step_values = [r.total_steps for r in records]
    reward_values = [r.total_reward for r in records]
    endings = Counter(r.ending_reason for r in records)

    progress_values_sorted = sorted(progress_values)
    n = len(progress_values_sorted)

    w("## SUMMARY")
    w(f"  Mean progress:   {sum(progress_values)/n:.1f}")
    w(f"  Median progress: {progress_values_sorted[n//2]}")
    w(f"  P25/P75/P90:     {progress_values_sorted[max(0,n//4-1)]} / "
      f"{progress_values_sorted[min(n-1, 3*n//4)]} / "
      f"{progress_values_sorted[min(n-1, int(n*0.9))]}")
    w(f"  Min/Max:         {min(progress_values)} / {max(progress_values)}")
    w(f"  Mean steps:      {sum(step_values)/n:.0f}")
    w(f"  Mean reward:     {sum(reward_values)/n:.2f}")
    w("")

    # ---- Progress Histogram ----
    w("## PROGRESS HISTOGRAM")
    progress_counts = Counter(progress_values)
    max_progress = max(PROGRESS_MAP.values())
    max_bar = max(progress_counts.values()) if progress_counts else 1
    for milestone in range(max_progress + 1):
        count = progress_counts.get(milestone, 0)
        bar = '█' * max(1, round(count / max_bar * 30)) if count > 0 else ''
        label = MILESTONE_NAMES.get(milestone, f"milestone {milestone}")
        w(f"  {milestone:2d} | {count:3d} | {bar:30s} | {label}")
    w("")

    # ---- Ending Breakdown ----
    w("## EPISODE ENDINGS")
    for reason, count in endings.most_common():
        pct = count / n * 100
        w(f"  {reason:30s}: {count:3d} ({pct:5.1f}%)")
    w("")

    # ---- Where Episodes End (room distribution at death/timeout) ----
    w("## WHERE EPISODES END (final room)")
    final_rooms = Counter()
    for r in records:
        if r.final_room:
            final_rooms[r.final_room] += 1
    for room, count in final_rooms.most_common(10):
        level, loc, cave = room
        progress = PROGRESS_MAP.get((level, loc), "?")
        w(f"  Level={level} Room=0x{loc:02x} Cave={cave}: {count:3d} episodes "
          f"(progress={progress})")
    w("")

    # ---- Stuck Detection ----
    stuck_episodes = [r for r in records if r.stuck_rooms]
    w(f"## STUCK DETECTION (>{STUCK_THRESHOLD} steps in one room)")
    if stuck_episodes:
        w(f"  {len(stuck_episodes)}/{n} episodes had stuck behavior:")
        stuck_room_counts = Counter()
        for r in stuck_episodes:
            for room, count in r.stuck_rooms:
                stuck_room_counts[room] += 1
        for room, ep_count in stuck_room_counts.most_common(10):
            level, loc, cave = room
            progress = PROGRESS_MAP.get((level, loc), "?")
            w(f"  Level={level} Room=0x{loc:02x} Cave={cave}: stuck in {ep_count} episodes "
              f"(progress={progress})")
    else:
        w("  No episodes showed stuck behavior.")
    w("")

    # ---- Reward Breakdown (aggregate across all episodes) ----
    w("## REWARD BREAKDOWN (aggregated across all episodes)")
    total_rewards = defaultdict(float)
    total_counts = defaultdict(int)
    for r in records:
        for name, val in r.reward_totals.items():
            total_rewards[name] += val
        for name, cnt in r.reward_counts.items():
            total_counts[name] += cnt

    # Sort by absolute value (most impactful first)
    sorted_rewards = sorted(total_rewards.items(), key=lambda x: abs(x[1]), reverse=True)
    w(f"  {'Name':45s} {'Total':>10s} {'Avg/Ep':>10s} {'Count':>8s} {'Avg Cnt':>8s}")
    w(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name, total in sorted_rewards:
        avg = total / n
        cnt = total_counts[name]
        avg_cnt = cnt / n
        w(f"  {name:45s} {total:10.3f} {avg:10.3f} {cnt:8d} {avg_cnt:8.1f}")
    w("")

    # ---- Per-Episode Details (condensed) ----
    w("## PER-EPISODE DETAILS")
    for r in records:
        w(f"  --- Episode {r.episode_idx + 1} ---")
        w(f"  Progress: {r.max_progress}  Steps: {r.total_steps}  "
          f"Reward: {r.total_reward:.2f}  Ending: {r.ending_reason}")
        w(f"  Health: {r.final_health}/{r.final_max_health}  "
          f"Steps w/o progress: {r.steps_without_progress}")

        # Room trace (compact)
        trace_parts = []
        for room in r.room_trace:
            level, loc, cave = room
            steps = r.room_step_counts[room]
            prefix = "C" if cave else f"L{level}"
            trace_parts.append(f"{prefix}:0x{loc:02x}({steps})")

        # Break into lines of ~100 chars
        trace_str = " → ".join(trace_parts)
        if len(trace_str) > 120:
            w(f"  Rooms: {len(r.room_trace)} unique transitions")
            # Show first 5 and last 3
            if len(trace_parts) > 8:
                w(f"    Start: {' → '.join(trace_parts[:5])}")
                w(f"    End:   {' → '.join(trace_parts[-3:])}")
            else:
                w(f"    {trace_str}")
        else:
            w(f"  Rooms: {trace_str}")

        # Top 3 rewards and penalties for this episode
        ep_sorted = sorted(r.reward_totals.items(), key=lambda x: x[1], reverse=True)
        top_rewards = [(n, v) for n, v in ep_sorted if v > 0][:3]
        top_penalties = [(n, v) for n, v in ep_sorted if v < 0][:3]
        if top_rewards:
            parts = [f"{n}={v:.2f}" for n, v in top_rewards]
            w(f"  Top rewards:   {', '.join(parts)}")
        if top_penalties:
            parts = [f"{n}={v:.2f}" for n, v in top_penalties]
            w(f"  Top penalties: {', '.join(parts)}")

        if r.stuck_rooms:
            for room, count in r.stuck_rooms:
                level, loc, cave = room
                w(f"  ⚠ STUCK: Level={level} Room=0x{loc:02x} for {count} steps")
        w("")

    w("=" * 72)
    w("END OF DIAGNOSTIC REPORT")
    w("=" * 72)

    return "\n".join(lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Diagnostic tool for trained triforce models")
    parser.add_argument("--model", required=True, help="Model name from triforce.json")
    parser.add_argument("--scenario", required=True, help="Scenario name from triforce.json")
    parser.add_argument("--model-path", required=True,
                        help="Path to .pt model file (supports glob patterns)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: stdout)")
    return parser.parse_args()


def resolve_model_path(pattern):
    """Resolve a model path, supporting glob patterns. Returns the latest by step count."""
    if os.path.isfile(pattern):
        return pattern

    matches = sorted(globmod.glob(pattern))
    if not matches:
        print(f"Error: No model files matching '{pattern}'")
        sys.exit(1)

    # Prefer highest step count
    def extract_steps(path):
        base = os.path.basename(path)
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
            try:
                return int(parts[1].replace('.pt', ''))
            except ValueError:
                pass
        return 0

    matches.sort(key=extract_steps, reverse=True)
    print(f"Using model: {matches[0]}")
    return matches[0]


def main():
    """Entry point."""
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    run_diagnostic(args.model, args.scenario, model_path, args.episodes, args.output)


if __name__ == "__main__":
    main()
