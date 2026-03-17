#!/usr/bin/env -S uv run python3
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

import gymnasium as gym

from triforce import ActionSpaceDefinition, ModelKindDefinition, Network, TrainingScenarioDefinition, make_zelda_env
from triforce.action_space import ActionKind
from triforce.critics import PBRS_SCALE
from triforce.observation_wrapper import infer_obs_kind
from triforce.room import Room
from triforce.zelda_enums import MapLocation, Direction

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

    # Load model metadata from the .pt file
    metadata = Network.load_metadata(model_path)
    model_kind = ModelKindDefinition.get(metadata["model_kind"] or model_name)
    action_space_def = ActionSpaceDefinition.get(metadata["action_space_name"] or "basic")
    multihead = getattr(model_kind.network_class, 'is_multihead', False)

    # Infer obs_kind and frame_stack from the saved observation space
    obs_kind, frame_stack = infer_obs_kind(metadata["obs_space"])

    # Create env WITHOUT GymTranslationWrapper, use our DiagnosticWrapper instead
    env = make_zelda_env(scenario_def, action_space_def.actions,
                         render_mode=None, multihead=multihead, translation=False,
                         obs_kind=obs_kind, frame_stack=frame_stack)
    env = DiagnosticWrapper(env)

    # Load model
    obs_space, act_space = Network.load_spaces(model_path)
    network = model_kind.network_class(obs_space, act_space)
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


# ---------------------------------------------------------------------------
# --infinite-pbrs mode: step-by-step PBRS diagnostic
# ---------------------------------------------------------------------------

@dataclass
class PbrsStepRecord:
    """Per-step PBRS calculation details."""
    step: int
    room: int
    tile_prev: tuple
    tile_curr: tuple
    objective: str
    wf_used: str            # "prev" or "curr" (which wavefront was used)
    old_dist: Optional[int]
    new_dist: Optional[int]
    shaped: float           # the PBRS value this step
    cumulative: float       # running total PBRS in this room
    health_lost: bool
    room_changed: bool
    wall_hit: bool
    other_rewards: str      # non-PBRS rewards this step
    num_items: int = 0      # items on screen this step
    num_enemies: int = 0    # enemies on screen this step
    item_tiles: tuple = ()  # item positions as tuple of (x,y)
    enemy_ids: tuple = ()   # enemy type IDs this step


def run_pbrs_diagnostic(model_name, scenario_name, model_path, episodes, tail, output_path):
    """Run episodes looking for stuck/timeout endings, then print step-by-step PBRS detail."""
    scenario_def = TrainingScenarioDefinition.get(scenario_name)
    metadata = Network.load_metadata(model_path)
    model_kind = ModelKindDefinition.get(metadata["model_kind"] or model_name)
    action_space_def = ActionSpaceDefinition.get(metadata["action_space_name"] or "basic")
    multihead = getattr(model_kind.network_class, 'is_multihead', False)

    obs_kind, frame_stack = infer_obs_kind(metadata["obs_space"])
    env = make_zelda_env(scenario_def, action_space_def.actions,
                         render_mode=None, multihead=multihead, translation=False,
                         obs_kind=obs_kind, frame_stack=frame_stack)
    env = DiagnosticWrapper(env)

    obs_space, act_space = Network.load_spaces(model_path)
    network = model_kind.network_class(obs_space, act_space)
    network.load(model_path)
    network.eval()

    lines = []
    w = lines.append
    w("=" * 90)
    w(f"PBRS DIAGNOSTIC: {model_name}")
    w(f"Model: {model_path}")
    w(f"Scenario: {scenario_name}")
    w("=" * 90)

    found_stuck = 0
    for ep_idx in range(episodes):
        result = _run_pbrs_episode(env, network, ep_idx)
        ep_steps, ending, all_steps, room_segments = result

        is_stuck = ending and "stuck" in ending or "no-next-room" in ending
        print(f"  Episode {ep_idx+1}/{episodes}: steps={ep_steps}, ending={ending}, "
              f"stuck={'YES' if is_stuck else 'no'}")

        if not is_stuck:
            continue

        found_stuck += 1
        w("")
        w(f"## EPISODE {ep_idx+1} — ended: {ending} ({ep_steps} steps)")

        # Find the stuck room (last room in the segment list with most steps)
        if not room_segments:
            continue

        last_room_id, last_room_steps = room_segments[-1]
        w(f"  Stuck room: 0x{last_room_id:02x}  ({len(last_room_steps)} steps in final visit)")

        # Room-level summary
        room_pbrs_total = sum(s.shaped for s in last_room_steps)
        room_pbrs_pos = sum(s.shaped for s in last_room_steps if s.shaped > 0)
        room_pbrs_neg = sum(s.shaped for s in last_room_steps if s.shaped < 0)
        skipped = sum(1 for s in last_room_steps if s.old_dist is None or s.new_dist is None)
        wall_hits = sum(1 for s in last_room_steps if s.wall_hit)
        health_lost_count = sum(1 for s in last_room_steps if s.health_lost)

        w(f"  PBRS net: {room_pbrs_total:+.3f}  (pos: {room_pbrs_pos:+.3f}, "
          f"neg: {room_pbrs_neg:+.3f})")
        w(f"  Steps skipped (off-wavefront): {skipped}/{len(last_room_steps)}  "
          f"Wall hits: {wall_hits}  Health lost: {health_lost_count}")

        # Show unique wavefront distances seen from the stuck room
        dist_set = set()
        for s in last_room_steps:
            if s.old_dist is not None:
                dist_set.add((s.tile_prev, s.old_dist))
            if s.new_dist is not None:
                dist_set.add((s.tile_curr, s.new_dist))
        if dist_set:
            # Group by tile, check if any tile has multiple distances (non-stationarity)
            tile_dists = defaultdict(set)
            for tile, dist in dist_set:
                tile_dists[tile].add(dist)
            non_stationary = {t: ds for t, ds in tile_dists.items() if len(ds) > 1}
            if non_stationary:
                w(f"\n  ⚠ NON-STATIONARY WAVEFRONT: {len(non_stationary)} tiles have "
                  f"varying distances!")
                for tile, ds in sorted(non_stationary.items(),
                                       key=lambda x: max(x[1]) - min(x[1]), reverse=True)[:10]:
                    w(f"    Tile {tile}: distances = {sorted(ds)}")
            else:
                w(f"  ✓ Wavefront appears stationary ({len(tile_dists)} unique tiles)")

        # Objective breakdown
        obj_counts = Counter(s.objective for s in last_room_steps)
        w(f"\n  Objectives: {dict(obj_counts)}")

        # Item/enemy summary for stuck room
        item_counts = [s.num_items for s in last_room_steps]
        enemy_counts = [s.num_enemies for s in last_room_steps]
        all_item_tiles = set()
        for s in last_room_steps:
            all_item_tiles.update(s.item_tiles)
        max_items = max(item_counts) if item_counts else 0
        max_enemies = max(enemy_counts) if enemy_counts else 0
        steps_with_items = sum(1 for c in item_counts if c > 0)
        # Enemy type breakdown
        all_enemy_ids = Counter()
        for s in last_room_steps:
            all_enemy_ids.update(s.enemy_ids)
        unique_enemy_types = sorted(set(eid for s in last_room_steps for eid in s.enemy_ids))
        w(f"  Enemies: max={max_enemies}  Types: {unique_enemy_types}")
        w(f"  Items: max={max_items}, "
          f"present in {steps_with_items}/{len(last_room_steps)} steps")
        if all_item_tiles:
            w(f"  Item locations seen: {sorted(all_item_tiles)}")

        # Non-PBRS reward totals in stuck room
        stuck_reward_totals = Counter()
        for s in last_room_steps:
            if s.other_rewards:
                for part in s.other_rewards.split(", "):
                    name, val = part.rsplit("=", 1)
                    stuck_reward_totals[name] += float(val)
        if stuck_reward_totals:
            w(f"  Non-PBRS rewards in stuck room:")
            for name, total in stuck_reward_totals.most_common():
                w(f"    {name}: {total:+.3f}")

        # Print the trailing steps
        show_steps = last_room_steps[-tail:]
        start_idx = len(last_room_steps) - len(show_steps)
        w(f"\n  Last {len(show_steps)} steps (of {len(last_room_steps)} in room):")
        w(f"  {'Step':>5s}  {'Tile':>10s} {'→':>1s} {'Tile':>10s}  "
          f"{'ObjKind':>7s}  {'OldD':>5s} {'NewD':>5s}  "
          f"{'PBRS':>7s}  {'Cumul':>7s}  {'Flags':>12s}  Other Rewards")
        w(f"  {'─'*5}  {'─'*10} {'─':>1s} {'─'*10}  "
          f"{'─'*7}  {'─'*5} {'─'*5}  "
          f"{'─'*7}  {'─'*7}  {'─'*12}  {'─'*30}")

        for i, s in enumerate(show_steps):
            flags = []
            if s.health_lost:
                flags.append("DMG")
            if s.wall_hit:
                flags.append("WALL")
            if s.room_changed:
                flags.append("ROOM")
            if s.old_dist is None or s.new_dist is None:
                flags.append("OFF-WF")
            # Flag item changes
            prev_step = show_steps[i-1] if i > 0 else (last_room_steps[start_idx-1]
                                                         if start_idx > 0 else None)
            if prev_step and s.num_items != prev_step.num_items:
                flags.append(f"ITM:{prev_step.num_items}→{s.num_items}")
            flag_str = ",".join(flags) if flags else ""

            old_d = str(s.old_dist) if s.old_dist is not None else "None"
            new_d = str(s.new_dist) if s.new_dist is not None else "None"

            w(f"  {start_idx+i:5d}  {str(s.tile_prev):>10s} → {str(s.tile_curr):>10s}  "
              f"{s.objective:>7s}  {old_d:>5s} {new_d:>5s}  "
              f"{s.shaped:+7.3f}  {s.cumulative:+7.3f}  {flag_str:>12s}  {s.other_rewards}")

        w("")

        # Only show first 3 stuck episodes to keep output manageable
        if found_stuck >= 3:
            w(f"(Showing first 3 stuck episodes, {episodes - ep_idx - 1} episodes remaining)")
            break

    env.close()

    if found_stuck == 0:
        w("\n  No stuck/timeout episodes found. Model did not get stuck in any episode.")

    w("")
    w("=" * 90)
    w("END OF PBRS DIAGNOSTIC")
    w("=" * 90)

    report = "\n".join(lines)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport written to: {output_path}")
    else:
        print(report)


def _run_pbrs_episode(env, network, ep_idx):
    """Run one episode, capturing per-step PBRS detail. Returns (steps, ending, all_steps, room_segments)."""
    obs, info = env.reset()
    state = env.last_reset_state

    all_steps = []
    # room_segments: list of (room_id, [PbrsStepRecord...]) for each contiguous visit
    room_segments = []
    current_room_id = state.location
    current_room_steps = []
    room_cumulative = 0.0
    step_num = 0
    ending = None

    terminated = truncated = False
    while not terminated and not truncated:
        action_mask = info.get('action_mask', None)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(0) if action_mask.dim() == 1 else action_mask

        action = network.get_action(obs, action_mask)
        action = action.squeeze(0)
        obs, reward_value, terminated, truncated, info = env.step(action)

        sc = env.last_state_change
        rewards = env.last_rewards
        prev = sc.previous
        curr = sc.state

        # Detect room change
        room_changed = prev.full_location != curr.full_location

        # Recompute PBRS calculation to capture internals
        health_lost = sc.health_lost
        wall_hit = prev.link.position == curr.link.position
        tile_prev = prev.link.tile
        tile_curr = curr.link.tile

        # Objective
        obj_kind = curr.objectives.kind.name if curr.objectives else "NONE"

        # PBRS calculation (mirrors critics.py logic)
        old_dist = None
        new_dist = None
        shaped = 0.0
        if not health_lost and not room_changed and not wall_hit:
            wf = prev.pbrs_wavefront if hasattr(prev, 'pbrs_wavefront') else prev.wavefront
            if wf is not None:
                old_dist = wf.get(tile_prev)
                new_dist = wf.get(tile_curr)
                if old_dist is not None and new_dist is not None:
                    shaped = (old_dist - new_dist) / PBRS_SCALE

        # Track room segments
        if room_changed:
            if current_room_steps:
                room_segments.append((current_room_id, current_room_steps))
            current_room_id = curr.location
            current_room_steps = []
            room_cumulative = 0.0

        room_cumulative += shaped

        # Collect non-PBRS rewards
        other_parts = []
        for outcome in rewards:
            if 'pbrs' not in outcome.name:
                other_parts.append(f"{outcome.name}={outcome.value:+.2f}")
        other_str = ", ".join(other_parts) if other_parts else ""

        # Track items and enemies
        items = curr.items if hasattr(curr, 'items') else []
        enemies = curr.enemies if hasattr(curr, 'enemies') else []
        item_tiles = tuple((i.tile[0], i.tile[1]) for i in items)
        enemy_ids = tuple(e.id.name if hasattr(e.id, 'name') else str(e.id) for e in enemies)

        record = PbrsStepRecord(
            step=step_num,
            room=curr.location,
            tile_prev=(tile_prev[0], tile_prev[1]) if tile_prev else None,
            tile_curr=(tile_curr[0], tile_curr[1]) if tile_curr else None,
            objective=obj_kind,
            wf_used="prev",
            old_dist=old_dist,
            new_dist=new_dist,
            shaped=shaped,
            cumulative=room_cumulative,
            health_lost=health_lost > 0 if health_lost else False,
            room_changed=room_changed,
            wall_hit=wall_hit,
            other_rewards=other_str,
            num_items=len(items),
            num_enemies=len(enemies),
            item_tiles=item_tiles,
            enemy_ids=enemy_ids,
        )
        all_steps.append(record)
        current_room_steps.append(record)
        step_num += 1

        if rewards.ending is not None:
            ending = str(rewards.ending)

    # Final segment
    if current_room_steps:
        room_segments.append((current_room_id, current_room_steps))

    return step_num, ending, all_steps, room_segments


# ---------------------------------------------------------------------------
# --invariants mode: movement & reward invariant checker
# ---------------------------------------------------------------------------

@dataclass
class InvariantViolation:
    """A single invariant violation detected during gameplay."""
    episode: int
    step: int
    kind: str               # short tag: "zero-movement", "no-wavefront", etc.
    description: str        # human-readable detail
    room: int
    level: int
    tile: tuple
    position: tuple
    action_kind: str
    action_direction: str


def run_invariant_checker(model_name, scenario_name, model_path, episodes, output_path=None):
    """Run episodes checking movement and reward invariants at every step."""
    scenario_def = TrainingScenarioDefinition.get(scenario_name)
    metadata = Network.load_metadata(model_path)
    model_kind = ModelKindDefinition.get(metadata["model_kind"] or model_name)
    action_space_def = ActionSpaceDefinition.get(metadata["action_space_name"] or "basic")
    multihead = getattr(model_kind.network_class, 'is_multihead', False)

    obs_kind, frame_stack = infer_obs_kind(metadata["obs_space"])
    env = make_zelda_env(scenario_def, action_space_def.actions,
                         render_mode=None, multihead=multihead, translation=False,
                         obs_kind=obs_kind, frame_stack=frame_stack)
    env = DiagnosticWrapper(env)

    obs_space, act_space = Network.load_spaces(model_path)
    network = model_kind.network_class(obs_space, act_space)
    network.load(model_path)
    network.eval()

    all_violations = []
    for ep_idx in range(episodes):
        violations = _check_one_episode(env, network, ep_idx)
        all_violations.extend(violations)
        tag = f"{len(violations)} violations" if violations else "clean"
        print(f"  Episode {ep_idx+1}/{episodes}: {tag}")

    env.close()

    report = _generate_invariant_report(all_violations, model_name, scenario_name,
                                        model_path, episodes)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport written to: {output_path}")
    else:
        print(report)

    return all_violations


def _check_one_episode(env, network, ep_idx):
    """Run one episode, checking invariants at every step. Returns list of violations."""
    violations = []
    obs, info = env.reset()
    state = env.last_reset_state

    terminated = truncated = False
    step_num = 0

    while not terminated and not truncated:
        action_mask = info.get('action_mask', None)
        if action_mask is not None:
            action_mask_batched = action_mask.unsqueeze(0) if action_mask.dim() == 1 else action_mask
        else:
            action_mask_batched = None

        action = network.get_action(obs, action_mask_batched)
        action = action.squeeze(0)
        obs, _, terminated, truncated, info = env.step(action)

        sc = env.last_state_change
        rewards = env.last_rewards
        prev = sc.previous
        curr = sc.state
        act = sc.action

        def _make_violation(kind, description):
            return InvariantViolation(
                episode=ep_idx, step=step_num, kind=kind,
                description=description, room=curr.location,
                level=curr.level,
                tile=(curr.link.tile.x, curr.link.tile.y),
                position=(curr.link.position.x, curr.link.position.y),
                action_kind=act.kind.name, action_direction=act.direction.name
            )

        # --- Invariant 1: Zero-pixel movement on MOVE action ---
        if act.kind == ActionKind.MOVE:
            if prev.full_location == curr.full_location:  # same room
                px_prev = prev.link.position
                px_curr = curr.link.position
                if px_prev == px_curr and not sc.health_lost:
                    # Classify: did our tile prediction think Link could move?
                    location = MapLocation(prev.level, prev.location,
                                           prev.full_location.in_cave)
                    room = Room.get(location)
                    # Check tile walkability, doorway constraint, and UW room boundary.
                    predicted_can_move = (room is not None and
                                         room.can_link_move_from(px_prev.x, px_prev.y,
                                                                 act.direction))
                    if predicted_can_move:
                        pi = prev.info
                        doorway_dir = pi.get('doorway_dir', 0)
                        # In a doorway, NES only allows movement in the doorway direction.
                        if doorway_dir != 0 and act.direction.value != doorway_dir:
                            predicted_can_move = False
                        # UW BoundByRoom: check direction depends on gridOffset.
                        elif prev.level != 0 and doorway_dir == 0:
                            grid_offset = pi.get('link_grid_offset', 0)
                            if grid_offset == 0:
                                check_dir = act.direction
                            elif abs(grid_offset) >= 4:
                                obj_dir = pi.get('link_direction', 0)
                                try:
                                    check_dir = Direction(obj_dir)
                                except ValueError:
                                    check_dir = None
                            else:
                                check_dir = None
                            if check_dir is not None:
                                if check_dir == Direction.N and px_prev.y < 0x5E:
                                    predicted_can_move = False
                                elif check_dir == Direction.S and px_prev.y >= 0xBD:
                                    predicted_can_move = False
                                elif check_dir == Direction.W and px_prev.x < 0x21:
                                    predicted_can_move = False
                                elif check_dir == Direction.E and px_prev.x >= 0xD0:
                                    predicted_can_move = False
                    if predicted_can_move:
                        # BUG: We predicted Link could move but the NES blocked it.
                        # Capture detailed NES state for debugging.
                        pi = prev.info
                        ci = curr.info
                        n_frames = len(sc.frames) if sc.frames else 0
                        enemies = [e for e in prev.enemies if e.is_active]
                        enemy_str = ", ".join(
                            f"{e.id.name}@({e.position.x},{e.position.y})" for e in enemies
                        ) if enemies else "none"
                        grid_off = pi.get('link_grid_offset', -1)
                        obj_timer = pi.get('link_obj_timer', -1)
                        shove_dir = pi.get('link_shove_dir', -1)

                        # Dump the hotspot tile values for debugging
                        px_x, px_y = px_prev.x, px_prev.y
                        tile_info = ""
                        match act.direction:
                            case Direction.N:
                                r = (px_y - 61) // 8
                                c = px_x // 8
                                t1 = int(room.tiles[c, r]) if 0 <= c < 32 and 0 <= r < 22 else -1
                                t2 = int(room.tiles[c+1, r]) if 0 <= c+1 < 32 and 0 <= r < 22 else -1
                                tile_info = f"N tiles[{c},{r}]=0x{t1:02x} [{c+1},{r}]=0x{t2:02x}"
                            case Direction.S:
                                r = (px_y - 45) // 8
                                c = px_x // 8
                                t1 = int(room.tiles[c, r]) if 0 <= c < 32 and 0 <= r < 22 else -1
                                t2 = int(room.tiles[c+1, r]) if 0 <= c+1 < 32 and 0 <= r < 22 else -1
                                tile_info = f"S tiles[{c},{r}]=0x{t1:02x} [{c+1},{r}]=0x{t2:02x}"
                            case Direction.W:
                                r = (px_y - 53) // 8
                                c = (px_x - 8) // 8
                                t1 = int(room.tiles[c, r]) if 0 <= c < 32 and 0 <= r < 22 else -1
                                tile_info = f"W tile[{c},{r}]=0x{t1:02x}"
                            case Direction.E:
                                r = (px_y - 53) // 8
                                c = (px_x + 16) // 8
                                t1 = int(room.tiles[c, r]) if 0 <= c < 32 and 0 <= r < 22 else -1
                                tile_info = f"E tile[{c},{r}]=0x{t1:02x}"

                        violations.append(_make_violation(
                            "movement-prediction-wrong",
                            f"MOVE {act.direction.name}: predicted can_move=True but "
                            f"NES blocked. pos={px_prev}, tile={prev.link.tile}\n"
                            f"      link_status=0x{pi.get('link_status',0):02x} "
                            f"mode=0x{pi.get('mode',0):02x} "
                            f"sword_anim=0x{pi.get('sword_animation',0):02x} "
                            f"link_dir=0x{pi.get('link_direction',0):02x} "
                            f"grid_offset={grid_off}\n"
                            f"      obj_timer={obj_timer} "
                            f"shove_dir=0x{shove_dir:02x} "
                            f"curr_link_status=0x{ci.get('link_status',0):02x} "
                            f"curr_mode=0x{ci.get('mode',0):02x} "
                            f"curr_pos=({ci.get('link_x',0)},{ci.get('link_y',0)}) "
                            f"curr_grid_offset={ci.get('link_grid_offset', -1)}\n"
                            f"      curr_obj_timer={ci.get('link_obj_timer', -1)} "
                            f"curr_shove_dir=0x{ci.get('link_shove_dir', 0):02x}\n"
                            f"      doorway_dir=0x{pi.get('doorway_dir', 0):02x} "
                            f"cur_opened_doors=0x{pi.get('cur_opened_doors', 0):02x} "
                            f"triggered_door_cmd=0x{pi.get('triggered_door_cmd', 0):02x}\n"
                            f"      {tile_info}\n"
                            f"      frames={n_frames} enemies=[{enemy_str}]"
                        ))
                    # Don't flag wall-hit (predicted False) — that's expected with
                    # multihead's always-True direction mask.

        # --- Invariant 2: No wavefront available ---
        if act.kind == ActionKind.MOVE and prev.full_location == curr.full_location:
            wf = prev.pbrs_wavefront if hasattr(prev, 'pbrs_wavefront') else None
            if wf is None:
                violations.append(_make_violation(
                    "no-wavefront",
                    f"No pbrs_wavefront on prev state. room=0x{prev.location:02x}"
                ))
            else:
                old_dist = wf.get(prev.link.tile)
                new_dist = wf.get(curr.link.tile)

                # --- Invariant 3: Wavefront has no distance for Link's tile ---
                if old_dist is None:
                    violations.append(_make_violation(
                        "wavefront-missing-tile",
                        f"Wavefront has no distance for prev tile {prev.link.tile}. "
                        f"room=0x{prev.location:02x}"
                    ))
                if new_dist is None and px_prev != px_curr if act.kind == ActionKind.MOVE else False:
                    violations.append(_make_violation(
                        "wavefront-missing-tile",
                        f"Wavefront has no distance for curr tile {curr.link.tile}. "
                        f"room=0x{curr.location:02x}"
                    ))

                # --- Invariant 4: Movement changed tiles but PBRS=0 ---
                if (old_dist is not None and new_dist is not None
                        and not sc.health_lost
                        and prev.link.position != curr.link.position
                        and prev.link.tile != curr.link.tile):  # tile must change
                    shaped = (old_dist - new_dist) / PBRS_SCALE
                    if shaped == 0.0:
                        has_pbrs = any('pbrs' in o.name for o in rewards)
                        if not has_pbrs:
                            violations.append(_make_violation(
                                "zero-pbrs-tile-change",
                                f"Moved {act.direction.name} (tile {prev.link.tile}→"
                                f"{curr.link.tile}) but PBRS=0. "
                                f"old_dist={old_dist}, new_dist={new_dist}"
                            ))

        # --- Invariant 5: No valid move directions in action mask ---
        if action_mask is not None:
            # For multihead: first K entries are action types, last 4 are directions
            # For flat: check that MOVE indices have at least one True
            mask_np = action_mask.numpy() if hasattr(action_mask, 'numpy') else action_mask
            # Check if total mask is all False (catastrophic)
            if not mask_np.any():
                violations.append(_make_violation(
                    "empty-mask",
                    f"Action mask is entirely False! No valid actions at all."
                ))

        # --- Invariant 6: MOVE action was taken but mask said it was invalid ---
        if action_mask is not None and act.kind == ActionKind.MOVE:
            # The action space wrapper checks can_link_move per direction.
            # If the model took a move that was masked, that's a bug in the
            # mask→action pipeline.
            pass  # This is enforced by the model's logit masking; skip for now.

        # --- Invariant 7: Room changed on non-edge tile ---
        if prev.full_location != curr.full_location and act.kind == ActionKind.MOVE:
            tile = prev.link.tile
            # Normal room transitions happen at screen edges
            at_edge = (tile.x <= 0 or tile.x >= 0x1e or tile.y <= 0 or tile.y >= 0x15)
            if not at_edge and not prev.full_location.in_cave and not curr.full_location.in_cave:
                # Could be wallmaster, staircase, etc. — note but don't flag as critical
                pass

        step_num += 1
        state = curr

    return violations


def _generate_invariant_report(violations, model_name, scenario_name, model_path, episodes):
    """Generate a text report of invariant violations."""
    lines = []
    w = lines.append

    w("=" * 72)
    w(f"INVARIANT CHECK REPORT: {model_name}")
    w(f"Scenario: {scenario_name}")
    w(f"Model: {model_path}")
    w(f"Episodes: {episodes}")
    w(f"Total violations: {len(violations)}")
    w("=" * 72)
    w("")

    if not violations:
        w("No invariant violations detected. All checks passed.")
        w("")
        return "\n".join(lines)

    # Summary by kind
    w("## VIOLATION SUMMARY")
    kind_counts = Counter(v.kind for v in violations)
    for kind, count in kind_counts.most_common():
        w(f"  {kind:30s}: {count:5d}")
    w("")

    # Summary by room
    w("## VIOLATIONS BY ROOM")
    room_counts = Counter((v.level, v.room) for v in violations)
    for (level, room), count in room_counts.most_common(15):
        w(f"  Level={level} Room=0x{room:02x}: {count:5d} violations")
    w("")

    # Detailed violations (cap at 50 to keep output manageable)
    # Sort so movement-prediction-wrong appears first (most important).
    priority = {'movement-prediction-wrong': 0, 'no-wavefront': 1, 'empty-action-mask': 2}
    sorted_violations = sorted(violations, key=lambda v: (priority.get(v.kind, 99), v.episode, v.step))
    w("## VIOLATION DETAILS")
    shown = min(len(sorted_violations), 50)
    w(f"  Showing {shown} of {len(sorted_violations)} violations:")
    w("")
    for v in sorted_violations[:shown]:
        w(f"  [{v.kind}] Episode {v.episode+1}, Step {v.step}")
        w(f"    Room: Level={v.level} 0x{v.room:02x} | Tile: {v.tile} | Pos: {v.position}")
        w(f"    Action: {v.action_kind} {v.action_direction}")
        w(f"    {v.description}")
        w("")

    w("=" * 72)
    w("END OF INVARIANT CHECK REPORT")
    w("=" * 72)

    return "\n".join(lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Diagnostic tool for trained triforce models")
    parser.add_argument("--model", required=True, help="Model name from triforce.yaml")
    parser.add_argument("--scenario", required=True, help="Scenario name from triforce.yaml")
    parser.add_argument("--model-path", required=True,
                        help="Path to .pt model file (supports glob patterns)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: stdout)")
    parser.add_argument("--infinite-pbrs", action="store_true",
                        help="Run PBRS diagnostic: play until stuck/timeout, print step-by-step "
                             "PBRS calculation for the stuck room")
    parser.add_argument("--tail", type=int, default=50,
                        help="Number of trailing steps to show in --infinite-pbrs mode (default: 50)")
    parser.add_argument("--invariants", action="store_true",
                        help="Run movement/reward invariant checker: detect zero-movement, "
                             "missing wavefronts, empty masks, and zero-PBRS violations")
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

    if args.infinite_pbrs:
        run_pbrs_diagnostic(args.model, args.scenario, model_path,
                            args.episodes, args.tail, args.output)
    elif args.invariants:
        run_invariant_checker(args.model, args.scenario, model_path,
                              args.episodes, args.output)
    else:
        run_diagnostic(args.model, args.scenario, model_path, args.episodes, args.output)


if __name__ == "__main__":
    main()
