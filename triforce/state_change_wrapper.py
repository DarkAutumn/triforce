"""This wrapper tracks the ongoing state of the game."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

from .objectives import ObjectiveSelector
from .action_space import ActionTaken
from .zelda_enums import AnimationState, ZeldaItemKind, ZeldaAnimationKind
from .zelda_game import ZeldaGame

# Predictions older than this many frames are discarded.  This prevents stale predictions
# from eating real damage credit if a weapon misses (e.g., enemy moved out of the way).
PREDICTION_EXPIRY_FRAMES = 200


@dataclass
class Prediction:
    """A single look-ahead prediction of future effects from a weapon activation.

    Each look-ahead creates one Prediction.  When real effects are later observed,
    matching entries are consumed (subtracted) so we don't double-count.
    """
    frame: int
    enemy_hits: Dict[int, int] = field(default_factory=dict)
    enemy_stuns: List[int] = field(default_factory=list)
    items: List[ZeldaItemKind] = field(default_factory=list)

    @property
    def is_empty(self):
        """True when all predicted effects have been consumed or were zero."""
        return not self.enemy_hits and not self.enemy_stuns and not self.items


class FutureCreditLedger:
    """Tracks predicted future effects so they aren't double-counted as rewards.

    When a weapon is fired, the look-ahead simulates forward and credits the damage to the
    current step.  A Prediction is recorded here.  On subsequent steps, when that damage
    actually materializes in the real game state, the Prediction is consumed so we don't
    reward it again.

    Each Prediction is independent — beam and bomb predictions for the same enemy don't
    merge.  This prevents a missed beam from eating a later bomb's real damage credit.
    """

    def __init__(self):
        self._predictions: List[Prediction] = []

    def clear(self):
        """Remove all predictions (e.g., on room change or reset)."""
        self._predictions.clear()

    def add_prediction(self, prediction: Prediction):
        """Record a new look-ahead prediction."""
        if not prediction.is_empty:
            self._predictions.append(prediction)

    def discount(self, current_frame: int, enemies_hit: Dict[int, int],
                 enemies_stunned: List[int], items_gained: List[ZeldaItemKind]):
        """Subtract already-predicted effects from actual observations.

        Modifies enemies_hit, enemies_stunned, and items_gained in place to remove
        effects that were already credited via look-ahead.  Consumed predictions are
        cleaned up.  Expired predictions are discarded.
        """
        # Expire old predictions first
        self._predictions = [p for p in self._predictions
                             if current_frame - p.frame < PREDICTION_EXPIRY_FRAMES]

        for pred in self._predictions:
            # Discount enemy damage — match by slot index, subtract up to predicted amount
            for index in list(pred.enemy_hits.keys()):
                if index in enemies_hit:
                    subtract = min(enemies_hit[index], pred.enemy_hits[index])
                    enemies_hit[index] -= subtract
                    pred.enemy_hits[index] -= subtract
                    if enemies_hit[index] <= 0:
                        del enemies_hit[index]
                    if pred.enemy_hits[index] <= 0:
                        del pred.enemy_hits[index]

            # Discount stuns
            for index in list(pred.enemy_stuns):
                if index in enemies_stunned:
                    enemies_stunned.remove(index)
                    pred.enemy_stuns.remove(index)

            # Discount items
            for item in list(pred.items):
                if item in items_gained:
                    items_gained.remove(item)
                    pred.items.remove(item)

        # Remove fully consumed predictions
        self._predictions = [p for p in self._predictions if not p.is_empty]

class StateChange:
    """Tracks the changes between two Zelda game states."""
    def __init__(self, env, prev : ZeldaGame, curr : ZeldaGame, action, frames,
                 ledger: FutureCreditLedger, ignore_health):
        self.action : ActionTaken = action
        self.frames : List[np.ndarray] = frames
        self.previous : ZeldaGame = prev
        self.state : ZeldaGame = curr
        self.action_mask : Optional[torch.Tensor] = None
        self.actions_available = None

        self.health_lost = (max(0, prev.link.health - curr.link.health + ignore_health) \
                           if prev.link.max_health == curr.link.max_health \
                           else max(0, prev.link.max_health - curr.link.max_health + ignore_health))

        self.health_gained = (max(0, curr.link.health - prev.link.health - ignore_health) \
                             if prev.link.max_health == curr.link.max_health \
                             else max(0, curr.link.max_health - prev.link.max_health - ignore_health))

        self.enemies_hit : Dict[int, int] = {}
        self.enemies_stunned : List[int] = []
        self.items_gained : List[ZeldaItemKind] = []

        if self.changed_location:
            ledger.clear()

        else:
            # 1. Observe what actually happened this step
            self._observe_damage(prev, curr)
            self._observe_items(prev, curr)

            # 2. Subtract effects already credited by previous look-aheads
            ledger.discount(curr.frames, self.enemies_hit, self.enemies_stunned, self.items_gained)

            # 3. Run look-aheads for newly activated weapons, credit and record predictions
            self._detect_future_damage(env, prev, curr, ledger)

    def __str__(self):
        # return a multiline string with the most important information
        return f"Action: {self.action}\n" \
                f"ActionMask: {self.action_mask}" \
                f"Health lost: {self.health_lost}\n" \
                f"Health gained: {self.health_gained}\n" \
                f"Enemies hit: {self.enemies_hit}\n" \
                f"Enemies stunned: {self.enemies_stunned}\n" \
                f"Items gained: {self.items_gained}\n" \
                f"Changed location: {self.changed_location}\n" \
                f"Previous: {self.previous}\n" \
                f"Current: {self.state}\n"

    @property
    def gained_triforce(self):
        """Returns True if the player gained a triforce piece."""
        return self.previous.link.triforce_pieces < self.state.link.triforce_pieces or \
               not self.previous.link.triforce_of_power and self.state.link.triforce_of_power

    @property
    def changed_location(self):
        """Returns True if the location changed."""
        return self.previous.full_location != self.state.full_location

    @property
    def damage_dealt(self):
        """The total damage dealt by link to enemies this turn."""
        return sum(self.enemies_hit.values())

    @property
    def hits(self):
        """The total number of enemies hit by link this turn."""
        return len(self.enemies_hit)

    def _observe_damage(self, prev: ZeldaGame, curr: ZeldaGame):
        """Compare enemy health between frames to detect hits, kills, and stuns."""
        for enemy in prev.enemies:
            if enemy.is_dying:
                continue

            curr_enemy = curr.get_enemy_by_index(enemy.index)
            if curr_enemy:
                dmg = enemy.health - curr_enemy.health
                if dmg > 0:
                    self.enemies_hit[enemy.index] = self.enemies_hit.get(enemy.index, 0) + dmg
                elif curr_enemy.is_dying:
                    health = enemy.health if enemy.health > 0 else 1
                    self.enemies_hit[enemy.index] = self.enemies_hit.get(enemy.index, 0) + health

                if curr_enemy.is_stunned and not enemy.is_stunned:
                    self.enemies_stunned.append(enemy.index)
            else:
                dmg = enemy.health if enemy.health > 0 else 1
                self.enemies_hit[enemy.index] = self.enemies_hit.get(enemy.index, 0) + dmg

    def _observe_items(self, prev: ZeldaGame, curr: ZeldaGame):
        """Detect items that were picked up between frames."""
        elapsed_frames = curr.frames - prev.frames
        for item in prev.items:
            curr_item = curr.get_item_by_index(item.index)
            if not curr_item and elapsed_frames < item.timer:
                self.items_gained.append(item.id)

    def _detect_future_damage(self, env, prev, curr, ledger: FutureCreditLedger):
        """Run look-aheads for all newly activated weapons."""
        for equipment in (ZeldaAnimationKind.BEAMS, ZeldaAnimationKind.MAGIC,
                          ZeldaAnimationKind.BOMB_1, ZeldaAnimationKind.BOMB_2,
                          ZeldaAnimationKind.FLAME_1, ZeldaAnimationKind.FLAME_2,
                          ZeldaAnimationKind.ARROW, ZeldaAnimationKind.BOOMERANG):
            self._handle_future_effects(env, prev, curr, equipment, ledger)

    def _handle_future_effects(self, env, prev: ZeldaGame, curr: ZeldaGame,
                               equipment, ledger: FutureCreditLedger):
        """Run look-ahead if this weapon just became active (INACTIVE → ACTIVE transition)."""
        curr_ani = curr.link.get_animation_state(equipment)
        if curr_ani != AnimationState.ACTIVE:
            return

        prev_ani = prev.link.get_animation_state(equipment)
        if prev_ani != AnimationState.ACTIVE:
            self._predict_future_effects(env, curr, equipment, ledger)

    def _predict_future_effects(self, env, start: ZeldaGame, equipment, ledger: FutureCreditLedger):
        """Simulate forward until weapon deactivates, credit damage, record prediction."""
        unwrapped = env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        self._disable_others(data, equipment)

        action = np.zeros(9, dtype=bool)
        curr = start
        while not curr.game_over and start.location == curr.location \
              and curr.link.get_animation_state(equipment) != AnimationState.INACTIVE:

            data.set_value('hearts_and_containers', 0xff)

            _, _, terminated, truncated, info = unwrapped.step(action)
            if terminated or truncated:
                break

            curr = ZeldaGame(env, info, curr.frames + 1)

        # Build a prediction from the simulated outcome
        prediction = Prediction(frame=start.frames)
        self._observe_damage_into(start, curr, prediction)

        if equipment in (ZeldaAnimationKind.BOOMERANG, ZeldaAnimationKind.ARROW):
            self._observe_items_into(start, curr, prediction)

        # Credit the damage to this step AND record the prediction for future discounting
        for index, dmg in prediction.enemy_hits.items():
            self.enemies_hit[index] = self.enemies_hit.get(index, 0) + dmg
        self.enemies_stunned.extend(prediction.enemy_stuns)
        self.items_gained.extend(prediction.items)

        ledger.add_prediction(prediction)

        unwrapped.em.set_state(savestate)

    @staticmethod
    def _observe_damage_into(prev: ZeldaGame, curr: ZeldaGame, prediction: Prediction):
        """Compare enemy health and populate a Prediction (not self)."""
        for enemy in prev.enemies:
            if enemy.is_dying:
                continue

            curr_enemy = curr.get_enemy_by_index(enemy.index)
            if curr_enemy:
                dmg = enemy.health - curr_enemy.health
                if dmg > 0:
                    prediction.enemy_hits[enemy.index] = \
                        prediction.enemy_hits.get(enemy.index, 0) + dmg
                elif curr_enemy.is_dying:
                    health = enemy.health if enemy.health > 0 else 1
                    prediction.enemy_hits[enemy.index] = \
                        prediction.enemy_hits.get(enemy.index, 0) + health

                if curr_enemy.is_stunned and not enemy.is_stunned:
                    prediction.enemy_stuns.append(enemy.index)
            else:
                dmg = enemy.health if enemy.health > 0 else 1
                prediction.enemy_hits[enemy.index] = \
                    prediction.enemy_hits.get(enemy.index, 0) + dmg

    @staticmethod
    def _observe_items_into(prev: ZeldaGame, curr: ZeldaGame, prediction: Prediction):
        """Compare items and populate a Prediction."""
        elapsed_frames = curr.frames - prev.frames
        for item in prev.items:
            curr_item = curr.get_item_by_index(item.index)
            if not curr_item and elapsed_frames < item.timer:
                prediction.items.append(item.id)

    def _disable_others(self, data, equipment):
        all_names = ['beam_animation', 'bomb_or_flame_animation', 'bomb_or_flame_animation2',
                     'bait_or_boomerang_animation', 'arrow_magic_animation']

        match equipment:
            case ZeldaAnimationKind.BEAMS:
                all_names.remove('beam_animation')
            case ZeldaAnimationKind.MAGIC:
                # Keep beam slot for rod shot. bomb_or_flame is zeroed to clear
                # stale objects — NES will write fire ($22) when rod shot hits.
                all_names.remove('beam_animation')
            case ZeldaAnimationKind.BOMB_1:
                all_names.remove('bomb_or_flame_animation')
            case ZeldaAnimationKind.BOMB_2:
                all_names.remove('bomb_or_flame_animation2')
            case ZeldaAnimationKind.FLAME_1:
                all_names.remove('bomb_or_flame_animation')
            case ZeldaAnimationKind.FLAME_2:
                all_names.remove('bomb_or_flame_animation2')
            case ZeldaAnimationKind.ARROW:
                all_names.remove('arrow_magic_animation')
            case ZeldaAnimationKind.BOOMERANG:
                all_names.remove('bait_or_boomerang_animation')
            case _:
                raise ValueError("Invalid equipment")

        for n in all_names:
            data.set_value(n, 0)

class StateChangeWrapper(gym.Wrapper):
    """Keeps track of the state of the game."""
    def __init__(self, env, scenario):
        super().__init__(env)
        self._ledger = FutureCreditLedger()
        self._objective_type = scenario.objective if scenario else None
        self._objectives : ObjectiveSelector = None
        self._prev_state = None

        self.per_reset = []
        self.per_room = []
        self.per_frame = []

        if scenario is not None:
            for key, value in scenario.per_reset.items():
                self.per_reset.append((key, value))

            for key, value in scenario.per_room.items():
                self.per_room.append((key, value))

            for key, value in scenario.per_frame.items():
                self.per_frame.append((key, value))

    def reset(self, **kwargs):
        frames, info = self.env.reset(**kwargs)
        self._ledger.clear()
        self._prev_state = None
        if self._objective_type:
            self._objectives = self._objective_type()

        state = self._update_state(None, frames, info)
        return frames, state

    def step(self, action):
        frames, rewards, terminated, truncated, info = self.env.step(action)
        state = self._update_state(action, frames, info)
        return frames, rewards, terminated, truncated, state

    def _create_and_set_state(self, info) -> Tuple[ZeldaGame, ZeldaGame]:
        prev = self._prev_state
        state = ZeldaGame(self, info, info['total_frames'])
        self._prev_state = state
        return prev, state

    def _update_state(self, action, frames, info):
        prev, state = self._create_and_set_state(info)
        health_change_ignore = self._apply_modifications(prev, state)

        if self._objectives:
            objectives = self._objectives.get_current_objectives(prev, state)
            state.objectives = objectives
            state.wavefront = state.room.calculate_wavefront_for_link(objectives.targets)

        if prev:
            return StateChange(self, prev, state, action, frames, self._ledger, health_change_ignore)

        info['initial_frame'] = frames[-1]
        return state

    def _apply_modifications(self, prev : ZeldaGame, curr : ZeldaGame) -> float:
        health = curr.link.health

        if prev is None:
            for name, value in self.per_reset:
                self._set_value(curr, name, value)

            for name, value in self.per_room:
                self._set_value(curr, name, value)

        elif prev.full_location != curr.full_location:
            for name, value in self.per_room:
                self._set_value(curr, name, value)

        for name, value in self.per_frame:
            self._set_value(curr, name, value)

        return curr.link.health - health

    def _set_value(self, state, name, value):
        order = [state, state.link]
        if hasattr(state.link, name):
            order = [state.link, state]

        obj = order.pop(0)
        if not hasattr(obj, name):
            obj = order.pop(0)

        if isinstance(value, str):
            value = getattr(obj, value)

        setattr(obj, name, value)
