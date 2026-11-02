"""Tracks the changes between two consecutive controllable game states, with look-ahead credit assignment.

Ported from v1's ``state_change_wrapper.py``.  Delayed weapons (beams, bombs, arrows, boomerang) are simulated
forward on the emulator when they fire, the predicted effects are credited to the firing step, and a
``FutureCreditLedger`` subtracts them again when they really happen (design §6, v1 ai-design §7.8).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .actions import Action, ActionMask
from .zelda_enums import AnimationState, ZeldaItemKind, ZeldaAnimationKind
from .zelda_game import ZeldaGame

# Safety cap on a single look-ahead (v1 relied on the weapon animation ending, about 200 frames max).
LOOK_AHEAD_MAX_FRAMES = 400

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
    # pylint: disable=too-many-instance-attributes
    def __init__(self, emu, prev : ZeldaGame, curr : ZeldaGame, action : Action, option,
                 ledger: FutureCreditLedger, ignore_health=0.0):
        self.action : Action = action
        self.option = option                      # frame_skip.OptionResult
        self.frames : int = option.frames
        self.interrupt : Optional[str] = option.interrupt
        self.previous : ZeldaGame = prev
        self.state : ZeldaGame = curr
        self.action_mask : Optional[ActionMask] = None

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
            self._detect_future_damage(emu, prev, curr, ledger)

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

    def _detect_future_damage(self, emu, prev, curr, ledger: FutureCreditLedger):
        """Run look-aheads for all newly activated weapons."""
        for equipment in (ZeldaAnimationKind.BEAMS, ZeldaAnimationKind.MAGIC,
                          ZeldaAnimationKind.BOMB_1, ZeldaAnimationKind.BOMB_2,
                          ZeldaAnimationKind.FLAME_1, ZeldaAnimationKind.FLAME_2,
                          ZeldaAnimationKind.ARROW, ZeldaAnimationKind.BOOMERANG):
            self._handle_future_effects(emu, prev, curr, equipment, ledger)

    def _handle_future_effects(self, emu, prev: ZeldaGame, curr: ZeldaGame,
                               equipment, ledger: FutureCreditLedger):
        """Run look-ahead if this weapon just became active (INACTIVE → ACTIVE transition)."""
        curr_ani = curr.link.get_animation_state(equipment)
        if curr_ani != AnimationState.ACTIVE:
            return

        prev_ani = prev.link.get_animation_state(equipment)
        if prev_ani != AnimationState.ACTIVE:
            self._predict_future_effects(emu, curr, equipment, ledger)

    def _predict_future_effects(self, emu, start: ZeldaGame, equipment, ledger: FutureCreditLedger):
        """Simulate forward until weapon deactivates, credit damage, record prediction.  Restores the emulator
        (state and frame counter) afterwards, so the look-ahead is invisible to the caller."""
        _ = start.enemies, start.items     # materialize from RAM while start is the active state
        savestate = emu.save_state()
        saved_frame = emu.frame

        self._disable_others(emu, equipment)

        curr = start
        for _ in range(LOOK_AHEAD_MAX_FRAMES):
            if curr.game_over or start.location != curr.location \
                    or curr.link.get_animation_state(equipment) == AnimationState.INACTIVE:
                break

            emu.set_value('hearts_and_containers', 0xff)
            emu.step(None, 1)
            curr = ZeldaGame(emu, emu.info, emu.frame)
            _ = curr.enemies, curr.items

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

        emu.load_state(savestate)
        emu.frame = saved_frame
        start.reactivate()

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

