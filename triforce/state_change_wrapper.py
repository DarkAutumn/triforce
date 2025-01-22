"""This wrapper tracks the ongoing state of the game."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

from .objectives import ObjectiveSelector
from .action_space import ActionTaken
from .zelda_enums import AnimationState, ZeldaItemKind, ZeldaAnimationKind
from .zelda_game import ZeldaGame

class StateChange:
    """Tracks the changes between two Zelda game states."""
    def __init__(self, env, prev : ZeldaGame, curr : ZeldaGame, action, frames, discounts, ignore_health):
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
            discounts.clear()

        else:
            # Check for changes to health and items
            self._compare_health_status(prev, curr, self.__dict__)
            self._compare_items(prev, curr, self.__dict__)

            # But discount any effects that we already rewarded in the past
            self._discount_damage(self.__dict__, discounts)

            # Find if this action will cause future damage
            self._detect_future_damage(env, prev, curr, discounts)

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

    def _compare_health_status(self, prev : ZeldaGame, curr : ZeldaGame, result):
        # Walk all enemies alive (and not dying) in the previous snapshot
        for enemy in prev.enemies:
            if enemy.is_dying:
                continue

            # Find the corresponding enemy in the current snapshot
            curr_enemy = curr.get_enemy_by_index(enemy.index)
            if curr_enemy:
                dmg = enemy.health - curr_enemy.health
                if dmg > 0:
                    enemies_hit = result.setdefault('enemies_hit', {})
                    enemies_hit[enemy.index] = enemies_hit.get(enemy.index, 0) + dmg

                elif curr_enemy.is_dying:
                    health = enemy.health if enemy.health > 0 else 1
                    enemies_hit = result.setdefault('enemies_hit', {})
                    enemies_hit[enemy.index] = enemies_hit.get(enemy.index, 0) + health

                if curr_enemy.is_stunned and not enemy.is_stunned:
                    enemies_stunned = result.setdefault('enemies_stunned', [])
                    enemies_stunned.append(enemy.index)

            else:
                dmg = enemy.health if enemy.health > 0 else 1
                enemies_hit = result.setdefault('enemies_hit', {})
                enemies_hit[enemy.index] = enemies_hit.get(enemy.index, 0) + dmg

    def _compare_items(self, prev : ZeldaGame, curr : ZeldaGame, result):
        # Walk all items in the previous snapshot
        elapsed_frames = curr.frames - prev.frames
        for item in prev.items:
            curr_item = curr.get_item_by_index(item.index)
            if not curr_item and elapsed_frames < item.timer:
                result.setdefault('items_gained', []).append(item.id)

    def _discount_damage(self, actual, discounts):
        discounted_hits = discounts.get('enemies_hit', {})
        hits = actual.get('enemies_hit', {})
        for index, dmg in discounted_hits.items():
            if index in hits:
                diff = min(hits[index], dmg)
                discounted_hits[index] -= diff
                hits[index] -= diff
                if hits[index] <= 0:
                    del hits[index]

        discounted_stunned = discounts.get('enemies_stunned', [])
        stunned = actual.get('enemies_stunned', [])
        for index in discounted_stunned.copy():
            if index in stunned:
                stunned.remove(index)
                discounted_stunned.remove(index)

        discounted_items = discounts.get('items_gained', None)
        if discounted_items:
            items = actual.get('items_gained', [])
            for item in discounted_items.copy():
                if item in items:
                    items.remove(item)
                    discounted_items.remove(item)

    def _detect_future_damage(self, env, prev, curr, discounts):
        # check if beams, bombs, arrows, etc are active and if they will hit in the future,
        # as we need to count them as rewards/results of this action so the model trains properly
        self._handle_future_effects(env, prev, curr, ZeldaAnimationKind.BEAMS, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationKind.BOMB_1, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationKind.BOMB_2, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationKind.ARROW, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationKind.BOOMERANG, discounts)

    def _handle_future_effects(self, env, prev : ZeldaGame, curr : ZeldaGame,  equipment, discounts):
        # If the current state is not active, we don't need to check for future effects
        curr_ani = curr.link.get_animation_state(equipment)
        if curr_ani != AnimationState.ACTIVE:
            return

        # If the previous state was already active, we already handled it
        prev_ani = prev.link.get_animation_state(equipment)
        if prev_ani != AnimationState.ACTIVE:
            self._predict_future_effects(env, curr, equipment, discounts)

    def _predict_future_effects(self, env, start : ZeldaGame, equipment, discounts):
        unwrapped = env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        # Disable all other equipment
        self._disable_others(data, equipment)

        action = np.zeros(9, dtype=bool)
        curr = start
        while not curr.game_over and start.location == curr.location  \
              and curr.link.get_animation_state(equipment) != AnimationState.INACTIVE:

            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            # Be sure to use unwrapped.step so we don't capture any state of this timeline digression
            _, _, terminated, truncated, info = unwrapped.step(action)
            if terminated or truncated:
                break

            curr = ZeldaGame(env, info, curr.frames + 1)

        # Check if we hit any enemies and store those into our damage counters and the discount of future hits
        self._compare_health_status(start, curr, self.__dict__)
        self._compare_health_status(start, curr, discounts)

        # If we fired a boomerang or arrow, check if we hit any items as they will be picked up
        if equipment in [ZeldaAnimationKind.BOOMERANG, ZeldaAnimationKind.ARROW]:
            self._compare_items(start, curr, self.__dict__)
            self._compare_items(start, curr, discounts)

        # Restore the state.  We only touched data.values, so we should be clear of any modifications
        unwrapped.em.set_state(savestate)

    def _disable_others(self, data, equipment):
        all_names = ['beam_animation', 'bomb_or_flame_animation', 'bomb_or_flame_animation2',
                     'bait_or_boomerang_animation', 'arrow_magic_animation']

        match equipment:
            case ZeldaAnimationKind.BEAMS:
                all_names.remove('beam_animation')
            case ZeldaAnimationKind.BOMB_1:
                all_names.remove('bomb_or_flame_animation')
            case ZeldaAnimationKind.BOMB_2:
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
        self._discounts = {}
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
        self._discounts.clear()
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
            return StateChange(self, prev, state, action, frames, self._discounts, health_change_ignore)

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
