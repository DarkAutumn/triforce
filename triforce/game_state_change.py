"""This wrapper tracks the ongoing state of the game."""

from typing import Dict, List

import numpy as np

from .zelda_enums import AnimationState, ZeldaItemId, ZeldaAnimationId
from .zelda_game_state import ZeldaGameState

class ZeldaStateChange:
    """Tracks the changes between two Zelda game states."""
    def __init__(self, env, prev : ZeldaGameState, curr : ZeldaGameState, discounts):
        self.prev : ZeldaGameState = prev
        self.curr : ZeldaGameState = curr

        self.enemies_hit : Dict[int, int] = {}
        self.enemies_stunned : List[int] = []
        self.items_gained : List[ZeldaItemId] = []

        self.changed_location = prev.full_location != curr.full_location

        if self.changed_location:
            discounts.clear()

        else:
            self._compare_health_status(prev, curr, self.__dict__)
            self._compare_items(prev, curr, self.__dict__)
            self._detect_future_damage(env, prev, curr, discounts)
            self._discount_damage(self.__dict__, discounts)

    def _compare_health_status(self, prev : ZeldaGameState, curr : ZeldaGameState, result):
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

                if enemy.is_stunned and not curr_enemy.is_stunned:
                    enemies_stunned = result.setdefault('enemies_stunned', [])
                    enemies_stunned.append(enemy.index)

            else:
                dmg = enemy.health if enemy.health > 0 else 1
                enemies_hit = result.setdefault('enemies_hit', {})
                enemies_hit[enemy.index] = enemies_hit.get(enemy.index, 0) + dmg

    def _compare_items(self, prev : ZeldaGameState, curr : ZeldaGameState, result):
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
        self._handle_future_effects(env, prev, curr, ZeldaAnimationId.BEAMS, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationId.BOMB_1, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationId.BOMB_2, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationId.ARROW, discounts)
        self._handle_future_effects(env, prev, curr, ZeldaAnimationId.BOOMERANG, discounts)

    def _handle_future_effects(self, env, prev : ZeldaGameState, curr : ZeldaGameState,  equipment, discounts):
        # If the current state is not active, we don't need to check for future effects
        curr_ani = curr.link.get_animation_state(equipment)
        if curr_ani != AnimationState.ACTIVE:
            return

        # If the previous state was already active, we already handled it
        prev_ani = prev.link.get_animation_state(equipment)
        if prev_ani != AnimationState.ACTIVE:
            self._predict_future_effects(env, curr, equipment, discounts)

    def _predict_future_effects(self, env, start : ZeldaGameState, equipment, discounts):
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
            _, _, terminated, truncated, info = unwrapped.step(action)
            if terminated or truncated:
                break

            curr = ZeldaGameState(env, info, curr.frames + 1)

        # Check if we hit any enemies and store those into discounts
        self._compare_health_status(start, curr, discounts)

        # If we fired a boomerang or arrow, check if we hit any items as they will be picked up
        if equipment in [ZeldaAnimationId.BOOMERANG, ZeldaAnimationId.ARROW]:
            self._compare_items(start, curr, discounts)

        # Restore the state.  We only touched data.values, so we should be clear of any modifications
        unwrapped.em.set_state(savestate)

    def _disable_others(self, data, equipment):
        all_names = ['beam_animation', 'bomb_or_flame_animation', 'bomb_or_flame_animation2',
                     'bait_or_boomerang_animation', 'arrow_magic_animation']

        match equipment:
            case ZeldaAnimationId.BEAMS:
                all_names.remove('beam_animation')
            case ZeldaAnimationId.BOMB_1:
                all_names.remove('bomb_or_flame_animation')
            case ZeldaAnimationId.BOMB_2:
                all_names.remove('bomb_or_flame_animation2')
            case ZeldaAnimationId.ARROW:
                all_names.remove('arrow_magic_animation')
            case ZeldaAnimationId.BOOMERANG:
                all_names.remove('bait_or_boomerang_animation')
            case _:
                raise ValueError("Invalid equipment")

        for n in all_names:
            data.set_value(n, 0)
