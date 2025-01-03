"""
Performs all hit-detection for the game.  This fills entries in the info dictionary such as 'step-hit'.
"""

import gymnasium as gym

from .zelda_game import STUN_FLAG, ZeldaObjectData, AnimationState, is_mode_death, \
    get_bomb_state, get_beam_state, get_boomerang_state, get_arrow_state

class ZeldaHitDetect(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env):
        super().__init__(env)
        self._last_frame = None
        self._prev_health = None
        self._prev_items = {}
        self._state = {}

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_frame = info['total_frames']
        self._prev_health = None
        self._prev_items.clear()
        self._state.clear()
        return obs, info

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)

        if info['new_location']:
            self._prev_health = None
            self._state.clear()
            hits, stuns, items = 0, 0, []
        else:
            # Only check hits if we didn't move room locations
            hits, stuns, items = self._get_step_hits(action, info)

        info['step_hits'] = hits
        info['step_stuns'] = stuns
        info['step_items'] = items

        return obs, rewards, terminated, truncated, info

    def _get_step_hits(self, act, info):
        step_hits = 0
        step_stuns = 0
        step_items = []

        # capture enemy health data
        curr_enemy_health = {}
        objects = info['objects']
        for eid in objects.enumerate_enemy_ids():
            health = objects.get_obj_health(eid)

            # Some enemies, like gels and keese, do not have health.  This makes calculating hits very challenging.
            # Instead, just set those 0 health enemies to 1 health, which doesn't otherwise affect the game.  The
            # game will set them to 0 health when they die.
            if not health and (not self._prev_health or self._prev_health.get(eid, 0) == 0):
                data = self.env.unwrapped.data
                data.set_value(f'obj_health_{eid:x}', 0x10)
                health = 1

            curr_enemy_health[eid] = health

        # check if we killed or injured anything
        if self._prev_health:
            for eid, health in self._prev_health.items():
                if eid in curr_enemy_health and curr_enemy_health[eid] < health:
                    step_hits += 1

        # capture item information
        curr_items = {x: objects.get_obj_timer(x) for x in objects.enumerate_item_ids()}

        # check if we picked up any items
        frames_elapsed = info['total_frames'] - self._last_frame
        for item, timer in self._prev_items.items():
            if frames_elapsed < timer and item not in curr_items:
                step_items.append(objects.get_object_id(item))


        # check if beams, bombs, arrows, etc are active and if they will hit in the future,
        # as we need to count them as rewards/results of this action so the model trains properly
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'beam_hits',
                                    lambda st: get_beam_state(st) == AnimationState.ACTIVE, self._set_beams_only)
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'bomb1_hits',
                                    lambda st: get_bomb_state(st, 0) == AnimationState.ACTIVE, self._set_bomb1_only)
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'bomb2_hits',
                                    lambda st: get_bomb_state(st, 1) == AnimationState.ACTIVE, self._set_bomb2_only)

        # arrows can pick up items but not stun
        step_hits, _, step_items = self._handle_future_effects(act, info, objects, step_hits, step_stuns, step_items,
                                                               'arrow_hits',
                                                               lambda st: get_arrow_state(st) == AnimationState.ACTIVE,
                                                               self._set_arrow_only)

        # boomerangs can stun, kill, and pick up items
        step_hits, step_stuns, step_items = self._handle_future_effects(act, info, objects, step_hits, step_stuns,
                                                step_items, 'boomerang_hits',
                                                lambda st: get_boomerang_state(st) == AnimationState.ACTIVE,
                                                self._set_boomerang_only)

        self._prev_health = curr_enemy_health
        self._prev_items = curr_items
        self._last_frame = info['total_frames']
        return step_hits, step_stuns, step_items

    def _capture_items(self, objects):
        items = {}
        for item in objects.enumerate_item_ids():
            items[item] = objects.get_obj_timer(item)

    def _handle_future_effects(self, act, info, objects, step_hits, step_stuns, items_obtained, name,
                               condition_check, disable_others):
        already_active_name = name + '_already_active'
        discounted_hits = name + '_discounted_hits'
        discounted_stuns = name + '_discounted_stuns'
        discounted_items = name + '_discounted_items'

        # check if boomerang is active and if it will hit in the future
        if condition_check(info):
            if not self._state.get(already_active_name, False):
                future_hits, future_stuns, future_items = self._predict_future_effects(act, info, objects,
                                                   condition_check, disable_others)

                step_hits += future_hits
                step_stuns += future_stuns
                self._state[discounted_hits] = future_hits
                self._state[discounted_stuns] = future_stuns
                self._state[discounted_items] = future_items
                self._state[already_active_name] = True

        else:
            self._state[already_active_name] = False

            step_hits = self._discount(step_hits, discounted_hits)
            step_stuns = self._discount(step_stuns, discounted_stuns)

            # discount items we already picked up with the boomerang
            discounted_items = self._state.get(discounted_items, None)
            if discounted_items and items_obtained:
                for item in discounted_items.copy():
                    if item in items_obtained:
                        items_obtained.remove(item)
                        discounted_items.remove(item)

        return step_hits, step_stuns, items_obtained

    def _discount(self, curr_total, name):
        to_discount = self._state.get(name, 0)
        if to_discount and curr_total:
            discount = min(to_discount, curr_total)
            to_discount -= discount

            self._state[to_discount] = to_discount
            curr_total -= discount

        return curr_total

    def _predict_future_effects(self, act, info, objects, should_continue, disable_others):
        # pylint: disable=too-many-locals
        unwrapped = self.env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        # disable beams, bombs, or other active damaging effects until the current one is resolved
        disable_others(data)

        start_enemies = list(objects.enumerate_enemy_ids())
        start_health = {x: objects.get_obj_health(x) for x in start_enemies}
        unstunned_enemies = [eid for eid in start_enemies if objects.get_obj_stun_timer(eid) == 0]

        item_timers = {}
        for item in objects.enumerate_item_ids():
            item_timers[item] = objects.get_obj_timer(item)

        # Step over until should_continue is false, or until we left this room or hit a termination condition.
        # Update info at each iteration.
        location = (info['level'], info['location'])

        frames = 1
        while should_continue(info) and not is_mode_death(info['mode']) and \
                location == (info['level'], info['location']):
            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            _, _, terminated, truncated, info = unwrapped.step(act)
            frames += 1
            if terminated or truncated:
                break

        objects = ZeldaObjectData(unwrapped.get_ram())

        # check stun
        stuns = 0
        for enemy in unstunned_enemies:
            if objects.get_obj_stun_timer(enemy):
                stuns += 1

        # check health
        hits = 0
        end_health = {x: objects.get_obj_health(x) for x in objects.enumerate_enemy_ids()}
        for enemy in start_enemies:
            start = start_health.get(enemy, 0)
            end = objects.get_obj_health(enemy)

            if enemy not in end_health or end < start:
                hits += 1

        # check items
        items_obtained = []
        remaining_items = list(objects.enumerate_item_ids())
        for item, timer in item_timers.items():
            if frames < timer and item not in remaining_items:
                items_obtained.append(item)

        unwrapped.em.set_state(savestate)
        return hits, stuns, items_obtained

    def _handle_future_hits(self, act, info, objects, step_hits, name, condition_check, disable_others):
        info[name] = 0

        already_active_name = name + '_already_active'
        discounted_hits = name + '_discounted_hits'

        if condition_check(info):
            already_active = self._state.get(already_active_name, False)
            if not already_active:
                # check if beams will hit something
                future_hits = self._predict_future_hits(act, info, objects, condition_check, disable_others)
                info[name] = future_hits

                # count the future hits now, discount them from the later hit
                step_hits += future_hits

                self._state[discounted_hits] = future_hits
                self._state[already_active_name] = True

        else:
            # If we got here, either beams aren't active at all, or we stepped past the end of
            # the beams.  Make sure we are ready to process them again, and discount any kills
            # we found.
            self._state[already_active_name] = False
            step_hits = self._discount(step_hits, discounted_hits)

        return step_hits

    def _predict_future_hits(self, act, info, objects, should_continue, disable_others):
        # pylint: disable=too-many-locals
        unwrapped = self.env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        # disable beams, bombs, or other active damaging effects until the current one is resolved
        disable_others(data)

        start_enemies = list(objects.enumerate_enemy_ids())
        start_health = {x: objects.get_obj_health(x) for x in start_enemies}

        # Step over until should_continue is false, or until we left this room or hit a termination condition.
        # Update info at each iteration.
        location = (info['level'], info['location'])

        while should_continue(info) and not is_mode_death(info['mode']) and \
                location == (info['level'], info['location']):
            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            _, _, terminated, truncated, info = unwrapped.step(act)
            if terminated or truncated:
                break

        hits = 0

        objects = ZeldaObjectData(unwrapped.get_ram())
        end_health = {x: objects.get_obj_health(x) for x in objects.enumerate_enemy_ids()}
        for enemy in start_enemies:
            start = start_health.get(enemy, 0)
            end = objects.get_obj_health(enemy)

            if enemy not in end_health or end < start:
                hits += 1

        unwrapped.em.set_state(savestate)
        return hits

    def _set_beams_only(self, data):
        self._set_only(data, 'beam_animation')

    def _set_bomb1_only(self, data):
        self._set_only(data, 'bomb_or_flame_animation')

    def _set_bomb2_only(self, data):
        self._set_only(data, 'bomb_or_flame_animation2')

    def _set_arrow_only(self, data):
        self._set_only(data, 'arrow_magic_animation')

    def _set_boomerang_only(self, data):
        self._set_only(data, 'bait_or_boomerang_animation')

    def _set_only(self, data, name):
        all_names = ['beam_animation', 'bomb_or_flame_animation', 'bomb_or_flame_animation2',
                     'bait_or_boomerang_animation', 'arrow_magic_animation']

        assert name in all_names
        for n in all_names:
            if n != name:
                data.set_value(n, 0)
