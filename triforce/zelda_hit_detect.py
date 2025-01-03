"""
Performs all hit-detection for the game.  This fills entries in the info dictionary such as 'step-hit'.
"""

import gymnasium as gym

from .zelda_game import ZeldaObjectData, AnimationState, is_mode_death, \
    get_bomb_state, get_beam_state, get_boomerang_state, get_arrow_state, ITEM_MAP, is_spawn_state_dying

class HitsDetected:
    """Contains information about hits that occurred during a step."""
    def __init__(self):
        self.hits = 0
        self.damage = 0
        self.stuns = 0
        self.items = []

class ZeldaHitDetect(gym.Wrapper):
    """
    Determines whether attacks hit and whether link picked up an item.  Fills in:
    info['step_damage']
    info['step_hits']
    info['step_stuns']
    info['step_items']
    """

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

        # Only check hits if we didn't move room locations
        if info['new_location']:
            self._prev_health = None
            self._state.clear()
            result = None

        else:
            result = self._get_step_hits(action, info)

        info['step_damage'] = result.damage if result else 0
        info['step_hits'] = result.hits if result else 0
        info['step_stuns'] = result.stuns if result else 0
        info['step_items'] = result.items if result else []

        return obs, rewards, terminated, truncated, info

    def _get_step_hits(self, act, info):
        detected = HitsDetected()

        # capture enemy health data
        objects = info['objects']
        curr_enemy_health = self._get_live_enemy_health(objects)
        curr_dmg, curr_hits = self._compare_health(self._prev_health, curr_enemy_health)
        detected.hits += curr_hits
        detected.damage += curr_dmg

        # capture item information
        curr_items = {x: objects.get_obj_timer(x) for x in objects.enumerate_item_ids()}


        # check if we picked up any items
        frames_elapsed = info['total_frames'] - self._last_frame
        for item, timer in self._prev_items.items():
            if frames_elapsed < timer and item not in curr_items:
                detected.items.append(ITEM_MAP[objects.get_object_id(item)])

        # check if beams, bombs, arrows, etc are active and if they will hit in the future,
        # as we need to count them as rewards/results of this action so the model trains properly

        # Beams and bombs can only damage, not pick up items or stun.  (The sword swing can pick up items,
        # but that will occur within a single action and detected the same way as link walking over one.)
        self._handle_future_hits(detected, act, info, objects,  'beam_hits',
                                get_beam_state, self._set_beams_only)
        self._handle_future_hits(detected, act, info, objects, 'bomb1_hits',
                                lambda st: get_bomb_state(st, 0), self._set_bomb1_only)
        self._handle_future_hits(detected, act, info, objects, 'bomb2_hits',
                                lambda st: get_bomb_state(st, 1), self._set_bomb2_only)

        # arrows can pick up items but not stun
        self._handle_future_effects(detected, act, info, objects, 'arrow_hits',
                                    get_arrow_state,
                                    self._set_arrow_only)

        # boomerangs can stun, kill, and pick up items
        self._handle_future_effects(detected, act, info, objects, 'boomerang_hits',
                                    get_boomerang_state,
                                    self._set_boomerang_only)

        self._prev_health = curr_enemy_health
        self._prev_items = curr_items
        self._last_frame = info['total_frames']
        return detected

    def _get_live_enemy_health(self, objects):
        result = {}
        for eid in objects.enumerate_enemy_ids():
            health = objects.get_obj_health(eid)
            spawn_state = objects.get_obj_spawn_state(eid)
            if is_spawn_state_dying(spawn_state):
                continue

            if health == 0 and spawn_state == 0:
                # enemy is alive but doesn't have a health value
                health = True

            if health:
                result[eid] = health

        return result

    def _compare_health(self, prev_health, curr_health):
        dmg = 0
        hits = 0

        if prev_health:
            for eid, health in prev_health.items():
                if eid in curr_health:
                    # If we have a health value, calculate the damage.  If we don't have a health value,
                    # then we rely on whether the value is in the dictionary to determine if the enemy is alive.
                    if health is not True:
                        diff = health - curr_health[eid]
                        if diff > 0:
                            hits += 1
                            dmg += diff
                else:
                    hits += 1
                    dmg += health if health is not True else 1

        return dmg, hits

    def _capture_items(self, objects):
        items = {}
        for item in objects.enumerate_item_ids():
            items[item] = objects.get_obj_timer(item)

    def _handle_future_effects(self, detected, act, info, objects, name, get_animation, disable_others):
        already_active_name = name + '_already_active'
        discounted_damage = name + '_discounted_damage'
        discounted_hits = name + '_discounted_hits'
        discounted_stuns = name + '_discounted_stuns'
        discounted_items = name + '_discounted_items'

        # check if boomerang is active and if it will hit in the future
        if get_animation(info) == AnimationState.ACTIVE:
            if not self._state.get(already_active_name, False):
                result = self._predict_future_effects(act, info, objects, get_animation, disable_others)

                future_damage, future_hits, future_stuns, future_items = result

                detected.damage += future_damage
                detected.hits += future_hits
                detected.stuns += future_stuns
                detected.items.extend(future_items)

                self._state[discounted_damage] = future_damage
                self._state[discounted_hits] = future_hits
                self._state[discounted_stuns] = future_stuns
                self._state[discounted_items] = future_items
                self._state[already_active_name] = True

        else:
            self._state[already_active_name] = False

            detected.damage = self._discount(detected.damage, discounted_damage)
            detected.hits = self._discount(detected.hits, discounted_hits)
            detected.stuns = self._discount(detected.stuns, discounted_stuns)

            # discount items we already picked up with the boomerang
            discounted_items = self._state.get(discounted_items, None)
            if discounted_items and detected.items:
                for item in discounted_items.copy():
                    if item in detected.items:
                        detected.items.remove(item)
                        discounted_items.remove(item)

    def _discount(self, curr_total, name):
        to_discount = self._state.get(name, 0)
        if to_discount and curr_total:
            discount = min(to_discount, curr_total)
            to_discount -= discount

            self._state[to_discount] = to_discount
            curr_total -= discount

        return curr_total

    def _predict_future_effects(self, act, info, objects, get_animation, disable_others):
        # pylint: disable=too-many-locals
        unwrapped = self.env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        # disable beams, bombs, or other active damaging effects until the current one is resolved
        disable_others(data)

        start_health = self._get_live_enemy_health(objects)
        unstunned_enemies = [eid for eid in start_health if objects.get_obj_stun_timer(eid) == 0]

        item_timers = {}
        for item in objects.enumerate_item_ids():
            item_timers[item] = objects.get_obj_timer(item)

        # Step over until should_continue is false, or until we left this room or hit a termination condition.
        # Update info at each iteration.
        location = (info['level'], info['location'])

        frames = 1
        while get_animation(info) != AnimationState.INACTIVE and not is_mode_death(info['mode']) and \
                location == (info['level'], info['location']):
            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            _, _, terminated, truncated, info = unwrapped.step(act)
            frames += 1
            if terminated or truncated:
                break

        # Update data
        objects = ZeldaObjectData(unwrapped.get_ram())

        # check stun
        stuns = 0
        for enemy in unstunned_enemies:
            if objects.get_obj_stun_timer(enemy):
                stuns += 1

        # check health
        end_health = self._get_live_enemy_health(objects)
        dmg, hits = self._compare_health(start_health, end_health)

        # check items
        items_obtained = []
        remaining_items = list(objects.enumerate_item_ids())
        for item, timer in item_timers.items():
            if frames < timer and item not in remaining_items:
                item_id = objects.get_object_id(item)
                items_obtained.append(ITEM_MAP[item_id])

        unwrapped.em.set_state(savestate)
        return dmg, hits, stuns, items_obtained

    def _handle_future_hits(self, detected, act, info, objects, name, get_animation, disable_others):
        info[name] = 0

        already_active_name = name + '_already_active'
        discounted_damage = name + '_discounted_damage'
        discounted_hits = name + '_discounted_hits'

        if get_animation(info) == AnimationState.ACTIVE:
            already_active = self._state.get(already_active_name, False)
            if not already_active:
                # check if beams will hit something
                dmg, hits = self._predict_future_hits(act, info, objects, get_animation, disable_others)

                # count the future hits now, discount them from the later hit
                detected.hits += hits
                detected.damage += dmg

                self._state[discounted_hits] = hits
                self._state[discounted_damage] = dmg
                self._state[already_active_name] = True

        else:
            # If we got here, either beams aren't active at all, or we stepped past the end of
            # the beams.  Make sure we are ready to process them again, and discount any kills
            # we found.
            self._state[already_active_name] = False
            detected.hits = self._discount(detected.hits, discounted_hits)
            detected.damage = self._discount(detected.damage, discounted_damage)

    def _predict_future_hits(self, act, info, objects, get_animation, disable_others):
        # pylint: disable=too-many-locals
        unwrapped = self.env.unwrapped
        savestate = unwrapped.em.get_state()
        data = unwrapped.data

        # disable beams, bombs, or other active damaging effects until the current one is resolved
        disable_others(data)

        start_health = self._get_live_enemy_health(objects)

        # Step over until the animation is no longer active, or until we left this room or hit a termination condition.
        # Update info at each iteration.
        location = (info['level'], info['location'])

        while get_animation(info) != AnimationState.INACTIVE  and not is_mode_death(info['mode']) and \
                location == (info['level'], info['location']):
            data.set_value('hearts_and_containers', 0xff) # make sure we don't die

            _, _, terminated, truncated, info = unwrapped.step(act)
            if terminated or truncated:
                break

        objects = ZeldaObjectData(unwrapped.get_ram())
        end_health = self._get_live_enemy_health(objects)
        dmg, hits = self._compare_health(start_health, end_health)

        unwrapped.em.set_state(savestate)
        return dmg, hits

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
