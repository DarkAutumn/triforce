"""
Performs all hit-detection for the game.  This fills entries in the info dictionary such as 'step-hit'.
"""

import gymnasium as gym

from .zelda_game import ZeldaObjectData, get_beam_state, AnimationState, get_bomb_state, is_mode_death

class ZeldaHitDetect(gym.Wrapper):
    """Interprets the game state and produces more information in the 'info' dictionary."""
    def __init__(self, env):
        super().__init__(env)
        self._prev_health = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_health = None
        return obs, info


    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)

        if info['new_location']:
            self._prev_health = None
            self._clear_variables('beam_hits')
            self._clear_variables('bomb1_hits')
            self._clear_variables('bomb2_hits')
            info['step_hits'] = 0
        else:
            # Only check hits if we didn't move room locations
            info['step_hits'] = self._get_step_hits(action, info)

        return obs, rewards, terminated, truncated, info

    def _get_step_hits(self, act, info):
        step_hits = 0

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

        # check if beams, bombs, arrows, etc are active and if they will hit in the future,
        # as we need to count them as rewards/results of this action so the model trains properly
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'beam_hits',
                                    lambda st: get_beam_state(st) == AnimationState.ACTIVE, self._set_beams_only)
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'bomb1_hits',
                                    lambda st: get_bomb_state(st, 0) == AnimationState.ACTIVE, self._set_bomb1_only)
        step_hits = self._handle_future_hits(act, info, objects, step_hits, 'bomb2_hits',
                                    lambda st: get_bomb_state(st, 1) == AnimationState.ACTIVE, self._set_bomb2_only)

        self._prev_health = curr_enemy_health
        return step_hits


    def _clear_variables(self, name):
        self._clear_item(name + '_already_active')
        self._clear_item(name + '_discounted_hits')

    def _clear_item(self, name):
        if name in self.__dict__:
            del self.__dict__[name]

    def _handle_future_hits(self, act, info, objects, step_hits, name, condition_check, disable_others):
        info[name] = 0

        already_active_name = name + '_already_active'
        discounted_hits = name + '_discounted_hits'

        if condition_check(info):
            already_active = self.__dict__.get(already_active_name, False)
            if not already_active:
                # check if beams will hit something
                future_hits = self._predict_future(act, info, objects, condition_check, disable_others)
                info[name] = future_hits

                # count the future hits now, discount them from the later hit
                step_hits += future_hits

                self.__dict__[discounted_hits] = future_hits
                self.__dict__[already_active_name] = True

        else:
            # If we got here, either beams aren't active at all, or we stepped past the end of
            # the beams.  Make sure we are ready to process them again, and discount any kills
            # we found.
            self.__dict__[already_active_name] = False

            # discount hits if we already counted as beam hits
            discounted_hits = self.__dict__.get(discounted_hits, 0)
            if discounted_hits and step_hits:
                discount = min(discounted_hits, step_hits)
                discounted_hits -= discount

                self.__dict__[discounted_hits] = discounted_hits
                step_hits -= discount

        return step_hits

    def _predict_future(self, act, info, objects, should_continue, disable_others):
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
        data.set_value('bomb_or_flame_animation', 0)
        data.set_value('bomb_or_flame_animation2', 0)

    def _set_bomb1_only(self, data):
        data.set_value('beam_animation', 0)
        data.set_value('bomb_or_flame_animation2', 0)

    def _set_bomb2_only(self, data):
        data.set_value('beam_animation', 0)
        data.set_value('bomb_or_flame_animation1', 0)
