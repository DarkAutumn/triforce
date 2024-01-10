# Zelda has a very complicated combat system.  This class is responsible for detecting when the
# agent has killed or injured an enemy.
#
# This consumes some state and produces 'total_kills' and 'total_injuries'.

import gymnasium as gym
from .zelda_game import is_mode_death, get_beam_state

class DamageDetector(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self._clear()

    def reset(self, **kwargs):
        self._clear()
        return self.env.reset(**kwargs)
    
    def _clear(self):
        self.beams_handled = False
        self.consume_kill = False
        self.consume_injury = False

        self.total_kills = 0
        self.last_kill_streak = 0
        self.total_injuries = 0

        self.damage_table = [0] * 12
        self.prev_location = (-1, -1)
        

    def step(self, act):
        obs, rewards, terminated, truncated, state = self.env.step(act)

        # do nothing if this is a new screen
        location = (state['level'], state['location'])
        if location != self.prev_location:
            self.prev_location = location
            self.beams_handled = False
            self.consume_kill = False
            self.consume_injury = False
            self.fill_damage_table(state, self.damage_table)
            return obs, rewards, terminated, truncated, state

        beam_state = get_beam_state(state)

        if self.last_kill_streak < state['kill_streak']:
            self.total_kills += state['kill_streak'] - self.last_kill_streak
            
            if beam_state == 2 and self.consume_kill:
                self.total_kills -= 1
                self.consume_kill = False

        if beam_state == 0:
            self.consume_kill = False
            self.beams_handled = False
            self.consume_injury = False

        elif beam_state == 1:
            if not self.beams_handled:
                kill, injury = self.did_beams_kill(act, state)
                self.consume_kill = kill
                self.consume_injury = injury
                if kill:
                    self.total_kills += 1
                if injury:
                    self.total_injuries += 1

        else:
            self.beams_handled = False

        # injuries are  hits that are not fatal
        prev_damage = self.damage_table.copy()
        self.fill_damage_table(state, self.damage_table)
        injuries = self.detect_injuries(prev_damage, self.damage_table)
        if injuries:
            if self.consume_injury:
                injuries -= 1
                self.consume_injury = False
                
            self.total_injuries += injuries

        # don't let anyone use kill_streak, they should use total_kills
        self.last_kill_streak = state['kill_streak']
        del state['kill_streak']
        state['total_kills'] = self.total_kills
        state['total_injuries'] = self.total_injuries
        
        return obs, rewards, terminated, truncated, state

    def detect_injuries(self, prev_health, curr_health):
        injuries = 0
        for x in range(12):
            prev = prev_health[x] >> 4
            curr = curr_health[x] >> 4
            if curr > 0 and curr < prev:
                injuries += 1

        return injuries
    
    def fill_damage_table(self, state, table):
        for x in range(12):
            table[x] = state[f"enemy_health_{x}"]

    def did_beams_kill(self, act, state):
        self.beams_handled = True
        savestate = self.env.em.get_state()
        original_kills = state['kill_streak']

        damage_table = [0] * 12
        self.fill_damage_table(state, damage_table)

        frames = 0
        beams = get_beam_state(state)
        while beams == 1 and not is_mode_death(state['mode']):
            frames += 1
            original_kills = state['kill_streak'] # update in case we get damaged and streak resets
            
            self.fill_damage_table(state, damage_table)
            obs, rewards, terminated, truncated, state = self.env.step(act)
            beams = get_beam_state(state)

        self.env.em.set_state(savestate)
        kills = state['kill_streak'] > original_kills

        # check damage
        prev_damage = damage_table.copy()
        self.fill_damage_table(state, damage_table)
        injuries = self.detect_injuries(prev_damage, damage_table) > 0

        return kills, injuries
            
