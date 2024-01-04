from .zelda_memory import ZeldaMemory
from .zelda_environment import *
from .rewards import ZeldaScoreBasic, ZeldaScoreDungeon

import gym
gym.envs.registration.register(id='Zelda-NoMenu-v0', entry_point='gym_zelda_nomenu:ZeldaNoMenuEnv', nondeterministic=True)
gym.envs.registration.register(id='Zelda-SmartItems-v0', entry_point='gym_zelda_nomenu:ZeldaSmartItemsEnv', nondeterministic=True)