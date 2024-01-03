from .zelda_memory import ZeldaMemory
from .constants import *
from .zelda_environment import *


import gym
gym.envs.registration.register(id='Zelda-NoMenu-v0', entry_point='gym_zelda_nomenu:ZeldaNoMenuEnv', nondeterministic=True)
gym.envs.registration.register(id='Zelda-SmartItems-v0', entry_point='gym_zelda_nomenu:ZeldaSmartItemsEnv', nondeterministic=True)
