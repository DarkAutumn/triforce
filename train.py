import tqdm
import gym
import gym_zelda_nomenu

#env = gym.make("Zelda-SmartItem-v0")
env = gym_zelda_nomenu.ZeldaSmartItemEnv()
env.reset(options={"random_delay" : False})

env.zelda_memory.sword = 1

while True:
    state, reward, done, info = env.step(0)
    if done:
        env.reset()

    env.render()

