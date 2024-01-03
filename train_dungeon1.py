import gym_zelda_nomenu

game = gym_zelda_nomenu.ZeldaSmartItemEnv()

# move to the first dungeon
game.reset_to_first_dungeon()
game.zelda_memory.sword = 1
game.zelda_memory.bombs = 2
game.save_state()

while True:
    state, reward, done, info = game.step(0)
    if done:
        game.reset()

    game.render()

