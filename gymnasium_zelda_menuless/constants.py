zelda_mode_gameplay = 5
zelda_mode_gameover = 0x11

import gymnasium
gymnasium.envs.registration.register(
    id='ZeldaMenuless-v0',
    entry_point='gymnasium_zelda_menuless:ZeldaEnv',
    nondeterministic=True,
)