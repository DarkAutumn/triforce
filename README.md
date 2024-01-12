# ZeldaML - a deep learning agent for The Legend of Zelda (nes)

Example program usage:

    triforce learn gauntlet 1000000

Example library usage:

    # or define your own
    scenario = 'gauntlet'

    # create the model using the PPO learning algorithm and train on the 'gauntlet' scenario
    zelda_ml = ZeldaML('./models/', scenario, 'ppo', framestack=3, color=False, render_mode='human')
    zelda_ml.load()

    zelda.train(iterations=100000, progress_bar=True)

    # or

    mean_reward, std_reward = zelda_ml.evaluate(args.iterations, render=True)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

Instead of passing a scenario name, you can define your own by inheriting from ZeldaScenario, ZeldaCritic, and ZeldaEndCondition.

## Architecture

This lib has a few concepts that are decoupled from each other:  Models, scenarios, and critics.

### ZeldaML

This piece of the library keeps track of what algorithm it's using, loading and saving the raw models, and any modifications to the observation space that is used (e.g. grayscale, frame stacking, etc), and the environment itself.

Even though the ZeldaML class is the entrypoint and takes in a scenario as a parameter, the class itself is completely decoupled from the scenario it is running.

Things which ZeldaML handles:

- Algorithm (ppo, a2c, etc)
- Observation space (grayscale, frame stacking)
- Model hyperparamters
- Saving/loading the underlying model

### ZeldaScenario

A scenario is a set of choices about how to train the agent.  ZeldaScenario objects are decoupled from ZeldaML.  These can be defined in user code.

Things which scenarios modify:

- How random or deterministic the start of the scenario is.
- How often an agent acts (implemented as frameskip ranges).
- The action space (what buttons or combinations are allowed).
- The starting state (rom .state savestate).
- A list of ZeldaCritic to ask for reward values.
- A list of ZeldaEndCondition to ask for termination/truncation conditions.

### ZeldaCritic

Critics define rewards.  They are often built for a particular scenario, but are decoupled in implementation.  You can use any critic with any scenario or learning model and the library will run, but you might get weird results.

Critics are specifically not gym wrappers.  They are an interface which can be called and tested separately from scenarios and training.  Critics are driven off of a restart/update_and_get_reward interface.  These names were deliberately chosen to be different from reset/step to not be confused with gym wrappers, even though they are similar in design.

The basic set of rewards are implemented in [ZeldaGameplayCritic](triforce_lib/critic.py).  This defines the rewards that *most* scenarios will want.  The indivdual rewards for each action can be set with the ZeldaGameplayCritic.*reward properties.

Critics:

- Take in gameplay state as a dictionary of names to memory values.
- Produce a reward between [-1, 1] inclusive for every call to update_and_get_reward.

### ZeldaEndCondition

End conditions define when a scenario is terminated or truncated.  The basic end condition set is defined in [ZeldaGameplayEndCondition](triforce_lib/end_condition.py) which ends the game if the Triforce of Power is obtained (after killing Ganon) or if the agent hits the Game Over screen.

### Gameplay state used for Critics/EndConditions

Do not directly modify [data.json](custom_integrations/Zelda-NES/data.json).  Instead, update [zelda_game_data.txt](triforce_lib/zelda_game_data.txt) and run [update_memory_definitions.py](../scripts/update_memory_definitions.py) to change what data is put into the dictionary fed to critics.
