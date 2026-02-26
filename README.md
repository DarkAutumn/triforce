# Triforce - a deep learning agent for The Legend of Zelda (nes)

This repo contains code to train neural networks to play The Legend of Zelda.  It also includes some pretrained models that can play the game from the start through the end of the first dungeon (though often it doesn't play as well as I might hope).

## Getting Started

The main branch is pretty unstable/may not work correctly.  The [Original Design Branch](https://github.com/DarkAutumn/triforce/tree/original-design/) has a working copy.

This repo does *not* include a copy of the Legend of Zelda rom.  You must find your own copy of the rom and place it (uncompressed) at `triforce/custom_integrations/Zelda-NES/rom.nes`.

This project requires Python 3.12 (stable-retro does not support 3.13+).  On Ubuntu you may also need `libgl1-mesa-glx` for `run.py`.

This repo does include a set of pre-trained models.  Once the Zelda rom is in place, simply run `./run.py` to see the model play the game up through the end of the first dungeon.  (It does not usually win.)

![run.py output](./img/run.png)

Here's a video of the agent [beating the first dungeon](https://www.youtube.com/watch?v=yERh3IJ54dU), which was trained with 5,000,000+ steps. At 38 seconds, it learned that it's invulnerable at the screen edge, it exploits that to avoid damage from a projectile. At 53 seconds it steps up to avoid damage, even though it takes a -0.06 penalty for moving the wrong way (taking damage would be a larger penalty.) At 55 seconds it walks towards the rock projectile to block it. And so on, lots of little things the model does is easy to miss if you don't know the game inside and out.

As a TLDR, [here's an early version of my new (single) model](https://youtu.be/3AJXfBnmgVk). This doesn't make it quite as far, but if you watch closely it's combat is already *far* better, and is only trained on 320,000 steps (~6% less than the first model's training steps).

This is pretty far along from my [very first model](https://www.youtube.com/watch?v=KXPMwehTOf0).

## Architecture

My first pass through this project was basically "imagine playing Zelda with your older sibling telling you where to go and what to do". I give the model an objective vector which points to where I want it to go on the screen (as a bird flies, the agent still had to learn path finding to avoid damage and navigate around the map). This includes either point at the nearest enemy I want it to kill or a NSEW vector if it's supposed to move to the next room.

Due a few limitations with stable-baselines (especially around action masking), I ended up training unique models for traversing the overworld vs the dungeon (since they have entirely different tilesets). I also trained a different model for when we have sword beams vs not. In the video above you can see what model is being used onscreen.

In my current project I've removed this objective vector as it felt too much like cheating. Instead I give it a one-hot encoded objective (move north to the next room, pickup items, kill enemies, etc). So far it's working quite well without that crutch. The new project also does a much better job of combat even without multiple models to handle beams vs not.

## Observation/Action Space

**Image** - The standard neural network had a really tough time being fed the entire screen. No amount of training seemed to help. I solved this by creating a [viewport around Link](https://github.com/DarkAutumn/triforce/blob/dea241219ff17b386e368bc25adfbc171207888a/notebooks/torch_viewport.ipynb) that keeps him centered. This REALLY helped the model learn.

I also had absolutely zero success with stacking frames to give Link a way to see enemy/projectile movement. The model simply never trained with stable-baselines when I implemented frame stacking and I never figured out why. I just added it to my current neural network and it seems to be working...

Though my early experiments show that giving it 3 frames (skipping two in between, so frames curr, curr-3, curr-6) doesn't *really* give us that much better performance. It might if I took away some of the vectors. We'll see.

**Vectors** - Since the model cannot see beyond its little viewport, I gave the model a vector to the closest item, enemy, and projectile onscreen. This made it so the model can shoot enemies across the room outside of its viewport. My new model gives it multiple enemies/items/projectiles and I plan to try to use an attention mechanism as part of the network to see if I can just feed it all of that data.

**Information** - It also gets a couple of one-off datapoints like whether it currently has sword beams. The new model also gives it a "source" room (to help better understand dungeons where we have to backtrack), and a one-hot encoded objective.

**Action Space**

My original project just has a few actions, 4 for moving in the cardinal directions and 4 for attacking in each direction (I also added bombs but never spent any time training it). I had an idea to use masking to help speed up training. I.E. if link bumps into a wall, don't let him move in that direction again until he moves elsewhere, as the model would often spend an entire memory buffer running headlong straight into a wall before an update...better to do it once and get a huge negative penalty which is essentially the same result but faster.

Unfortunately SB made it really annoying architecturally to pass that info down to the policy layer. I could have hacked it together, but eventually I just reimplemented PPO and my own neural network so I could properly mask actions in the new version. For example, when we start training a fresh model, it cannot attack when there aren't enemies on screen and I can disallow it from leaving certain areas.

The new model actually understands splitting swinging the sword short range vs firing sword beams as two different actions, though I haven't yet had a chance to fully train with the split yet.

**Frameskip/Cooldowns** - In the game I don't use a fixed frame skip for actions. Instead I use the internal ram state of game to know when Link is animation locked or not and only allow the agent to take actions when it's actually possible to give meaningful input to the game. This greatly sped up training. We also force movement to be between tiles on the game map. This means that when the agent decides to move it loses control for longer than a player would...a player can make more split second decisions. This made it easier to implement movement rewards though and might be something to clean up in the future.

### The Guts of the Project

[make_zelda_env](triforce/zelda_env.py) is a function which creates a Zelda retro environment with all the appropriate wrappers.  Most uses of stable-retro are confined to this one file/function (with some exceptions where it made sense).

[FrameSkipWrapper](triforce/frame_skip_wrapper.py) ensures that the agent only acts when it's actually possible to affect the outcome of the game.  (I.E. not framelocked.)

[ZeldaGame](triforce/zelda_game.py), [Link](triforce/link.py), [Enemy](triforce/enemy.py), etc all build an object model over the game so the rest of the code can interact with what arrows, sword, health link has, etc.

[StateChangeWrapper](triforce/state_change_wrapper.py) keeps track of the change of state between one action and the next.  This also does some complicated work to ensure that when Link uses actions that will impact later (sword beams, bombs, etc) the frame he uses it on will get the rewards by running the simulation forward, then rewinding time,

[ZeldaActionSpace](triforce/action_space.py) defines the actions that can be taken.  The project supports using all equipment but the current models are only trained on movement and sword until it the kinks are ironed out.

[ObservationWrapper](triforce/observation_wrapper.py) converts from the full screen into an observation.  In this case, the last N frames (skipping some) are modified to be a viewport around Link so that the shared nature CNN works properly.  We also add some one-hot encoded objectives and some vectors to the nearest objects an enemies.

[ScenarioWrapper](triforce/scenario_wrapper.py) keeps track of the scenario we are running or training on, and all of the critics and end conditions associated with it.

[Objectives](triforce/objectives.py) sets the current goal of the model.  The overall idea of the project is that the model should roughly know where dungeons and items exist (like you have a game guide...few people beat zelda1 without one), but doesn't fully spoon feed the model what to do.  It's a work in progress.


### Critics and End Conditions

[ZeldaCritic](triforce/critics.py) classes define rewards.  Scenarios have exactly one critic associated with them, and that is used to reward the model for good gameplay and discourage it from bad gameplay.  This is implemented by giving the critics the game state prior to an action it took, the game state after that action, and it fills in a dictionary of rewards.  We use a reward dictionary (not just a number) so that we can debug rewards to figure out issues with training.

Critics also define a **`progress`** that is separate from rewards.  Where rewards are all about how well the model is *playing the game*, and are used to train the model, the `progress` is used to determine how well the model is *completing the scenario*.  In particular, the progress counts things like health lost and how far the model progressed through rooms.   We don't use progress when training the model.  Instead, progress is used during evaluation to make sure the model doesn't just get a high reward value but die in the first room or two.

[ZeldaEndCondition](triforce/end_conditions.py) classes track when the scenario is won, lost, or should be truncated (if the AI gets stuck in one place, for example).  Scenarios have multiple end conditions.

### Scenarios and Models

[triforce.json](triforce/triforce.json) this *defines* models and the scenarios used to train and evaluate them.

Models define what their action space is (for example, basic combat models do not use items like bombs, so the B button is entirely disabled for them.)  They define their training scenario and how many iterations should be used by default to train them.  Models also define a series of conditions and a priority for selecting them.  This project uses multiple models to play the game, so each model defines the equipment needed to use it (beams, bombs, etc) and the level that the model should be used (dungeon1, overworld, etc).  Lastly, it also defines a priority.  When multiple models match the criteria, priority determines which is picked.

Scenarios are used to train and evaluate models.  They define what critic, end conditions, and starting room(s) are used to train them.  They also allow us to change aspects of gameplay.  The `data` section can change RAM values of the game to do a variety of things.  The `fixed` section is like `data` but those values are reset every frame.  This is how we train the model which knows how to use sword beams.  No matter if that model takes damage, it is always set to full health.  For the no-beam model, we always set health lower than full so that it never has sword beams when training.

## Running and Debugging the Models

I develop on Linux, but in the past this project has worked fine on an up-to-date WSL2 Ubuntu 22 instance on Windows (the pygame based GUI also works).  I do not test on Windows though, so your experience may vary if I've accidently broken something.

Be sure to install dependencies:

    uv venv --python 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt

Be sure to place The Legend of Zelda rom at `triforce/custom_integrations/Zelda-NES/rom.nes`.

### Running and Debugging Scenarios

This repo includes a set of pre-trained models.  To run the full game simply use `run.py`.  By default it will run the full game, but you can specify a scenario with the first argument (e.g. `run.py dungeon1beams`).  You can specify the `--model-path` to be a location other than the checked in models, which is helpful to do after training (e.g. `run.py --model-path training/`).

    ./run.py
    ./run.py overworld1beams
    ./run.py overworld1nobeams --model-path training/

Keybindings for `run.py`:

* **u** - Uncap frame rate to make the game (hopefully) run faster.
* **q** - Quit.
* **p** - Pause.
* **c** - Continue (unpause).
* **n** - Run the next step.
* **o** - Show tile/pathfinding overlay
* **m** - Change between models.  Useful when pointing at a training directory.
* **F4** - Enable recording.  Recording files are dropped to recording/
* **F10** - Enable in-memory recording, only saving the recording if the scenario is successfully completed.  **DANGER**: This can eat 100gb+ of RAM.  I use this to capture videos of successful runs, but should not be used unless your machine has a massive amount of RAM.

As the program runs, per-step rewards are displayed on the right hand side.  Clicking on those individual rewards will cause the program to re-run the ZeldaCritic responsible for generating that reward and will print the values to the console.  This is used to debug rewards.  When you see a reward that doesn't make sense, use VS Code (or your favorite editor/debugger) to set a breakpoint on the function and click the reward in the UI to debug through it.

### Training Models

Training the model can be accomplished using `train.py`.  By default it will train all models, single-threaded, using the default number of iterations defined in `triforce.json`.  You can also specify which models to train, where to output the models, how many subprocesses to use, etc.

    ./train.py
    ./train.py overworld1beams overworld1nobeams
    ./train.py overworld1beams overworld1nobeams --iterations 7500000
    ./train.py overworld1beams overworld1nobeams --iterations 7500000 --parallel 8
    ./train.py overworld1beams overworld1nobeams --iterations 7500000 --parallel 8 --output ~/new_models/

By default, the new models will be placed in `training/`. Use `run.py --model-path training/` to test the new models.

### Evaluating Models

Rewards teach the model how to play, but it's the scoring function that tries to figure out how well the model is doing in completing the scenario.  It doesn't matter how many enemies Link kills or movements in the "wrong" direction as long as he completes the scenario without dying.  The `evaluate.py` module will run a given scenario 100 times (by default) and calculate the percentage of runs that were completed successfully, as well as the average progress and rewards.

After completing a training run, you should use `evaluate.py` and compare it to `models/evaluation.csv`.  You can run this in parallel (but it's graphics card RAM intensive, so keep the number low), and the number of episodes to run.  For example, if you trained the dungeon1beams and dungeon1nobeams models and saved them to the default training/ folder:

    usage: evaluate.py model-path scenarios

    ./evaluate.py training/ dungeon1beams dungeon1nobeams
    ./evaluate.py training/ dungeon1beams dungeon1nobeams --parallel 4
    ./evaluate.py training/ dungeon1beams dungeon1nobeams --episodes 250 --parallel 4

## Contributing

Contributions to this repo are more than welcome, just submit a pull request. If you are making large edits or changes that are more than just a bugfix or tweak, feel free to open an issue to discuss beforehand. That will make sure that a giant change won't be rejected after you get through it.

When submitting a PR, please be sure to run `pylint *py triforce/` and clean up/silence any errors.  It's ok if functions are slightly too long, have too many if statements, etc.  If it's possible to follow the pylint rules reasonably, please do so, if not then just make sure you aren't adding any new warnings.

If your change modifies how the model is rewarded (e.g. any modifications to [critics.py](triforce/critics.py)), please include the output of `evaluate.py` on a newly trained model with those changes, even if we aren't updating the checked in model.

You are also more than welcome to fork this project and take it in your own direction.  Have fun!
