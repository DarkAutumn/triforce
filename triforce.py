import argparse
import os

from triforce_lib import ZeldaScenario, ZeldaML, pygame_render

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments for training, testing, evaluating, or recording.")

    scenarios = ZeldaScenario.get_all_scenarios()

    parser.add_argument("action", choices=['train', 'evaluate', 'help', 'display'], help="Action to perform.")
    parser.add_argument("scenario", choices=scenarios, help="The scenario to run.")
    parser.add_argument("iterations", type=int, help="Number of iterations to run.")

    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel environments to run.")
    parser.add_argument("--render", action='store_true', help="Render the environment.")
    parser.add_argument("--color", action='store_true', help="Record the environment.")
    parser.add_argument("--frame-stack", type=int, default=3, help="Number of frames to stack in the observation.")
    parser.add_argument("--record", action='store_true', help="Whether to record playback or not.")
    parser.add_argument("--load", help="Load a specific saved model.")
    parser.add_argument("--debug-scenario", action='store_true', help="Debug the scenario by printing out rewards.")
    parser.add_argument("--ent-coef", type=float, default=0.00, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport', help="The kind of observation to use.")

    try:
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)
        parser.print_help()
        exit(0)


def main():
    # load environment variables
    base_dir = './models/'
    if 'TRIFORCE_MODEL_DIR' in os.environ:
        base_dir = os.environ['TRIFORCE_MODEL_DIR']

    # parse arguments
    args = parse_args()

    # create the agent and load the model

    render_mode = None
    if args.action == 'test' or args.action == 'evaluate':
        render_mode = 'human'
    elif args.action == 'display':
        render_mode = 'rgb_array'

    record = args.record
    debug_scenario = args.debug_scenario or args.action == 'evaluate'
    zelda_ml = ZeldaML(base_dir, args.scenario, args.frame_stack, args.color, args.parallel, record=record, render_mode=render_mode, verbose=args.verbose, debug_scenario=debug_scenario, ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)

    if args.load:
        zelda_ml.load(args.load)
    else:
        zelda_ml.load()

    try:
        if args.action == 'train' or args.action == 'learn':
            # learn automatically saves both the best model and the model at the end of training
            zelda_ml.learn(args.iterations, progress_bar=True)

        elif args.action == 'evaluate' or args.action == 'record':
            mean_reward, std_reward = zelda_ml.evaluate(args.iterations, deterministic=False, render=True)
            print(f'Mean reward: {mean_reward} +/- {std_reward}')

        elif args.action == 'display':
            pygame_render(zelda_ml)

    finally:
        zelda_ml.close()


if __name__ == '__main__':
    main()
