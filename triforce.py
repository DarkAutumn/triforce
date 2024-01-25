import argparse
import os

from triforce_lib import ZeldaScenario, ZeldaML, pygame_render

def parse_args():
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true', help="Record the environment.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack in the observation.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport', help="The kind of observation to use.")

    subparsers = parser.add_subparsers(dest='action', required=True)

    parser_train = subparsers.add_parser('train', aliases=['learn'])
    parser_train.add_argument('models', nargs='*', help='List of models to train')
    parser_train.add_argument("--output", type=str, help="Location to write to.")
    parser_train.add_argument("--render", action='store_true', help="Render the environment.")
    parser_train.add_argument("--iterations", type=int, default=-1, help="Number of iterations to run.")
    parser_train.add_argument("--parallel", type=int, default=1, help="Number of parallel environments to run.")

    parser_display = subparsers.add_parser('run', aliases=['show', 'display'])
    parser_display.add_argument('scenario', nargs=1, help='Scenario name')
    parser_display.add_argument('model_path', nargs=1, help='Model path')

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
    elif args.action in ['display', 'run', 'show']:
        render_mode = 'rgb_array'

    zelda_ml = ZeldaML(args.frame_stack, args.color, render_mode=render_mode, verbose=args.verbose, ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)

    if args.action in ['train', 'learn']:
        iterations = None if args.iterations <= 0 else args.iterations
        models = args.models if args.models else None
        zelda_ml.train(args.output, models, iterations, args.parallel)

    elif args.action in ['display', 'run', 'show']:
        pygame_render(zelda_ml, args.scenario[0], args.model_path[0])

if __name__ == '__main__':
    main()
