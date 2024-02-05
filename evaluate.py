#! /usr/bin/python

import argparse
import os
import pandas as pd
from tqdm import tqdm
from triforce_lib import ZeldaML, ZeldaModel, ZeldaScenario

def main(args):
    model_path = args.model_path[0] if args.model_path else os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    zelda_ml = ZeldaML(args.color, render_mode=None, verbose=args.verbose, ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)
    zelda_ml.load_models(model_path)

    columns = ['Model', 'Kind', 'Success%', 'Score', 'Total Reward', 'Rewards', 'Penalties']
    results = []


    for model in ZeldaModel.get_loaded_models():
        for i in range(len(model.models)):
            model_kind = model.model_kinds[i]
            loaded_model = model.models[i]

            print(f"Evaluating model: {model.name} ({model_kind})")

            model_result = []

            for ep in tqdm(range(args.episodes)):
                env = zelda_ml.make_env(ZeldaScenario.get(model.training_scenario), model.action_space, 1)
                obs, info = env.reset()

                episode_rewards = 0
                episode_penalties = 0
                episode_total_reward = 0
                episode_score = 0
                success = False

                terminated = truncated = False

                while not terminated and not truncated:
                    action, _ = loaded_model.predict(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_total_reward += reward

                    if 'score' in info:
                        episode_score = info['score']

                    if 'rewards' in info:
                        for kind, rew in info['rewards'].items():
                            rew_kind = kind.split('-', 1)[0]
                            if rew_kind == 'reward':
                                episode_rewards += abs(rew)
                            elif rew_kind == 'penalty':
                                episode_penalties -= abs(rew)
                            else:
                                raise ValueError(f"Unknown reward kind: {kind}")
                    
                    if 'end' in info:
                        success = info['end'].startswith("success")

                    model_result.append((ep, success, episode_score, episode_total_reward, episode_rewards, episode_penalties))

                if args.verbose:
                    print(f"Episode {ep}: {'Success' if success else 'Failure'} - Score: {episode_score} - Total Reward: {episode_total_reward} - Rewards: {episode_rewards} - Penalties: {episode_penalties}")

                env.close()

            success_rate = round(100 * sum([1 for x in model_result if x[1]]) / len(model_result), 1)
            score = round(sum([x[2] for x in model_result]) / len(model_result), 1)
            total_reward = round(sum([x[3] for x in model_result]) / len(model_result), 1)
            rewards = round(sum([x[4] for x in model_result]) / len(model_result), 1)
            penalties = round(sum([x[5] for x in model_result]) / len(model_result), 1)

            results.append((model.name, model_kind, success_rate, score, total_reward, rewards, penalties))

    data_frame = pd.DataFrame(results, columns=columns)
    print(data_frame.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true', help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport', help="The kind of observation to use.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test.")
    parser.add_argument("--parallel", type=bool, default=False, help="Use parallel environments to evaluate the models.")

    parser.add_argument('model_path', nargs=1, help='The director containing the models to evaluate')

    try:
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)
        parser.print_help()
        exit(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
