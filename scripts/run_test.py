"""
To test a trained agent on an environment from the carl/ directory run

python3 -m scripts.run_test
"""

import argparse

from src.environment import Environment
from src.agent import DQLAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sensors', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--crash_value', type=float, default=-3)
    parser.add_argument('--model', type=str)
    parser.add_argument('--gif', type=str, default='')
    args = parser.parse_args()

    to_gif = args.gif != ''
    env = Environment(
        num_sensors=args.num_sensors, render=True, to_movie=to_gif)

    agent = DQLAgent(
        state_size=args.num_sensors + 1, action_size=len(env.actions),
        gamma=args.gamma, max_steps=args.max_steps)
    if agent.load(args.model):
        agent.run_once(env, train=False, greedy=True)
        if to_gif:
            env.ui.toGIF(args.gif)
