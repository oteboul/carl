"""
To test a trained agent on an environment from the carl/ directory run

python3 -m scripts.run_test
"""

import argparse

from src.agent import DQLAgent
from src.circuit import Circuit
from src.environment import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--crash_value', type=float, default=-3)
    parser.add_argument('--model', type=str)
    parser.add_argument('--gif', type=str, default='')
    args = parser.parse_args()

    agent = DQLAgent(gamma=args.gamma, max_steps=args.max_steps)

    circuit = Circuit([
        (-0.5, 0), (0.5, 0.5), (0.5, 1), (0, 2), (2.5, 2.5), (3, 0.5), (4.5, 1.),
        (6, 0.5), (6, -0.5), (5, -1), (5, -2), (0, -2), (-0.5, -0.5)],
        width=0.3)

    to_gif = args.gif != ''
    env = Environment(circuit, render=True, to_movie=to_gif)

    if agent.load(args.model):
        agent.run_once(env, train=False, greedy=True)
        if to_gif:
            env.ui.toGIF(args.gif)
