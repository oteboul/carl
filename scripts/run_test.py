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
    args = parser.parse_args()

    env = Environment(
        num_sensors=args.num_sensors, crash_value=args.crash_value, render=True)

    agent = DQLAgent(
        state_size=args.num_sensors + 1, action_size=len(env.actions),
        gamma=args.gamma)
    if agent.load(args.model):
        agent.run_once(env, train=False, greedy=True)
