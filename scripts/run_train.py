"""
To train a DQL Agent to drive a car, from the carl/ directory run

python3 -m scripts.run_train
"""

import argparse

from src.agent import DQLAgent
from src.circuit import Circuit
from src.environment import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sensors', type=int, default=5)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--minibatch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=1.00)
    parser.add_argument('--epsilon_decay', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--crash_value', type=float, default=-3)
    parser.add_argument('--output', type=str, default='weights.h5')
    parser.add_argument('--ui', type=str, default='true')
    args = parser.parse_args()

    circuit = Circuit(
        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)], width=0.3)

    render = args.ui.lower() != 'false'
    env = Environment(
        circuit=circuit, crash_value=args.crash_value, render=render)

    agent = DQLAgent(
        state_size=args.num_sensors + 1, action_size=len(env.actions),
        gamma=args.gamma, epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay, learning_rate=args.learning_rate,
        output=args.output)

    agent.train(
        env, episodes=args.num_episodes, minibatch=args.minibatch_size)
