import argparse

from src.environment import Environment
from src.agent import DQLAgent


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
    args = parser.parse_args()

    env = Environment(num_sensors=args.num_sensors, render=True)
    agent = DQLAgent(
        state_size=args.num_sensors + 1, action_size=len(env.actions),
        gamma=args.gamma, epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay, learning_rate=args.learning_rate)
    agent.train(
        env, episodes=args.num_episodes, minibatch=args.minibatch_size)
