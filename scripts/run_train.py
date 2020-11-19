"""
To train a DQL Agent to drive a car, from the carl/ directory run

python3 -m scripts.run_train
"""

import argparse, os

from carl.agent import DQLAgent
from carl.circuit import Circuit
from carl.environment import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The number of episodes
    parser.add_argument('--num_episodes', type=int, default=1000)
    # The max number of steps per episodes
    parser.add_argument('--max_steps', type=int, default=300)
    # The batch_size
    parser.add_argument('--minibatch_size', type=int, default=32)
    # The gamma of the model
    parser.add_argument('--gamma', type=float, default=1.0)
    # The learning_rate
    parser.add_argument('--learning_rate', type=float, default=0.1)
    # The path of the output
    parser.add_argument('--output', type=str, default='weights.h5')
    # Render the user interface every iteration without closing window
    parser.add_argument('--ui', type=str, default='true')
    # Render the user interface periodicly during training
    parser.add_argument('--rendering_period', type=int, default=0)
    # Change the circuit to be increasingly difficult
    parser.add_argument('--increasing_circuits', type=str, default='false')
    args = parser.parse_args()

    circuit = Circuit([(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)], width=0.3)

    render = args.ui.lower() != 'false'
    increasing_circuits = args.increasing_circuits.lower() == 'true'
    env = Environment(circuit=circuit, render=render)

    agent = DQLAgent(
        state_size=len(env.current_state), action_size=len(env.actions),
        gamma=args.gamma, learning_rate=args.learning_rate, max_steps=args.max_steps)

    agent.train(
        env, episodes=args.num_episodes, minibatch=args.minibatch_size,
        output=os.path.join('models', args.output), rendering_period=args.rendering_period, increasing_circuits=increasing_circuits)
