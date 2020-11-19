"""
To test a trained agent on an environment from the carl/ directory run

python3 -m scripts.run_test
"""

import argparse
import os.path
import numpy as np

from carl.agent import DQLAgent
from carl.circuit import Circuit
from carl.environment import Environment

def generateCircuitPoints(n_points=16, difficulty=0, circuit_size=(5, 2)):
    n_points = min(25, n_points)
    angles = [-np.pi/4 + 2*np.pi*k/n_points for k in range(3*n_points//4)]
    points = [(circuit_size[0]/2, 0.5), (3*circuit_size[0]/2, 0.5)]
    points += [(circuit_size[0]*(1+np.cos(angle)), circuit_size[1]*(1+np.sin(angle))) for angle in angles]
    for i, angle in zip(range(n_points), angles):
        rd_dist = 0
        if difficulty > 0:
            rd_dist = min(circuit_size) * np.random.vonmises(mu=0, kappa=32/difficulty)/np.pi
        points[i+2] = tuple(np.array(points[i+2]) + rd_dist*np.array([np.cos(angle), np.sin(angle)]))
    return points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--model', type=str, default=os.path.join('models', 'weights.h5'))
    args = parser.parse_args()

    agent = DQLAgent(gamma=args.gamma, max_steps=args.max_steps)
    points = generateCircuitPoints(n_points=16, difficulty=32, circuit_size=(5, 3))
    # points = [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)]
    circuit = Circuit(points, width=0.3)
    env = Environment(circuit, render=True)
    if agent.load(args.model):
        name = os.path.basename(args.model)
        agent.run_once(env, train=False, greedy=True, name=name[:-3])
        print("{:.2f} laps in {} steps".format(
            circuit.laps + circuit.progression, args.max_steps))
