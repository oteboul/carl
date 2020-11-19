"""
To test a trained agent on an environment from the carl/ directory run

python3 -m scripts.run_tournament
"""
import numpy as np
import argparse

from carl.circuit import Circuit
from carl.tournament import TournamentEnvironment, Tournament


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_laps', type=int, default=3)
    args = parser.parse_args()
    n_points = 5
    points = [(0, 0), (6, 0)]
    for _ in range(n_points):
        points.append(tuple(np.array(points[-1])+ (np.random.randint(-4, 5), np.random.randint(-4, 5))))
    print(points)
    points = [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)]
    circuit = Circuit(points, width=0.3)
    env = TournamentEnvironment(circuit, render=True, laps=args.num_laps)
    tournament = Tournament(env, 10000)
    tournament.run()
    print('\n'.join(map(str, tournament.scores)))
