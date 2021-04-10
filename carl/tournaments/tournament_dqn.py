import os

import numpy as np
from learnrl import Agent, Playground

from carl.utils import generate_circuit
from carl.environment import Environment
from carl.agents.tensorflow.DQN import DQNAgent
from carl.agents.callbacks import ScoreCallback

class MultiAgent(Agent):

    def __init__(self, agents):
        self.agents = agents

    def act(self, observations, greedy=False):
        if len(self.agents) > 1:
            actions = []
            for obs, agent in zip(observations, self.agents):
                actions.append(agent.act(obs, greedy))
            return actions
        else:
            return self.agents[0].act(observations, greedy)

circuits = [
    # [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
    # [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
    # [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)],
    # [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
    #  (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
    # [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
    #  (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
    generate_circuit(n_points=50, difficulty=0),
    generate_circuit(n_points=20, difficulty=8),
    generate_circuit(n_points=20, difficulty=16),
    generate_circuit(n_points=20, difficulty=32),
    generate_circuit(n_points=30, difficulty=32),
]

# filenames = os.listdir(os.path.join('models', 'DQN', 'test_done'))
filenames = ['Beholder10.h5']
teamnames = [name.split('.')[0].capitalize() for name in filenames]

n_agents = len(filenames)
env = Environment(circuits, n_agents, names=teamnames,
    action_type='discrete', n_sensors=7, fov=np.pi*210/180)

agents = [DQNAgent(env.action_space) for _ in range(n_agents)]

for agent, filename in zip(agents, filenames):
    filepath = os.path.join('models', 'DQN', filename)
    agent.load(filepath)

multi_agent = MultiAgent(agents)
pg = Playground(env, multi_agent)
pg.test(3 * len(circuits), callbacks=[ScoreCallback()], verbose=2)
