import os

import numpy as np
from learnrl import Agent, Playground

from carl.utils import generate_circuit, teams_from_csv
from carl.environment import Environment
from carl.agents.tensorflow.DDPG import DDPGAgent
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
    [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
    [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
    [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)],
    [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
     (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
    [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
     (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
]

# teams = teams_from_csv(
#     models_path = os.path.join('models', 'DDPG', 'challenge_ddpg'),
#     csv_path = os.path.join('models', 'DDPG', 'challenge_ddpg', 'teams.csv')
# )

# modelnames, filepaths = [], []

# for team in teams:
#     team_modelnames, team_filepaths = teams[team]
#     team_modelnames = [f"{team}({modelname})" for modelname in team_modelnames]
#     modelnames += team_modelnames
#     filepaths += team_filepaths

filenames = os.listdir(os.path.join('models', 'DDPG'))
filepaths = [os.path.join('models', 'DDPG', filename) for filename in filenames]
modelnames = [filename.split('.')[0] for filename in filenames]

n_agents = len(modelnames)
env = Environment(circuits, n_agents, names=modelnames,
    action_type='continueous', n_sensors=9, fov=np.pi*220/180)

agents = [DDPGAgent(env.action_space) for _ in range(n_agents)]
for agent, filepath in zip(agents, filepaths):
    agent.load(filepath)

multi_agent = MultiAgent(agents)
playground = Playground(env, multi_agent)
scorecallback = ScoreCallback()

playground.test(len(circuits), callbacks=[scorecallback])
