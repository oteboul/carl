import os

import numpy as np
from learnrl import Agent, Playground, Callback

from carl.utils import generate_circuit
from carl.environment import Environment
from carl.agents.tensorflow.DQN import DQNAgent

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

class RandomAgent(Agent):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observations, greedy=False):
        return self.action_space.sample()

class ScoreCallback(Callback):
    def __init__(self, **kwargs):
        self.step = 0

    def on_run_begin(self, logs):
        self.score = np.zeros(self.playground.env.n_cars)

    def on_step_end(self, step, logs):
        self.step += 1

    def on_episode_begin(self, step, logs):
        self.step = 0

    def on_episode_end(self, episode, logs):
        env = self.playground.env
        circuit = env.current_circuit

        progressions = circuit.laps + circuit.progression
        crashed = env.cars.crashed

        bonus = max(0, (2 - self.step / 200))
        score = np.where(crashed, progressions, 2 + bonus)
        self.score += score

    def on_run_end(self, logs):
        score = self.score
        if len(score) == 1:
            score = score[0]
        print(f"score:{score}")

circuits = [
    [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
    [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
    [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)],
    [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
     (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
    [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
     (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
    generate_circuit(n_points=25, difficulty=0),
    generate_circuit(n_points=20, difficulty=5),
    generate_circuit(n_points=15, difficulty=20),
    generate_circuit(n_points=20, difficulty=50),
    generate_circuit(n_points=20, difficulty=100),
]


filenames = os.listdir(os.path.join('models', 'DQN'))
teamnames = [name.split('.')[0].capitalize() for name in filenames]

n_agents = len(filenames)
env = Environment(circuits, n_agents, names=teamnames, action_type='discrete', n_sensors=7, fov=np.pi*210/180)
agents = [DQNAgent(env.action_space) for _ in range(n_agents)]

for agent, filename in zip(agents, filenames):
    filepath = os.path.join('models', 'DQN', filename)
    agent.load(filepath)

multi_agent = MultiAgent(agents)
pg = Playground(env, multi_agent)
pg.test(len(circuits), callbacks=[ScoreCallback()])
