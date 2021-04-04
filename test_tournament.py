import os, sys, math
import tensorflow as tf
from learnrl import Agent, Playground, Callback

from carl.circuit import generate_circuit
from carl.environment import Environment

from models.DQN import DQNAgent

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
        self.t = 0

    def on_step_end(self, step, logs):
        self.t += logs['dt_step']

    def on_episode_end(self, episode, logs):
        env = self.playground.env

        progression = env.circuit.laps[0] + env.circuit.progression[0]
        crashed = env.cars.crashed[0]

        score = progression if crashed else 2 + math.exp(- self.t / 10)
        print(f"score:{score}")

# circuit_points = generate_circuit(difficulty=16)
circuit_points = [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)]
model = sys.argv[1]
filenames = [model]

n_agents = len(filenames)
env = Environment(circuit_points, n_agents, action_type='discrete')
agents = [DQNAgent(env.action_space) for _ in range(n_agents)]

for agent, filename in zip(agents, filenames):
    filepath = os.path.join('models', filename)
    agent.load(filepath)

multi_agent = MultiAgent(agents)
pg = Playground(env, multi_agent)
pg.test(1, render=False, verbose=1, logger=ScoreCallback())
