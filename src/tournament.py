import colorsys
import glob
import logging
import os.path
import random

from src.agent import DQLAgent
from src.environment import Environment


class TournamentEnvironment(Environment):
    """The tournament environment is a tournament, which on goes till n laps"""
    def __init__(self, circuit, render, laps):
        super().__init__(circuit, render)
        self.laps = laps

    def isEnd(self) -> bool:
        return not self.car.in_circuit() or self.circuit.laps >= self.laps


class Performance(object):
    def __init__(self, laps, steps, name):
        self.laps = laps
        self.steps = steps
        self.name = name

    def __lt__(self, other):
        return (self.laps, -self.steps) < (other.laps, -other.steps)

    def __repr__(self):
        return "{name}: {laps} laps in {steps} steps".format(**vars(self))


class Tournament(object):
    """The tournament itself: reads all the models from the folder and the cars
    on the circuit till they finish and sort their performance"""

    def __init__(self, env, max_num_steps, folder='models/'):
        self.env = env
        self.circuit = self.env.circuit
        self.max_num_steps = max_num_steps
        self.folder = folder
        self.agent = DQLAgent(max_steps=self.max_num_steps)

    def ranking(self):
        return "\n".join(map(
            lambda p: "{}. {}".format(p[0] + 1, p[1]), enumerate(self.scores)))

    @staticmethod
    def makeColor(h):
        return "#{:02x}{:02x}{:02x}".format(
            *map(lambda x: int(255*x), colorsys.hsv_to_rgb(h, 0.8, 0.8)))

    def save(self, filename='ranking.txt'):
        with open(filename, 'w') as fp:
            fp.write(self.ranking())

    def run(self):
        self.scores = []
        filenames = glob.glob(os.path.join(self.folder, '*.h5'))
        for k, filename in enumerate(filenames):
            self.agent.load(filename)
            name = os.path.basename(filename)[:-3]
            debug = "{}\n\n{}".format(self.ranking(), name)

            self.env.car.color = self.makeColor(random.random())
            perf = Performance(0, 0, name)
            try:
                _, steps = self.agent.run_once(
                    self.env, train=False, greedy=True, name=debug)
                perf.laps = self.circuit.laps + self.circuit.progression
                perf.steps = steps
            except Exception as e:
                logging.error('model {} failed: {}'.format(name, e))

            self.scores.append(perf)
            self.scores.sort(reverse=True)
            self.save()
