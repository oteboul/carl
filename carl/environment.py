from carl.car import Car
from carl.ui import Interface
from carl.circuit import Circuit, generateCircuitPoints
import numpy as np
import gym


class Environment(gym.Env):
    NUM_SENSORS = 5

    def __init__(self, circuit, render=False):
        self.circuit = circuit
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)

        # To render the environment
        self.render_init = False
        self.render_ui = render
        if self.render_ui:
            self.initRender(self.circuit, self.car)

        # Build the possible actions of the environment
        self.actions = []
        for turn_step in range(-2, 3, 1):
            for speed_step in range(-1, 2, 1):
                self.actions.append((speed_step, turn_step))

        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(0, 1, (self.NUM_SENSORS,))

        self.count = 0
        self.progression = 0

    def reward(self) -> float:
        """Computes the reward at the present moment"""
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        reward = 0
        if isCrash or hasStopped:
            return -.5
        if self.circuit.laps + self.circuit.progression > self.progression:
            reward += (self.circuit.laps + self.circuit.progression - self.progression)
            self.progression = self.circuit.laps + self.circuit.progression
        return reward + self.car.speed/20

    def isEnd(self) -> bool:
        """Is the episode over ?"""
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        return isCrash or hasStopped

    def reset(self):
        self.count = 0
        self.progression = 0
        self.car.reset()
        self.circuit.reset()
        return self.current_state

    @property
    def current_state(self):
        result = self.car.distances()
        result.append(self.car.speed)
        return result

    def step(self, i: int):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        self.count += 1
        self.car.action(*self.actions[i])

        state = self.current_state
        isEnd = self.isEnd()
        reward = self.reward()

        if self.render_ui:
            self.ui.update()
            self.render_ui = False

        return state, reward, isEnd, {}

    def mayAddTitle(self, title):
        if self.render_ui:
            self.ui.setTitle(title)
        
    def getNewCircuit(self, n_points=16, difficulty=0):
        print('-------------------\nNew Circuit !\n-------------------')
        self.points = generateCircuitPoints(n_points, difficulty)
        self.circuit = Circuit(self.points)
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)
        self.car.reset()
        self.circuit.reset()
        if self.render_ui:
            self.ui = Interface(self.circuit, self.car)
            self.ui.show(block=False)

    def render(self):
        self.render_ui = True
        if not self.render_init:
            self.initRender(self.circuit, self.car)
    
    def initRender(self, circuit, car):
        self.render_init = True
        if self.render_ui:
            self.ui = Interface(circuit, car)
            self.ui.show(block=False)

