from carl.car import Cars
from carl.ui import Interface
from carl.circuit import Circuit
import numpy as np
import gym
import matplotlib.pyplot as plt
import random
import colorsys


class Environment(gym.Env):
    NUM_SENSORS = 5

    def __init__(self, circuit_points, n_cars=1, render=False, action_type='discrete', render_sensors=None):
        self.n_cars = n_cars
        self.render_sensors = render_sensors if render_sensors else n_cars < 6

        self.circuit = Circuit(circuit_points, n_cars=self.n_cars)
        self.cars = Cars(self.circuit,
                         n_cars=self.n_cars,
                         num_sensors=self.NUM_SENSORS,
                         render_sensors=self.render_sensors)

        self.render_ui = render
        self.render_init = False
        if render:
            self.init_render()

        # Build individual action space
        self.action_type = action_type
        
        if action_type == 'discrete':
            self.actions = []
            for turn_step in range(-2, 3, 1):
                for speed_step in range(-1, 2, 1):
                    self.actions.append((speed_step, turn_step))
            self.action_space = gym.spaces.Discrete(len(self.actions))
        else:
            self.action_space = gym.spaces.Box(low=np.array([-1, -2]), high=np.array([1, 2]))

        # Build individual observation space
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.NUM_SENSORS,))

        self.time = 0
        self.progression = np.array([0 for _ in range(self.n_cars)])
    
    def init_render(self):
        self.render_init = True
        if self.render_ui:
            self.ui = Interface(self.circuit, self.cars)
            self.ui.show(block=False)

    def reset(self):
        self.time = 0
        self.progression = np.array([0 for _ in range(self.n_cars)])
        self.cars.reset()
        self.circuit.reset()

        if not self.render_ui:
            plt.close('all')
            self.cars.reset_render()
            self.circuit.reset_render()
            self.render_init = False

        if self.n_cars > 1:
            return self.current_state
        else:
            return self.current_state[0]

    @staticmethod
    def makeColor(h):
        return "#{:02x}{:02x}{:02x}".format(
            *map(lambda x: int(255*x), colorsys.hsv_to_rgb(h, 0.8, 0.8)))

    @property
    def current_state(self):
        return np.concatenate((self.cars.distances, np.expand_dims(self.cars.speeds, -1)), axis=-1).astype(np.float32)

    def step(self, actions):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        if self.n_cars == 1:
            actions = [actions]
        self.time += 1

        if self.action_type == 'discrete':
            actions = [self.actions[action_id] for action_id in actions]
        actions = np.array(actions)
        self.cars.action(actions)
        self.cars.step()

        done = self.done
        reward = self.reward
        obs = self.current_state

        if self.render_ui:
            self.ui.update()
            self.render_ui = False

        if self.n_cars > 1:
            return obs, reward, done, {}
        else:
            return obs[0], reward, done, {}

    @property
    def reward(self) -> float:
        """Computes the reward at the present moment"""
        reward = 0.0
        crashed = self.cars.crashed[0]
        if crashed:
            reward -= 0.5
        if self.circuit.laps[0] + self.circuit.progression[0] > self.progression[0]:
            reward += (self.circuit.laps[0] + self.circuit.progression[0] - self.progression[0])
            self.progression[0] = self.circuit.laps[0] + self.circuit.progression[0]
        reward += self.cars.speeds[0]/20
        return np.float(reward)

    @property
    def done(self) -> bool:
        """Is the episode over ?"""
        return np.all(np.logical_or(self.cars.crashed, self.circuit.laps >= 2))

    def render(self):
        self.render_ui = True
        if not self.render_init:
            self.init_render()
