from carl.car import Cars
from carl.ui import Interface
from carl.circuit import Circuit
import numpy as np
import gym
import matplotlib.pyplot as plt
import random
import colorsys


class Environment(gym.Env):

    def __init__(self, circuits, n_cars=1, action_type='discrete',
        render_sensors=None, n_sensors=5, fov=np.pi, names=None, road_width=0.3, max_steps=1000):
        self.render_sensors = render_sensors if render_sensors else n_cars < 6
        self.NUM_SENSORS = n_sensors
        self.FOV = fov
        self.max_steps = max_steps

        if isinstance(circuits, Circuit) or isinstance(circuits[0][0], (int, float)):
            circuits = [circuits]

        self.circuits = circuits
        for i, circuit in enumerate(self.circuits):
            if isinstance(circuit, Circuit):
                self.n_cars = circuit.n_cars
                self.circuits[i] = circuit
            else:
                self.n_cars = n_cars
                self.circuits[i] = Circuit(circuit, n_cars=self.n_cars, width=road_width)
        
        self.current_circuit_id = -1
        self.cars = Cars(
            self.circuits[self.current_circuit_id],
            names=names,
            n_cars=self.n_cars,
            num_sensors=self.NUM_SENSORS,
            render_sensors=self.render_sensors,
            fov=self.FOV
        )

        self.render_ui = False
        self.render_init = False

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

    @property
    def current_circuit(self):
        return self.circuits[self.current_circuit_id % len(self.circuits)]

    @property
    def current_state(self):
        normalized_speeds = np.expand_dims(self.cars.speeds, -1) / (10 * self.cars.SPEED_UNIT)
        distances = self.cars.get_distances(self.current_circuit)
        return np.concatenate((distances, normalized_speeds), axis=-1).astype(np.float32)

    def step(self, actions):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        if self.n_cars == 1:
            actions = [actions]
        self.time += 1

        if self.action_type == 'discrete':
            actions = [self.actions[action_id] for action_id in actions]
        actions = np.array(actions)
        self.cars.action(actions, self.current_circuit)

        done = self.done
        reward = self.reward
        obs = self.current_state

        if self.render_ui:
            self.ui.update(self.cars, self.current_circuit, self.time)
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

        circuit = self.current_circuit
        if circuit.laps[0] + circuit.progression[0] > self.progression[0]:
            reward += (circuit.laps[0] + circuit.progression[0] - self.progression[0])
            self.progression[0] = circuit.laps[0] + circuit.progression[0]
        reward += self.cars.speeds[0]/20
        return np.float(reward)

    @property
    def done(self) -> bool:
        """Is the episode over ?"""
        return self.time >= self.max_steps \
            or np.all(np.logical_or(self.cars.crashed, self.current_circuit.laps >= 2))

    def reset(self):
        self.time = 0
        self.progression = np.array([0 for _ in range(self.n_cars)])

        if self.render_init:
            self.current_circuit.remove_plot(self.ui.ax)

        self.current_circuit_id += 1
        circuit = self.current_circuit
        start = circuit.start.x , circuit.start.y
        self.cars.reset(start)
        circuit.reset()

        if self.render_init:
            self.cars.reset_render()
            circuit.plot(self.ui.ax)

        if self.n_cars > 1:
            return self.current_state
        else:
            return self.current_state[0]

    def render(self, render_mode="human"):
        self.render_ui = True
        if not self.render_init:
            self.init_render()

    def init_render(self):
        self.render_init = True
        if self.render_ui:
            self.ui = Interface()
            self.ui.plot(self.cars, self.current_circuit)
            self.ui.show(block=False)
