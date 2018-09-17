from src.car import Car
from src.ui import Interface


class Environment(object):
    NUM_SENSORS = 5

    def __init__(self, circuit, render=False, crash_value=0, to_movie=False):
        self.circuit = circuit
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)

        # To render the environment
        self.render = render
        if render:
            self.ui = Interface(self.circuit, self.car, save_frames=to_movie)
            self.ui.show(block=False)

        # Build the possible actions of the environment
        self.actions = []
        for turn_step in range(-2, 3, 1):
            for speed_step in range(-1, 2, 1):
                self.actions.append((speed_step, turn_step))

        # Used for the rewards
        self.crash_value = crash_value
        self.count = 0

    def reward(self) -> float:
        """Computes the reward at the present moment"""
        isCrash = self.car.car not in self.circuit
        unit = self.car.speed / self.car.SPEED_UNIT
        return unit ** 1.1 + int(isCrash) * self.crash_value

    def isEnd(self) -> bool:
        """Is the episode over ?"""
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        return isCrash or hasStopped

    def reset(self):
        self.count = 0
        self.car.reset()
        return self.current_state

    @property
    def current_state(self):
        result = self.car.distances()
        result.append(self.car.speed)
        return result

    def step(self, i: int, greedy):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        self.count += 1
        self.car.action(*self.actions[i])

        state = self.current_state
        isEnd = self.isEnd()
        reward = self.reward()

        if self.render:
            self.ui.update()

        return state, reward, isEnd

    def mayAddTitle(self, title):
        if self.render:
            self.ui.setTitle(title)
