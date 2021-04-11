import os
import numpy as np
import learnrl as rl

class ScoreCallback(rl.Callback):

    def __init__(self, print_circuits=False):
        self.step = 0
        self.print_circuits = print_circuits

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
        if len(score) == 1:
            score = score[0]
        if self.print_circuits:
            print(f"circuit n°{env.current_circuit_id}:{score}")
        self.score += score

    def on_run_end(self, logs):
        if not self.print_circuits:
            print(f"score:{self.score}")

class CheckpointCallback(rl.Callback):

    def __init__(self, filename, save_every_cycle=False, run_test=True):
        self.filename = filename
        self.save_every_cycle = save_every_cycle
        self.run_test = run_test

    def on_episodes_cycle_end(self, episode, logs):
        path = self.filename
        if self.save_every_cycle:
            path = os.path.join(path, f'episode_{episode}')
        self.playground.agents[0].save(path)
        if self.run_test:
            self.playground.test(1, callbacks=[ScoreCallback(print_circuits=True)])
