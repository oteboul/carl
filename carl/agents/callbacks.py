import os
import numpy as np
import learnrl as rl

class ScoreCallback(rl.Callback):

    def __init__(self, print_circuits=False, print_names=False):
        self.step = 0
        self.print_circuits = print_circuits
        self.print_names = print_names

    def on_run_begin(self, logs):
        self.score = np.zeros(self.playground.env.n_cars)
        self.total_bonuses = np.zeros(self.playground.env.n_cars)

    def on_episode_begin(self, step, logs):
        self.step = 0
        self.bonuses = np.zeros(self.playground.env.n_cars)

    def on_step_end(self, step, logs):
        self.step += 1
        finished =  self.playground.env.current_circuit.laps >= 2
        self.bonuses = np.where(
            np.logical_and(finished, self.bonuses == 0),
            max(0, (2 - self.step / 200)),
            self.bonuses
        )

    def on_episode_end(self, episode, logs):
        env = self.playground.env
        circuit = env.current_circuit
        progressions = circuit.laps + circuit.progression
        self.score += np.minimum(2, progressions) + self.bonuses
        self.total_bonuses += self.bonuses
        if self.print_circuits:
            print(f"circuit n°{env.current_circuit_id}:{self.score}")
        self.scores_by_names = {name: score for score, name in zip(self.score, env.cars.names)}

    def on_run_end(self, logs):
        if self.print_names:
            names_and_scores = [
                [name, round(score, 4), round(bonus, 4)]
                for name, score, bonus in zip(
                        self.playground.env.cars.names,
                        self.score,
                        self.total_bonuses
                    )
            ]
            names_and_scores = np.array(names_and_scores)
            ranks = np.argsort(-names_and_scores[:, 1].astype(np.float32))
            print(names_and_scores[ranks])
        else:
            if not self.print_circuits:
                score = self.score if len(self.score) > 1 else self.score[0]
                print(f"score:{score}")

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
