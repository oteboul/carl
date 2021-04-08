import learnrl as rl

class ValidationCallback(rl.Callback):

    def on_episodes_cycle_end(self, episode, logs):
        self.playground.test(1)

class CheckpointCallback(rl.Callback):

    def __init__(self, filename):
        self.filename = filename

    def on_episodes_cycle_end(self, episode, logs):
        self.playground.agents[0].save(self.filename)
