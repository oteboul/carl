import logging
import keras.models
import os
import random
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQLAgent(object):
    def __init__(
            self, state_size, action_size, max_steps=200,
            gamma=0.95, epsilon=1.0, epsilon_decay=0.99, learning_rate=0.1,
            output='weights.h5'):
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.memory = deque(maxlen=2000)
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.output = output
        self.count = 0

    def _build_model(self):
        """Neural Net for Deep-Q learning Model"""
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def save(self):
        self.model.save(self.output)

    def load(self, filename):
        if os.path.isfile(filename):
            self.model = keras.models.load_model(filename)
            return True
        else:
            logging.error('no such file {}'.format(filename))
            return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, greedy=True):
        if np.random.rand() <= self.epsilon and not greedy:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_once(self, env, train=True, greedy=False):
        self.count += 1
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        returns = 0
        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            action = self.act(state, greedy=greedy)
            next_state, reward, done = env.step(action, greedy)
            next_state = np.reshape(next_state, [1, self.state_size])

            if train:
                self.remember(state, action, reward, next_state, done)

            returns = returns * self.gamma + reward
            state = next_state
            if done:
                return returns

            env.mayAddTitle(
                'Iter {} ($\epsilon$={:.2f})\nlaps {}, return {:.2f}, '.format(
                    self.count, self.epsilon, env.circuit.laps, returns))

        return returns

    def train(self, env, episodes, minibatch, render=False):
        for e in range(episodes):
            r = self.run_once(env, train=True, greedy=False)
            print("episode: {}/{}, return: {}, e: {:.2}".format(
                e, episodes, r, self.epsilon))

            if len(self.memory) > minibatch:
                self.replay(minibatch)
                self.save()

        # Finally runs a greedy one
        r = self.run_once(env, train=False, greedy=True)
        self.save()
        print("Greedy return: {}".format(r))
