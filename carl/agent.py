import logging
import os
import random
import numpy as np
import time

import collections
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Softmax, Dropout, Activation
from tensorflow.keras.regularizers import l2
from copy import copy
import tensorflow.keras.backend as K


class DQLAgent(object):
    def __init__(
            self, state_size=-1, action_size=-1,
            max_steps=200, gamma=1.0, epsilon=0.8, learning_rate=0.1, max_memory_len=1e4):
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.memory_keys = ('state','action', 'reward', 'next_state', 'done')
        self.memory = {}
        self.max_memory_len = max_memory_len
        self.gamma = gamma   # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = learning_rate  # learning_rate
        if self.state_size > 0 and self.action_size > 0:
            self.model = self.build_model()

        self.count = 0
        self.time_per_run = 0
        self.replays = 0
        self.difficulty = None
        self.exploration_period = 1000

    def build_model(self):
        """Neural Net for Deep-Q learning Model."""
        model = Sequential()
        # model.add(Activation('sigmoid'))
        model.add(Dense(128, kernel_regularizer=l2(1e-5)))
        model.add(Activation('tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def updateEpsilon(self):
        """This function change the value of self.epsilon to deal with the
        exploration-exploitation tradeoff as time goes"""
        self.epsilon = 0.001 + 0.29*np.cos(self.replays*2*np.pi/self.exploration_period)**2

    def save(self, output: str):
        self.model.save(output)

    def load(self, filename):
        if os.path.isfile(filename):
            self.model = load_model(filename)
            # self.model.summary()
            self.state_size = self.model.layers[0].input_shape[1]
            self.action_size = self.model.layers[-1].output.shape[1]
            return True
        else:
            logging.error('no such file {}'.format(filename))
            return False

    def remember(self, state, action, reward, next_state, done):
        for key, value in zip(self.memory_keys, (state, action, reward, next_state, done)):
            if isinstance(value, collections.Sequence):
                value = np.array(value)
            elif type(value) != np.ndarray:
                value = np.array([value])
            if key not in self.memory:
                self.memory[key] = value
            else:
                if len(self.memory[key]) <= self.max_memory_len:
                    self.memory[key] = np.concatenate((self.memory[key], value))
                else:
                    self.memory[key] = np.roll(self.memory[key], shift=-1, axis=0)
                    self.memory[key][-1] = np.array(value)

    def act(self, state, greedy=True):
        Q = self.model.predict(state)
        take_random = random.random() <= self.epsilon
        if greedy or not take_random:
            action = np.argmax(Q)
        else:
            action = np.random.choice(range(Q.size))
        return action

    def replay(self, batch_size):
        batch_memory = {}
        idx = np.random.choice(np.arange(len(self.memory['state'])), batch_size, replace=False)
        for key in self.memory_keys:
            batch_memory[key] = self.memory[key][idx]
        
        expected_return = batch_memory['reward'] +\
                  (1-batch_memory['done']) * self.gamma * np.amax(self.model.predict(batch_memory['next_state']), axis=1)
        
        target_batch = self.model.predict(batch_memory['state'])
        target_batch[np.arange(len(batch_memory['state'])), batch_memory['action']] = expected_return
        self.model.fit(batch_memory['state'], target_batch, epochs=1, batch_size=batch_size, verbose=0)
        self.updateEpsilon()

    def setTitle(self, env, train, name, num_steps, returns):
        h = name
        if train:
            h = 'Iter {} ($\epsilon$={:.2f})'.format(self.count, self.epsilon)
        end = '\nreturn {:.2f}'.format(returns) if train else ''

        env.mayAddTitle('{}\nsteps: {} | {}{}'.format(
            h, num_steps, env.circuit.debug(), end))

    def run_once(self, env, train=True, greedy=False, name=''):
        self.count += 1
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        returns = 0
        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            action = self.act(state, greedy=greedy)
            next_state, reward, done, _ = env.step(action, greedy)
            next_state = np.reshape(next_state, [1, self.state_size])

            if train:
                self.remember(state, action, reward, next_state, done)

            returns = returns * self.gamma + reward
            state = next_state
            if done:
                return returns, num_steps

            self.setTitle(env, train, name, num_steps, returns)

        return returns, num_steps

    def train(self, env, episodes, minibatch, output='weights.h5', render=False, rendering_period=0, increasing_circuits=False, max_circuit_life=3000):
        greedy = False
        for episode in range(episodes):
            t0 = time.time()
            if increasing_circuits:
                if episode==0:
                    self.difficulty = 0
                    env.getNewCircuit(difficulty=self.difficulty)
                    circuit_life = max_circuit_life
                    
            if rendering_period > 0 and episode % rendering_period == 0 and env.render==False:
                env.initRender()
                greedy = True
                
            r, n = self.run_once(env, train=True, greedy=greedy)            
            print("Episode: {}/{}, Return: {:.2f} \tin {} steps\t e: {:.2} \t Time left: {:.0f}min \t Difficulty: {}\t lr: {:.1e}".format(
                episode, episodes, r, n, self.epsilon, (episodes-episode)*self.time_per_run/60, self.difficulty, self.learning_rate))
            
            
            if increasing_circuits:
                circuit_life -= 1
                if circuit_life <= 0:
                    env.getNewCircuit(difficulty=self.difficulty)
                    circuit_life = max_circuit_life
                if env.circuit.progression >= 2 or n >= self.max_steps:
                    self.difficulty += 1    
                    env.getNewCircuit(difficulty=self.difficulty)
                    circuit_life = max_circuit_life
                    self.learning_rate *= .99
                    K.set_value(self.model.optimizer.lr, self.learning_rate)

            if rendering_period > 0:
                env.render = False
                greedy = False
                if env.ui is not None:
                    env.ui.close()

            if len(self.memory['state']) > minibatch:
                self.replay(minibatch)
                if episode % 100 == 0:
                    self.save(output)
                    print('Model Saved')
                self.replays += 1

                time_taken = time.time() - t0
                self.time_per_run += (time_taken - self.time_per_run)/self.replays

        # Finally runs a greedy one
        r, n = self.run_once(env, train=False, greedy=True)
        self.save(output)
        print("Greedy return: {} in {} steps".format(r, n))
