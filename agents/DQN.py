import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import learnrl as rl
import tensorflow as tf
from copy import deepcopy

class Memory():

    def __init__(self, max_memory_len):
        self.max_memory_len = max_memory_len
        self.memory_len = 0
        self.MEMORY_KEYS = ('observation', 'action', 'reward', 'done', 'next_observation')
        self.datas = {key:None for key in self.MEMORY_KEYS}

    def remember(self, observation, action, reward, done, next_observation):
        for val, key in zip((observation, action, reward, done, next_observation), self.MEMORY_KEYS):
            batched_val = tf.expand_dims(val, axis=0)
            if self.memory_len == 0:
                self.datas[key] = batched_val
            else:
                self.datas[key] = tf.concat((self.datas[key], batched_val), axis=0) #pylint: disable=all
            self.datas[key] = self.datas[key][-self.max_memory_len:]
        
        self.memory_len = len(self.datas[self.MEMORY_KEYS[0]])

    def sample(self, sample_size, method='random'):
        if method == 'random':
            indexes = tf.random.shuffle(tf.range(self.memory_len))[:sample_size]
            datas = [tf.gather(self.datas[key], indexes) for key in self.MEMORY_KEYS] #pylint: disable=all
        elif method == 'last':
            datas = [self.datas[key][-sample_size:] for key in self.MEMORY_KEYS]
        else:
            raise ValueError(f'Unknowed method {method}')
        return datas
    
    def __len__(self):
        return self.memory_len

class Control():

    def __init__(self, exploration=0, exploration_decay=0, exploration_minimum=0):
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_minimum = exploration_minimum
    
    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimum)

    def act(self, Q):
        raise NotImplementedError('You must define act(self, Q) when subclassing Control')

    def __call__(self, Q, greedy):
        if greedy:
            return tf.argmax(Q, axis=-1, output_type=tf.int32)
        else:
            return self.act(Q)

class EpsGreedy(Control):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.exploration <= 1 and self.exploration >= 0, \
            "Exploration must be in [0, 1] for EpsGreedy"

    def act(self, Q):
        batch_size = Q.shape[0]
        action_size = Q.shape[1]

        actions_random = tf.random.uniform((batch_size,), 0, action_size, dtype=tf.int32) #pylint: disable=all
        actions_greedy = tf.argmax(Q, axis=-1, output_type=tf.int32)

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd <= self.exploration, actions_random, actions_greedy)

        return actions


class Evaluation():

    def __init__(self, discount):
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        raise NotImplementedError('You must define eval when subclassing Evaluation')

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)

class QLearning(Evaluation):

    def eval(self, rewards, dones, next_observations, action_value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_values = tf.reduce_max(action_value(next_observations[ndones]), axis=-1)

            ndones_indexes = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(futur_rewards, ndones_indexes, self.discount * next_values)
        
        return futur_rewards


class DQNAgent(rl.Agent):

    def __init__(self, action_value:tf.keras.Model=None,
                    control:Control=None,
                    memory:Memory=None,
                    evaluation:Evaluation=None,
                    sample_size=32,
                    learning_rate=1e-4,
                    cql_weight=0,
                    freezed_steps=0):
        
        self.action_value = action_value
        self.action_value_learner = deepcopy(action_value)
        self.action_value_opt = tf.keras.optimizers.Adam(learning_rate)

        self.control = Control() if control is None else control
        self.memory = memory
        self.evaluation = evaluation

        self.sample_size = sample_size
        self.cql_weight = cql_weight

        self.freezed_steps = freezed_steps
        self._freezed_steps = freezed_steps

    @tf.function
    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        Q = self.action_value(observations)
        action = self.control(Q, greedy)[0]
        return action
    
    def learn(self):
        observations, actions, rewards, dones, next_observations = self.memory.sample(self.sample_size)
        expected_futur_rewards = self.evaluation(rewards, dones, next_observations, self.action_value)

        with tf.GradientTape() as tape:
            Q = self.action_value_learner(observations)

            action_index = tf.stack( (tf.range(len(actions)), actions) , axis=-1)
            Q_action = tf.gather_nd(Q, action_index)

            Q_softmax = tf.math.log(tf.reduce_sum(tf.math.exp(Q), axis=-1))

            cql_loss = tf.reduce_mean(Q_softmax) - tf.reduce_mean(Q_action)
            mse_loss = tf.keras.losses.mse(expected_futur_rewards, Q_action)
            loss = mse_loss + self.cql_weight * cql_loss

        grads = tape.gradient(loss, self.action_value_learner.trainable_weights)
        self.action_value_opt.apply_gradients(zip(grads, self.action_value_learner.trainable_weights))

        if self._freezed_steps == 0:
            self.action_value.set_weights(self.action_value_learner.get_weights())
            self._freezed_steps = self.freezed_steps
        else:
            self._freezed_steps -= 1

        metrics = {
            'value': tf.reduce_mean(Q_action).numpy(),
            'mse_loss': mse_loss.numpy(),
            'cql_loss': cql_loss.numpy(),
            'loss': loss.numpy(),
            'exploration': self.control.exploration,
            'learning_rate': self.action_value_opt.lr.numpy()
        }

        self.control.update_exploration()
        return metrics

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        self.memory.remember(observation, action, reward, done, next_observation)
    
    def save(self, filename):
        tf.keras.models.save_model(self.action_value, filename, save_format='h5')
        print(f'Model saved at {filename}')
    
    def load(self, filename):
        self.action_value = tf.keras.models.load_model(filename)

class RewardScaler(rl.RewardHandler):

    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def reward(self, observation, action, reward, done, info, next_observation):
        return self.scaling_factor * reward

if __name__ == "__main__":
    import wandb
    from carl.environment import Environment
    from agents.callbacks import ValidationCallback, CheckpointCallback, WandbCallback
    kl = tf.keras.layers

    default_config = {
        'max_memory_len': 40960,

        'exploration': 0.1,
        'exploration_decay': 3e-4,
        'exploration_minimum': 5e-3,

        'discount': 0.90,

        'dense_1_size': 256,
        'dense_1_activation': 'selu',
        'dense_2_size': 128,
        'dense_2_activation': 'selu',
        'dense_3_size': 64,
        'dense_3_activation': 'selu',

        'sample_size': 2048,
        'learning_rate': 2e-4,

        'freezed_steps': 0,
        'cql_weight': 2e-2,
    }

    run = wandb.init(config=default_config)
    config = run.config

    env = Environment([(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)])

    memory = Memory(config.max_memory_len)
    control = EpsGreedy(
        config.exploration,
        config.exploration_decay,
        config.exploration_minimum
    )
    evaluation = QLearning(config.discount)

    action_value = tf.keras.models.Sequential((
        kl.Dense(config.dense_1_size, activation=config.dense_1_activation),
        kl.Dense(config.dense_2_size, activation=config.dense_2_activation),
        kl.Dense(config.dense_3_size, activation=config.dense_3_activation),
        kl.Dense(env.action_space.n, activation='linear')
    ))

    agent = DQNAgent(
        action_value=action_value,
        control=control,
        memory=memory,
        evaluation=evaluation,
        sample_size=config.sample_size,
        learning_rate=config.learning_rate,
        cql_weight=config.cql_weight,
        freezed_steps=config.freezed_steps
    )
    
    metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'cql_loss~cql',
        'mse_loss~mse',
        'loss',
        'exploration~exp',
        'value~Q'
    ]

    valid = ValidationCallback()
    check = CheckpointCallback(os.path.join('models', 'poulet29.h5'))
    wandbcallback = WandbCallback(run, metrics)

    pg = rl.Playground(env, agent)
    pg.fit(
        1000, verbose=2, metrics=metrics,
        callbacks=[valid, check, wandbcallback],
        reward_handler=RewardScaler(1)
    )
