import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import learnrl as rl
from learnrl.callbacks.logging_callback import MetricList
import tensorflow as tf

from agents.memory import Memory
from gym.spaces import Box

class DDPGAgent(rl.Agent):

    def __init__(self, actions_space:Box,
                       actor:tf.keras.Model=None, actor_lr=1e-4,
                       value:tf.keras.Model=None, value_lr=1e-4,
                       memory:Memory=None,
                       sample_size=32, discount=0.99, 
                       exploration=0, exploration_decay=0, exploration_minimal=0):
        self.actor = actor
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)

        self.value = value
        self.value_opt = tf.keras.optimizers.Adam(value_lr)
        self.discount = discount

        self.memory = memory
        self.sample_size = sample_size

        assert exploration >= 0
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.exploration_minimal = exploration_minimal

        assert isinstance(actions_space, Box)
        self.actions_space = actions_space

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        action = self.actor(observations)[0]
        if not greedy or self.exploration > 0:
            action = self.explore(action)
        return tf.clip_by_value(action, self.actions_space.low, self.actions_space.high)
    
    def explore(self, action):
        return action + tf.random.normal(action.shape, 0, self.exploration)

    def learn(self):
        datas = self.memory.sample(self.sample_size)

        metrics = {
            'exploration': self.update_exploration(),
        }

        metrics.update(self.update_value(datas))
        metrics.update(self.update_actor(datas))
        return metrics
    
    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimal)
        return self.exploration

    def update_value(self, datas):
        observations, actions, rewards, dones, next_observations = datas
        expected_futur_rewards = self.evaluate(rewards, dones, next_observations, self.value)

        with tf.GradientTape() as tape:
            inputs = tf.concat([observations, actions], axis=-1) #pylint: disable=all
            Q = self.value(inputs, training=True)[..., 0]
            loss = tf.keras.losses.mse(expected_futur_rewards, Q)
        
        self.update_network(self.value, loss, tape, self.value_opt)
        return {'value_loss': loss.numpy()}

    def update_actor(self, datas):
        observations, _, _, _, _ = datas

        with tf.GradientTape() as tape:
            actions = self.actor(observations, training=True)
            inputs = tf.concat((observations, actions), axis=-1)
            Q = self.value(inputs)
            loss = - tf.reduce_mean(Q)
        
        self.update_network(self.actor, loss, tape, self.actor_opt)
        return {'actor_loss': loss.numpy()}

    def update_network(self, network:tf.keras.Model, loss, tape, opt:tf.keras.optimizers.Optimizer):
        grads = tape.gradient(loss, network.trainable_weights)
        opt.apply_gradients(zip(grads, network.trainable_weights))

    def evaluate(self, rewards, dones, next_observations, value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_action = self.actor(next_observations[ndones])
            next_inputs = tf.concat((next_observations[ndones], next_action), axis=-1)
            next_value = self.value(next_inputs)[..., 0]

            ndones_ind = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(
                futur_rewards,
                ndones_ind,
                self.discount * next_value
            )
        
        return futur_rewards

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        self.memory.remember(observation, action, reward, done, next_observation)

if __name__ == "__main__":
    from agents.callbacks import ValidationCallback, WandbCallback, CheckpointCallback
    from carl.environment import Environment
    from carl.circuit import generate_circuit
    kl = tf.keras.layers

    circuit_point = generate_circuit(n_points=25, difficulty=3)
    env = Environment(circuit_point, action_type='continuous')

    memory = Memory(10000)

    action_value = tf.keras.Sequential((
        kl.Dense(128, activation='selu'),
        kl.Dense(64, activation='selu'),
        kl.Dense(1, activation='linear')
    ))

    actor = tf.keras.Sequential((
        kl.Dense(128, activation='selu'),
        kl.Dense(64, activation='selu'),
        kl.Dense(2, activation='tanh')
    ))

    agent = DDPGAgent(
        actions_space=env.action_space,
        actor=actor,
        value=action_value,
        memory=memory,
        sample_size=512,
        exploration=2,
        actor_lr=5e-5,
        value_lr=5e-4,
        exploration_decay=2e-4,
        exploration_minimal=5e-2,
        discount=0.99
    )

    metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'value_loss~vloss',
        'actor_loss~aloss',
        'exploration~exp'
    ]

    import wandb
    run = wandb.init()
    
    wandbcallback = WandbCallback(run, metrics)
    valid = ValidationCallback()

    pg = rl.Playground(env, agent)
    pg.fit(3000, verbose=2, episodes_cycle_len=10,
        callbacks=[valid, wandbcallback], metrics=metrics, reward_handler=reward_handler)

