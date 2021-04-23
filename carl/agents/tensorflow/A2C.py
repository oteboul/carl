import os
import learnrl as rl
import tensorflow as tf
import tensorflow_probability as tfp

from carl.agents.tensorflow.memory import Memory
from gym.spaces import Box

class A2CAgent(rl.Agent):

    def __init__(self, actions_space:Box,
                       actor:tf.keras.Model=None, actor_lr=1e-4,
                       value:tf.keras.Model=None, value_lr=1e-4,
                       memory:Memory=None,
                       sample_size=32, discount=0.99, entropy_reg=0,
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

        self.entropy_reg = entropy_reg

        assert isinstance(actions_space, Box)
        self.actions_space = actions_space

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        action_dist = self.actor(observations)
        if greedy:
            action = action_dist.mean()[0]
        else:
            action = action_dist.sample()[0]
            action = self.explore(action)
        return tf.clip_by_value(action, self.actions_space.low, self.actions_space.high)

    def explore(self, action):
        return action + tf.random.normal(action.shape, 0, self.exploration)

    def learn(self):
        datas = self.memory.sample(self.sample_size)

        metrics = {
            'exploration': self.update_exploration(),
        }

        metrics.update(self.update_actor(datas))
        metrics.update(self.update_value(datas))
        return metrics
    
    def update_exploration(self):
        self.exploration *= 1 - self.exploration_decay
        self.exploration = max(self.exploration, self.exploration_minimal)
        return self.exploration

    def update_value(self, datas):
        observations, actions, rewards, dones, next_observations = datas
        expected_futur_rewards = self.evaluate(rewards, dones, next_observations, self.value)

        with tf.GradientTape() as tape:
            V = self.value(observations, training=True)[..., 0]
            loss = tf.keras.losses.mse(expected_futur_rewards, V)

        self.update_network(self.value, loss, tape, self.value_opt)
        value_metrics = {
            'value_loss': loss.numpy(),
            'value': tf.reduce_mean(V).numpy()
        }
        return value_metrics

    def update_actor(self, datas):
        observations, actions, rewards, dones, next_observations = datas

        futur_rewards = self.evaluate(rewards, dones, next_observations, self.value)
        V = self.value(observations)
        critic = futur_rewards - V

        with tf.GradientTape() as tape:
            actions_dist = self.actor(observations, training=True)
            log_prob = actions_dist.log_prob(actions)
            entropy = - tf.exp(log_prob) * log_prob
            loss = - tf.reduce_mean(critic * log_prob - self.entropy_reg * entropy)

        self.update_network(self.actor, loss, tape, self.actor_opt)
        return {
            'actor_loss': loss.numpy(),
            'entropy': tf.reduce_mean(entropy).numpy()
        }

    def update_network(self, network:tf.keras.Model, loss, tape, opt:tf.keras.optimizers.Optimizer):
        grads = tape.gradient(loss, network.trainable_weights)
        opt.apply_gradients(zip(grads, network.trainable_weights))

    def evaluate(self, rewards, dones, next_observations, value):
        futur_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            next_value = self.value(next_observations[ndones])[..., 0]
            ndones_ind = tf.where(ndones)
            futur_rewards = tf.tensor_scatter_nd_add(
                futur_rewards,
                ndones_ind,
                self.discount * next_value
            )
        
        return futur_rewards

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        self.memory.remember(observation, action, reward, done, next_observation)

    @staticmethod
    def _get_filenames(filename):
        if filename.endswith('.h5'):
            return filename, None
        actor_path = os.path.join(filename, "actor.h5")
        value_path = os.path.join(filename, "value.h5")
        return actor_path, value_path

    def save(self, filename):
        actor_path, value_path = self._get_filenames(filename)
        tf.keras.models.save_model(self.actor, actor_path)
        tf.keras.models.save_model(self.value, value_path)
        print(f'Models saved under {filename}')

    def load(self, filename):
        actor_path, value_path = self._get_filenames(filename)
        self.actor = tf.keras.models.load_model(actor_path, custom_objects={'tf': tf})
        if value_path is not None:
            self.value = tf.keras.models.load_model(value_path, custom_objects={'tf': tf})

if __name__ == "__main__":
    import numpy as np
    from carl.environment import Environment
    from carl.utils import generate_circuit
    kl = tf.keras.layers
    tfpl = tfp.layers

    circuits = [
        # [(0.5, 0), (2.5, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 2), (0, 1)],
        # # [(0, 0), (1, 2), (0, 4), (3, 4), (2, 2), (3, 0)],
        [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)],
        # [(1, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3),
        # (4, 3), (4, 2), (2, 2), (2, 3), (0, 3), (0, 1)],
        # [(2, 0), (5, 0), (5.5, 1.5), (7, 2), (7, 4), (6, 4), (5, 3), (4, 4),
        # (3.5, 3), (3, 4), (2, 3), (1, 4), (0, 4), (0, 2), (1.5, 1.5)],
    ]
    env = Environment(circuits, action_type='continueous', 
        n_sensors=5, fov=np.pi*180/180, car_width=0.1, speed_unit=0.05)

    memory = Memory(10000)

    feature_extractor = tf.keras.Sequential((
        kl.Dense(256, activation='relu'),
        kl.Dense(128, activation='relu'),
        kl.Dense(64, activation='relu'),
        kl.Dense(64, activation='relu'),
    ))

    value = tf.keras.Sequential((
        feature_extractor,
        kl.Dense(1, activation='linear')
    ))

    event_shape = [2]
    actor = tf.keras.models.Sequential([
        feature_extractor,
        kl.Dense(tfpl.IndependentNormal.params_size(event_shape), activation=None),
        tfpl.IndependentNormal(event_shape)
    ])

    agent = A2CAgent(
        actions_space=env.action_space,
        actor=actor,
        value=value,
        memory=memory,
        sample_size=256,
        exploration=1,
        actor_lr=1e-5,
        value_lr=1e-3,
        exploration_decay=5e-4,
        exploration_minimal=0,
        discount=0.99,
        entropy_reg=1e-2
    )

    filename = "test"
    from carl.agents.callbacks import CheckpointCallback
    check = CheckpointCallback(
        os.path.join('models', 'A2C', filename),
        save_every_cycle=True,
        run_test=True,
    )

    metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'actor_loss~aloss',
        'value_loss~vloss',
        'entropy',
        'exploration~exp',
        'value~V'
    ]

    pg = rl.Playground(env, agent)
    pg.fit(3000, verbose=2, episodes_cycle_len=100,
        callbacks=[check], metrics=metrics,
        reward_handler=lambda reward, **kwargs: reward
    )
