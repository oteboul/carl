import tensorflow as tf

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
