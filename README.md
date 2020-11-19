# Self-driving car with Deep-Q Network

Start by installing the requirements:
```
sudo pip3 install -r requirements
```

Then you must implement in:

1. environment.py:
  - rewards
  - end of the episode.
2. agent.py
  - an update scheme for epsilon
  - The epsilon-greedy policy itself
  - The neural network mapping states to values Q(s, a)


To train your reinforcement learning agent with some parameters:
```
python3 -m scripts.run_train --num_episodes=X --output='my_weights.h5'
```

To test your trained agent in a greedy way (saved in the .h5 file):
```
python3 -m scripts.run_test --model='my_weights.h5'
```
