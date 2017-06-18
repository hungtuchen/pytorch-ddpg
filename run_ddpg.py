import gym
import torch.optim as optim

BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
Q_WEIGHT_DECAY = 0.01
TAU = 0.001

actor_optimizer_spec = OptimizerSpec(
    constructor=optim.Adam,
    kwargs=dict(lr=ACTOR_LEARNING_RATE),
)

critic_optimizer_spec = OptimizerSpec(
    constructor=optim.Adam,
    kwargs=dict(lr=CRITIC_LEARNING_RATE, weight_decay=Q_WEIGHT_DECAY),
)
