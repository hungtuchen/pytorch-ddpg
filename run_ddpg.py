from collections import namedtuple
import gym
import torch.optim as optim

from utils.random_process import OrnsteinUhlenbeckProcess
from utils.normalized_env import NormalizedEnv
from agents.ddpg_low_dim import DDPG
from ddpg_learning import ddpg_learning

NUM_EPISODES = 5000
BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
Q_WEIGHT_DECAY = 0.01
TAU = 0.001
THETA = 0.15
SIGMA = 0.2
LOG_EVERY_N_EPS = 10

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

actor_optimizer_spec = OptimizerSpec(
    constructor=optim.Adam,
    kwargs=dict(lr=ACTOR_LEARNING_RATE),
)

critic_optimizer_spec = OptimizerSpec(
    constructor=optim.Adam,
    kwargs=dict(lr=CRITIC_LEARNING_RATE, weight_decay=Q_WEIGHT_DECAY),
)

random_process = OrnsteinUhlenbeckProcess(theta=THETA, sigma=SIGMA)

env = NormalizedEnv(gym.make('Pendulum-v0'))

agent = DDPG(
    actor_optimizer_spec=actor_optimizer_spec,
    critic_optimizer_spec=critic_optimizer_spec,
    num_feature=env.observation_space.shape[0],
    num_action=env.action_space.shape[0],
    replay_memory_size=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    tau=TAU
)

stats = ddpg_learning(
    env=env,
    random_process=random_process,
    agent=agent,
    num_episodes=NUM_EPISODES,
    gamma=GAMMA,
    log_every_n_eps=LOG_EVERY_N_EPS,
)
