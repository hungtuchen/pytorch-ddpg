import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory
from utils import plotting

def ddpg_learning(
    env,
    random_process,
    agent,
    num_episodes,
    gamma=1.0,
    ):

    """The Deep Deterministic Policy Gradient algorithm.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    random_process: Defined in utils.random_process
        The process that add noise for exploration in deterministic policy.
    agent:
        a DDPG agent consists of a actor and critic.
    num_episodes:
        Number of episodes to run for.
    gamma: float
        Discount Factor
    """
    ###############
    # RUN ENV     #
    ###############
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        state = env.reset()
        random_process.reset_states()
        for t in count(1):
            action = agent.select_action(state)
            # Add noise for exploration
            noise = random_process.sample()[0]
            action += noise
            action = np.clip(action, -1.0, 1.0)
            next_state, reward, done, _ = env.step(action)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            # Store transition in replay memory
            agent.replay_memory.push(state, action, reward, next_state, done)
            # Update
            agent.update(gamma)
            if done:
                break
            else:
                state = next_state

    return stats
