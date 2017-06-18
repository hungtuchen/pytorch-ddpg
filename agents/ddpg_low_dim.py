import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory, Transition

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def init_fanin(tensor):
    fanin = tensor.size(1)
    v = 1. / np.sqrt(fanin)
    init.uniform.uniform_(tensor, -v, v)

class Actor(nn.Module):
    def __init__(self, in_features, num_actions):
        """
        Initialize a Actor for low dimensional environment.
            in_features: number of features of input.
            num_actions: number of available actions in the environment.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_features, 400)
        init_fanin(self.fc1.weight)
        self.fc2 = nn.Linear(400, 300)
        init_fanin(self.fc2.weight)
        self.fc3 = nn.Linear(300, num_actions)
        init.uniform(self.fc3.weight, -3e-3, 3e-3)
        init.uniform(self.fc3.bias, -3e-3, 3e-3)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, in_features, num_actions):
        """
        Initialize a Critic for low dimensional environment.
            in_features: number of features of input.

        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_features, 400)
        init_fanin(self.fc1.weight)
        self.fc2 = nn.Linear(400, 300)
        init_fanin(self.fc2.weight)
        self.fc3 = nn.Linear(300, num_actions)
        init.uniform(self.fc3.weight, -3e-3, 3e-3)
        init.uniform(self.fc3.bias, -3e-3, 3e-3)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class DDPG():
    """
    The Deep Deterministic Policy Gradient (DDPG) Agent
    Parameters
    ----------
        actor_optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate and other
            parameters for the optimizer
        critic_optimizer_spec: OptimizerSpec
        num_feature: int
            The number of features of the environmental state
        num_action: int
            The number of available actions that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        tau: float
            The update rate that target networks slowly track the learned networks.
    """
    def __init__(self,
                 actor_optimizer_spec,
                 critic_optimizer_spec,
                 num_feature,
                 num_action,
                 replay_memory_size=1000000,
                 batch_size=64,
                 tau=0.001):
        ###############
        # BUILD MODEL #
        ###############
        self.num_feature = num_feature
        self.num_action = num_action
        self.batch_size = batch_size
        # Construct actor and critic
        self.actor = Actor().type(dtype)
        self.target_actor = Actor().type(dtype)
        self.critic = Critic().type(dtype)
        self.target_critic = Critic().type(dtype)
        # Construct the optimizers for actor and critic
        self.actor_optimizer = actor_optimizer_spec.constructor(self.actor.parameters(), **actor_optimizer_spec.kwargs)
        self.critic_optimizer = critic_optimizer_spec.constructor(self.critic.parameters(), **critic_optimizer_spec.kwargs)
        # Construct the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size)

    def select_action(self):
        pass
