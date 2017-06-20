import numpy as np
import random

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
    v = 1.0 / np.sqrt(fanin)
    init.uniform(tensor, -v, v)

class Actor(nn.Module):
    def __init__(self, num_feature, num_action):
        """
        Initialize a Actor for low dimensional environment.
            num_feature: number of features of input.
            num_action: number of available actions in the environment.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_feature, 400)
        init_fanin(self.fc1.weight)
        self.fc2 = nn.Linear(400, 300)
        init_fanin(self.fc2.weight)
        self.fc3 = nn.Linear(300, num_action)
        init.uniform(self.fc3.weight, -3e-3, 3e-3)
        init.uniform(self.fc3.bias, -3e-3, 3e-3)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, num_feature, num_action):
        """
        Initialize a Critic for low dimensional environment.
            num_feature: number of features of input.

        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_feature, 400)
        init_fanin(self.fc1.weight)
        # Actions were not included until the 2nd hidden layer of Q.
        self.fc2 = nn.Linear(400 + num_action, 300)
        init_fanin(self.fc2.weight)
        self.fc3 = nn.Linear(300, 1)
        init.uniform(self.fc3.weight, -3e-3, 3e-3)
        init.uniform(self.fc3.bias, -3e-3, 3e-3)


    def forward(self, states, actions):
        x = F.relu(self.fc1(states))
        # Actions were not included until the 2nd hidden layer of Q.
        x = torch.cat((x, actions), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        self.tau = tau
        # Construct actor and critic
        self.actor = Actor(num_feature, num_action).type(dtype)
        self.target_actor = Actor(num_feature, num_action).type(dtype)
        self.critic = Critic(num_feature, num_action).type(dtype)
        self.target_critic = Critic(num_feature, num_action).type(dtype)
        # Construct the optimizers for actor and critic
        self.actor_optimizer = actor_optimizer_spec.constructor(self.actor.parameters(), **actor_optimizer_spec.kwargs)
        self.critic_optimizer = critic_optimizer_spec.constructor(self.critic.parameters(), **critic_optimizer_spec.kwargs)
        # Construct the replay memory
        self.replay_memory = ReplayMemory(replay_memory_size)

    def select_action(self, state):
        state = torch.from_numpy(state).type(dtype).unsqueeze(0)
        action = self.actor(Variable(state, volatile=True)).data.cpu()[0, 0]
        return action

    def update(self, gamma=1.0):
        if len(self.replay_memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_mask = \
            self.replay_memory.sample(self.batch_size)
        state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
        action_batch = Variable(torch.from_numpy(action_batch).type(dtype)).unsqueeze(1)
        reward_batch = Variable(torch.from_numpy(reward_batch).type(dtype))
        next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

        ### Critic ###
        # Compute current Q value, critic takes state and action choosen
        current_Q_values = self.critic(state_batch, action_batch)
        # Compute next Q value based on which action target actor would choose
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        target_actions = self.target_actor(state_batch)
        next_max_q = self.target_critic(next_state_batch, target_actions).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = reward_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        critic_loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ### Actor ###
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        self.update_target(self.target_critic, self.critic)
        self.update_target(self.target_actor, self.actor)

    def update_target(self, target_model, model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
