import torch
import torch.optim as optim
import numpy as np
from ActorCritic.ActorNetwork import ActorNetwork
from ActorCritic.CriticNetwork import CriticNetwork
from Agent import Agent
import utils
import random

class DDPG(Agent):

    def __init__(self, actor_hidden_size: int,
                critic_hidden_size: int,
                lr: float, gamma: float,
                epsilon: float, tau: float,
                num_random_actions: int = 1000,
                state_dim: int = 4, action_dim: int = 1,
                batch_size: int = 200, buffer_min_size: int = 1000,
                buffer_max_size: int = 10000) -> None:
        
        super(DDPG,self).__init__(batch_size,buffer_min_size,buffer_max_size,action_type = torch.float)

        self.state_dim = state_dim
        self.action_size = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.num_random_actions = num_random_actions

        self.train_rounds = 0
        self.action_rounds = 0

        self.actor_network:torch.nn.Module  = ActorNetwork(state_dim,action_dim,actor_hidden_size)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),lr=lr)

        self.actor_target_network:torch.nn.Module = ActorNetwork(state_dim,action_dim,actor_hidden_size)
        utils.target_update(self.actor_target_network,self.actor_network,1,True)

        self.critic_network:torch.nn.Module  = CriticNetwork(state_dim+action_dim,1,critic_hidden_size)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=lr)

        self.critic_target_network:torch.nn.Module = CriticNetwork(state_dim+action_dim,1,critic_hidden_size)
        utils.target_update(self.critic_target_network,self.critic_network,1,True)

    def get_action(self,state,train):

        state_tensor = torch.tensor(state, dtype = torch.float)
        action = self.actor_target_network.forward(state_tensor)

        if not train:
            return action.item()

        self.action_rounds += 1
        rand = random.random()

        if self.action_rounds <= self.num_random_actions  or rand < self.epsilon:
            return torch.rand_like(action).item()
        else:
            return action.item()
    
    def train_agent(self):

        if not self.buffer.ready():
            return

        s_batch,a_batch,r_batch,next_s_batch,notdone_batch = self.buffer.sample()

        with torch.no_grad():
            next_a_batch = self.actor_target_network.forward(next_s_batch)
            next_q = self.critic_target_network.forward(next_s_batch,next_a_batch)
            label = r_batch + self.gamma*next_q*notdone_batch

        q_predict = self.critic_network.forward(s_batch,a_batch)

        critic_loss = torch.nn.functional.mse_loss(label,q_predict)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_predict =  self.actor_network.forward(s_batch)

        actor_loss = -self.critic_network.forward(s_batch,a_predict).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.train_rounds += 1
        utils.target_update(self.critic_target_network,self.critic_network,self.tau,False)
        utils.target_update(self.actor_target_network,self.actor_network,self.tau,False)