import torch
import torch.optim as optim
import numpy as np
from ActorCritic.ActorNetwork import ActorNetwork
from ActorCritic.CriticNetwork import CriticNetwork
from Agent import Agent
import utils

class TDDDPG(Agent):

    def __init__(self,actor_hidden_size: int,
                critic_hidden_size: int,lr: float,
                gamma: float, sigma: float,
                tau: float, policy_delay: int,
                num_random_actions: int = 1000,
                state_dim: int = 4, action_dim: int = 1,
                batch_size: int = 200, buffer_min_size: int = 1000,
                buffer_max_size: int = 10000) -> None:
        
        super(TDDDPG,self).__init__(batch_size,buffer_min_size,buffer_max_size,action_type = torch.float)

        self.state_dim = state_dim
        self.action_size = action_dim
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.num_random_actions = num_random_actions
        self.policy_delay = policy_delay

        self.train_rounds = 0
        self.action_rounds = 0

        self.actor_network:torch.nn.Module  = ActorNetwork(state_dim,action_dim,actor_hidden_size)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),lr=lr)

        self.actor_target_network:torch.nn.Module = ActorNetwork(state_dim,action_dim,actor_hidden_size)
        utils.target_update(self.actor_target_network,self.actor_network,1,True)

        self.critic_network1:torch.nn.Module  = CriticNetwork(state_dim+action_dim,1,critic_hidden_size)
        self.critic_optimizer1 = optim.Adam(self.critic_network1.parameters(),lr=lr)

        self.critic_target_network1:torch.nn.Module = CriticNetwork(state_dim+action_dim,1,critic_hidden_size)
        utils.target_update(self.critic_target_network1,self.critic_network1,1,True)
        
        self.critic_network2:torch.nn.Module  = CriticNetwork(state_dim+action_dim,1,critic_hidden_size)
        self.critic_optimizer2 = optim.Adam(self.critic_network2.parameters(),lr=lr)

        self.critic_target_network2:torch.nn.Module = CriticNetwork(state_dim+action_dim,1,critic_hidden_size)
        utils.target_update(self.critic_target_network2,self.critic_network2,1,True)

    def get_action(self,state,train):

        state_tensor = torch.tensor(state, dtype = torch.float)
        action = self.actor_target_network.forward(state_tensor)

        if not train:
            return action.item()

        self.action_rounds += 1

        if self.action_rounds <= self.num_random_actions:
            return torch.rand_like(action).item()
        else:
            noise = np.random.normal(0,self.sigma)
            noise = torch.clamp(torch.ones(action.shape)*noise,-0.1,0.1)
            action = torch.clamp(noise+action,-1,1)

            return action.item()
    
    def train_agent(self):

        if not self.buffer.ready():
            return

        s_batch,a_batch,r_batch,next_s_batch,notdone_batch = self.buffer.sample()

        with torch.no_grad():
            next_a_batch = self.actor_target_network.forward(next_s_batch)
            q1 = self.critic_target_network1.forward(next_s_batch,next_a_batch)
            q2 = self.critic_target_network2.forward(next_s_batch,next_a_batch)
            label = r_batch + self.gamma*torch.min(q1,q2)*notdone_batch

        q_predict1 = self.critic_network1.forward(s_batch,a_batch)
        q_predict2 = self.critic_network2.forward(s_batch,a_batch)

        critic_loss1 = torch.nn.functional.mse_loss(label,q_predict1)
        critic_loss2 = torch.nn.functional.mse_loss(label,q_predict2)

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        self.train_rounds += 1
        utils.target_update(self.critic_target_network1,self.critic_network1,self.tau,False)
        utils.target_update(self.critic_target_network2,self.critic_network2,self.tau,False)

        if self.train_rounds % self.policy_delay == 0:
            a_predict =  self.actor_network(s_batch)

            q1 = self.critic_network1.forward(s_batch,a_predict)
            q2 = self.critic_network2.forward(s_batch,a_predict)
            actor_loss = -torch.min(q1,q2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            utils.target_update(self.actor_target_network,self.actor_network,self.tau,False)