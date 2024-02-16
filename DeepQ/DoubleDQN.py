import torch
import torch.optim as optim
from DeepQ.DQNetwork import DQNetwork
from Agent import Agent
import utils
import random

class DoubleDQN(Agent):

    def __init__(self,hidden_size: int,lr: float,
                gamma: float, epsilon: float,
                tau: float,
                num_random_actions: int = 1000,
                state_dim: int = 4, action_size: int = 2,
                batch_size: int = 200, buffer_min_size: int = 1000,
                buffer_max_size: int = 10000) -> None:
        
        super(DoubleDQN,self).__init__(batch_size,buffer_min_size,buffer_max_size)

        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.num_random_actions = num_random_actions

        self.train_rounds = 0
        self.action_rounds = 0

        self.network:torch.nn.Module  = DQNetwork(state_dim,action_size,hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(),lr=lr)

        self.target_network:torch.nn.Module = DQNetwork(state_dim,action_size,hidden_size)
        utils.target_update(self.target_network,self.network,1,True)

    def get_action(self,state,train):

        if not train:
            state_tensor = torch.tensor(state, dtype = torch.float)
            reward = self.target_network.forward(state_tensor)
            return reward.argmax().item()

        self.action_rounds += 1

        rand = random.random()

        if self.action_rounds <= self.num_random_actions or rand < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.tensor(state, dtype = torch.float)
            reward = self.target_network.forward(state_tensor)
            return reward.argmax().item()
    
    def train_agent(self):

        if not self.buffer.ready():
            return

        s_batch,a_batch,r_batch,next_s_batch,notdone_batch = self.buffer.sample()

        with torch.no_grad():
            qmax = torch.max(self.target_network.forward(next_s_batch),dim = 1,keepdim=True).values
            label = r_batch + self.gamma*qmax*notdone_batch

        q_predict = self.network(s_batch).gather(1,a_batch)

        loss = torch.nn.functional.mse_loss(label,q_predict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_rounds += 1
        utils.target_update(self.target_network,self.network,self.tau,False)