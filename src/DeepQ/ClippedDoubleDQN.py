import torch
import torch.optim as optim
from DeepQ.DQNetwork import DQNetwork
from Agent import Agent
import utils
import random

class ClippedDoubleDQN(Agent):

    def __init__(self,hidden_size: int,lr: float,
                gamma: float, epsilon: float,
                tau: float,
                num_random_actions: int = 1000,
                state_dim: int = 4, action_size: int = 2,
                batch_size: int = 200, buffer_min_size: int = 1000,
                buffer_max_size: int = 10000) -> None:
        
        super(ClippedDoubleDQN,self).__init__(batch_size,buffer_min_size,buffer_max_size)

        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.num_random_actions = num_random_actions

        self.action_rounds = 0

        self.network1:list[torch.nn.Module]  = DQNetwork(state_dim,action_size,hidden_size)
        self.optimizer1 = optim.Adam(self.network1.parameters(),lr=lr)
        self.target_network1:list[torch.nn.Module]  = DQNetwork(state_dim,action_size,hidden_size)
        utils.target_update(self.target_network1,self.network1,1,True)

        self.network2:list[torch.nn.Module]  = DQNetwork(state_dim,action_size,hidden_size)
        self.optimizer2 = optim.Adam(self.network2.parameters(),lr=lr)
        self.target_network2:list[torch.nn.Module]  = DQNetwork(state_dim,action_size,hidden_size)
        utils.target_update(self.target_network2,self.network2,1,True)

    def get_action(self,state,train):

        if not train:
            state_tensor = torch.tensor(state, dtype = torch.float)
            reward = self.target_network1.forward(state_tensor)
            return reward.argmax().item()
        
        self.action_rounds += 1

        rand = random.random()

        if self.action_rounds <= self.num_random_actions or rand < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.tensor(state, dtype = torch.float)
            reward = self.target_network1.forward(state_tensor)
            return reward.argmax().item()
    
    def train_agent(self):

        if not self.buffer.ready():
            return

        s_batch,a_batch,r_batch,next_s_batch,notdone_batch = self.buffer.sample()

        with torch.no_grad():
            qval1 = torch.max(self.target_network1.forward(next_s_batch),dim = 1,keepdim=True).values
            qval2 = torch.max(self.target_network2.forward(next_s_batch),dim = 1,keepdim=True).values
            qval = torch.min(qval1,qval2)
            label = r_batch + self.gamma*qval*notdone_batch

        q_predict1 = self.network1(s_batch).gather(1,a_batch)
        q_predict2 = self.network2(s_batch).gather(1,a_batch)

        loss1 = torch.nn.functional.mse_loss(label,q_predict1)
        loss2 = torch.nn.functional.mse_loss(label,q_predict2)

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        utils.target_update(self.target_network1,self.network1,self.tau,False)
        utils.target_update(self.target_network2,self.network2,self.tau,False)