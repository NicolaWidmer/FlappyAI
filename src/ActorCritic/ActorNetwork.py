import torch
import torch.nn as nn
import numpy as np

def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

class ActorNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim:int,hidden_size:int):
        super(ActorNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=input_dim,out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.out_layer = torch.nn.Linear(in_features=hidden_size,out_features=output_dim)
        self.apply(weights_init_uniform_rule)


    def forward(self, s: torch.Tensor) -> torch.Tensor:

        x = torch.nn.functional.relu(self.fc1(s))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.tanh(self.out_layer(x))
        return x