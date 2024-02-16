import torch
import torch.nn as nn

class DQNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim:int,hidden_size:int):
        super(DQNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=input_dim,out_features=hidden_size)
        self.out_layer = torch.nn.Linear(in_features=hidden_size,out_features=output_dim)


    def forward(self, s: torch.Tensor) -> torch.Tensor:

        x = torch.nn.functional.relu(self.fc1(s))
        x = self.out_layer(x)
        return x