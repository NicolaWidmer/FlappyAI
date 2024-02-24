import torch
import torch.nn as nn

class SoftActorNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim:int,hidden_size:int, log_std_bounds:tuple[float,float] = (-20,2)):
        super(SoftActorNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=input_dim,out_features=hidden_size)
        self.mean_out_layer = torch.nn.Linear(in_features=hidden_size,out_features=output_dim)
        self.std_out_layer = torch.nn.Linear(in_features=hidden_size,out_features=output_dim)
        self.log_std_min,self.log_std_max = log_std_bounds


    def forward(self, s: torch.Tensor) -> torch.Tensor:

        x = torch.nn.functional.relu(self.fc1(s))
        mean = self.mean_out_layer(x)
        log_std = self.std_out_layer(x)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)
        return mean,log_std
    
    def sample(self,s,train):
        mean,log_std = self.forward(s)

        if not train:
            return torch.nn.functional.tanh(mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(0,1)
        eps = normal.rsample()
        a_sample = mean + std*eps
        action = torch.nn.functional.tanh(a_sample)

        log_prob = normal.log_prob(eps) - torch.log(1 - torch.pow(action,2) + 1e-5)

        return action,log_prob

