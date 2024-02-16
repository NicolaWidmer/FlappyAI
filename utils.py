from collections import deque
import random
import torch

class ReplayBuffer:

    def __init__(self,batch_size: int,
                min_size: int,
                max_size: int) -> None:
        
        self.queue = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.min_size = min_size

    def put(self,transition):
        self.queue.append(transition)

    def sample(self):
        mini_batch = random.sample(self.queue, self.batch_size)
        s_list, a_list, r_list, next_s_list, notdone_list = [], [], [], [], []

        for s,a,r,next_s,notdone in mini_batch:
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            next_s_list.append(next_s)
            notdone_list.append([notdone])
        
        s_batch = torch.tensor(s_list, dtype=torch.float)
        a_batch = torch.tensor(a_list, dtype=torch.int64)
        r_batch = torch.tensor(r_list, dtype=torch.float)
        s_prime_batch = torch.tensor(next_s_list, dtype=torch.float)
        notdone_batch = torch.tensor(notdone_list, dtype=torch.float)

        # normalize reward
        r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)


        return s_batch,a_batch,r_batch,s_prime_batch,notdone_batch
    
    def ready(self):
        return len(self.queue) >= self.min_size
        

def target_update(target,base,tau: float,hard_update: bool):
    
    for param_target, param in zip(target.parameters(), base.parameters()):
        if hard_update:
            param_target.data.copy_(param.data)
        else:
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)