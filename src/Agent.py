from utils import ReplayBuffer

class Agent():

    def __init__(self,batch_size: int,
                buffer_min_size: int,
                buffer_max_size: int) -> None:
        
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(batch_size,buffer_min_size,buffer_max_size)
    
    def get_action(self,state,train):
        pass

    def put(self,state,action,reward,next_state,notdone):
        self.buffer.put((state,action,reward,next_state,notdone))

    def train_agent(self):
        pass