
import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, ob_space, action_space) -> None:
        super().__init__()
        self.fc1 = nn.Linear(ob_space, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_space)
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return F.tanh(x)
        
        
        


class Critic(nn.Module):
    def __init__(self, state_space, action_space) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_space + action_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x