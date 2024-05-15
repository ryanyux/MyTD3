
import torch
import torch.nn as nn


class ReplayBuffer():
    def __init__(self, capacity, state_shape, action_shape, device="cpu") -> None:
        self.size = 0
        self.capacity = capacity
        self.cur = 0
        
        self.state_tensor = torch.zeros((0, state_shape), dtype=torch.float32).to(device)
        self.next_state_tensor = torch.zeros((0, state_shape), dtype=torch.float32).to(device)
        self.action_tensor = torch.zeros((0, action_shape), dtype=torch.float32).to(device)
        self.reward_tensor = torch.zeros((0, 1), dtype=torch.float32).to(device)
        self.done_tensor = torch.zeros((0, 1), dtype=torch.float32).to(device)
    
    
    def add(self, s, ns, a, r, d):
        s, ns, a, r, d =    torch.from_numpy(s), \
                            torch.from_numpy(ns), \
                            torch.from_numpy(a), \
                            torch.tensor((r,), dtype=torch.float32), \
                            torch.tensor((d,), dtype=torch.float32)
        
        s, ns, a, r, d =    s.unsqueeze(0), \
                            ns.unsqueeze(0), \
                            a.unsqueeze(0), \
                            r.unsqueeze(0), \
                            d.unsqueeze(0)
        
        if self.size < self.capacity:
            self.state_tensor = torch.cat((self.state_tensor, s), dim=0)
            self.next_state_tensor = torch.cat((self.next_state_tensor, ns), dim=0)
            self.action_tensor = torch.cat((self.action_tensor, a), dim=0)
            self.reward_tensor = torch.cat((self.reward_tensor, r), dim=0)
            self.done_tensor = torch.cat((self.done_tensor, d), dim=0)
            self.size += 1
            self.cur = (self.cur + 1) % self.capacity
        
        else:
            self.state_tensor[self.cur] = s
            self.next_state_tensor[self.cur] = ns
            self.action_tensor[self.cur] = a
            self.reward_tensor[self.cur] = r
            self.done_tensor[self.cur] = d
            
            self.cur = (self.cur + 1) % self.capacity
            
    
    def sample(self, batch_size):
        count = min(self.size, batch_size)
        indeces = torch.randperm(self.size)[:count]
        return self.state_tensor[indeces],\
                self.next_state_tensor[indeces],\
                self.action_tensor[indeces],\
                self.reward_tensor[indeces],\
                self.done_tensor[indeces]
    
        