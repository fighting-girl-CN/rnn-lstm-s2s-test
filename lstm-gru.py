import torch
import torch.nn as nn
from torch.nn import Embedding
import urllib.request
from pathlib import Path



class Gate(nn.Module):
    def __init__(self, hidden_size,input_size):
        super().__init__()
        self.wah = nn.Linear(hidden_size,hidden_size,bias=False)
        self.wax = nn.Linear(input_size,hidden_size,bias=True)

    def forward(self,x, h):
        h_h = self.wah(h)
        h_x = self.wax(x)
        output = torch.sigmoid(h_h + h_x)
        return output


