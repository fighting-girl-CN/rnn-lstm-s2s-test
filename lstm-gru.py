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

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_gate = Gate(hidden_size= HIDDEN_SIZE, input_size= EMBEDDING_DIME)
        self.f_gate = Gate(hidden_size= HIDDEN_SIZE, input_size= EMBEDDING_DIME)
        self.o_gate = Gate(hidden_size= HIDDEN_SIZE, input_size= EMBEDDING_DIME)

HIDDEN_SIZE = 256
EMBEDDING_DIME = 128