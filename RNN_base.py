from tkinter.constants import HIDDEN

import torch
import torch.nn as nn
from torch._dynamo.variables import optimizer
from torch.nn import Embedding




class MyRNN(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.wah = nn.Linear(hidden_size, hidden_size, bias= False)
        self.wax = nn.Linear(input_size, hidden_size, bias= True)


    def forward(self, x_in, ht):
        outputs = []
        for e in x_in:
            wah = self.wah(ht)
            wax = self.wax(e)
            ht = torch.tanh(wah + wax)
            outputs.append(ht)
        outputs = torch.stack(outputs, dim=1)

        return outputs, ht

class Simple_TXT_Generator(nn.Module):
    def __init__(self,vocab,hidden_size):
        super().__init__()
        self.char2id = {c:i for i,c in enumerate(vocab)}
        self.id2char = {i:c for i,c in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.rnn = MyRNN(hidden_size,input_size=EMBEDDING_DIM)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, EMBEDDING_DIM)

    def forward(self,input_x, ht):
        x_embedding = self.embedding(input_x)
        y, ht = self.rnn(x_embedding, ht)
        outputs = self.fc(y)
        return outputs,ht


    def init_hidden(self,batch_size, hidden_size):
        return torch.zeros(batch_size, hidden_size)

    def char2id_fun(self, input):
        outputs = []
        for i in input:
            index = self.char2id[i]
            outputs.append((index))
        return torch.tensor(outputs)

    def id2char_fun(self,input):
        outputs = []
        for i in range(input):
            char = self.id2char[i]
            outputs.append(char)
        return torch.tensor(outputs)



    def txt_generate(self,start_char, gen_len = 1):
        generate_txt = start_char
        x = self.char2id[generate_txt]
        x_input = self.embedding(x)
        h_first = self.init_hidden(batch_size=1)
        out_put, ht = self.forward(generate_txt)



juan = "hello juan juan, you are a pretty girl."
hei = "hello little hei, you are an ugly boy!"
HIDDEN_SIZE = 128
EMBEDDING_DIM = 256

epoch_size = 10000
vocab = sorted(set(juan))
model = Simple_TXT_Generator(vocab,hidden_size= HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model.train()
juan_id = model.char2id_fun(juan)
input_x = juan_id[:-1]
target_y = juan_id[1:]
for e in range(epoch_size):
    ht = model.init_hidden(batch_size=1,hidden_size=HIDDEN_SIZE)
    outputs, ht = model(input_x,ht)
    loss = criterion(outputs.permute(0,2,1),target_y.unsqueeze(dim = 0))



