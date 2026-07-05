import numpy as np
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence
from random import shuffle
import math
import os
import time

# 与3.16日相比，增加模型评估并生成名字功能
# 与3.15日相比，调整batch大小为50，删除最后total_loss=0,并做相应调整

# 第 1 步：数据预处理
with open('dinos.txt','r') as f:
    #data 是一个list 包含了1500多个元素，每个元素就是一个恐龙的名字并以回车键结尾
    data = f.readlines()
dataset = [i.lower().rstrip() for i in data]

#制作字符表：将数据列表里的每一个元素依次取出，改为全小写，去除空格后，存放到一个set里去重
char = set()
for name in dataset:
    char = char.union(set(name.strip()))
char = sorted(char)
vocable_size = len(char)


# 制作字符与索引的对应表
char_to_idx = {i:h for h,i in enumerate(char)}
idx_to_char = {h:i for h,i in enumerate(char)}

# 制作每一个单词的独热编码
def word2onehot(x):
    one_hot = torch.zeros((len(x), vocable_size))
    for i,c in enumerate(x):
        one_hot[i,char_to_idx[c]] =1
    return one_hot

# 按50为一个batch封装数据
def data_transit(x):
    input_seq = word2onehot(x[:-1])
    target = torch.tensor([char_to_idx[c] for c in list(x)[1:]])
    return input_seq, target

def batch_bloom(input,target):
    batch_size = 50
    batches = []
    for s in range(len(input) // batch_size):
        batch_data = pad_sequence([torch.tensor(seq) for seq in input[s * batch_size: (s + 1) * batch_size]])
        batch_target = pad_sequence([torch.tensor(seq) for seq in target[s * batch_size: (s + 1) * batch_size]])
        batches.append((batch_data,batch_target))
    return batches

input_set = []
target_set = []
for name in dataset:
    input_s, target_s = data_transit(name)
    input_set.append(input_s)
    target_set.append(target_s)
batches = batch_bloom(input_set,target_set)

#搭建RNN模型
class DinoRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layer):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layer,batch_first= True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,h0=None):
        y,h = self.rnn(x, h0)
        output = self.fc(y)
        return output, y

#设置超参数
input_size = 26
hidden_size = 128
num_layer = 2
output_size = 26
epoch = 1500
# 训练
model = DinoRNN(input_size, hidden_size, output_size, num_layer)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.00001)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

model.train()
for k in range(epoch):
    total_loss = 0
    for i in range(len(batches)):
        input_x = batches[i][0].permute(1,0,2)
        target = batches[i][1]
        h0 = torch.zeros(50,hidden_size)

        optimizer.zero_grad()
        # start_event.record()
        # t1 = time.time()
        out_put, h = model(input_x)

        out_put = out_put.reshape(-1, out_put.size(-1))
        target = target.reshape(-1)
        loss = criterion(out_put, target)

        # print(f"运行时间时间：{time.time()- t1:.5f}")
        # end_event.record()
        # torch.cuda.synchronize()
        # print(f"运行时间：{start_event.elapsed_time(end_event):.5f}")
        total_loss += loss
        loss.backward()
        optimizer.step()
    scheduler.step()

    if (k+1) % 30 == 0:
        print(f"Epoch {k + 1}/{epoch}, Loss:{total_loss.item()/len(batches):.4f}")
#     print(f'lr: {optimizer.param_groups[0]['lr']}')
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             grad_norm = param.grad.data.norm(2)
#             print(f"Layer: {name} | Gradient Norm: {grad_norm}                        ")
torch.save(model.state_dict(),'best_dino_model.pth')


model = DinoRNN(input_size=26, hidden_size=128, output_size=26, num_layer=2)
model.load_state_dict(torch.load('best_dino_model.pth'))
model.eval()
m = nn.Softmax(dim = 1)
name_out = []
def name_generator(start_c,length,temp):

    for i in range(length):
        c = start_c
        name_out.append(c)
        input_x = word2onehot(c)
        output_c,_ = model(input_x)
        output_f = torch.multinomial(m(output_c/temp),num_samples=1)
        start_c = idx_to_char[output_f.item()]
    word = ''.join(name_out)
    return word

name_1 = name_generator('a',6,1.2)
name_2 = name_generator('g',9,0.8)
print(name_1, name_2)

# # ----------------------------
# # 第 2 步：定义 GRU 模型
# # ----------------------------
# class DinoNameGenerator(nn.Module):
#     def __init__(self, vocab_size, hidden_size=128, num_layers=2):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.embedding = nn.Embedding(vocab_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, x, hidden=None):
#         batch_size = x.size(0)
#         if hidden is None:
#             hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#
#         embedded = self.embedding(x)  # (B, L) -> (B, L, H)
#         gru_out, hidden = self.gru(embedded, hidden)  # (B, L, H)
#         logits = self.fc(gru_out)  # (B, L, V)
#         return logits, hidden
#
#
# # ----------------------------
# # 第 3 步：准备训练数据 & 训练
# # ----------------------------
# def create_batches(data_indices, seq_length=25, batch_size=64):
#     """创建 (input, target) 批次"""
#     n = len(data_indices) - 1
#     # 确保能整除
#     total_chars = (n // (batch_size * seq_length)) * batch_size * seq_length
#     x_data = data_indices[:total_chars]
#     y_data = data_indices[1:total_chars + 1]
#
#     x = torch.tensor(x_data, dtype=torch.long).view(batch_size, -1)
#     y = torch.tensor(y_data, dtype=torch.long).view(batch_size, -1)
#
#     batches = []
#     for i in range(0, x.size(1), seq_length):
#         xb = x[:, i:i + seq_length]
#         yb = y[:, i:i + seq_length]
#         if xb.size(1) == seq_length:
#             batches.append((xb, yb))
#     return batches
#
#
# # 超参数
# SEQ_LENGTH = 25
# BATCH_SIZE = 64
# HIDDEN_SIZE = 128
# NUM_LAYERS = 2
# EPOCHS = 30
# LR = 0.003
#
# # 创建批次
# batches = create_batches(data_indices, SEQ_LENGTH, BATCH_SIZE)
# print(f"共 {len(batches)} 个批次")
#
# # 初始化模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DinoNameGenerator(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
# # 训练循环
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#
#     for x_batch, y_batch in batches:
#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)
#
#         optimizer.zero_grad()
#         logits, _ = model(x_batch)
#
#         # 重塑为 (N*L, V) 和 (N*L,)
#         logits = logits.reshape(-1, vocab_size)
#         targets = y_batch.reshape(-1)
#
#         loss = criterion(logits, targets)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     avg_loss = total_loss / len(batches)
#     print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Perplexity: {math.exp(avg_loss):.2f}")
#
#
# # ----------------------------
# # 第 4 步：生成新恐龙名字
# # ----------------------------
# def generate_name(model, start_char=None, max_len=20, temperature=0.8):
#     model.eval()
#     with torch.no_grad():
#         # 随机选一个起始字母（如果不是指定）
#         if start_char is None:
#             letters = [c for c in chars if c.isalpha()]
#             start_char = random.choice(letters)
#
#         input_idx = char_to_idx[start_char]
#         input_tensor = torch.tensor([[input_idx]], device=device)
#         hidden = None
#         name = start_char
#
#         for _ in range(max_len):
#             logits, hidden = model(input_tensor, hidden)
#             # 应用温度
#             logits = logits.squeeze() / temperature
#             probs = torch.softmax(logits, dim=-1)
#
#             # 采样
#             next_idx = torch.multinomial(probs, 1).item()
#             next_char = idx_to_char[next_idx]
#
#             if next_char == '\n' or len(name) >= max_len:
#                 break
#             name += next_char
#             input_tensor = torch.tensor([[next_idx]], device=device)
#
#         return name.capitalize()  # 首字母大写
#
#
# # 生成 10 个新名字
# print("\n🦕 生成的新恐龙名字:")
# for i in range(10):
#     name = generate_name(model, temperature=0.7)
#     print(f"{i + 1}. {name}")

# with open('dinos.txt', 'r') as f:
#     data = f.readlines()
# data = [x.lower().strip() for x in data]
#
# shuffle(data)
# ch_to_id = {ch:i for i,ch in enumerate(vocab)}
# id_to_ch = {i:ch for i,ch in enumerate(vocab)}
#
# #将data按照每批次50个进行封装
# batch_size = 50
# batches = [data[i:i + batch_size] for i in range(0,len(data)-9,batch_size)]
#
#
# #手动one_hot
#
# def one_hot_decoder(batch,vocab_size,ch_to_id):
#     one_hot = np.zeros((len(batch),vocab_size))
#     t = 0
#     for i in batch:
#         one_hot[t,ch_to_id[i]] = 1
#     return one_hot
#
# #搭建RNN模型
# class Rnn(nn.Module):
#     def __init__(self,input_size,hidden_size,output_size):
#         super().__init__()
#
#
#
#
#
#
# #训练
#
#
# #给定名字长度，生成新的名字