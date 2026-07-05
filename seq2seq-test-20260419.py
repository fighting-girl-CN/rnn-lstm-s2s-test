from collections import Counter
import torch
import torch.nn as nn
from torch.nn import attention
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import embedding
from faker import Faker
import random
from babel.dates import format_date
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle


fake = Faker()
fake.seed_instance(42)
FORMATS = [
    'short',
    'medium',
    'long',
    'full',
    'd MMM YYYY',
    'd MMMM YYYY',
    'd/MM/YY',
    'MMMM, d, YYYY'
]
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def creat_dataset(num_samples):
    dataset = []
    human_set = []
    machine_set = []
    for i in range(num_samples):
        dt = fake.date_object()
        human_readable = format_date(dt,format=random.choice(FORMATS),locale='en_US')
        human_readable = human_readable.lower().replace(',', '')
        machine_readable = dt.isoformat()
        dataset.append((human_readable,machine_readable))
        human_set.append(human_readable)
        machine_set.append(machine_readable)
    return dataset,human_set,machine_set

def incidents_generator(datasets):
    dataset = []
    for i in range(len(datasets)):
        data_process = re.split(r'[-\s/]+',datasets[i])
        dataset.extend(data_process)
    return dataset

def unique_word_index(datasets):
    word_process = [PAD_TOKEN]+[SOS_TOKEN] + [EOS_TOKEN] + [UNK_TOKEN] + list(set(datasets))
    word2idx = {w : i for i,w in enumerate(word_process)}
    idx2word = {i : w for i,w in enumerate(word_process)}
    return word2idx,idx2word


dataset,human_dates,machine_dates = creat_dataset(10000)
human_dataset = incidents_generator(human_dates)
machine_dataset = incidents_generator(machine_dates)
hum2idx, idx2hum = unique_word_index(human_dataset)
mac2idx, idx2mac = unique_word_index(machine_dataset)


class DatesDateset(Dataset):
    def __init__(self, dataset, hum2idx, mac2idx):
        super().__init__()
        self.dataset = dataset
        self.hum2idx = hum2idx
        self.mac2idx = mac2idx
        self.length = len(dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item < self.length:
            # 训练数据只加[EOS_IDX]，标签数据要同时加[SOS_IDX]和[EOS_IDX]，[SOS_IDX]是专门给解码器用的启动信号
            hum_data = [hum2idx[i] for i in re.split(r'[-\s/]+',dataset[item][0])] + [EOS_IDX]
            mac_data = [SOS_IDX] + [mac2idx[i] for i in re.split(r'[-\s/]+',dataset[item][1])] + [EOS_IDX]
        else:
            raise IndexError
        # 这里不要写成torch.tensor([hum_data]),torch.tensor([mac_data])，因为这样会在外面多一层括号，导致维度不对
        # 如果一个函数的 return 语句写成 return a, b，Python 会自动将这两个元素打包成一个元组返回
        return torch.tensor(hum_data), torch.tensor(mac_data)

def collate_fn(batch):
    # 这里的collate_fn的输入batch是getitem的输出，也就是一个列表，列表中的每个元素是一个元组，元组的第一个元素是hum_data，第二个元素是mac_data
    hum_data = [item[0] for item in batch]
    mac_data = [item[1] for item in batch]
    hum_data_pad = pad_sequence(hum_data, batch_first=True, padding_value= PAD_IDX)
    mac_data_pad = pad_sequence(mac_data, batch_first=True, padding_value= PAD_IDX)
    return hum_data_pad, mac_data_pad

#
# class BasicEncoder(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, vocab_size):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
#
#     def forward(self, x):
#         input_embedding = self.embedding(x)
#         out_puts, (hidden, cell) = self.lstm(input_embedding)
#         return hidden, cell
#
# class BasicDecoder(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, output_size):
#         super().__init__()
#         self.embedding = nn.Embedding(output_size, embed_dim)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_size)
#
#     def forward(self, input, hidden, cell):
#         input = input.unsqueeze(1)
#         output_embedding = self.embedding(input)
#         outputs, (hidden, cell) = self.lstm(output_embedding, (hidden, cell))
#         output = self.fc(outputs.squeeze(1))
#         return output


class AttentionEncoder(nn.Module):
    def __init__(self,embed_dim,hidden_dim,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: [batch_size, seq_len]
        input_embedding = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(input_embedding)
        # outputs: [batch_size, seq_len, hidden_dim] -> 所有的隐藏状态
        return outputs, hidden, cell


class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_dim)
        self.attn = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim,OUTPUT_SIZE)

    def forward(self, input, last_hidden, last_cell, encoder_outputs):
        # input[20,], last_hidden[1,20,256], last_cee[1,20,256], encoder_outputs[20,5,256]
        input = self.embedding(input.unsqueeze(1)) # input[20,1,128]
        # [20,5,256] 和 [20,5,256] 合并成[20,5,512]
        cat = torch.cat((last_hidden.permute(1,0,2).repeat(1, encoder_outputs.size(1), 1), encoder_outputs),dim = 2)
        # energy [20,5,256]
        energy = torch.tanh(self.attn(cat))
        # scores [20,5,1]
        scores = self.v(energy)
        weights = torch.softmax(scores,dim = 1)
        # context [20,1,256]
        context = torch.bmm(weights.permute(0,2,1), encoder_outputs)
        # input [ 20,1,384]
        input = torch.cat((context, input),dim = 2)
        output,(hidden,cell) = self.lstm(input,(last_hidden,last_cell))
        prediction = self.fc(output)
        return prediction,hidden,cell


class AttenSequence(nn.Module):
    def __init__(self, aec, adc):
        super().__init__()
        self.aec = aec
        self.adc = adc

    def forward(self, src, trg):
        word_length = len(trg[1])
        total_output = torch.zeros(BATCH_SIZE,word_length,OUTPUT_SIZE)
        outputs, hidden, cell = self.aec(src)
        input = trg[:,0]
        for i in range(1, word_length):
            output,hidden,cell = self.adc(input, hidden, cell, outputs)
            total_output[:,i,:] = output.squeeze(1)
            input = trg[:,i]
        return total_output


EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 20
ENPOCHS = 50
VOCAB_SIZE = len(hum2idx)
OUTPUT_SIZE = len(mac2idx)



train_data = DatesDateset(dataset, hum2idx, mac2idx)
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# ecd = BasicEncoder(embed_dim= EMBED_DIM, hidden_dim= HIDDEN_DIM, vocab_size=len(hum2idx))
# dcd = BasicDecoder(embed_dim= EMBED_DIM,hidden_dim= HIDDEN_DIM,output_size= len(mac2idx))
aec = AttentionEncoder(EMBED_DIM,HIDDEN_DIM,VOCAB_SIZE)
adc = AttentionDecoder(EMBED_DIM,HIDDEN_DIM,OUTPUT_SIZE)
model = AttenSequence(aec,adc)
loss = nn.CrossEntropyLoss(ignore_index= PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for t in range(ENPOCHS):
    epoch_loss = 0
    for i,(src,trg) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(src,trg)
        output_dim = len(mac2idx)
        loss_output = outputs[:,1:,:].reshape(-1, output_dim)
        loss_target = trg[:,1:].reshape(-1)
        batch_loss = loss(loss_output, loss_target)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    if t % 5 == 0:
        print(f'Epoch: {t+1}, Loss:{epoch_loss / len(dataloader):.4f}')

# class AttnDecoder(nn.Module):
#     def __init__(self, output_dim, emb_dim, hid_dim):
#         super().__init__()
#         self.output_dim = output_dim
#         self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
#         self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
#
#         # 注意力层：用来计算分数
#         # 输入是 hidden( hid_dim ) + context( hid_dim )，输出 是 1 个分数
#         self.attn = nn.Linear(hid_dim * 2, 1)
#
#         self.fc = nn.Linear(hid_dim + hid_dim, output_dim) # 拼接后的维度翻倍
#
#     def forward(self, input, hidden, cell, encoder_outputs):
#         # input: [batch_size]
#         # encoder_outputs: [batch_size, src_len, hid_dim]
#
#         input = input.unsqueeze(1) # [batch_size, 1]
#         embedded = self.embedding(input) # [batch_size, 1, emb_dim]
#
#         # 1. 运行 LSTM，得到当前步的隐藏状态
#         output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
#         # output: [batch_size, 1, hid_dim]
#
#         # 2. 计算注意力分数
#         # 我们需要把当前的 hidden 复制到 src_len 那么长，以便和 encoder_outputs 拼接
#         # hidden: [batch_size, 1, hid_dim] -> [batch_size, src_len, hid_dim]
#         hidden_expanded = hidden.repeat(1, encoder_outputs.size(1), 1)
#
#         # 拼接：[batch_size, src_len, hid_dim*2]
#         # 这里我们把“当前的想法”和“原文的所有位置”拼在一起，让网络去学习它们的相关性
#         concat = torch.cat((hidden_expanded, encoder_outputs), dim=2)
#
#         # 计算能量值（分数）：[batch_size, src_len, 1]
#         energy = self.attn(concat)
#
#         # 3. 归一化分数 (Softmax)
#         # 让分数变成 0-1 之间的概率分布，加起来等于 1
#         attention = torch.softmax(energy, dim=1) # [batch_size, src_len, 1]
#
#         # 4. 应用注意力：加权求和
#         # 用分数乘以原文输出，得到上下文向量
#         # [batch_size, src_len, 1] * [batch_size, src_len, hid_dim] -> [batch_size, 1, hid_dim]
#         context = torch.bmm(attention.transpose(1, 2), encoder_outputs)
#
#         # 5. 融合信息并预测
#         # 把“上下文”和“当前状态”拼起来，作为预测依据
#         output = torch.cat((context, output), dim=2) # [batch_size, 1, hid_dim*2]
#         prediction = self.fc(output.squeeze(1)) # [batch_size, output_dim]
#
#         return prediction, hidden, cell
# class AttnSeq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, source, target):
#         batch_size = source.shape[0]
#         target_len = target.shape[1]
#         target_vocab_size = self.decoder.output_dim
#
#         outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
#
#         # 1. Encoder 输出所有时间步的结果
#         encoder_outputs, hidden, cell = self.encoder(source)
#
#         input = target[:, 0] # <SOS>
#
#         for t in range(1, target_len):
#             # 2. 把 encoder_outputs 传给 Decoder
#             output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
#
#             outputs[:, t, :] = output
#
#             teacher_force = random.random() < 0.5
#             top1 = output.argmax(1)
#             input = target[:, t] if teacher_force else top1
#
#         return outputs


#
#
#
# # 写到主类的时候，要想到后期是通过dataloader 把数据一个batch 一个batch喂进来
# class Seq2Seq(nn.Module):
#     def __init__(self, ecd, dcd):
#         super().__init__()
#         self.ecd = ecd
#         self.dcd = dcd
#
#     def forward(self, src, trg):
#         word_length = trg.shape[1]
#         trg_size = len(mac2idx)
#         outputs_batch = torch.zeros((BATCH_SIZE, word_length, trg_size))
#         hidden, cell = self.ecd(src)
#         input = trg[:, 0]
#         for i in range(1, word_length):
#             output = dcd(input, hidden, cell)
#             outputs_batch[:,i,:] = output
#             input = trg[:, i]
#         return outputs_batch


