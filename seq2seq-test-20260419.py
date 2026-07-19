from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import embedding
from faker import Faker
import random
from babel.dates import format_date
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
from datasets import load_dataset



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

def incidents_generator(datasets): #针对生成日期的清洗
    dataset = []
    for i in range(len(datasets)):
        data_process = re.split(r'[-\s/]+',datasets[i])
        dataset.extend(data_process)
    return dataset

def clean_and_tokenize(text): #针对从hugging face里直接导入的数据的清洗
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens

def unique_word_index(datasets):
    word_process = [PAD_TOKEN]+[SOS_TOKEN] + [EOS_TOKEN] + [UNK_TOKEN] + list(set(datasets))
    word2idx = {w : i for i,w in enumerate(word_process)}
    idx2word = {i : w for i,w in enumerate(word_process)}
    return word2idx,idx2word



class HugDataset(Dataset):
    def __init__(self, hum_data, machine_data, hum2idx, mac2idx):
        super().__init__()
        self.hum = hum_data
        self.mac = machine_data
        self.hum2idx = hum2idx
        self.mac2idx = mac2idx
        self.length = len(hum_data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item < self.length:
            # 训练数据只加[EOS_IDX]，标签数据要同时加[SOS_IDX]和[EOS_IDX]，[SOS_IDX]是专门给解码器用的启动信号
            hum_data = [hum2idx[i] for i in self.hum[item]]+ [EOS_IDX]
            mac_data = [SOS_IDX] + [mac2idx[i] for i in self.mac[item]] + [EOS_IDX]
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
        # input[20,], last_hidden[1,20,256], last_cell[1,20,256], encoder_outputs[20,5,256]
        input = self.embedding(input.unsqueeze(1)) # input[20,1,128]
        # [20,5,256] 和 [20,5,256] 合并成[20,5,512]
        cat = torch.cat((last_hidden.permute(1,0,2).repeat(1, encoder_outputs.size(1), 1), encoder_outputs),dim = 2)
        # energy [20,5,256]
        energy = torch.tanh(self.attn(cat))
        # scores [20,5,1] 是decoder当前的隐藏状态与encoder所有时间步的隐藏状态对齐后计算得到的原始相关性得分，这里用的是Bahdanau加性注意力
        scores = self.v(energy)
        weights = torch.softmax(scores,dim = 1) # 上述scores经过softmax变换后得到注意力权重
        # context [20,1,256]， torch.bmm 是矩阵乘法，也就是获得加权求和的效果 A（batch_size, n, m),B(batch_size, m, p) C= torch.mm(A,B）获得C（batch_size，n,p)
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
        cur_batchsize = src.shape[0]
        word_length = trg.shape[1]
        device = src.device
        total_output = torch.zeros(cur_batchsize,word_length,OUTPUT_SIZE, device=device)
        outputs, hidden, cell = self.aec(src)
        input = trg[:,0]
        for i in range(1, word_length):
            output,hidden,cell = self.adc(input, hidden, cell, outputs)
            total_output[:,i,:] = output.squeeze(1)
            input = trg[:,i]
        return total_output


PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def creat_data(num_data):
    train_dataset = []
    human_set = []
    machine_set = []
    data_files = {
        "train": "https://huggingface.co/datasets/dap305/processed_europarlv7_subset50k/resolve/main/data/train-00000-of-00001.parquet"
    }
    print("正在直接从云端安全建立连接并载入 Parquet 静态数据...")
    dataset = load_dataset("parquet", data_files=data_files)
    print("\n🎉 远程数据集直接调用成功！")
    # sample = dataset['train'][0]
    # print("英文:", sample['translation']['en'])
    for i in range(num_data):
        hum_data = re.findall(r"\w+|[^\w\s]", dataset['train'][i]['translation']['en'].lower())
        machine_data = re.findall(r"\w+|[^\w\s]", dataset['train'][i]['translation']['es'].lower())
        train_dataset.append((hum_data,machine_data))
        human_set.append(hum_data)
        machine_set.append(machine_data)
    return train_dataset,human_set,machine_set

data_set, hum_data, machine_data = creat_data(1000)
hum_data_flat = [item for sublist in hum_data for item in sublist]
machine_data_flat = [item for sublist in machine_data for item in sublist]
hum2idx, idx2hum = unique_word_index(hum_data_flat)
mac2idx, idx2mac = unique_word_index(machine_data_flat)



EMBED_DIM = 512
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 50
VOCAB_SIZE = len(hum2idx)
OUTPUT_SIZE = len(mac2idx)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前训练设备: {device}")

train_data = HugDataset(hum_data,machine_data, hum2idx, mac2idx)
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# ecd = BasicEncoder(embed_dim= EMBED_DIM, hidden_dim= HIDDEN_DIM, vocab_size=len(hum2idx))
# dcd = BasicDecoder(embed_dim= EMBED_DIM,hidden_dim= HIDDEN_DIM,output_size= len(mac2idx))
aec = AttentionEncoder(EMBED_DIM,HIDDEN_DIM,VOCAB_SIZE).to(device)
adc = AttentionDecoder(EMBED_DIM,HIDDEN_DIM,OUTPUT_SIZE).to(device)
model = AttenSequence(aec,adc).to(device)
loss = nn.CrossEntropyLoss(ignore_index= PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for t in range(EPOCHS):
    epoch_loss = 0
    for i,(src,trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        outputs = model(src,trg)
        output_dim = len(mac2idx)
        loss_output = outputs[:,1:,:].reshape(-1, output_dim)
        loss_target = trg[:,1:].reshape(-1)
        batch_loss = loss(loss_output, loss_target)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    if t % 1 == 0:
        print(f'Epoch: {t+1}, Loss:{epoch_loss / len(dataloader):.4f}')





# class DatesDateset(Dataset):  # 日期数据集
#     def __init__(self, dataset, hum2idx, mac2idx):
#         super().__init__()
#         self.dataset = dataset
#         self.hum2idx = hum2idx
#         self.mac2idx = mac2idx
#         self.length = len(dataset)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, item):
#         if item < self.length:
#             # 训练数据只加[EOS_IDX]，标签数据要同时加[SOS_IDX]和[EOS_IDX]，[SOS_IDX]是专门给解码器用的启动信号
#             hum_data = [hum2idx[i] for i in re.split(r'[-\s/]+',dataset[item][0])] + [EOS_IDX]
#             mac_data = [SOS_IDX] + [mac2idx[i] for i in re.split(r'[-\s/]+',dataset[item][1])] + [EOS_IDX]
#         else:
#             raise IndexError
#         # 这里不要写成torch.tensor([hum_data]),torch.tensor([mac_data])，因为这样会在外面多一层括号，导致维度不对
#         # 如果一个函数的 return 语句写成 return a, b，Python 会自动将这两个元素打包成一个元组返回
#         return torch.tensor(hum_data), torch.tensor(mac_data)

#===============基本 Seq2Seq 模型类 ===============================
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

# # 生成表示日期的数据库
# fake = Faker()
# fake.seed_instance(42)
# FORMATS = [
#     'short',
#     'medium',
#     'long',
#     'full',
#     'd MMM YYYY',
#     'd MMMM YYYY',
#     'd/MM/YY',
#     'MMMM, d, YYYY'
# ]


# dataset,human_dates,machine_dates = creat_dataset(10000)
# human_dataset = incidents_generator(human_dates)
# machine_dataset = incidents_generator(machine_dates)
# hum2idx, idx2hum = unique_word_index(human_dataset)
# mac2idx, idx2mac = unique_word_index(machine_dataset)


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



# =====================================================
def creat_data_splits(num_train=1000, num_val=200, num_test=200):
    # 1. 🔗 将三个文件的云端直链全部配置好
    data_files = {
        "train": "https://huggingface.co/datasets/dap305/processed_europarlv7_subset50k/resolve/main/data/train-00000-of-00001.parquet",
        "validation": "https://huggingface.co/datasets/dap305/processed_europarlv7_subset50k/resolve/main/data/validation-00000-of-00001.parquet",
        "test": "https://huggingface.co/datasets/dap305/processed_europarlv7_subset50k/resolve/main/data/test-00000-of-00001.parquet"
    }

    print("正在从云端安全载入完整的三分流 Parquet 数据...")
    dataset = load_dataset("parquet", data_files=data_files)
    print("🎉 训练/验证/测试集全部直接调用成功！")

    # 定义内部清洗和提取小函数
    def process_split(split_name, num_samples):
        pairs, hum_list, mac_list = [], [], []
        # 防止请求数量超过单集上限
        max_available = len(dataset[split_name])
        actual_samples = min(num_samples, max_available)

        for i in range(actual_samples):
            hum_data = re.findall(r"\w+|[^\w\s]", dataset[split_name][i]['translation']['en'].lower())
            machine_data = re.findall(r"\w+|[^\w\s]", dataset[split_name][i]['translation']['es'].lower())
            pairs.append((hum_data, machine_data))
            hum_list.append(hum_data)
            mac_list.append(machine_data)
        return pairs, hum_list, mac_list

    # 2. ⚡ 分别抽取对应数量的数据（调试阶段建议 train: 1000, val: 200, test: 200）
    train_pairs, train_hum, train_mac = process_split('train', num_train)
    val_pairs, _, _ = process_split('validation', num_val)
    test_pairs, _, _ = process_split('test', num_test)

    # 3. 🚨 注意：词表（Vocab）建立必须【只能】用训练集（train_hum/train_mac）！
    # 绝对不能把验证集和测试集的词放进词表，否则属于“数据泄露”
    return train_pairs, val_pairs, test_pairs, train_hum, train_mac


# ========================================================
# 🔄 实例化数据通道
# ========================================================
train_pairs, val_pairs, test_pairs, hum_data_flat_src, mac_data_flat_src = creat_data_splits(1000, 200, 200)

# 展平训练集建立词表
hum_data_flat = [item for sublist in hum_data_flat_src for item in sublist]
machine_data_flat = [item for sublist in mac_data_flat_src for item in sublist]
hum2idx, idx2hum = unique_word_index(hum_data_flat)
mac2idx, idx2mac = unique_word_index(machine_data_flat)

# 封装成 3 个不同的 Dataset 和 DataLoader
train_dataset = HugDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs], hum2idx, mac2idx)
val_dataset = HugDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs], hum2idx, mac2idx)
test_dataset = HugDataset([p[0] for p in test_pairs], [p[1] for p in test_pairs], hum2idx, mac2idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)