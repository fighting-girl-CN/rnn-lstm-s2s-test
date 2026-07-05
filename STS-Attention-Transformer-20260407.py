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

fake = Faker()
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

# 生成n个 [('h1','m1'),('h2','m2')...('hn','mn')]个日期数据
# 这里我感觉可以不要放在一个列表里，分成两个，最后可以试着调整一下
def creat_dataset(num_samples):
    dataset = []
    for i in range(num_samples):
        dt = fake.date_object()
        human_readable = format_date(dt,format=random.choice(FORMATS),locale='en_US')
        human_readable = human_readable.lower().replace(',', '')
        machine_readable = dt.isoformat()
        dataset.append((human_readable,machine_readable))
    return dataset

# 将生成的日期数据输入，输出的是两个切分好的列表
def split_word(mixdata):
    human_seq = []
    machine_seq = []
    for i in range(len(mixdata)):
        # re模块，利用正则表达式拆分字符串 r''表示一个原始字符串，[]表示字符集
        # \s代表任意空白字符，包括空格、制表符、换行符等，/ 代表普通斜杠，+表示一个或多个，即[]内的字符出现多次算作一个分隔符
        # 将最终切分好的单个字符串元素存入列表中[‘a', 'b', ...]
        human_data = re.split(r'[\s/]+',mixdata[i][0])
        machine_data = re.split(r'[\s\-]+',mixdata[i][1])
        human_data = [t for t in human_data if t]
        machine_data = [t for t in machine_data if t]
        human_seq.extend(human_data)
        machine_seq.extend(machine_data)
        # total_data.extend(machine_data)
        # final_data.append((human_data,machine_data))
    return human_seq, machine_seq

# 定义全局变量
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# 提取唯一词汇
def creat_incidies(human_dates, machine_dates):
    unique_hum = list(set(human_dates))
    unique_mac = list(set(machine_dates))
    # count_hum = Counter(human_dates)
    # count_mac = Counter(machine_dates)
    # unique_hum = list(count_hum.keys())
    # unique_mac = list(count_mac.keys())

    # 把特殊字符放在最前面
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    unique_hum = special_tokens + unique_hum
    unique_mac = special_tokens + unique_mac
    # 生成字典
    hum2idx = {w : i for i,w in enumerate(unique_hum)}
    idx2hum = {i : w for i,w in enumerate(unique_hum)}
    mac2idx = {w : i for i,w in enumerate(unique_mac)}
    idx2mac = {i : w for i,w in enumerate(unique_mac)}
    return hum2idx, idx2hum, mac2idx, idx2mac


class DateDataset(Dataset):
    def __init__(self, datas, hum2idx, mac2idx):
        super().__init__()
        # self.human_date = human_date
        # self.machine_date = machine_date
        self.datas = datas
        self.hum2idx = hum2idx
        self.mac2idx = mac2idx
        self.length = len(datas)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item < self.length:
            # 这里要重新拆分是因为之前的split_word(mixdata)函数只是制作了词汇表，并没有按照一一对应的关系，比如10000个单词，生成的词汇表有30000多个元素
            # 并没有按照10000个元素这样的批次来整齐排列 现在这个__getitem__就是从10000个原始数据对里，每次取1对，分别对datas[item][0]和[1]单独拆分，放入列表
            # 生成了一个完成序列转换的数据对
            human_date = re.split(r'[\s/]+', self.datas[item][0])
            machine_date = re.split(r'[\s\-]+', self.datas[item][1])
            human_set = [hum2idx[i] for i in human_date]
            machine_set = [mac2idx[i] for i in machine_date]
            human_idx = human_set + [EOS_IDX]
            machine_idx = [SOS_IDX] + machine_set
        else:
            raise IndexError
        return torch.tensor(human_idx), torch.tensor(machine_idx)

def collate_fn(batch):
    human_batch = [item[0] for item in batch]
    machine_batch = [item[1] for item in batch]
    human_padded = pad_sequence(human_batch, batch_first=True)
    machine_padded = pad_sequence(machine_batch, batch_first=True)
    return human_padded, machine_padded


# dataset,voc_data = split_word(my_data)
# # 这里Couter()能够接收list对象，但是不能接受[[1,1],[2,2]]这种嵌套的列表，所以要进行转换，用extend替代append或者
# # flattened_list = [item for sublist in nested_list for item in sublist]
# count_dataset = Counter(voc_data)
# unique_words = list(count_dataset.keys())
# word2idx = {w : i for i,w in enumerate(unique_words)}
# idx2word = {i : w for i,w in enumerate(unique_words)}
# # 将嵌套在list里的元组 每个元素按照规则进行编码转换
# converted_list = [[[word2idx[element] for element in sublist] for sublist in pair] for pair in dataset]
# print(converted_list)

class BasicEncoder(nn.Module):
    def __init__(self,vocab_size,emb_dim,hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emb_dim)
        self.lstm = nn.LSTM(emb_dim,hid_dim,batch_first= True)

    def forward(self,x):
        embedded = self.embedding(x)
        outputs,(hidden,cell) = self.lstm(embedded)
        # outputs 是所有时间步的输出，(hidden, cell) 是最后的压缩“思想”
        return hidden,cell

class BasicDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        # output_dim 是 machine_vocab 的大小
        self.embedding = nn.Embedding(output_dim,emb_dim)
        self.lstm = nn.LSTM(emb_dim,hid_dim,batch_first=True)
        self.fc = nn.Linear(hid_dim,output_dim)

    def forward(self,input,hidden,cell):
        # input shape: [batch_size] (当前步输入的单个字符索引)
        input = input.unsqueeze(1) # 变为 [batch_size, 1],emnedding()函数的输入要求是(Batch, Seq_Len)
        embedded = self.embedding(input)
        # 核心：Decoder 每一拍都拿着上一步的 hidden 和 cell 继续工作
        output, (hidden,cell) = self.lstm(embedded,(hidden,cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    # 创建实例的时候，需要明确encoder和decoder结构，调用的时候需要输入两个参数 source：一个句子序列化之后的数据，target：对应句子的翻译序列化之后的数据，按照批次数大小叠加在一起
    def __init__(self,encoder, decoder, teacher_forcing_radio = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocable_size = OUTPUT_DIM
        # 创建一个0张量用来后续存放预测结果 outputs 可以理解为有多少个地方需要预测，批次数（一次处理多少个句子） * 每个句子有多长 * 每一位可能的取值数（也就是目标词汇表的唯一单词数）
        outputs = torch.zeros(batch_size,target_len,target_vocable_size)
        hidden, cell = self.encoder(source)
        # target[:,0]是指target数组的第一列
        input = target[:,0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden,cell)
            # 把每一个时间步的输出结果按照顺序存放到outputs里
            outputs[:,t,:] = output
            # 教师强制，是训练的一个技巧，以50%的概率选择训练集中的“标准答案”或者模型当前时间步预测的结果作为下一步的输入
            teacher_force = random.random() < 0.5
            top1 = output.argmax(1)
            input = target[:,t] if teacher_force else top1
        return outputs

my_data = creat_dataset(10000)
human_dates, machine_dates = split_word(my_data)
hum2idx, idx2hum, mac2idx, idx2mac = creat_incidies(human_dates,machine_dates)
dataset = DateDataset(my_data, hum2idx, mac2idx)
INPUT_DIM = len(hum2idx)
OUTPUT_DIM = len(mac2idx)
EMBED_DIM = 128
HID_DIM = 256
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

enc = BasicEncoder(INPUT_DIM,EMBED_DIM,HID_DIM)
dec = BasicDecoder(OUTPUT_DIM,EMBED_DIM,HID_DIM)
model = Seq2Seq(enc,dec)

optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index= PAD_IDX)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i,(src,trg) in enumerate(dataloader):
        # scr:[batch_size, src_len] trg:[batch_size, trg_len]
        optimizer.zero_grad()
        # output的大小应该是一个（batch_size,target_len,target_vocable_size）的张量，这里是（32,4,92）
        output = model(src,trg)
        output_dim = output.shape[-1]
        # loss函数预测值的输入形状必须是二维张量[N,C],N是这一批次中所有单词总数，C代表词表大小
        loss_input = output[:,1:,:].reshape(-1, output_dim)  # (32*3 = 96 , 92) 32个句子同时训练，每个句子有3个位置需要预测，每个位置有92种可能的结果
        loss_target = trg[:,1:].reshape(-1)  # (96,) 32个句子同时训练，每个句子有3个位置需要预测，每个位置就是1种确定的结果
        loss = criterion(loss_input, loss_target)  #
        loss.backward()
        #梯度修剪
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch:{epoch+1:02}, Loss:{epoch_loss/len(dataloader):.4f}')

def evaluate(model , sentence, hum2idx , idx2mac , max_len = 20):
    model.eval()
    with torch.no_grad():
        tokens = re.split(r'[\s/]+', sentence.lower().replace(',', ' '))
        tokens = [t for t in tokens if t]
        scr_indexes = [hum2idx.get(token, UNK_TOKEN) for token in tokens] + [EOS_IDX]
        scr_tensor = torch.LongTensor(scr_indexes).unsqueeze(0)
        hidden, cell = model.encoder(scr_tensor)
        inputs = torch.tensor([SOS_IDX])
        res_tokens = []
        for i in range(max_len):
            output, hidden, cell = model.decoder(inputs,hidden,cell)
            pred_idx = output.argmax(1).item()
            if pred_idx == EOS_IDX:
                break
            res_tokens.append(idx2mac[pred_idx])
            inputs = torch.tensor([pred_idx])
    return "-".join(res_tokens)

test_data = creat_dataset(10)
for i in range(len(test_data)):
    scr_str, trg_str = test_data[i]
    prediction = evaluate(model, scr_str, hum2idx, idx2mac)
    print(f"Input : { scr_str}")
    print(f"Output : { trg_str}")
    print(f"Predict : { prediction}")

