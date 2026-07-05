import torch
import re
import pickle
import torch.nn as nn
import random
from faker import Faker
from babel.dates import format_date

with open('seq_voc.pkl', 'rb') as f:
    vocab = pickle.load(f)
    hum2idx = vocab['hum2idx']
    idx2hum = vocab['idx2hum']
    mac2idx = vocab['mac2idx']
    idx2mac = vocab['idx2mac']

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
INPUT_DIM = len(hum2idx)
OUTPUT_DIM = len(mac2idx)
EMBED_DIM = 128
HID_DIM = 256
EPOCHS = 30
BATCH_SIZE = 32
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

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
        input = input.unsqueeze(1)# 变为 [batch_size, 1],emnedding()函数的输入要求是(Batch, Seq_Len)
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



def creat_dataset(num_samples):
    dataset = []
    for i in range(num_samples):
        dt = fake.date_object()
        human_readable = format_date(dt,format=random.choice(FORMATS),locale='en_US')
        human_readable = human_readable.lower().replace(',', '')
        machine_readable = dt.isoformat()
        dataset.append((human_readable,machine_readable))
    return dataset

def evaluate(model , sentence, hum2idx , idx2mac , max_len = 20):
    model.eval()
    with torch.no_grad():
        tokens = re.split(r'[\s/]+', sentence.lower().replace(',', ' '))
        scr_indexes = [hum2idx.get(token, UNK_IDX) for token in tokens] + [EOS_IDX]
        scr_tensor = torch.LongTensor(scr_indexes).unsqueeze(0)
        hidden, cell = model.encoder(scr_tensor)
        inputs = torch.tensor([SOS_IDX])
        res_tokens = []
        last_pred = -1
        repeat_count = 0

        for i in range(max_len):
            output, hidden, cell = model.decoder(inputs,hidden,cell)
            pred_idx = output.argmax(1).item()

            if pred_idx == EOS_IDX:
                # break会立即终止包含它最近一层的循环for 或者while，继续执行循环体后的代码
                break

            if pred_idx == last_pred:
                repeat_count += 1
                if repeat_count >= 2:
                    break
            else:
                repeat_count = 0

            last_pred = pred_idx
            res_tokens.append(idx2mac[pred_idx])
            inputs = torch.tensor([pred_idx])
    return "-".join(res_tokens)

test_data = creat_dataset(10)
enc = BasicEncoder(INPUT_DIM,EMBED_DIM,HID_DIM)
dec = BasicDecoder(OUTPUT_DIM,EMBED_DIM,HID_DIM)
model = Seq2Seq(enc,dec)
model.load_state_dict(torch.load('seq_to_seq_model.pth'))
model.eval()

for i in range(len(test_data)):
    scr_str, trg_str = test_data[i]
    prediction = evaluate(model, scr_str, hum2idx, idx2mac)
    print(f"Input : { scr_str}")
    print(f"Output : { trg_str}")
    print(f"Predict : { prediction}")