import torch
import torch.nn as nn
from torch.nn.functional import embedding
from torch.utils.data import  DataLoader, Dataset
import re
from collections import Counter
import numpy as np


# ！！！非常重要！！！
# LSTM模型里，outputs, hidden, cell的关系：ht = 输出门*tanh(Ct) outputs= [h1,h2,...ht]
# cell是内部长期记忆因素是hidden的上游，决定hidden能提取什么信息;
# hidden是当前时间步的输出，是cell经过tanh函数并被输出门筛选的结果，传递给下一个时间步参与门控计算
# outputs：是每一个时间步hidden的集合
# LSTM完整的调用是：output, (h_n, c_n) = lstm(input, (h_0, c_0))。如果仅仅写lstm（input）就是默认（h0，c0）是从0开始的

def read_and_clean_text(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        text = f.read()
    text = text.lower()
    # re是专门处理正则表达式的工具包，相当于超级查找替换工具 re.sub(找什么，换成什么，在哪里找)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # text = re.sub(r'\s+', ' ', text)
    #text.split()按照空格来切分，并存入一个list，连续空格也可以
    words = text.split()
    return words

# words是一个有17976个元素的列表,也就是按照空格将所有的单词截取后存入一个列表
words = read_and_clean_text('shakespeare.txt')

# Counter()自动统计每个元素出现了多少次,输出一个增强版的字典，默认按次数从多到少排序。能用[]直接调用，是因为有--getitem--这个方法，类似的有--len--、--iter--
count_words = Counter(words)
#unique_words是一个有3086个元素的列表，也就是所有单词中不重复的单词，是用来制作序列对照表的依据
unique_words = list(count_words.keys())
word2idx = {w : i for i,w in enumerate(unique_words)}
idx2word = {i : w for i,w in enumerate(unique_words)}
vocab_size = len(unique_words)

class ShakespeareDateset(Dataset):
    def __init__(self,sentences, seq_length):
        self.sentences = sentences
        self.seq_length = seq_length

    def __len__(self):
        # 数据集共有多少对，总长度要预留一个seq——length给输入，比如sentences长度是10，seq_length是3，那么可索引的范围是0到7，可以防止越界
        return len(self.sentences) - self.seq_length

    def __getitem__(self, idx):
        #定义了每次取第idx个数据时，具体应该返回什么,先截取输入序列X，长度为seq_lenth，
        # 然后再提取紧跟在X后面的那个词作为目标单词
        x_seq = self.sentences[idx : idx + self.seq_length]
        y_target = self.sentences[idx + 1 : idx + self.seq_length + 1]
        x_tensor = torch.tensor(x_seq,dtype=torch.long)
        y_tensor = torch.tensor(y_target,dtype= torch.long)
        return  x_tensor, y_tensor

#尤其注意，这里在创建实例的时候，processed_data应该是什么？
#应该是整个语料库被展平后的序列数字列表，也是一个有17976个元素的列表，相当于用一个数字代替了原语料库的一个单词
#seq_length = 8 相当于拿着长度为8的滑动窗口，在整个长文本上从头滑到尾
# 样本0：【1,2,3,4,5,6,7,8】，样本2【2,3,4,5,6,7,8,9】一直到样本N【N,N+7】
processed_data = [word2idx[i] for i in words]
seq_length = 8
poetry_dataset = ShakespeareDateset(processed_data,seq_length)
batch_size = 50
dataloader = DataLoader(dataset=poetry_dataset, batch_size= batch_size, shuffle=True,drop_last=True)
print(f"数据集总样本数{len(poetry_dataset)}")
print(f"划分成了{len(dataloader)}个批次")


#定义语言模型（LSTM）
#这里继承nn.Module，能够实现参数自动收集、设备自动迁移、训练评估模式切换
class LstmModle(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size):
        super().__init__()
        # nn.Embedding(3028, 100), 创建了一个embedding实例, PyTorch会创建了一个(3028, 100)的矩阵
        # 一开始会随机生成一些数字填进去作为从不同维度理解单词的权重值，每一行对应一个词，行数就是词汇量的大小，随着模型训练，词嵌入矩阵会随着反向传播不断调整
        # 最后会变成有意义的向量， 比如Hi和hello 两个词的词向量在数学空间里会非常接近
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,batch_first= True)
        self.fc = nn.Linear(hidden_size,vocab_size)

    def forward(self,x,hidden = None):
        # 这里是具体调用刚才创建的emnedding实例的forward（）函数，emnedding()函数的输入要求就是(Batch, Seq_Len)
        # 就是给定索引号，去词嵌入矩阵中提取对应单词的词向量。
        embedded = self.embedding(x)
        lstm_out,_ = self.lstm(embedded,hidden)
        out_put = self.fc(lstm_out)
        return out_put

embed_size = 300
hidden_size = 128
# 这里只是创建了一个模型实例，到后面的model(x)才是把x传给forward函数
model = LstmModle(vocab_size, embed_size, hidden_size)

#训练模型

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 500
for epoch in range(epochs):
    step_loss  = 0
    #这里x和y的形状都是(50,8)
    for i, (x,y) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x)
        #loss函数的输入形状一定要搞清楚output 和 y_target 到底应该是什么形状，原则就是每一个需要预测的位置，都要对应一组分数logits和一个目标
        #1.序列预测时，输入中的output是lstm模型最后经过fc函数的输出，形状是(N,S,C),N是批次大小，S是序列长度/时间步，C是词汇表大小/分类数，尤其是C代表了整个训练集唯一不重复的单词个数，也就是预测分类数
        #输入中的y_target的形状应该是（N,S） 如果是情感分析类或者图像分类，整个句子只预测一个类别，y形状应该是(N,)
        #这里就涉及到很重要的一步：维度对齐，把N和S压扁到一起，可以理解为“一共有多少个位置需要预测”,把(N,S,C)变成(N*S,C)一共有N*S个位置要预测，每个位置有C种可能的类别；把(N,S）变成（N*S,）,每个需要预测的位置都有一个明确的标签值
        #最后计算出的loss的形状是？
        loss = criterion(output.reshape(-1,vocab_size),y.reshape(-1))
        #loss反向传播计算模型中每一个可训练参数的梯度
        loss.backward()
        #优化器优化更新参数 = 参数 - 学习率*梯度
        optimizer.step()
        step_loss += loss.item()
        # print(i)
    avg_loss = step_loss / len(dataloader)
    # print(i)
    if(epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs},Loss:{avg_loss:.4f}")

