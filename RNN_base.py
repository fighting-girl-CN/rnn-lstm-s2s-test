
import torch
import torch.nn as nn
from torch.nn import Embedding
import urllib.request
from pathlib import Path

class MyRNN(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.wah = nn.Linear(hidden_size, hidden_size, bias= False)
        self.wax = nn.Linear(input_size, hidden_size, bias= True)


    def forward(self, x_in, ht):
        outputs = []
        seq_len = x_in.size(1)
        for t in range(seq_len):
            x_t = x_in[:,t,:] #x_t 表示第t个时间步的x取值
            wah = self.wah(ht)
            wax = self.wax(x_t)
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
        for i in input:
            char = self.id2char[i.item()] # 这里i是tensor（15），不是数字15，不能直接用作索引号
            outputs.append(char)
        return "".join(outputs)

    def txt_generate(self,start_char, gen_len = 1):
        outputs = [start_char]
        ht = self.init_hidden(batch_size=1, hidden_size=HIDDEN_SIZE)
        for i in range(gen_len-1):
            x = torch.tensor([self.char2id[start_char]], dtype=torch.long)# torch.tensor()默认创建一个Float数据类型，但embedding层需要的是 Long, Int类型
            out_put, ht = self.forward(x.unsqueeze(0), ht)
            out_idx = torch.argmax(out_put,dim = 2)
            y = self.id2char_fun(out_idx)
            start_char = y
            outputs.append(y)
        return outputs

    def model_train(self,txt,epoch_size,batch_size,seq_len):
        vocab = sorted(set(txt))
        input_id = self.char2id_fun(txt)
        batch_num = len(txt) // (batch_size * seq_len)
        input_seq = input_id[:batch_size * batch_num * seq_len].view(-1,batch_size,seq_len) # 转换成这种形状方便下面切片
        target_seq = input_id[1:batch_size * batch_num * seq_len + 1].view(-1,batch_size,seq_len)

        # input_x = input_id[:-1].unsqueeze(0)
        # target_y =input_id[1:].unsqueeze(0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        for e in range(epoch_size):
            ht = self.init_hidden(batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE) # 隐藏状态应该在一个 Epoch 内持续传递。
            # 处理完第 i 个 Batch 后得到的 ht，应该作为处理第 i+1 个 Batch 的初始状态。
            # 这样，模型才能学习到跨越多个 Batch 的长距离依赖关系。
            for i in range(batch_num):
                input_x = input_seq[i]
                target = target_seq[i]
                optimizer.zero_grad()
                outputs, ht = self(input_x, ht)
                ht = ht.detach()  # 不写这句会报错 在pytorch动态计算图机制中，执行完一次loss.backward()后会计算图会被销毁，
                # ht在batch_num循环外，并在不同的batch之间传递，当处理第一个batch时模型构建了一个计算图，backward后这个图就被销毁了，
                # 但是此时ht仍然记录着第一个batch的信息，当处理第二个batch时，把旧的ht传进去再次调用backward由于之前的图已经被释放，
                # 无法回溯，会报错，需要切断ht与历史计算图的联系（BPTT截断反向传播）
                loss = criterion(outputs.permute(0, 2, 1), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if i % 50 == 0:
                    print(f"Epoch{e}, {i}/{batch_num}, Loss:{loss.item() :.4f}")


# epoch_size = 1000
# vocab = sorted(set(juan))
# model = Simple_TXT_Generator(vocab,hidden_size= HIDDEN_SIZE)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#
# model.train()
# juan_id = model.char2id_fun(juan)
# input_x = juan_id[:-1]
# target_y = juan_id[1:]
# for e in range(epoch_size):
#     # for i in range(len(input_x)):
#     optimizer.zero_grad()
#     ht = model.init_hidden(batch_size=1, hidden_size=HIDDEN_SIZE)
#     outputs, ht = model(input_x, ht)
#     loss = criterion(outputs.permute(0,2,1),target_y.unsqueeze(dim = 0))
#     loss.backward()
#     optimizer.step()
#
#     if e % 50 == 0:
#         print(f"Epoch {e}/{epoch_size}, Loss:{loss.item() :.4f}")


juan = "hello juan juan, you are a pretty girl."
hei = "hello little hei, you are an ugly boy!"

# =================================================
# 步骤1 下载并准备莎士比亚数据库
# ================================================
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
shakespeare_path = Path("resource/shakespeare.txt")

if not shakespeare_path.exists():
    print("正在从网络下载莎士比亚文本...")
    urllib.request.urlretrieve(url, shakespeare_path)

big_txt = []
with shakespeare_path.open(mode='r', encoding='utf-8') as f:
    shakespeare_string = f.read()
    big_txt.extend(line.strip() for line in shakespeare_string)
print(f"文本总长度: {len(shakespeare_string)} 个字符")



HIDDEN_SIZE = 512
EMBEDDING_DIM = 64
BATCH_SIZE = 64
txt = big_txt
vocab = sorted(set(txt))
model = Simple_TXT_Generator(vocab,hidden_size= HIDDEN_SIZE)
model.model_train(txt= txt,epoch_size=30,batch_size=BATCH_SIZE,seq_len=100)

gen_txt = model.txt_generate("a",20)
gen_txt = "".join(gen_txt)
print(gen_txt)




