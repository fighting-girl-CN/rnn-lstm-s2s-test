import torch
import torch.nn as nn
import pickle

# ===== 1. 加载词汇表 =====
with open('dino_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
char_to_idx = vocab['char_to_idx']
idx_to_char = vocab['idx_to_char']
vocable_size = vocab['vocable_size']


# ===== 2. 定义模型结构（必须和训练时完全一致！）=====
class DinoRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layer):
        super().__init__()
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        # self.num_layer = num_layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layer,batch_first= False)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,h0=None):
        y,h = self.rnn(x, h0)
        output = self.fc(y)
        return output, y


# ===== 3. 加载训练好的模型 =====
input_size = vocable_size  # 应该是 26 或 27（含 \n？）
hidden_size = 128
output_size = vocable_size
num_layer = 2

model = DinoRNN(input_size, hidden_size, output_size, num_layer)
model.load_state_dict(torch.load('best_dino_model.pth', map_location='cpu'))
model.eval()  # 切换到评估模式！
m = nn.Softmax(dim=1)


def word2onehot(x):
    one_hot = torch.zeros((len(x), vocable_size))
    for i,c in enumerate(x):
        one_hot[i,char_to_idx[c]] =1
    return one_hot

def name_generator(start_c,length,temp):
    name_out = [start_c]
    current_c = start_c
    with torch.no_grad():
        for i in range(length):
            input_x = word2onehot(current_c)
            output,_ = model(input_x)
            output_f = torch.multinomial(m(output/temp),num_samples=1)
            next_c = idx_to_char[output_f.item()]
            name_out.append(next_c)
            current_c = next_c
        word = ''.join(name_out)
    return word

if __name__ == "__main__":
    name_1 = name_generator('a',6,1.2)
    name_2 = name_generator('h',10,2)
    print(name_1, name_2)

    # import torch
    # import torch.nn as nn
    # import pickle
    #
    # # ===== 1. 加载词汇表 =====
    # with open('dino_vocab.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    # char_to_idx = vocab['char_to_idx']
    # idx_to_char = vocab['idx_to_char']
    # vocable_size = vocab['vocable_size']
    #
    #
    # # ===== 2. 定义模型结构（必须和训练时完全一致！）=====
    # class DinoRnn(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size, num_layer):
    #         super().__init__()
    #         self.rnn = nn.RNN(input_size, hidden_size, num_layer, batch_first=False)  # 注意: batch_first=False
    #         self.fc = nn.Linear(hidden_size, output_size)
    #
    #     def forward(self, x, h0=None):
    #         y, h = self.rnn(x, h0)
    #         output = self.fc(y)
    #         return output, h  # 返回 h 而不是 y！
    #
    #
    # # ===== 3. 加载训练好的模型 =====
    # input_size = vocable_size  # 应该是 26 或 27（含 \n？）
    # hidden_size = 128
    # output_size = vocable_size
    # num_layer = 2
    #
    # model = DinoRnn(input_size, hidden_size, output_size, num_layer)
    # model.load_state_dict(torch.load('best_dino_model.pth', map_location='cpu'))
    # model.eval()  # 切换到评估模式！
    #
    #
    # # ===== 4. 修复 word2onehot 函数 =====
    # def word2onehot(char):
    #     """将单个字符转为 one-hot 向量 (1, 1, vocab_size)"""
    #     one_hot = torch.zeros((1, 1, vocable_size))
    #     idx = char_to_idx[char]
    #     one_hot[0, 0, idx] = 1
    #     return one_hot
    #
    #
    # # ===== 5. 修复生成函数 =====
    # def name_generator(start_char, length, temperature=1.0):
    #     """
    #     生成恐龙名字
    #     :param start_char: 起始字符（如 'a'）
    #     :param length: 生成总长度（包含起始字符）
    #     :param temperature: 温度值
    #     :return: 生成的名字
    #     """
    #     if start_char not in char_to_idx:
    #         raise ValueError(f"字符 '{start_char}' 不在词汇表中！")
    #
    #     name = [start_char]
    #     current_char = start_char
    #     h = None  # 隐藏状态
    #
    #     with torch.no_grad():
    #         for _ in range(length - 1):  # 已有1个字符，再生成 (length-1) 个
    #             # 准备输入 (1, 1, vocab_size)
    #             input_tensor = word2onehot(current_char)
    #
    #             # 前向传播
    #             output, h = model(input_tensor, h)  # output: (1, 1, vocab_size)
    #
    #             # 获取 logits 并应用 temperature
    #             logits = output[0, 0, :] / temperature
    #
    #             # 转概率并采样
    #             probs = torch.softmax(logits, dim=-1)
    #             next_idx = torch.multinomial(probs, num_samples=1).item()
    #
    #             # 获取下一个字符
    #             next_char = idx_to_char[next_idx]
    #             name.append(next_char)
    #             current_char = next_char
    #
    #     return ''.join(name).capitalize()
    #
    #
    # # ===== 6. 生成名字 =====
    # if __name__ == "__main__":
    #     print("🦖 恐龙名字生成器（加载已训练模型）")
    #     print("生成示例：")
    #     print(name_generator('a', 7, 0.8))
    #     print(name_generator('t', 8, 1.0))
    #     print(name_generator('s', 9, 1.2))