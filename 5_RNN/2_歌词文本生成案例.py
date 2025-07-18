import jieba
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import traceback  # 用于打印详细错误信息


# 构建词表
def build_vocab():
    all_words = []
    unique_words = []
    file_path = r'F:\Maker\Learn_Systematically\6_Deep_learning\5_RNN\jaychou_lyrics.txt'
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return [], {}, 0, []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = jieba.lcut(line.strip())
                all_words.append(words)
                for word in words:
                    if word not in unique_words:
                        unique_words.append(word)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], {}, 0, []

    special_tokens = ["<PAD>", "<START>", "<END>"]
    for token in special_tokens:
        if token not in unique_words:
            unique_words.append(token)

    word2index = {word: i for i, word in enumerate(unique_words)}
    print(f"词表大小: {len(unique_words)}")

    print(f"分词结果示例: {all_words[:5]}")
    corpus_id = []
    for words in all_words:
        temp = []
        for word in words:
            temp.append(word2index.get(word, word2index["<PAD>"]))
        temp.append(word2index[" "])
        corpus_id.extend(temp)

    print(f"语料库长度: {len(corpus_id)}")
    return unique_words, word2index, len(unique_words), corpus_id


# 构建数据集
class LyricsDataset(Dataset):
    def __init__(self, corpus_id, num_char):
        self.corpus_id = corpus_id
        self.num_char = num_char
        self.word_count = len(self.corpus_id)
        self.num = max(0, self.word_count - self.num_char - 1)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        start = idx
        end = start + self.num_char
        x = self.corpus_id[start:end]
        y = self.corpus_id[start + 1:end + 1]
        x = x + [0] * (self.num_char - len(x))
        y = y + [0] * (self.num_char - len(y))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# 模型构建
class TextGenerator(nn.Module):
    def __init__(self, word_count, embedding_dim=128, hidden_size=256, num_layers=1):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(num_embeddings=word_count, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(in_features=hidden_size, out_features=word_count)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, inputs, hidden):
        embeds = self.embed(inputs)
        out, hid = self.rnn(embeds, hidden)
        out = self.out(out.reshape(-1, self.hidden_size))
        return out, hid

    def init_hidden(self, bs):
        return torch.zeros(self.num_layers, bs, self.hidden_size)


# 模型训练
def train(dataset, model, epochs=10, batch_size=32, lr=0.001):
    if len(dataset) == 0:
        print("错误：数据集为空，无法训练")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cri = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_path = r'F:\Maker\Learn_Systematically\6_Deep_learning\5_RNN\model.pth'

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_num = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            h0 = model.init_hidden(batch_size).to(device)

            out, _ = model(x, h0)
            loss = cri(out, y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_num += 1

            if batch_num % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_num}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / batch_num
        print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    return model


# 模型预测
def predict(model, start_word, len, unique_words, word2index):
    model.eval()
    model_path = r'F:\Maker\Learn_Systematically\6_Deep_learning\5_RNN\model.pth'

    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在，请先训练模型")
        return

    try:
        # 先加载到CPU，再移动到目标设备
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # 检查模型参数是否匹配
        model_keys = set(model.state_dict().keys())
        saved_keys = set(state_dict.keys())

        # 打印不匹配的参数
        if model_keys != saved_keys:
            print("模型参数不匹配:")
            print("当前模型有但保存模型没有的参数:", model_keys - saved_keys)
            print("保存模型有但当前模型没有的参数:", saved_keys - model_keys)

        model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        traceback.print_exc()
        return

    if start_word not in word2index:
        print(f"错误：起始词 '{start_word}' 不在词表中")
        return

    wor_index = word2index[start_word]
    h0 = model.init_hidden(bs=1).to(device)
    words_list = [start_word]

    with torch.no_grad():
        for _ in range(len):
            input_tensor = torch.tensor([[wor_index]], dtype=torch.long).to(device)
            out, h0 = model(input_tensor, h0)
            wor_index = torch.argmax(out, dim=1).item()
            next_word = unique_words[wor_index]
            words_list.append(next_word)

    print("生成的歌词:")
    for word in words_list:
        print(word, end=' ')
    print()


if __name__ == '__main__':
    unique_words, word2index, word_count, corpus_id = build_vocab()

    if word_count == 0:
        print("词表构建失败，程序退出")
    else:
        dataset = LyricsDataset(corpus_id, num_char=10)
        print(f"数据集大小: {len(dataset)}")

        model = TextGenerator(word_count)
        print(model)

        model_path = r'F:\Maker\Learn_Systematically\6_Deep_learning\5_RNN\model.pth'
        if not os.path.exists(model_path):
            print("开始训练模型...")
            train(dataset, model, epochs=10, batch_size=32, lr=0.001)

        predict(model, '青春', 50, unique_words, word2index)