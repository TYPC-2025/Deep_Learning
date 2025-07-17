# 一、自然语言概述

- 自然语言处理研究的主要是通过计算机算法来理解自然语言。对于自然语言来说，处理的数据主要就是人类的语言，该类型的数据不像我们前面接触的结构化数据或图像数据可以很方便的进行数值化

# 二、词嵌入层

- 词嵌入层的作用就是将文本转换为向量，词嵌入层首先会根据输入的词的数量构建一个词向量矩阵，例如：我们有5个词，每个词希望转换成3维度的向量，那么构建的矩阵的形状即为5*3，输入的每个词都对应了一个该矩阵中的一个向量



在Pytorch中，使用nn.Embedding词嵌入层来实现输入词的向量化

```python
nn.Embedding(num_embeddings = 10, embedding_dim = 4)
```

nn.Embedding对象构建时，最主要有两个参数：

- num_embeddings表示**词的数量**
- embedding_dim表示**用多少维的向量来表示每个词**



将词转换为词向量的步骤：

1. 先将语料进行分词，构建词与索引的映射，我们可以把这个映射叫做词表，词表中的每个词都对应了一个唯一的索引
2. 然后使用nn,Embedding构建词嵌入矩阵，词索引对应的向量即为该词对应的数值化后的向量表示

```python
import jieba
import torch
import torch.nn as nn

# 分词
text = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途"
words = jieba.lcut(text)
print(words)

print("-"*50)
# 去重
un_words = list(set(words))
print(un_words)
print("-"*50)
num = len(un_words)
print(num)

# 调用embedding
embeds = nn.Embedding(num_embeddings=num, embedding_dim=3)
print(embeds(torch.tensor(4))) # 获取  '已经'  的词向量

print("-"*50)

for i,word in enumerate(un_words):
    print(word)
    print(embeds(torch.tensor(i)))
```

