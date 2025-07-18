# 一、RNN网络原理

- 文本数据是具有序列特性的。为了表示出数据的序列关系，需要使用循环神经网络（Recurrent Nearal Networks,RNN）来对数据进行建模，RNN是一个作用于处理带有序列特点的样本数据

## 1.RNN的计算过程

- h表示隐藏状态，每一次的输入都会包含两个值：上一个时间步的隐藏状态，当前状态的输入值，输出当前时间步的隐藏状态和当前时间步的预测结果

实际上，字是重复输入到同一个神经元中的

## **2.神经元内部的计算过程**

$h_t= tanh(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh})$

- $W_{ih}$表示输入数据的权重
- $b_{ih}$表示输入数据的偏置
- $W_{hh}$表示输入隐藏状态的权重
- $b_{hh}$表示输入隐藏状态的偏置
- 最后对输出结果使用tanh激活函数进行计算，得到该神经元的输出

## 3.API

```python
RNN = torch.nn.RNN(input_size, hidden_size, num_layer)
```

- input_size：输入数据的维度，一般设为词向量的维度
- hidden_size：隐藏层h的维数，也是当前层神经元的输出维度
- num_layer：隐藏层h的层数，默认为1

将RNN实例化就可以将数据送入进行处理，处理方式如下：

```python
output, hn = RNN(x, h0)
```

- 输入数据：输入主要包括词嵌入的x，初始的隐藏层h0

- - x的表示形式为[seq_len, batch, input_size]，即[句子的长度，batch的大小，词向量的维度]
  - h0的表示形式为[num_layers, batch, hidden_size]，即[隐藏层的层数, batch的大小,隐藏层h的维数]（初始化设置为全0）

- 输出结果：主要包括输出结果output，最后一层的hn

- - output的表示形式为[seq_len, batch, input_size]，即[句子的长度，batch的大小，词向量的维度]
  - hn的表示形式为[num_layers, batch, hidden_size]，即[隐藏层的层数, batch的大小,隐藏层h的维数]