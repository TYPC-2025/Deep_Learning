# 一、张量

- 张量其实就是数组，不过是在深度学习中是这样的叫法

## 1.张量的创建

### （1）基本创建方式

- `torch.tensor()`：根据指定数据创建张量

```python
import torch 
import numpy as np
"""创建张量标量"""
data = torch.tensor(10)
print(data) # 输出结果tensor(10)

"""numpy数组"""
data = np.random.randn(2, 3)
data = torch.tensor(data)
print(data) # 输出结果tensor([[0.1345,0.1149,0.2435],
            #                 [0.8026,-0.6744,-1.0918]],dtype                                             = torch.float64)
    
"""列表，下面代码使用默认元素类型"""
data = [[10.,20.,30.],[40.,50.,60.]]
data = torch.tensor(data)
print(data) # 输出结果tensor([[10.,20.,30.],
            #                [40.,50.,60.])
```

- `torch.Tensor()`：根据指定形状创建张量，也可以用来创建指定数据的张量

```python
"""创建2行3列的张量"""
data = torch.Tensor(2, 3)
"""注意：如果传递列表，则创建包含指定元素的张量"""
data = torch.Tensor([10])

data = torch.Tensor([10, 20]) # tensor([10., 20.])
```

- `torch.IntTensor()、torch.FloatTensor()、torch.DoubleTensor()`：创建指定类型的张量

```python
"""创建2行3列"""
torch.IntTensor(2, 3)

"""列表"""
torch.IntTensor([2,3]) # 输出 tensor([2.,3.])

"""如果类型不符合，则会强制转换"""
torch.IntTensor([2.43,3.42]) # 输出 tensor([2.,3.])
```

### （2）创建线性和随机张量

- `torch.arange`和`torch.linspace`创建线性张量

```python
data = torch.arange(0,10,2) # 输出结果 tensor([0,2,4,6,8])
data = torch.linspace(0,9,10) 
```

- `torch.random.init_seed`和`torch.random.manual_seed`随机种子设置
- `torch.randn`创建随机张量

```python
data = torch.randn(2,3) # 创建2行3列张量

"""查看随机数种子"""
print('随机数种子'：torch.random.initial_seed())

"""随机数种子设置"""
torch.random.manual_seed(100)
data = torch.randn(2,3)
print(data)
print('随机数种子：',torch.random.initial_seed())
```

### （3）创建0-1张量

- `torch.ones`和`torch.ones_like`创建全1张量
- `torch.zeros`和`torch.zeros_like`创建全0张量
- `torch.full`和`torch.full_like`创建全为指定值张量

```python
data = torch.randn(2,3)
torch.zeros(4,5) # 生成4行5列全为0的二维数组
torch.zeros_like(data) #  生成2行3列全为0的二维数组

torch.ones(4,5)
torch.ones_like(data)

torch.full([4,5],10) # 生成4行5列 全为10的二维数组
torch.full_like(data,20)
```

## 2.张量的类型转换

### （1）张量元素的类型转换

- `data.type(torch.DoubleTensor)`：ShortTensor,IntTensor, LongTensor, FloatTensor
- `data.double()`:short, int, long, float

```python
data = torch.randn(4, 5)
print(data)
print(data.dtype) # torch.float32
print(data.type(torch.IntTensor).dtype) # torch.int32
print(data.int().dtype) # torch.int32
```

### （2）张量转换成Numpy数组

- 使用Tensor.numpy函数可以将张量转换为ndarray数组，但是共享内存，可是使用copy函数避免共享

```python
# 共享空间
import torch
import numpy as np

torch.random.manual_seed(2)
data_tensor = torch.randint(0, 10, [2, 3])
print(type(data_tensor))

data_numpy = data_tensor.numpy()
print(type(data_numpy))

data_numpy[0][0] = 100
print(data_numpy)
print(data_tensor)


# 使用copy函数避免空间共享
import torch
import numpy as np

torch.manual_seed(2)
data_tensor = torch.randint(0, 10, [2, 3])
print(type(data_tensor))

data_numpy = data_tensor.numpy().copy()
print(type(data_numpy))

data_numpy[0][0] = 100
print(data_tensor)
print(data_numpy)
```

### （3）Numpy数组转换为张量

- 使用from_numpy可以将naddray数组转换为Tensor，默认共享内存，同样可以使用copy函数避免内存共享
- 使用torch.tensor可以将ndarray数组转换为Tensor，默认不共享内存

```python
# 内存共享
import torch
import numpy as np

data_numpy = np.array([1, 2, 3])
data_tensor = torch.from_numpy(data_numpy)
data_tensor[0] = 10
print(data_numpy)
print(data_tensor)
```

```lua
[10  2  3]
tensor([10,  2,  3], dtype=torch.int32)
```

```python
# 避免内存共享
import torch
import numpy as np

data_numpy = np.array([1, 2, 3])
data_tensor = torch.from_numpy(data_numpy.copy())
data_tensor[0] = 10
print(data_numpy)
print(data_tensor)
```

```lua
[1 2 3]
tensor([10,  2,  3], dtype=torch.int32)
```

```python
# 避免内存共享
import torch
import numpy as np

data_numpy = np.array([1, 2, 3])
data_tensor = torch.Tensor(data_numpy)
data_tensor[0] = 10
print(data_numpy)
print(data_tensor)
```

```lua
[1 2 3]
tensor([10.,  2.,  3.])
```

### （4）标量张量和数字转换

- 对于只有一个元素的张量，使用item()函数将该值从张量中提取出来
  - 注意：在后面的反向传播中，必须要加上item()，否则模型会报错

```python
import torch

data = torch.tensor(30)
print(data) # tensor(30)
print(data.item()) # 30

data1 = torch.tensor([30])
print(data1) # tensor([30])
print(data1.item()) # 30 (得到的不是[30],最终只能得到数值)
```

## 3.张量数值计算

### （1）张量的基本运算

加减乘除取负号

- add, sub, mul, div, neg

- add_,  sub_, mul_, div_, neg_（其中带下划线的版本会修改原数据）

```python
import torch

torch.random.manual_seed(22)
data = torch.randint(0, 10, [2, 3])
print(data)

print(data.add(10))
print(data) # 原数据并没有发生改变
```

```lua
tensor([[9, 6, 6],
        [4, 2, 2]])
tensor([[19, 16, 16],
        [14, 12, 12]])
tensor([[9, 6, 6],
        [4, 2, 2]])
```

```python
import torch

torch.random.manual_seed(22)
data = torch.randint(0, 10, [2, 3])
print(data)

print(data.add_(10))
print(data) # 原数据会发生改变
```

```lua
tensor([[9, 6, 6],
        [4, 2, 2]])
tensor([[19, 16, 16],
        [14, 12, 12]])
tensor([[19, 16, 16],
        [14, 12, 12]])
```

### （2）点乘运算

- 定义：是两个同维矩阵对应位置的元素相乘，使用mul和运算符*实现

```python
import torch

torch.random.manual_seed(22)
data1 = torch.randint(0, 10, [2, 3])
print(data1)

torch.random.manual_seed(23)
data2 = torch.randint(0, 10, [2, 3])
print(data2)

# 点乘
print(torch.mul(data1, data2))

print(data1 * data2)
```

```lua
输出结果：
tensor([[9, 6, 6],
        [4, 2, 2]])
tensor([[1, 6, 6],
        [7, 0, 2]])
tensor([[ 9, 36, 36],
        [28,  0,  4]])
tensor([[ 9, 36, 36],
```

### （3）矩阵乘法

- 矩阵乘法运算要求第一个矩阵shape:(n, m)，第二个矩阵shape:(m ,p)，两个矩阵点积运算shape为(n, p)
  - 运算符@用于进行两个矩阵的乘积运算
  - torch.matmul对进行乘积运算的两矩阵形状没有限定，对数输入的shape不同的张量，对应的最后几个维度必须符合矩阵运算规则

```python
import torch

torch.random.manual_seed(22)
data1 = torch.randint(0, 10, [2, 4])
print(data1)

torch.random.manual_seed(23)
data2 = torch.randint(0, 10, [4, 5])
print(data2)

# 矩阵乘法
print(data1 @ data2)

print(torch.matmul(data1, data2))
```

```lua
输出结果：
tensor([[9, 6, 6, 4],
        [2, 2, 2, 1]])
tensor([[1, 6, 6, 7, 0],
        [2, 7, 1, 4, 7],
        [5, 6, 3, 7, 8],
        [7, 5, 2, 9, 8]])
tensor([[ 79, 152,  86, 165, 122],
        [ 23,  43,  22,  45,  38]])
tensor([[ 79, 152,  86, 165, 122],
        [ 23,  43,  22,  45,  38]])
```

## 4.常见运算函数

- 均值
- 平方根
- 指数计算
- 对数计算

```python
import torch

data = torch.randint(0, 10, [2, 3],dtype = torch.float64)
print(data)

"""计算均值（注意：tensor必须为Float或者Double类型）"""
print(data.mean())
print(data.mean(dim = 0)) # 按列计算均值
print(data.mean(dim = 1)) # 按行进行计算

"""计算总和"""
print(data.sum())
print(data.sum(dim = 0))
print(data.sum(dim = 1))

"""计算平方"""
print(torch.pow(data, 2))

"""计算平方根"""
print(data.sqrt())

"""指数计算"""
print(data.exp())

"""对数计算"""
print(data.log())
print(data.log2())
print(data.log10())
```

## 5.张量索引操作

### （1）索引操作

- 行列索引
- 列表索引

```python
print(data[[0, 2], [1, 2]]) #返回(0, 1)，(2, 2)两个位置的元素

print(data[[[0], [1]], [1, 2]]) # 返回0，1行的1，2列共4个元素
```

- 范围索引

```python
print(data[:3, :2]) # 前3行前2列数据

print(data[2:, :2]) # 第2行到最后的前2列数据
```

- 布尔索引

```python
tensor([[0, 7, 6, 5, 9],
       [6, 8, 3, 1, 0],
       [6, 3, 8, 7, 3],
       [4, 9, 5, 3, 1]])
print(data[data[:, 2] > 5])
print(data[:, data[1] > 5])
```

```lua
输出结果：
tensor([[0, 7, 6, 5, 9],
    [6, 3, 8, 7, 3]])

tensor([[0, 7],
    [6, 8],
    [6, 3],
    [4, 9]])
```

- 多维索引

```python
data = torch.randint(0, 10, [3, 4, 5])
print(data)
#获取0轴上的第一个数据
print(data[0, :, :])

# 获取1轴上的第一个数据
print(data[:, 0, :])

# 获取2轴上的第一个数据
print(data[:, :, 0])
```

## 6.张量形状操作

### （1）reshape函数

```python
import torch

data = torch.tensor([[10, 20, 30], [40, 50, 60]])

# 1.使用shape属性或者size方法都可以获得张量的形状
print(data.shape, data.shape[0], data.shape[1])
print(data.size(), data.size(0), data.size(1))

# 2.使用reshape函数修改张量形状
new_data = data.reshape(1, 6)
print(new_data.shape)
```

### （2）`squeeze()`和`unsqueeze()`函数

- squeeze函数删除形状为1的维度（降维），unsqueeze函数添加形状为1的维度（升维）

```python
import torch

torch.random.manual_seed(22)
data = torch.randint(0, 10, [3, 4, 5])

# 添加维度
data1 = data.unsqueeze(dim = 1).unsqueeze(dim = -1)
print(data1.shape)

# 降低维度
print(data1.squeeze().shape)
```

```lua
输出结果：
torch.Size([3, 1, 4, 5, 1])
torch.Size([3, 4, 5])
```

### （3）`transpose()`和`permute()`函数

- transpose函数可以实现交换张量形状的指定维度；permute函数可以一次交换更多的维度

```python
import torch

torch.random.manual_seed(22)
data = torch.randint(0, 10, [4, 2, 3, 5])
print(data.shape)

# 转换成[3, 4, 5, 2]
data1 = torch.transpose(data, 0, 2)
data2 = torch.transpose(data1, 1, 2)
data3 = torch.transpose(data2, 2, 3)
print(data3.shape)

data4 = torch.permute(data, [2, 0, 3, 1])
print(data4.shape)

print(data.permute([2, 0, 3, 1]).shape)
```

```lua
输出结果：
torch.Size([4, 2, 3, 5])
torch.Size([3, 4, 5, 2])
torch.Size([3, 4, 5, 2])
torch.Size([3, 4, 5, 2])  
```

### （4）view()和contiguous()函数

- view函数也可以用于修改张量的形状，只能用于存储在整块内存中的张量

- 一个张量经过了transpose或者permute函数的处理之后，就无法使用view函数进行形状操作，如果要使用view函数，需要使用contiguous()变得连续以后再使用view函数

```python
import torch

torch.random.manual_seed(22)
data = torch.randint(0, 10, [2, 3])
print(data.shape)

# 判断内存是否连续
print(data.is_contiguous())

print(data.view(-1).shape)

data1 = torch.transpose(data, 0, 1)
print(data1.is_contiguous()) # 内存不连续

data2 = data1.contiguous()
print(data2.view(-1).shape)

if data.is_contiguous():
    data.view(-1)
else:
    data.contiguous().view(-1)
```

```lua
输出结果：
torch.Size([2, 3])
True
torch.Size([6])
False
torch.Size([6])
```

## 7.张量拼接操作

- torch.cat()：可以将两个张量根据指定的维度拼接起来，不改变维度数

```py
import torch
data1 = torch.randint(0, 10, [1, 2, 3])
data2 = torch.randint(0, 10, [1, 2, 3])
print(data1)
print(data2)

# 1.按0维度拼接
new_data1 = torch.cat([data1, data2], dim = 0)
# print(new_data1)
print(new_data1.shape)

# 2.按1维度拼接
new_data2 = torch.cat([data1, data2], dim = 1)
# print(new_data2)
print(new_data2.shape)

# 3.按2维度拼接
new_data3 = torch.cat([data1, data2], dim = 2)
# print(new_data3)
print(new_data3.shape)
```

```lua
输出结果：
tensor([[[5, 1, 8],
         [8, 9, 5]]])
tensor([[[5, 2, 9],
         [0, 5, 6]]])
torch.Size([2, 2, 3])
torch.Size([1, 4, 3])
torch.Size([1, 2, 6])
```

## 8.自动微分模块

- 训练神经网络时，最常用的算法就是反向传播。在该算法中，参数（模型权重）会根据损失函数关于对应参数的梯度进行调整。为了计算这些梯度，pytorch内置了名为torch.autograd的微分引擎，它支持任意计算图的自动梯度计算

```python
"""w = w - L(w.grad)"""
import torch

# 数据  特征+目标
x = torch.tensor(5)
y = torch.tensor(0.)

# 权重  偏置
w = torch.tensor(1, requires_grad=True,dtype = torch.float32)
b = torch.tensor(3, requires_grad=True, dtype = torch.float32)

# 预测
z = w*x + b

# 损失
loss = torch.nn.MSELoss()
loss = loss(z, y)

# 微分
loss.backward()

#梯度
print(w.grad)
print(b.grad)
```

```lu
输出结果：
tensor(80.)
tensor(16.)
```

```python
import torch

x = torch.ones(2, 5)
y = torch.zeros(2, 3)

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b

loss = torch.nn.MSELoss()
loss = loss(z, y)

loss.backward()

print(w.grad)
print(b.grad)
```

```lu
输出结果：
tensor([[0.2782, 1.4126, 0.4037],
        [0.2782, 1.4126, 0.4037],
        [0.2782, 1.4126, 0.4037],
        [0.2782, 1.4126, 0.4037],
        [0.2782, 1.4126, 0.4037]])
tensor([0.2782, 1.4126, 0.4037])
```
