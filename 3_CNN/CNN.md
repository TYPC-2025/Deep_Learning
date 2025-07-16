# 一、图像基础知识

## 1.图像基本概念

- 图像是由像素点组成的，每个像素点的取值范围为：[0, 255]（无符号整型，8bytes）。像素值越接近于0，颜色越暗，越接近于黑色；像素值越接近于255，颜色越亮，接近于白色。
- 在深度学习中，我们使用的图像大多是彩色图，彩色图由RGB3个通道组成

## 2.图像的加载

```python
import matplotlib.pyplot as plt
import numpy as np

img1 = np.zeros([200, 300, 3]) # 全0，黑色图像
plt.imshow(img1)
plt.show()

img2 = np.full([200, 300, 3], 255) # 白色图像
plt.imshow(img2)
plt.show()

img3 = np.full([200, 300, 3], 128) # 灰色图像
plt.imshow(img3)
plt.show()
```

# 二、CNN

- 卷积神经网络 是含有卷积层的神经网络，卷积层的作用就是用来自动学习，提取图像的特征
- CNN网络主要由三部分构成：卷积层，池化层和全连接层

- - 卷积层负责提取图像中的局部特征
  - 池化层用来大幅降低参数量级（降维）
  - 全连接层用来输出想要的结果

## 1.卷积层

### （1）卷积计算

1. input 表示输入的图像
2. filter表示卷积核，也叫做卷积核（滤波矩阵）
3. input经过filter得到输出为最右侧的图像，该图叫做特征图

- 卷积运算本质就是在卷积核和输入数据的局部区域间做点积

### （2）Padding

通过上面的卷积计算过程，最终的特征图比原始图像小很多，如果想要保持经过卷积后的图像大小不变，可以在原图像周围添加padding来实现

### （3）多通道卷积计算

- 卷积核的高、宽是超参数，通道是由输入来决定的

### （4）特征图大小

输出特征图的大小和以下参数息息相关：

1. size：卷积核/过滤器大小，一般会选择奇数，比如有$1\times1$、$3\times3$、$5\times5$
2. Padding：零填充的方式
3. Stride：步长

计算方式：

1. 输入图像的大小：$W\times W$
2. 卷积核大小：$F\times F$
3.  $Stride$  $S$
4. $Padding$  $S$
5. 输出图像大小：$N\times N$
6. 则$N = \frac{W-F+2P}{S}+ 1$

### （5）API

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = plt.imread(r"F:\Maker\Learn_Systematically\6_Deep_learning\3_Convolutional_Neural_Networks_CNN\Meeting_at_the_Peak.jpg")
print(img.shape) # [H, W, C]

img = torch.tensor(img).permute(2, 0, 1)   # [H, W, C]--->[C, H, W]

img = img.to(torch.float32).unsqueeze(0) # [C, H, W]--->[B, C, H, W]
print(img.shape)

layer = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), stride = 1, padding=0)
fm = layer(img)
print(fm.shape)   # (W - Kernel_size + 2Padding) / Stride + 1
```

输出结果：

```lua
(5760, 2912, 3)
torch.Size([1, 3, 5760, 2912])
torch.Size([1, 5, 5758, 2910])
```

## 2.池化层

- 池化层（Pooling）降低维度，缩减模型大小，提高计算速度
- 池化过程不会改变特征的通道数
- 分为两种：最大池化和平均池化

### （1）API

```python
"""最大池化"""
nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)

"""平均池化"""
nn.AvgPool2d(kernel_size = 2, stride = 1, padding = 0)
```

#### A、单通道

```python
import torch
import torch.nn as nn

inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).float()
print(inputs.shape)

pooling = nn.MaxPool2d(kernel_size=2, stride = 1, padding=0)
print(pooling(inputs))

pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
print(pooling(inputs))
```

#### B、多通道

```python
import torch
import torch.nn as nn

inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                       [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                       [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]).float()
print(inputs.shape)

pooling = nn.MaxPool2d(kernel_size=2, stride = 1, padding=0)
print(pooling(inputs))

pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
print(pooling(inputs))
```

# 三、卷积神经网络案例

我们需要搭建的网络结果如下：

1. 输入形状：$32\times 32$
2. 第一个卷积层输入3个Channel，输出6个Channel，kernel Size为$3\times 3$
3. 第一个池化层输入$30\times30$，输出$15\times15$，Kernel Size为$2\times2$，Strides为：2
4. 第二个卷积层输入6个Channel，输出16个Channel，kernel Size为$3\times 3$
5. 第二个池化层输入$13\times13$，输出$6\times6$，Kernel Size为$2\times2$，Strides为：2
6. 第一个全连接层输入576维，输出120维
7. 第二个全连接层输入120维，输出84维
8. 最后的输出层输入84维，输出10维

在每个卷积计算之后应用relu激活函数来给网络增加非线性因素

构建网络代码实现如下：

```python
import matplotlib
from torch.utils.data import DataLoader

matplotlib.use('Agg')  # 解决兼容性问题
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import torch.nn as nn
import torch
from torchsummary import summary

# 加载数据集
train_data = CIFAR10(root='data', train=True, transform=Compose([ToTensor()]), download=True)
test_data = CIFAR10(root='data', train=False, transform=Compose([ToTensor()]), download=True)

# 查看数据集信息
print(train_data.data.shape)
print(test_data.data.shape)
print(train_data.classes)
print(train_data.class_to_idx)

# 显示图像
plt.imshow(train_data.data[100])
plt.savefig('cifar_image.png')  # 保存图像到文件
plt.close()

# 如果需要在控制台查看图像路径
print("图像已保存至: cifar_image.png")

"""模型构建"""


class imgClassification(nn.Module):
    # 初始化
    def __init__(self):
        super(imgClassification, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer3 = nn.Linear(in_features=576, out_features=120)
        self.layer4 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.pooling1(x)
        x = torch.relu(self.layer2(x))
        x = self.pooling2(x)
        x = torch.reshape(x, [x.size(0), -1])
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        out = self.out(x)
        return out


# 实例化
model = imgClassification()
summary(model, input_size=(3, 32, 32), batch_size=1)

"""模型训练"""


def train():
    pass
    # 损失函数
    cri = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.99))
    # 遍历每个轮次
    epochs = 10
    loss_mean = []
    for epoch in range(epochs):
        dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
        loss_sum = 0
        sample = 0.1
    # 每隔遍历
        for x, y in dataloader:
            y_pre = model(x)
            loss = cri(y_pre, y)
            loss_sum += loss.item()
            sample += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        loss_mean.append(loss_sum /sample)
        print(loss_sum / sample)

    print('-'*50)
    print(loss_mean)
    # 保存模型权重
    torch.save(model.state_dict(), r'F:\Maker\Learn_Systematically\6_Deep_learning'
                                   r'\3_Convolutional_Neural_Networks_CNN\model.pth')


train()
```

输出结果：

```lua
(50000, 32, 32, 3)
(10000, 32, 32, 3)
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
图像已保存至: cifar_image.png
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [1, 6, 30, 30]             168
         MaxPool2d-2             [1, 6, 15, 15]               0
            Conv2d-3            [1, 16, 13, 13]             880
         MaxPool2d-4              [1, 16, 6, 6]               0
            Linear-5                   [1, 120]          69,240
            Linear-6                    [1, 84]          10,164
            Linear-7                    [1, 10]             850
================================================================
Total params: 81,302
Trainable params: 81,302
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.08
Params size (MB): 0.31
Estimated Total Size (MB): 0.40
----------------------------------------------------------------
2.1190285682678223
2.0913889191367407
2.1235652403397993
2.134627428921786
2.109860506924716
2.0705240423029116
2.112814729863947
2.0445303483442827
2.124096263538707
2.1400482004339043
--------------------------------------------------
[2.1190285682678223, 2.0913889191367407, 2.1235652403397993, 2.134627428921786, 2.109860506924716, 2.0705240423029116, 2.112814729863947, 2.0445303483442827, 2.124096263538707, 2.1400482004339043]
```