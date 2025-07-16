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