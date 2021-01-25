# 使用神经网络完成逻辑回归

# 什么是逻辑回归？

假如现在有个方程：
$$
y=wx+b
$$
那么可以得到下面的式子：
$$
w_1x_1+w_2x_2+b>0
$$
即根据上面公式，如果结果大于0，那么就是正例，如果小于0就是负例。

即假如我们得到样本点，每个样本点有两个特征$x_1,x_2$那么带入到上面公式中，就可以得到结果

假如我们知道数据是线性可分的，即可以用一个超平面将数据分隔开来和多项式一样，所以

神经网络的输入层是2个，因为有两个特征，输出也有两个，因为有两个类别。

# 创建样本点

```
# 导入模块
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 创建样本点
data = zip(x1.data.numpy(), x2.data.numpy())

positive = []
negtive = []

def classification(data):
    for i in data:
        if (i[0] > 1.5+0.1*torch.rand(1).item()*(-1)**torch.randint(1,10,(1, 1)).item()):
            positive.append(i)
        else:
            negtive.append(i)
classification(data)

p_x = [i[0] for i in positive]
p_y = [i[1] for i in positive]
n_x = [i[0] for i in negtive]
n_y = [i[1] for i in negtive]

# 数据处理与神经网络的搭建
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, t):
        return self.sigmoid(self.fc(t))
func_loss = nn.CrossEntropyLoss()
model = LogisticRegression()
optimizer = optim.Adam(model.parameters(), lr=0.1)

p_x = torch.tensor(p_x).float()
p_y = torch.tensor(p_y).float()
n_x = torch.tensor(n_x).float()
n_y = torch.tensor(n_y).float()

# 创建数据特征
pdata = torch.stack([p_x, p_y], dim=1)
ndata = torch.stack([n_x, n_y], dim=1)

data = torch.cat([pdata, ndata])

# 为每个类别添加标签
py = torch.full((pdata.shape[0], 1), 1.0)
ny = torch.full((ndata.shape[0], 1), 0.0)
labels = torch.cat([py, ny])

for epoch in range(100):
    preds = model(data)
    optimizer.zero_grad()
    loss = func_loss(preds, labels.view(-1).long())
    loss.backward()
    optimizer.step()
    print(f'loss:{loss}')
    acc = preds.argmax(dim=1).eq(labels.flatten()).sum().item()
    print(f'acc:{acc/data.shape[0]*100:4f}%')
```
