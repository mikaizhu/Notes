# pytorch 实现线性回归

线性回归的任务是什么？

我们希望拟合出下面的函数,也就是函数$f(x)$

$$
f(x_i)=Wx_i+b
$$

完整代码如下：

```
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

x = torch.arange(20)
y = x*4 + torch.randint(1, 5, size=x.shape)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, t):
        return self.fc(t)

x = x.float()
y = y.float()

model = LinearRegression()
optimizer = optim.Adam(model.parameters(), lr=0.4)
loss_f = nn.MSELoss()

# 绘制学习图，并进行拟合
plt.figure()
for epoch in range(20):
    preds = model(x.unsqueeze(dim=1))
    optimizer.zero_grad()
    loss = loss_f(preds, y)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0 and epoch != 0:
        plt.plot(x.data.numpy(), preds.flatten().data.numpy())
        plt.scatter(x, y)
        plt.show()
```
# 疑问

## 为什么只要1，1神经元就好了？

从神经网络的基本结构就可以知道，根据公式，神经元的前向传播计算，其实就是矩阵相乘。输入的神经元只有1个，代表每个数据点x。输出的结果也只有1个，就是y的值。

一个神经元就可以满足训练线性方程。是因为参数只有一个w，和一个偏置b。所以这样设计

## 为什么要将数据变成2维的？

所以，神经网络是支持并行计算的，所以有个维度可以是batch。

我们可以看到在最下面进行拟合的时候，将x进行了unsqueeze操作。为什么要这样操作呢？

unsqueeze操作是干什么的？设置成dim=1，也就是将x从原先的[20,]变成了[20, 1]

这样的意义是什么？20表示有20个batch，1表示每个batch有1个数据点。

$$
\begin{bmatrix}x&1\end{bmatrix}\begin{bmatrix}w\\b\end{bmatrix}
$$

上面这是只有一个batch的情形，假如有多个batch的时候，是怎么计算的呢？

$$
\begin{bmatrix}x_1&1\\x_2&1\\\vdots&\vdots\\x_m&1\end{bmatrix}\begin{bmatrix}w\\b\end{bmatrix}=\begin{bmatrix}wx_1+b\\wx_2+b\\\vdots\\wx_m+b\end{bmatrix}
$$

然后对所有点的loss进行求和。并根据loss的公式，进行梯度下降，梯度下降的方法就是optimizer的方法，这里选择的是Adam。


Tips:因为输出的preds也是20，1, 所以在画图的时候要flatten下。 

## 只能使用Adam优化器吗？

不一定，还可以使用SGD，但adam优化器是比较好的，使用不同的优化器，学习率也不一样。
