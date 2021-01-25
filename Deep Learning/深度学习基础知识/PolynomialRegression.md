# 介绍下使用pytorch实现多项式回归

什么是多项式回归呢？多项式回归相对于线性回归，就是多项式回归的次数，不仅是1，还有高次项。

比如下面公式：
$$
y=-1.13x-2.14x^2+3.15x^3-0.01x^4+3
$$

我们接下来就是利用神经网络来拟合这个多项式。

怎么进行下去呢？大致思路如下：

- 搜集数据
- 将输入输入到神经网络中
- 然后进行拟合


# 怎么搭建网络？

我们知道神经网络其实就是在学习一个方程，也就是$y=f(x)$

x表示输入的样本点的值。这个函数有一个输入x，一个输出y，那是不是说明这个神经网络的输入是1， 输出也是1呢？只不过中间有很多神经元，加几层隐藏层？

**答案是否定的！**

我们可以这样看待这个问题。这个函数以x为自变量，但是有x的一次，二次，三次和四次项。所以我们可以看成有4个输入，只不过其他三个输入，都和第一个输入x有关。

所以，神经网络的架构出现了：**四个输入神经元，一个输出**

# 神经网络学的是什么？

如果已经知道数据点x和标签y，那么我们就是希望神经网络学习x和y之间的关系，也就是上面那个方程式。

# 创建数据

既然我们知道规则了，就是上面函数

$$
y=-1.13x-2.14x^2+3.15x^3-0.01x^4+3
$$

那么我们怎么构造数据呢？也就是先找到x，然后带入到上面的公式中就可以得到标签y

然后再将4个特征喂到神经网络中，然后和y计算loss，让网络迭代就好了

```
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

x = torch.linspace(-2, 2, 50)
y = -1.13*x + 2.13*torch.pow(x, 2) + 3.15*torch.pow(x, 3) - 0.01*torch.pow(x, 4) + 0.15

# 构建神经网络
class PolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
    def forward(self, t):
        return self.fc(t)

# 构建两个辅助函数
# 第一个辅助函数是构造其他几个输入，比如我们已知x，然后得到其他输入
def get_feature(x):
    # input x and get x^2, x^3, x^4
    res = []
    for i in range(len(x)):
        temp = torch.tensor([x[i]**j for j in range(1, 5)])
        res.append(temp.float().unsqueeze(dim=0))
    return torch.cat(res, dim=0)

# 和权重相乘，然后就可以得到标签
def get_labels(x, weights):
    res = torch.mm(x, weights.unsqueeze(dim=1))
    return res
 
x = torch.linspace(-2, 2, 50)
feature = get_feature(x) # 得到4个输入，feature的shape为50，4

weights = torch.tensor([-1.13, -2.14, 3.15, -0.01])
y = get_labels(feature, weights) + 0.15

model = PolynomialRegression()
optimizer = optim.Adam(model.parameters(), lr=0.4)
loss_func = nn.MSELoss()

plt.figure()
for epoch in range(100):
    preds = model(feature)
    optimizer.zero_grad()
    loss = loss_func(preds, y)
    loss.backward()
    optimizer.step()
    if epoch != 0 and epoch % 5 == 0:
        plt.scatter(x, preds.flatten().detach().numpy())
        plt.scatter(x, y)
        plt.show()
```

然后观察拟合情况即可
然后可以观察参数

```
for i in model.parameters():
    print(i)
```
输出:

```
Parameter containing:
tensor([[-1.1489, -2.1486,  3.1446, -0.0097]], requires_grad=True)
Parameter containing:
tensor([0.1508], requires_grad=True)
```

# 拟合心形线代码

- 注意要使用权重矩阵即weight
- 之前没使用权重矩阵，然后导致标签和数据不能对应，所以拟合的不好

```
t = torch.linspace(-10, 10, 1000)
x = 16*torch.sin(t)**3
y = 13*torch.cos(t)-5*torch.cos(2*t)-2*torch.cos(3*t)-torch.cos(4*t)
class PolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
    def forward(self, t):
        return self.fc(t)

def get_feature(t):
    res = []
    for i in range(len(t)):
        temp = torch.tensor([torch.cos(j*t[i]) for j in range(1, 5)])
        res.append(temp.float().unsqueeze(dim=0))
    return torch.cat(res, dim=0)
t = torch.linspace(-10, 10, 1000)
x = 16*torch.sin(t)**3

y = 13*torch.cos(t)-5*torch.cos(2*t)-2*torch.cos(3*t)-torch.cos(4*t)
feature = get_feature(t)
weight = torch.tensor([13, -5, -2, -1]).float() 
labels = torch.mm(feature, weight.reshape(-1, 1))
model = PolynomialRegression()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_func = nn.MSELoss()

plt.figure()
for epoch in range(10):
    preds = model(feature)
    optimizer.zero_grad()
    loss = loss_func(preds, labels)
    loss.backward()
    optimizer.step()
    if epoch != 0 and epoch % 1 == 0:
        plt.scatter(x, preds.flatten().detach().numpy())
        # plt.scatter(x, y)
        plt.show()
```

