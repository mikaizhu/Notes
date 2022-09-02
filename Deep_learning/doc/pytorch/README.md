<!--ts-->
* [pytorch autograd](#pytorch-autograd)
* [关于nn.Linear()的一点理解：](#关于nnlinear的一点理解)
* [计算机视觉](#计算机视觉)
   * [知识点记录](#知识点记录)
      * [内核，步长，填充，通道数的作用](#内核步长填充通道数的作用)
      * [保持输出不变](#保持输出不变)
      * [卷积核与全连层](#卷积核与全连层)
      * [特征图的生成过程](#特征图的生成过程)
      * [搭建CNN的技巧](#搭建cnn的技巧)
   * [对CNN中channel的理解](#对cnn中channel的理解)
   * [参考教程](#参考教程)
* [自然语言处理](#自然语言处理)
* [神经网络正则化](#神经网络正则化)
* [设置样本权重](#设置样本权重)
* [初始化权重](#初始化权重)
* [自定义Dataset](#自定义dataset)
* [early stop](#early-stop)
* [RNN](#rnn)
* [LSTM](#lstm)
* [pytorch参数注册](#pytorch参数注册)
* [pytorch 中的序列化checkpoint, 断点恢复](#pytorch-中的序列化checkpoint-断点恢复)
* [为什么要继承nn.Module](#为什么要继承nnmodule)
* [pytorch 自己定义网络层](#pytorch-自己定义网络层)
   * [nn.Parameter()](#nnparameter)
   * [self.register_parameter()](#selfregister_parameter)
* [教程推荐](#教程推荐)

<!-- Added by: zwl, at: 2021年 7月20日 星期二 21时10分06秒 CST -->

<!--te-->

# pytorch autograd

reference: https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95

# 关于nn.Linear()的一点理解：

这里假设nn.Linear(3, 4), 表示输入的神经元个数为3， 输出的神经元个数为4，
后面4个神经元，每个都会被前面三个所连接。因此权重矩阵为(3, 4), 偏置的长度为4

因此nn.Linear就相当于矩阵变换,初始化一个权重矩阵，然后与输入向量相乘，输入为1\*3, 输出为1\*4

<a href="https://sm.ms/image/nIbV7M2JcyU6oBP" target="_blank"><img src="https://i.loli.net/2021/06/02/nIbV7M2JcyU6oBP.png" ></a>

# 计算机视觉

## 知识点记录

### 内核，步长，填充，通道数的作用

内核大小的作用：

- 在神经网络中， 我们通常使用奇数大小的卷积核如7\*7，9\*9, 卷积核的设置需要自
  己调节，如果认为数据中有比较小的特征，通常用小卷积核取卷积。如3\*3, 5\*5。较
  小的卷积核可以提取更多局部的特征，较大的卷积核可以提取比较有代表的特征.

填充的作用：

- 填充指的是padding，也就是在输入的两边填充0或者其他数值。这样的好处是可以保留
  数据的边缘特征.

步长的作用：

- 步长的作用主要是帮助减小输入的特征，如果除不尽的话，内部会做四舍五入的处理。

通道数:

- 通道越多，使用的滤波器也就越多，这样可以让模型学习更多特征。

池化层：

- 对数据进行下采样，如Maxpooling，会取数据中的最大值，降低数据的维度

批归一化：

- 通常归一化后的输入，经过神经网络中的某一层后，输入变得太大或太小，所以引入批
  量归一化

- 批归一化的位置：激活函数层如ReLU之后，dropout层之前

### 保持输出不变

- 我们在卷积神经网络中使用奇数高宽的核，比如3×3，5×5的卷积核，对于高度（或宽度）为大小为2k+1的核，令步幅为1，在高（或宽）两侧选择大小为k的填充，便可保持输入与输出尺寸相同。

- 要想输出为输入的一半，可以设置卷机核大小等于步长大小，padding 设置为0即可

### 卷积核与全连层

卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出
则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接
层的输出变换为四维，1×1卷积层可以看成全连接层，其中空间维度（高和宽）上的每个
元素相当于样本，通道相当于特征。因此，NiN使用1×1卷积层来替代全连接层，从而使空
间信息能够自然传递到后面的层中去。

### 特征图的生成过程

参考2，假如是RGB三个颜色通道，那么每个卷积核都是3通道的，每个通道对RGB分别卷积再相加，就变成一维度了，假如有3个卷积核，那么
最后输出三个特征图。

### 搭建CNN的技巧

参考：
https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7

1. 首先使用较小的卷积核来提取局部特征，然后使用较大的卷积核提取高级的，全局的
   特征
2. 刚开始使用比较少的通道来提取特征，然后慢慢增加通道数。通道学习时候，通常是
   保持通道数量不变
3. 如果数据的边缘信息觉得很重要，使用padding参数
4. 不断增加网络的层数，直到过拟合，然后使用dropout或者正则来减小过拟合。
5. 构建模型时候，使用结构 conv-pool-conv-pool or
   conv-conv-pool-conv-conv-pool的结构，通道数使用32-64-64 or 32-64-128


## 对CNN中channel的理解

在pytorch中，通常第二个维度是channel，所以我们要自己决定什么作为channel维度。
假如现在有个文本数据，在自然语言中，我们通常是将文本数据转换成向量，假如现在有
个文本数据，维度为(N, 8, 10), 各个维度的意义是batch，一句话中有8个token，每个
token用10维度的one-hot向量表示。

那么由于pytorch中第二个维度是channel，所以我们要将数据变成(N, 10, 8)， 因为
one-hot向量是无关的，我们需要提取的是词与词之间的关系，所以使用8作为最后一个维
度.

## 参考教程

参考:
1. https://tianchi.aliyun.com/course/337/3988
2. https://blog.csdn.net/xys430381_1/article/details/82529397
3. https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7

# 自然语言处理

# 神经网络正则化

参考：https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/06_Linear_Regression.ipynb

```
# 神经网络的正则化，其实是优化器的一个参数, 默认是0，就不带惩罚
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-2)
```

在数据量比较大的时候，使用惩罚可以防止过拟合，同时提高准确率

# 设置样本权重

参考：https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/07_Logistic_Regression.ipynb

```

# Class weights
counts = np.bincount(y_train)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"counts: {counts}\nweights: {class_weights}")

class_weights_tensor = torch.Tensor(list(class_weights.values()))
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

# 初始化权重

如何初始化：
- https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
- https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

初始化对训练结果的影响：
- https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

初始化单一层的权重：

```
conv1 = torch.nn.Conv2d(...)
torch.nn.init.xavier_uniform(conv1.weight)
```

初始化指定网络层权重：

```
# 该方法主要应用了nn中的apply方法
# 一个很热门的权重初始化方式是xavier initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
```

面向对象的初始化写法:

```
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal(self.fc1.weight, gain=init.calculate_gain('relu')) 
        
    def forward(self, x_in, apply_softmax=False):
        z = F.relu(self.fc1(x_in)) # ReLU activaton function added!
        y_pred = self.fc2(z)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1) 
        return y_pred
```
# 自定义Dataset

自定义详细教程：

- https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00

有时候在使用我们自己的数据的时候，都要自己定义Dataset。通常我们像下面一样定义
dataset：

```
from torch.utils.data import DadaSet DataLoader
class MyDataSet(DataSet):
  def __init__(self, text, label):
    super().__init__()
    ...

  def __len__(self, label):
    return len(self.lable)

  def __getitem__(self, idx):
    x = self.text[idx]
    y = self.label[idx]
    return x, y
```

我们一般都是重新定义这三个方法, 其中getitem返回的总是特征和标签

```
__init__()
__len__()
__getitem__()
```

实际上，Dataset只是一个迭代器，可以通过dir(Dataset) 查看里面的方法, 除了上面的创建DS的方法以外，还有另一种方法：

纠正上面的说法，dataset不是迭代器，Dataloader才是迭代器，dl通过ds中的getitem方
法，迭代里面的所有数据，ds只是我们自己构造的一个数据结构

```
# DS只是迭代器，DL是用来遍历迭代器的
DL_DS = DataLoader(TD, batch_size=2, shuffle=True)
for (idx, batch) in enumerate(DL_DS):
    # Print the 'text' data of the batch
    print(idx, 'Text data: ', batch['Text'])
    # Print the 'class' data of batch
    print(idx, 'Class data: ', batch['Class'], '\n')
```

DataLoader 中还一个参数 collate_fn，会对所有迭代的数据先进行处理，然后再输出,
也可以看上面教程

# 自定义DataLoader

如果只是想单纯继承dataloader，可以按下面方式：

```
class MyDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
        
next(iter(train_loader))        
```

这里尝试理解下, `*args, **kwargs` 

```
class A:
    def __init__(self, a=1):
        self.a = a
class B(A):
    # 可以发现，使用下面写法，会继承父类中的所有参数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
a = A()
b = B()

a.a # 1
b.a # 1
```



# early stop 

神经网络的训练，并不需要固定的epoch次数，通常我们会使用early stop，即不完全训
练完，当loss不再收敛，就停止训练

```
best_val_loss = np.inf
        for epoch in range(num_epochs):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break
```

# RNN

rnn公式和pytorch官网：https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

rnn相比于DNN就是多了个隐藏层。假如现在有个100天的时间序列，那么输入就是[100, 1], 第一个维度是100，表示句子长度是100， 维度为1表示，每个句子用1个特征表示。
如果是一句话，那么[10, 128],10表示句子长度， 128表示每个单词可以用128维度的向
量表示。

rnn的训练：

```
# 1 RNN()
# 因为RNN是有时间步长的，如果通过RNN这个类，那么我们不用for循环.

# 2 RNNcell()
# cell就是一个很小的单元, 因此我们需要手动设置时间步长，采用for循环.
```

定义rnn:

```
x = torch.randn(100, 3, 128) # word_len, batch_size, imbedding_dim

# 这里input size应该等于word len， hidden size应该等于
rnn = RNN(input_size=100, hidden_size=10)

# 那么查看RNN的内部参数
rnn._parameters.keys()
rnn.weight_ih_l0.shape, rnn.weight_hh_l0.shape, rnn.bias_ih_l0.shape, rnn.bias_hh_l0.shape
(torch.Size([10, 100]),
 torch.Size([10, 10]),
 torch.Size([10]),
 torch.Size([10]))
```

再根据公式可以发现，rnn原理其实就是有很多个时间步长，每个步长只取一个单词的向
量，比如每个单词是128维度，那么每个步长的输入是[1, 128], 而hidden size没有限制

- input_size = embedding dim
- hidden size是你自己设置的超参数

```
out, h0 = rnn(x)
out.shape, h0.shape
(torch.Size([10, 3, 10]), torch.Size([1, 3, 10]))
```

- h0是最后一个时间步长的输出，也就是3个batch，每个输出是[1, 10]
- out是3个batch的输出集合起来, 果然`out[-1, :, :] == h0`
- 我们通常是取out后的最后几层，用来训练神经网络

# LSTM

lstm和rnn都是一样的，只不过内部的架构变了很多，rnn只有一个隐藏层状态h0， 而
lstm有两个隐藏层，h0和c0，因此lstm的输入有h0，c0，xt，输出有yt，ht，ct.

参考：https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

```
# 每个单词用128维度的向量表示
embedding_dim = 128

lstm = nn.LSTM(input_size=embedding_dim, hidden_size=10)
lstm._parameters.keys()
x = torch.randn(10, 3, 128) # word_len, batch_size, embedding_dim
out, (hn, cn) = lstm(x)
print(out.shape)
# torch.Size([10, 3, 10])
```

# pytorch参数注册

当我们想自己diy一个神经网络结构的时候，就需要将网络层参数注册到网络中。

查看哪些参数是注册的，可以print(model)

```
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = [nn.Linear(10, 256)]
    def forward(self, x):
        out = self.encoder[0](x)

print(Net())
# Net()
```

按上面形式定义网络，我们会发现网络结构是空的，并没有被注册。

解决办法：

```
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.ModuleList([nn.Linear(10, 256) for _ in range(10)])
  def forward(self, x):
    for layer in self.encoder:
      out = layer(x)
```

# pytorch 中的序列化checkpoint, 断点恢复

现在有个场景，当做试验的时候，因为实验时间需要很长，或者实验突然中断，导致我们需要从某个epoch开始训练，，这时候我们就需要保存checkpoint.

参考：
- https://zhuanlan.zhihu.com/p/53927068
- https://zhuanlan.zhihu.com/p/133250753


我们除了要保存模型此时的epoch参数，还要保存此时的optimizer的参数, 保存和加载的
方式如下：

```
#save
torch.save({
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict(),
            }, PATH)

#load
model = CivilNet()
optimizer = TheOptimizerClass(model.parameters())

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
start = checkpoint['epoch']
loss = checkpoint['loss']
scheduler.load_state_dict(checkpoint['scheduler'])
```

动手尝试实现：

```
# 自己定义线性层
class Mylayer(nn.Module):
    # 定义z=ax+by
    def __init__(self):
        super().__init__()
        # 定义权重a和b
        #self.weights = nn.Parameter(torch.randn(1, 2))
        self.register_parameter('weights', nn.Parameter(torch.randn(1, 2)))
        # 查看网络的参数
        # print(self._parameters)
    def forward(self, x):
        out = x @ self.weights.T
        return out

# 网络定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Mylayer()
    def forward(self, x):
        x = self.layer(x)
        return x


# 创建数据
data = [[1, 2], [2, 2], [2, 3], [3, 4], [3, 2]]
data = torch.tensor(data)

weights = torch.tensor([2, 3])
model = Net()

# 网络训练
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1000):
    for idx in range(len(label)):
        feature = data[idx].float()
        target = label[idx].float().unsqueeze(0)
        optimizer.zero_grad()
        preds = model(feature)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

checkpoint = torch.load('mymodel200.pt')
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.load_state_dict(checkpoint['model_state_dict'])
start = checkpoint['epoch']
loss = checkpoint['loss']
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()
for epoch in range(start+1, 1000):
    for idx in range(len(label)):
        feature = data[idx].float()
        target = label[idx].float().unsqueeze(0)
        optimizer.zero_grad()
        preds = model(feature)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
    if epoch % 200 == 0:
        print(model.layer.weights)
        print(f'loss: {loss:.4f}')

```

# 为什么要继承nn.Module

- [参考教程](https://zhuanlan.zhihu.com/p/34616199) 

![](https://pic2.zhimg.com/80/v2-ad82550f31457c187b59cf56cdcb3fe5_720w.jpg) 

从上图中可以看出，nn.Module 中含有很多基础和必要的内容

# pytorch 自己定义网络层

比如说我想自己定义一个公式，`z=ax+by` 其中只有两个参数是可以学习的`a` 和 `b`

## nn.Parameter()

先把基础框架搭好

```
class Mylayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x):
        pass
```

```
# 定义方式1
class Mylayer(nn.Module):
    # 定义z=ax+by
    def __init__(self):
        super().__init__()
        # 定义权重a和b
        self.weights = nn.Parameter(torch.randn(1, 2))
        # 查看网络的参数
        # print(self._parameters)
    def forward(self, x):
        # or use , * is element wise
        # out = (x * self.weights).sum(dim=1, keepdim=True)
        out = x @ self.weights.T
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Mylayer()
    def forward(self, x):
        x = self.layer(x)

# 我们可以发现Mylayer已经被注册进Net网络中了
print(Net())
# Net(
  (layer): Mylayer()
)
```

开始训练：

```
# 创建数据

- [参考](https://blog.csdn.net/goodxin_ie/article/details/84680433) 

data = [[1, 2], [2, 2], [2, 3], [3, 4], [3, 2]]
data = torch.tensor(data)

weights = torch.tensor([2, 3])
model = Net()

# 训练网络参数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    for idx in range(len(label)):
        feature = data[idx]
        target = label[idx]
        optimizer.zero_grad()
        preds = model(feature.long())
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
```

但提示出现下面错误：

```
---> 10         out = x @ self.weights.T
     11         return out

RuntimeError: expected scalar type Long but found Float
```

进行修改后，

```
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1000):
    for idx in range(len(label)):
        feature = data[idx].float()
        target = label[idx].float().unsqueeze(0) # 这里feature 和label的维度必
        须一样
        optimizer.zero_grad()
        preds = model(feature)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

print(model.layer.weights)
#Parameter containing:
#tensor([[2.0002, 2.9998]], requires_grad=True)
```

## self.register_parameter()

- [参考](https://blog.csdn.net/xinjieyuan/article/details/106951116) 

```
class Mylayer(nn.Module):
    # 定义z=ax+by
    def __init__(self):
        super().__init__()
        # 定义权重a和b
        #self.weights = nn.Parameter(torch.randn(1, 2))
        self.register_parameter('weights', nn.Parameter(torch.randn(1, 2)))
        # 查看网络的参数
        # print(self._parameters)
    def forward(self, x):
        out = x @ self.weights.T
        return out
```

# 教程推荐

**这里有个教程挺好的：**

https://www.jianshu.com/u/898c7641f6ea

**官网教程查询：**

https://pytorch.org/docs/stable/index.html

**pytorch中文文档：**

https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
