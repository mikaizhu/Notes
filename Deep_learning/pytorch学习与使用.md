[TOC]

# pytorch学习与使用

## 常用模块介绍

先介绍下常用的模块

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003110038214.png" alt="image-20201003110038214" style="zoom: 33%;" />

- torch模块是最基础的模块，包括了其他模块
- nn模块包括了各种类和神经网络模型
- autograd包括各种对张量求导求微分的函数
- nn.functional 包括了很多激活函数
- optim包括了很多算法，如随机梯度下降法
- utils包括了很多数据处理的函数，可以简化数据处理的过程
- torchivision包括了计算机视觉专门的模块，对图像进行处理

## GPU介绍

CPU通常用来计算，GPU计算速度大于CPU是取决于计算的数据内省的。

==最适合GPU的计算是可以并行的计算==

可以并行处理的数据，就是可以将数据分成很多小的部分，经过计算后，再将计算结果组合起来。分成多少个部分取决于处理单元有多少个内核。CPU通常有10多个内核。而GPU有上千个内核。所以GPU可以很高效处理并行的数据。

**为什么GPU通常使用在深度学习中？**

因为深度学习中的数据，通常都是并行处理的数据，例如处理图片数据，我们可以同时使用多个内核进行工作，每个内核都可以单独执行自己的任务

## 代码介绍

- 查看版本

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003132929469.png" alt="image-20201003132929469" style="zoom: 50%;" />

- 将张量放入到GPU上

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003133215527.png" alt="image-20201003133215527" style="zoom:50%;" />

==在调试的时候，我发现，将张量放置到GPU上，会花比较多的时间，是不是将数据都放入到GPU上计算，都会节省大量时间呢？==

> 答案是否定的。将数据从CPU上转移到GPU上，是要比较高的成本的，所以如果数据很小很简单，我们没必要将数据转移到GPU上，相反，数据比较复杂的时候，我们就可以利用GPU进行计算，这样会节省更多的时间。

==GPU是硬件，CUDA是基于GPU的软件==

- 什么是nd array？

n dimention array。说明张量就是n纬度的

- 张量的不同纬度的介绍

> 假如现在处理 CNN 图像（卷积神经网络）问题，那么张量因该有4个纬度，首先，照片应该有长宽，两个纬度就没了，其次，照片应该有颜色，颜色通道RGB三种。最后，应该是图片数量，我们叫 batch_size。所以，要定位到某个像素，我们要在4个纬度上设置好，然后就能找到。
>
> 参考视频 https://www.youtube.com/watch?v=k6ZF1TSniYk&list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG&index=7

## 知识点介绍

1. 数据类型与结构

   ==三种属性==

   - device。查看数据在什么设备上，不同设备上的数据是不能进行计算的
   - dtype。查看数据类型，不同类型的数据，也是不能计算的，现在更新版本，可以计算了
   - layout。查看数据的存储形式，有顺序存储和链式存储，创建的tensor存储默认是顺序存储。

   

   <img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003144559527.png" alt="image-20201003144559527" style="zoom: 50%;" />

   torch.layout 可选`torch.stried`或`torch.sparse_coo`。分别对应顺序储存、离散储存。

   一般说，稠密张量适用`torch.stried`，稀疏张量（0 比较多）适用`torch.sparse_coo`。

   ==数据类型==

   - 每个数据类型，都分为CPU版本和GPU版本

2. 将numpy数据转换成tensor

   ```python
   data = np.array([1,2,3])
   type(data)
   ```
   
   ```
   torch.Tensor(data)
   
   torch.tensor(data)
   
   torch.as_tensor(data)
   
   torch.from_numpy(data)
   ```

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003145940601.png" alt="image-20201003145940601" style="zoom: 50%;" />

> Tensor和tensor的区别，Tensor接口是constructer函数，下面三个都是factory函数。factory函数有更多的属性。
>
> Tensor创建tensor的时候，默认使用的数据类型是float

3. 创建tensor

   ```python
   torch.eye(2)
   
   torch.zeros(2,2)
   
   torch.ones(2,2)
   
   torch.rand(2,2)
   ```

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003150056954.png" alt="image-20201003150056954" style="zoom:50%;" />

4. 内存共享

   ```
   data = np.array([1,2,3])
   t1 = torch.Tensor(data)
   t2 = torch.tensor(data)
   t3 = torch.as_tensor(data)
   t4 = torch.from_numpy(data)
   print(t1)
   print(t2)
   print(t3)
   print(t4)
   data[0] = 0
   data[1] = 0
   data[2] = 0
   print(data)
   print(t1)
   print(t2)
   print(t3)
   print(t4)
   
   tensor([1., 2., 3.])
   tensor([1, 2, 3])
   tensor([1, 2, 3])
   tensor([1, 2, 3])
   [0 0 0]
   tensor([1., 2., 3.]) # torch.Tensor(data)
   tensor([1, 2, 3]) # torch.tensor(data)
   tensor([0, 0, 0]) # torch.as_tensor(data)
   tensor([0, 0, 0]) # torch.from_numpy(data)
   ```

   内存共享能够更加节省时间：

   - torch.tensor() # 比Tensor接口，可以设置更多的参数，比如可以改变数据类型，设置dtype参数
   - Torch.as_tensor() # 比from numpy更好用，因为可以接受其他不是numpy的数据，甚至数组也可以接受

   这两个函数我们在日常生活中使用的比较多。

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003154123856.png" alt="image-20201003154123856" style="zoom: 33%;" />

> 左边是共享内存的，右边不共享内存

## reshape

接下来介绍三个方法：

- reshape
- squeeze
- unsqueeze

1. Reshape

```
data = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
])
```

```
data.reshape(-1,2,3)
data.reshape(2,6)
```

2. squeeze and unsqueeze
```
print(data.reshape(1,12))
print(data.reshape(1,12).shape)
print(data.reshape(1,12).squeeze())
print(data.reshape(1,12).squeeze().shape)
print(data.reshape(1,12).squeeze().unsqueeze(dim=0))
print(data.reshape(1,12).squeeze().unsqueeze(dim=0).shape)

tensor([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
torch.Size([1, 12])
tensor([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
torch.Size([12])
tensor([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
torch.Size([1, 12])
```

> squeeze常用来构建flatten的张量。意味着只保留一个纬度，其他纬度都将被删除。如果将卷积层转换成全连接层的时候。
>
> `data.reshape(-1,1).squeeze()`  squeeze适用于任何只要有一个纬度是1的情况.但三维的时候好像不适用
>
> unsqueeze就是在指定纬度上加上一个纬度。dim=0表示从左边往右边数，加上一个纬度
>
> 参考教程：https://blog.csdn.net/xiexu911/article/details/80820028

## cat合并两个张量

```
t1 = torch.tensor([
    [1,1],
    [2,2]
])

t2 = torch.tensor([
    [3,3],
    [4,4]
])
```

```
torch.cat([t1,t2], dim=0)
torch.cat([t1,t2], dim=1)

tensor([[1, 1],
        [2, 2],
        [3, 3],
        [4, 4]])
        
tensor([[1, 1, 3, 3],
        [2, 2, 4, 4]])
```

## flatten

如果要将张量输入到全连层，我们必须将张量展开成一维的。假如现在我们有三个图片。要将他们输入到全连层中。

```
t1 = torch.ones(4,4)
t2 = t1*2
t3 = t1*3

tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
        
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
```

首先，因为有三张图片，所以要组成一个batch_size

```
t = torch.stack([t1,t2,t3])

tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[2., 2., 2., 2.],
         [2., 2., 2., 2.],
         [2., 2., 2., 2.],
         [2., 2., 2., 2.]],

        [[3., 3., 3., 3.],
         [3., 3., 3., 3.],
         [3., 3., 3., 3.],
         [3., 3., 3., 3.]]])
```

```
t.shape

torch.Size([3, 4, 4])
```

Emm...现在第0个纬度表示batch，然后，后面两个纬度表示长宽。因为是图片，还缺少颜色纬度。

```
t = t.reshape(3,1,4,4)

t.reshape(-1)
t.flatten()

tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3.,
        3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
```

emm，现在问题来了，上面两个确实展平了。但是上面将三个图片糅合在了一起

==继续使用flatten函数==

- 这个参数表示从第1个纬度开始，其他的全部flatten

```
t.flatten(start_dim=1)

tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]])
```

看，这才是我们想要的answer！



## bordcast

bordcast俗称广播机制。本来，不同纬度的张量，是不能直接相加的。要想运算必须设置成维度一样。有人甚至写for循环来进行不同纬度的张量运算，这样效率太低了。使用bordcast机制，可以节省很多时间。

```
t = torch.tensor([
    [1,1],
    [2,2]
])
```

```
t + 2
t / 2
t * 2

np.broadcast_to(2, t.shape)

array([[2, 2],
       [2, 2]])
```

使用np.broadcast_to 方法可以观察广播机制。可以发现广播机制会将低维的，拓展成高维的，然后再进行计算。

## 比较运算

- 首先创建一个张量

```
t = torch.tensor([
    [0,5,7],
    [6,0,7],
    [0,8,0]
], dtype=torch.float32)
```

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201003183714934.png" alt="image-20201003183714934" style="zoom: 33%;" />

也可以使用下面操作来实现

```
t > 0
t == 0
t >= 0
t < 0
t <= 0
```

## argmax

下面有一些基础运算

- max
- argmax
- mean
- sum

还有很多类似的函数。

```
t = torch.tensor([
    [0,1,0],
    [2,0,2],
    [0,3,0]
])

t.sum()
t.numel()
t.sum().numel()

tensor(8)
9
1
```

> 可以看到不过不对上面函数设置参数，则默认返回的是一个scaler

```
tensor([[1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]])
        
t.sum(dim=0) # tensor([6., 6., 6., 6.])
t.sum(dim=1) # tensor([ 4.,  8., 12.])
```

> dim=0表示这个维度计算后就没了，所以最后是1*4的tensor

```
t = torch.tensor([
    [1,0,0,2],
    [0,3,3,0],
    [4,0,0,5]
])

t.max() # tensor(5)
t.argmax() # tensor(11)
t.flatten() # tensor([1, 0, 0, 2, 0, 3, 3, 0, 4, 0, 0, 5])
```

> 直接使用max函数，返回的是整个tensor中最大的那个，返回的是scaler，argmax返回的是最大的元素的位置，返回的也是一个scaler。这里返回的是11，是flatten后最大元素的位置

```
t.max(dim=1) # 下面函数都返回了2个tensor第一个是值，第二个是index

# 维度设置为dim=1，表示要消除第1个纬度，原来是3*4的，随意最后会变成3*1的，也就是找到每一行的最大值。
# 注意下面返回了两个tensor，一个是值，另一个是indices,可以通过这两个属性访问tensor
torch.return_types.max(
values=tensor([2, 3, 5]),
indices=tensor([3, 2, 3]))

t.argmax(dim=1) # tensor([3, 2, 3])

t.max(dim=0)

torch.return_types.max(
values=tensor([4, 3, 3, 5]),
indices=tensor([2, 1, 1, 2]))

t.argmax(dim=0) # tensor([2, 1, 1, 2])
```

```
t = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype = torch.float32)

t.mean() # 注意这些都是直接获取scaler
t.mean().item() # 获得scaler的值，返回一个数
t.mean(dim=0).tolist() # 返回一个列表，可以转换成列表
t.mean(dim=0).numpy() # 可以转换成numpy
```

## trochvision

torchvision 是torch中加载数据的一个模块

- 导入模块

```
import torch
import torchvision
import torchvision.transforms as transforms
```

- 下载数据

```
train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
train=True, 
download=True, 
transform=transforms.Compose([transforms.ToTensor()])
)
```

## torch.nn.module



首先我们应该导入nn模块

```
import torch.nn as nn
```

nn模块是很基础的神经网络模块，我们要在这个模块的基础上自己定义模块

- 首先，我们要在init函数中继承nn.module类方法，然后我们就可以使用里面的所有方法。
- 然后，在这个模块的基础上进行修改。
- nn模块中有很多神经层。这些都是在初始化的时候自己定义的。

> - 下面代码中，只有conv1中的，in_channel=1和self.out中的 out feature=10是根据我们输入的数据确定的。因为输入的数据是灰度图，所以颜色通道是1，然后out feature是10分类，所以设置为10
> - 其他参数的数值，都是我们自己人为设置的，所以叫做**超参数**。conv1 out channel=6, then next conv2 in channel = 6
> - Conv2 out channel = 12 then fully connect layer in feature = 12 * 4 * 4，这个4*4是什么意思呢？我们以后再解释

```
class Network(nn.Module):
    def __init__(self):
        super().__init__() # 继承了nn.module模块中所有的属性和方法
        
        # 首先定义卷积层
        self.conv1 = nn.Conv2d(in_channel = 1, out_channel = 6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channel = 6, out_channel = 12,kernel_size=5)
        
        # 然后定义线性层，fc表示fully connect，因为线性层也叫全连层
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
    def forward(self,t):
        return t
```

> 上面哪些是parameter，哪些是argument？ in_features和out_features这些就是parameter，120，60这些就是argument真正有价值的是argument。

**现在来介绍下线性层和卷积层里面有什么？**

- Linear module

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201006132948701.png" alt="image-20201006132948701" style="zoom: 50%;" />

每一层都含有权重和前向传播函数。函数中含有输出特征量，输出特征量。

**注意下面，out feature的数量在一直下降，这是因为我们处理的只是10分类问题**

```
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
```

init中保存了该层的所有权重参数。并且torch会自动追踪这些权重，我们在自己拓展神经网络的时候，类方法就继承了这个特性。

> 什么是线性层呢？比如CNN中的卷积层，卷积的话就是利用卷积核进行扫描，然后相乘相加，进行线性运算。实现这样功能的神经网络就是线性层，线性层只是用来进行线性运算

- Conv2d (input_channel, output_channel, kennel_size, stride, padding)

这个函数表示2D卷积，用来处理图像数据比较好。

现在来介绍下这个函数的参数。

> input channel表示输入的图像的颜色通道。如果是灰度的图片，这里就设置成1就好了
>
> output channel表示要输出多少通道，形成输入到输出的一个转换。其实也就是设置使用多少个卷积核去进行卷积，每个卷积核都会产生一个通道。假如现在有三个颜色通道，4个卷积核，由于有4个卷积核，最后就会产生4个通道。现在是第一个卷积核，第一个卷积核会扫描这三个颜色通道，进行卷积操作，然后将扫描的结果，进行相加，再加上一个权重。最后一个通道结果完成，其余3个卷积核，也完成类似的操作。然后输出。
>
> 元素的卷积操作，参考。https://zhuanlan.zhihu.com/p/57575810

> kennel size=3，则说明卷积核是3*3大小的
>
> stride表示步长，每次卷积核移动多少
>
> padding表示在图像周围进行填充

例如在上面例子中，Fashion Minist 数据是灰色的，所以颜色通道是1，我们采用6个卷积核进行卷积

```
        self.conv1 = nn.Conv2d(1,6,kernel_size=5)
        self.conv2 = nn.Conv2d(6,12,kernel_size=5)
```

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201006141936410.png" alt="image-20201006141936410" style="zoom:50%;" />

<center>使用6个卷积核</center>

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201006142104628.png" alt="image-20201006142104628" style="zoom:50%;" />

可以发现，不同卷积核收集到的特征是不一样的

```
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__() # 继承了nn.module模块中所有的属性和方法
        
        # 首先定义卷积层
        self.conv1 = nn.Conv2d(1,6,kernel_size=5) # 这些也只是将类对象实例化而已
        self.conv2 = nn.Conv2d(6,12,kernel_size=5)
        
        # 然后定义线性层，fc表示fully connect，因为线性层也叫全连层
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
    def forward(self,t):
        return t
        
network = Network() # 实例化
print(network) # 然后打印这个参数

# 输出了这个网络的所有特征
Network(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=192, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=60, bias=True)
  (out): Linear(in_features=60, out_features=10, bias=True)
)
```

==注意，super(Network, self).__init__() 这里一定只能这样写，否则不会真正继承==

> 问题：上面打印network这个参数，为什么会打印出来一堆东西呢。
>
> 这是因为我们继承了nn.Module

```
    # 我们可以手动修改原来的，然后对他的方法进行覆盖
    def __repr__(self): # 表示represent
        return 'zhuweilin'
```

我们来观察下这些权重。

```
network.conv1.weight
```

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201006152106629.png" alt="image-20201006152106629" style="zoom:50%;" />

```
network.conv1.weight.shape # torch.Size([6, 1, 5, 5])
network.conv2.weight.shape # torch.Size([12, 6, 5, 5])
network.fc1.weight.shape # torch.Size([120, 192])
network.fc2.weight.shape # torch.Size([60, 120])
network.out.weight.shape # torch.Size([10, 60])

self.conv1 = nn.Conv2d(in_channel = 1, out_channel = 6,kernel_size=5)
self.conv2 = nn.Conv2d(in_channel = 6, out_channel = 12,kernel_size=5)
```

> 解释下这些权重的形状
>
> - conv1的形状，6表示有6个卷积核，1表示通道数量。5，5表示长宽。为什么形状是这样的呢？我们来看看代码。in channel = 1, then we have kennel 6,每个kennel都会对灰度图像进行卷积，然后产生1\*5\*5的张量，一共产生6个这样的。6也可以看成batch size
> - conv2形状也是同理。表示使用12个卷积核进行卷积
>
> ==综上所述：上面张量有4个维度，第一个维度表示卷积核的数量，第二个维度表示卷积核卷积的深度，第三四个维度表示长宽==

## 神经网络中的矩阵运算

我们来查看下面的代码。

```
in_feature = torch.tensor([1,2,3,4], dtype=torch.float32)

fc = nn.Linear(in_features=4, out_features=3)

fc(in_feature)

tensor([ 2.0580, -2.3204,  1.3129], grad_fn=<AddBackward0>)
```

- 输入是个1乘4的矩阵。
- 然后我们定义线性层。输入4个特征。输出3。相当于4*3 。会初始化一个这样形状的权重矩阵
- 然后线性层中会初始化权重，直接使用类对象进行运算即可

**为什么权重都要用Parameter类对象封装起来呢？**

==因为权重都是参数，所以必须要封装起来==

- 我们先手动生成两个tensor。然后进行矩阵运算

```
weight_feature = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
], dtype = torch.float32)

in_feature = torch.tensor([1,2,3,4], dtype=torch.float32)
```

```
weight_feature.matmul(in_feature) # tensor([30., 40., 50.])
```

- 假如现在使用线性层进行运算。会是怎么样的结果呢？

```
from torch.nn import Parameter

fc.weight

# 查看初始化的权重，可以发现是随机的，并且是Parameter类对象封装好的
Parameter containing:
tensor([[ 0.1431,  0.4674, -0.0564,  0.2586],
        [-0.3928, -0.0278, -0.1688, -0.3314],
        [ 0.3665,  0.0332, -0.0852,  0.3662]], requires_grad=True)

# 之前我们自己创建了一个权重，现在封装进去
fc.weight = Parameter(weight_feature)
fc.weight

Parameter containing:
tensor([[1., 2., 3., 4.],
        [2., 3., 4., 5.],
        [3., 4., 5., 6.]], requires_grad=True)


# 进行运算，但是为什么会有小数点呢？        
fc(in_feature)
tensor([30.1151, 39.9601, 49.6709], grad_fn=<AddBackward0>)
```

> 为什么会有偏差呢？因为Parameter参数会给矩阵加个bias。如果我们将bias设置为False。那么就是准确值了

```
fc = nn.Linear(in_features=4, out_features=3, bias=False)
fc(in_feature)
tensor([30., 40., 50.], grad_fn=<SqueezeBackward3>)
```

> 为什么直接使用fc然后就可以进行权重运算呢？
>
> 因为里面有个很重要的魔法方法 \_\_call\_\_方法。当我们直接layer(input),他会调用\_\_call\_\_(input).魔法方法会调用里面的forward方法

## 前向传播的实现

- 首先初始化的时候，要先定义好各种网络层
- 前向传播其实就是定义好网络层的矩阵运算

```
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # 定义两个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        # fully connect
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        # input layer
        t = t
        
        # hidden conv layer 
        t = self.conv1(t) # 调用这个会自动与权重进行矩阵运算，实现前向传播
        t = F.relu(t) # 将矩阵运算完后，输入到激活函数中
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        # hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        # 全连接层 。使用全连接层必须将矩阵拉直，变成一个行向量
        # hidden liner layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # hidden liner layer
        t = self.fc2(t)
        t = F.relu(t)
        
        return t
```

> 什么是池化层？
>
> 在前向传播的过程中，选择图像区域中的最大值作为池化后的值，可以降低信息的冗余。防止过拟合。
>
> 其实也是使用卷积核对矩阵进行卷积运算，只不过卷积运算使用的是挑选最大值。池化层通常使用在CNN中
>
> <img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201007150853599.png" alt="image-20201007150853599" style="zoom:50%;" />

> 现在来解释下为什么里面是12*4\*4
>
> ```
> t = t.reshape(-1, 12*4*4)
> ```
>
> 先观察下代码，只有1行，12*4\*4列。我们知道12使用的是上一层的输出量，4\*4表示的是每个内核的长和宽。长和宽变这么小是因为进行了卷积和池化运算
>
> ```
> conv1 input shape: torch.Size([1, 1, 28, 28])
> conv2 input shape: torch.Size([1, 6, 12, 12])
> fc1 input shape: torch.Size([1, 12, 4, 4])
> fc2 input shape: torch.Size([1, 120])
> out shape: torch.Size([1, 60])
> ```
>
> 可以发现是conv2到fc1这里出现问题：在 conv1 和conv2这里如果没设置stride，则默认的步长是1
>
> ```
> nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
> nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
> nn.Linear(in_features=12*4*4, out_features=120)
> nn.Linear(in_features=120, out_features=60)
> nn.Linear(in_features=60, out_features=10)
> ```
>
> 

**设置多个batch**

在实际的计算中，我们通常都是使用多个batch，也就是传入多个图片来训练网络。

那么我们需要重新定义网络层吗？

answer is not ！ torch已经帮我们封装好了，对没错，就是DataLoader模块。

```
from torch.utils.data import DataLoader

batch = next(iter(data_loader))

images, labels = batch

labels # tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5])
images.shape # 现在打包了10张图片 torch.Size([10, 1, 28, 28])
preds = network(images) # 然后放入到神经网络中训练即可
```

```
tensor([[-0.0276,  0.0466,  0.0940, -0.0181,  0.1199, -0.0581, -0.0511, -0.0340,  0.1157, -0.0545],
        [-0.0288,  0.0588,  0.0949, -0.0242,  0.1083, -0.0463, -0.0764, -0.0361,  0.1081, -0.0581],
        [-0.0344,  0.0606,  0.0938, -0.0253,  0.1145, -0.0619, -0.0570, -0.0254,  0.1081, -0.0523],
        [-0.0336,  0.0653,  0.0962, -0.0277,  0.1182, -0.0544, -0.0550, -0.0349,  0.1116, -0.0558],
        [-0.0289,  0.0590,  0.0966, -0.0311,  0.1201, -0.0597, -0.0521, -0.0393,  0.1111, -0.0480],
        [-0.0294,  0.0476,  0.0982, -0.0227,  0.1176, -0.0534, -0.0724, -0.0340,  0.1055, -0.0554],
        [-0.0278,  0.0497,  0.0804, -0.0165,  0.1172, -0.0700, -0.0482, -0.0352,  0.1127, -0.0546],
        [-0.0293,  0.0431,  0.0915, -0.0192,  0.1203, -0.0591, -0.0707, -0.0444,  0.1117, -0.0558],
        [-0.0309,  0.0592,  0.0899, -0.0156,  0.1141, -0.0566, -0.0634, -0.0256,  0.1151, -0.0577],
        [-0.0308,  0.0508,  0.0950, -0.0153,  0.1146, -0.0556, -0.0652, -0.0291,  0.1099, -0.0575]])
```

然后会获得一系列的预测值，获取预测值的最大的位置

```
preds.argmax(dim = 1)
```

同时和标签进行比较

```
preds.argmax(dim = 1).eq(labels)

tensor([False, False, False, False, False, False, False, False, False, False])
```

获得预测准确的数量

```
preds.argmax(dim = 1).eq(labels).sum()

tensor(0)
```

注意返回的是一个scaler。用item获得值

## 计算公式

神经网络中，卷积核卷积后的矩阵大小如何计算呢？
$$
O\;=\;\frac{n-f+2p}s+1
$$

> n 表示输入的矩阵shape
>
> f表示卷积核的shape
>
> p表示padding的shape
>
> s 表示stride

```
conv1 input shape: torch.Size([1, 1, 28, 28])
conv2 input shape: torch.Size([1, 6, 12, 12])
fc1 input shape: torch.Size([1, 12, 4, 4])
fc2 input shape: torch.Size([1, 120])
out shape: torch.Size([1, 60])

nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
nn.Linear(in_features=12*4*4, out_features=120)
nn.Linear(in_features=120, out_features=60)
nn.Linear(in_features=60, out_features=10)
```

> 我们来带入下公式：输入的图片大小shape是28，n=28.f=5，p=0，因为我们没有使用填充，所以是0，这里默认的步长为1带入计算后0刚好是24
>
> 再来计算下conv2的输入，因为conv2之前还有卷积核，也就是池化层
>
> ```
> t = F.max_pool2d(t, kernel_size = 2, stride = 2)
> ```
>
> 现在输入shape是24，n=24,f=2,p=0,s=2,计算后刚好是12

上面是shape一定的情况下，但是如果shape不一样怎么办呢？
$$
O_h\;=\;\frac{n_h-f_h+2p}s+1
$$

$$
O_w\;=\;\frac{n_w-f_w+2p}s+1
$$

分别带入公式计算长宽

## 神经网络的训练

现在，我们可以开始神经网络的训练了，就是开始计算和更新权重。

神经网络的训练流程大致如下：

1. 获得batch数据
2. 将batch数据送入到神经网络中
3. 计算loss。也就是真实值和预测值之间的差异
4. 计算loss的梯度
5. 更新weight
6. 重复1-5步骤，直到1个epoch完成
7. 重复1-6步骤，直到多个epoch完成。并且获得比较好的准确率

> epoch表示每个数据都学习了。
>
> 对于第三步，我们通过计算损失函数方程来计算损失函数
>
> 对于第四步，我们通过反向传播来计算损失函数的梯度
>
> 对于第五步，我们通过使用optimism algorithm来进行更新权重
>
> 第六和第七步只是单纯的for循环

- 导入模块

```
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
```

- 定义函数，判断预测的准确率

```
def get_correct_num(preds, labels):
    return preds.argmax(dim = 1).eq(labels).sum().item() 
```

- 定义神经网络

```
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.reshape(-1, 12*4*4) # 知道这里为什么不使用1吗？因为这里-1会自动设置好batch的大小
        print(t.shape)
        
        t = self.fc1(t)
        t = F.relu(t)
        
        t = self.fc2(t)
        t = F.relu(t)
        
        t = self.out(t)
#         t = F.softmax(t)
        
        return t
```

- 获得训练数据

```
 train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)
```

- 初始化神经网络

```
network = Network()
```

- 对数据进行封装

```
train_loader = DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader))
images, labels = batch
```

- 计算损失函数

```
# calculating the loss
preds = network(images)
loss = F.cross_entropy(preds, labels)
loss.item()
```

- 计算梯度

```
loss.backward() # torch会跟踪所有需要计算梯度的变量
```

- 利用optimizer里面的算法更新weight

```
optimizer = optim.Adam(network.parameters(), lr=0.01) # 要想权重进行优化，必须传入两个量，一个是神经网络的权重，另一个是学习率

get_correct_num(preds, labels)

optimizer.step() # 更新权重

preds = network(images)
loss = F.cross_entropy(preds, labels)

get_correct_num(preds, labels)
```

## 训练多个epoch

我们来看看如何训练多个epoch，以及怎么用代码实现

参考视频 https://www.youtube.com/watch?v=XfYmia3q2Ow&list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG&index=27

```
network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

total_loss = 0
total_correct = 0

for batch in train_loader:
    images, labels = batch
    
    preds = network(images)
    loss = F.cross_entropy(preds, labels)
    
    optimizer.zero_grad() # 因为梯度是会积累的，计算不同batch前，权重都应该清零
    # 因为神经网络只会计算一个batch的梯度并更新
    
    loss.backward()
    optimizer.step()
    
    total_correct += get_correct_num(preds, labels)
    total_loss += loss.item()
    
print('epoch:', 0, 'correct_num:', total_correct, 'loss:', total_loss)
```

> 1. 首先，初始化神经网络，并且加载数据
> 2. 我们利用for循环进行迭代。每次获取100张图片进行训练
> 3. 先看传入的100张图片，首先将图片的数据和标签分离
> 4. 将图片的数据传入神经网络中，进行前向传播，由于是第一次传入，此时神经网络的权重都是随机初始化的。所以刚开始预测，准确率肯定不是很高
> 5. 然后计算损失函数，计算损失函数会跟踪所有变量
> 6. 因为是首次for循环，所以权重还没有定义，这里清零没用，但是下一次循环有用
> 7. 进行loss.backward之后，梯度就会定义并且有值
> 8. 然后使用optimizer进行更新权重
> 9. 然后进入下一层循环。我们又获得了100张图片，这次利用上一次100张图片训练的权重，进行预测，这次准确率会稍微提高点
> 10. 计算损失函数
> 11. 梯度清0，如果不清0，则会和这批数据的梯度进行累加，然后更新梯度

- 训练多个epoch

```
network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad() # 因为梯度是会积累的，计算不同batch前，权重都应该清零
        # 因为神经网络只会计算一个batch的梯度并更新

        loss.backward()
        optimizer.step()

        total_correct += get_correct_num(preds, labels)
        total_loss += loss.item()
    
    print('epoch:', epoch, 'correct_num:', total_correct, 'loss:', total_loss)  
```

> 1. 首先进入第一个epoch。和上面一样，神经网络的权重已经固定了
> 2. 进入第二个epoch，先抽取100张图片，利用之前的权重进行预测，第一轮到第二轮的准确率，会有很高的提升
> 3. 往后的epoch，准确率提升幅度就没这么大了，因为权重更新很慢很慢了

> 这里解释下为什么optimizer是放在for循环外面
>
> ```
> optimizer = optim.Adam(network.parameters(), lr=0.01)
> ```
>
> 因为只要定义一次，这个network.parameters,传入给optimizer，是告诉优化器，哪些参数需要更新，然后torch会以这个算法更新神经网络的权重。

## 混淆矩阵

先来看看混淆矩阵张什么样子。

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201009175034002.png" alt="image-20201009175034002" style="zoom: 67%;" />

x 轴表示神经网络预测的标签，y轴是真实值，因为是10分类问题，x轴的坐标范围是0-9，y轴的坐标范围是0-9，如果现在预测值是1，真实值是9，那么1，9这个位置就会增加1.

首先创建神经网络，训练神经网络的参数。

```
for epoch in range(6):
    total_loss = 0
    total_correct = 0
    for feature, labels in data_loader:
        preds = network(feature)
        
        loss = F.cross_entropy(preds, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss
        total_correct += get_correct_num(preds, labels)
    print(f'epoch:{epoch}, loss:{total_loss}, acc:{total_correct/len(data_set)}')
    
epoch:0, loss:832.16015625, acc:0.7377333333333334
epoch:1, loss:550.2818603515625, acc:0.8339833333333333
epoch:2, loss:467.35186767578125, acc:0.8587333333333333
epoch:3, loss:422.51409912109375, acc:0.8707166666666667
epoch:4, loss:391.98907470703125, acc:0.87995
epoch:5, loss:369.84832763671875, acc:0.8859166666666667
```

> 可以发现训练结果的好坏，和batch size有很大的关系

训练好后，就可以开始构建confuse matrix了

刚开始我的想法就是，首先初始化一个cmt ，全是0的二维张量。然后使用batch size为1，使用for循环，也就是说要迭代6w此。这样效率太低了

```
data = DataLoader(data_set, batch_size=1)
for feature, labels in data:
    preds = network(feature)
    x = preds.argmax(dim=1).item()
    y = labels.item()
    cmt[x,y] += 1
```

**上面方法肯定是不行的**

神经网络是支持张量运算的，使用张量直接运算，会节省很多的时间。

- batch size充分利用了张量的运算。dtype设置为int32是因为，索引不支持float，而torch默认的创建的float类型

```
data = DataLoader(data_set, batch_size=10000)
all_preds = torch.tensor([], dtype=torch.int32)
all_labels = torch.tensor([], dtype=torch.int32)
for feature, labels in data:
    preds = network(feature).argmax(dim=1)
    all_preds = torch.cat([all_preds, preds], dim=0) 
    all_labels = torch.cat([all_labels, labels], dim=0)
    
for x, y in stack:
    cmt[x,y] += 1
```

我们这里要熟悉cat 和 stack用法

- stack创建一个新的维度，然后在这个维度上拼接起来，就像下面张量都是1维的，stack后就变成2维的了
- cat是在现有的维度上进行拼接，不会创建新的维度

```
a = torch.tensor([1,1,1])
b = torch.tensor([2,2,2])

torch.cat([a,b], dim=0)
```

```
tensor([1, 1, 1, 2, 2, 2])
```

```
torch.stack([a,b], dim=1)
```

```
tensor([[1, 2],
        [1, 2],
        [1, 2]])
```

```
torch.stack([a,b], dim=0)
```

```
tensor([[1, 1, 1],
        [2, 2, 2]])
```

## 使用GPU计算

- 查看显卡信息

```
nvidia-smi
```

- 查看显卡数量

```
torch.cuda.device_count()
```

- 使用CPU进行计算

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

data_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST/',
    download=True,
    train = True,
    transform=transforms.Compose([transforms.ToTensor()])
)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        t = F.relu(self.fc2(t))
        
        return self.out(t)
        
def correct_num(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
    
network = Network()
optimizer = optim.Adam(network.parameters(), lr = 0.01)

t = []
for epoch in range(10):
    total_loss = 0
    total_correct = 0
    time1 = time.time()
    for feature, labels in data_loader:
        preds = network(feature)
        
        loss = F.cross_entropy(preds, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss
        
        total_correct += correct_num(preds, labels)
    time2 = time.time()
    t.append(time2 - time1)
    print(f'epoch:{epoch}, acc:{total_correct/len(data_set)}, loss:{total_loss}, time:{time2 - time1}')
    
print(f'total time:{sum(t)}')
```

- 使用GPU进行计算

> - 要使用GPU进行计算，首先要将神经网络放到GPU上
>
> - 然后还要将数据放到GPU上进行计算

**主要修改有下面几个地方**

```
device = torch.device("cuda")
network = Network()
network.to(device) # 将神经网络放到cuda上

t = []
for epoch in range(10):
    total_loss = 0
    total_correct = 0
    time1 = time.time()
    for feature, labels in data_loader:
        feature = feature.to(device) # 将这两个数据放到cuda上
        labels = labels.to(device)
        preds = network(feature)
        
        loss = F.cross_entropy(preds, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss
        
        total_correct += correct_num(preds, labels)
    time2 = time.time()
    t.append(time2 - time1)
    print(f'epoch:{epoch}, acc:{total_correct/len(data_set)}, loss:{total_loss}, time:{time2 - time1}')
    
print(f'total time:{sum(t)}')
```

注意一定是下面这样

```
feature = feature.to(device) # 将这两个数据放到cuda上
labels = labels.to(device)
```

完整代码：

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

data = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST/',
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        t = F.relu(self.fc2(t))
        
        return self.out(t)

def correct_num(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

data_loader = DataLoader(
    data, 
    batch_size=1000, 
    shuffle=True,
)
network = Network()

device = torch.device('cuda:0')

network.to(device)

optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(5):
    total_correct = 0
    total_loss = 0
    time1 = time.time()
    for feature, labels in data_loader:
        feature = feature.to(device)
        labels = labels.to(device)
        preds = network(feature)
    
        optimizer.zero_grad()
        
        loss = F.cross_entropy(preds, labels)
        
        loss.backward()
        
        optimizer.step()
        
        total_correct += correct_num(preds, labels)
        total_loss += loss
    time2 = time.time()
    print(f'time:{time2-time1}, acc:{total_correct/len(data)}, loss:{total_loss}')
```

