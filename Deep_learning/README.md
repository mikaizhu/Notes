<!--ts-->
* [基础教程](#基础教程)
* [pytorch修改某层神经网络](#pytorch修改某层神经网络)
* [模型融合和投票使用方式](#模型融合和投票使用方式)
* [pytorch中间层特征提取](#pytorch中间层特征提取)
* [使用numpy手动实现卷积核运算](#使用numpy手动实现卷积核运算)

<!-- Added by: zwl, at: 2021年 9月 7日 星期二 14时08分51秒 CST -->

<!--te-->

# 基础教程

- [pytorch 基础教程](./doc/pytorch) 
- [tensorflow 基础教程](./doc/tensorflow) 

# pytorch修改某层神经网络

```
model = resnet18()
model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.maxpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
model.fc = nn.Linear(512, 30, bias=True)
```

注意不能这样修改:

```
model.fc.out_features = 30 # 这样只是修改了内部的字典，并没有真实修改参数
```

应该这样修改：

```
model.fc = nn.Linear(512, 30, bias=True)
```

# 模型融合和投票使用方式

参考:[](../Others/gjbc/stack) 

# pytorch中间层特征提取

模型如下, 现在要提取中间的某层特征，比如softmax之前的特征：

- 方法1:可以在原先的DNN的forward函数中修改，然后return 出来，但很多模型都封装
  好了
- 方法2: 使用model.layername.register_forward_hook()

```
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.dnn(x)
        return F.softmax(x, dim=1)
```

```
model = DNN().to(device)
model.load_state_dict(torch.load('./best_model0.point'))
print(model)
```

Output:

```
DNN(
  (dnn): Sequential(
    (0): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Linear(in_features=300, out_features=1024, bias=True)
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.2, inplace=False)
    (5): Linear(in_features=1024, out_features=256, bias=True)
    (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.2, inplace=False)
    (9): Linear(in_features=256, out_features=10, bias=True)
  )
)
```

```
# 使用下面方法获得中间层特征
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.dnn[9].register_forward_hook(get_activation(model.dnn[9]))
tensor = torch.FloatTensor(val_sp).to(device)
model(tensor).argmax(dim=1).cpu().numpy()

# 查看字典，会发现中间层特征已经存储到字典中了
print(activation)
```

- 使用dnn[9], 获得某一层的名字

# 使用numpy手动实现卷积核运算

```
#!/bin/python
# -*- coding: utf8 -*-
import numpy as np

class Solution:
    def conv2d(self, kernel, image, stride):
        # Write Code Here
        self.k_h = int(kernel.split(',')[0].split()[0])
        self.k_w = int(kernel.split(',')[0].split()[1])
        self.kernel_data = list(map(int, kernel.split(',')[1].split()))
        self.image_data = list(map(int, image.split(',')[1].split()))
        self.i_h = int(image.split(',')[0].split()[0])
        self.i_w = int(image.split(',')[0].split()[1])


        self.kernel_data = np.array(self.kernel_data).reshape(self.k_h, self.k_w)
        self.image_data = np.array(self.image_data).reshape(self.i_h, self.i_w)
        self.stride = int(stride)

        padding_h = int(np.floor(self.k_h/2))
        padding_w = int(np.floor(self.k_w/2))
        self.padding_image = np.zeros((self.i_h + 2*padding_h, self.i_w + 2*padding_w))
        self.padding_image[padding_w:padding_w + self.i_w, padding_h:padding_h + self.i_h] = self.image_data

        #开始卷积运算
        res = []
        self.H = (self.i_h - self.k_h + 2 * padding_h) // self.stride + 1
        self.W = (self.i_w - self.k_w + 2 * padding_w) // self.stride + 1
        for i in range(0, self.i_h, self.stride):
            for j in range(0, self.i_w, self.stride):
                t1 = self.padding_image[i : i + self.k_h, j : j + self.k_w]
                res.append(np.sum(t1 * self.kernel_data))
        print(np.array(res).reshape(self.H, self.W))

try:
    #_kernel = input()
    _kernel = '3 3,1 2 1 2 4 2 1 2 1'
except:
    _kernel = None

try:
    #_image = input()
    _image = '5 5,' + '1 ' * 25
except:
    _image = None

#_stride = int(input())
_stride = '1'

s = Solution()
res = s.conv2d(_kernel, _image, _stride)

print(res + "\n")
```
