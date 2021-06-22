<!--ts-->
* [基础教程](#基础教程)
* [模型融合和投票使用方式](#模型融合和投票使用方式)
* [pytorch中间层特征提取](#pytorch中间层特征提取)

<!-- Added by: zwl, at: 2021年 6月22日 星期二 20时45分59秒 CST -->

<!--te-->

# 基础教程

- [pytorch 基础教程](./doc/pytorch) 
- [tensorflow 基础教程](./doc/tensorflow) 

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
