<!--ts-->
* [高级编程期末作业](#高级编程期末作业)
   * [摘要](#摘要)
   * [DNN baseline模型讲解](#dnn-baseline模型讲解)
      * [导入对应的模块](#导入对应的模块)
      * [数据加载](#数据加载)
      * [模型搭建](#模型搭建)
      * [模型保存](#模型保存)
   * [59-60的提分技巧说明](#59-60的提分技巧说明)
   * [60-61分技巧说明](#60-61分技巧说明)
   * [61-62提分技巧说明](#61-62提分技巧说明)
      * [boost学习思想说明](#boost学习思想说明)
      * [boost+snapshot代码如下](#boostsnapshot代码如下)
      * [模型改进和多DNN模型融合](#模型改进和多dnn模型融合)
   * [思考与总结](#思考与总结)

<!-- Added by: zwl, at: 2021年 6月23日 星期三 22时12分41秒 CST -->

<!--te-->
# 高级编程期末作业

## 摘要

本代码使用的是DNN模型，其中baseline在测试集上可以达到59分的成绩，修改了学习DNN
神经网络的非线性层和激活函数后，单模型可以到60分，再经过数据的过采样后，单模型
可以达到61分的准确率。然后采用snapshot学习方式，进行多模型软投票融合，使得DNN模型可以达
到62分的准确率。

## DNN baseline模型讲解

本代码使用的是pytorch深度学习框架

DNN baseline能跑到59，总结下主要有以下几个提分点：

1. 数据处理技巧
2. DNN模型的搭建技巧
3. 选择比较好的模型参数保存技巧

接下来对baseline进行讲解

### 导入对应的模块
```
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
```

### 数据加载
```
train = np.load('../train/10type_sort_train_data_8192.npy')
val = np.load('../val/10type_sort_eval_data_8192.npy')
train_label = np.load('../train/10type_sort_train_label_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')
```

数据处理: 因为原始数据是时间序列，我们在对数据进行处理的时候, 进行了一下几点处理：

1. 将时间序列进行8192个点的fft变换。
2. 对fft数据取模
3. 对取模的数据进行归一化
4. 因为fft变换的数据是对称的，所以为了减小参数，将取模后的数据进行切片处理
5. 切片选择，这里我们做了很多试验，发现在峰值附近的300个点的训练效果是最好的，
   即6892-8192个点

```
def get_fft_and_scaler(data, start=6892, end=7192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]
```

随机数种子固定：这里一定要固定随机数种子，因为神经网络参数的随机性，如果不固定
随机数种子，模型的训练结果会每次都不一样，这样就不能复现模型
```
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

### 模型搭建
1. 这里只是简单地使用了三层神经网络进行学习，因为发现四层和三层的学习差不多，
   所以这里选择参数更少的三层神经网络
2. 提升DNN神经网络的有效方法之一就是，在输入的时候加BatchNorm，经过测试发现，
   如果不在每一层添加batchnorm，那么DNN的准确率会下降5-10个点左右
3. 为防止过拟合，这里在输入前加入Dropout层, 试验发现， 如果不加入Dropout，神经
   网络的准确率会下降，说明合适的Dropout既能起到防止过拟合，又能提高模型的泛化
   能力。
4. 最后分类必须经过softmax，将所有值映射成0-1之间的概率值进行分类，如果不进行
   映射直接分类，准确率也会有所下降

```
# 搭建DNN模型
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

### 模型保存

模型保存也有很重要的提分点，我们到底选择什么样的训练模型进行保存，才能得到比较
高的分数呢？

模型在训练的时候，有以下几个参数：
- 训练集的损失函数，越低表示拟合的越好
- 训练集的准确率
- 测试集的损失函数
- 测试集的准确率

一般认为, 损失函数越低，那么模型的拟合能力越高，所以这里有以下几种方式来保存模
型：
- 保存训练集准确率最高的模型
- 保存训练集损失函数最低的模型
- 保存测试集准确率最高的模型
- 保存测试剂损失函数最低的模型

经过我们试验发现，对于本题，我们选择保存测试集准确率最高的模型，效果会比较好。
同时我们发现，并不是损失函数最低，准确率就会越高

```
train_best = 0
best_model = None

for epoch in range(epochs):
    print('='*20 + f' Epoch: {epoch} '+ '='*20)
    train(train_loader, model, optimizer, criterion=criterion, labels=train_label)
    acc = predict(val_loader, model, criterion=criterion, labels=val_label)
    if acc >= train_best:
        train_best = loss
        # 保存测试集上准确率最高的模型
        torch.save(model.state_dict(), './best_model.point')
```

## 59-60的提分技巧说明

上面DNN神经网络是最基础的框架，可以尝试的可以有以下几点：
- 使用的学习器件是Adam，修改为其他学习器
- 使用的激活函数是ReLU，可以尝试其他激活函数
- 其他的网络层，比如损失函数，以及Softmax以外的分类层

经过我们的测试，发现以下修改后的网络框架，加上保存最佳模型，可以使得准确率达到
60.

对于DNN神经网络框架，我们修改了以下几点：

1. 将ReLU函数，换成了leakyReLU
2. 将Dropout rate, 换成了0.1

我们还测试了其他激活函数和分类层，只有以下是效果最好的

```
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.dnn(x)
        return x
```

我们对神经网络的学习器，进行了以下修改：
1. 将Adam学习器，换成了AdamW
2. 对AdamW中的参数进行了调节，将eps参数设置为1e-3，发现效果是最好的

```
lr = 0.0001
step_size = 1
epochs = 60
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
model = DNN().to(device)
eps = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=lr, eps=eps)
criterion = nn.CrossEntropyLoss()
```

## 60-61分技巧说明

这里提分点主要是做了数据增强，因为原来的数据经过统计发现，样本存在轻微的不均衡
。

```
Counter({0: 4257,
         1: 10081,
         2: 4270,
         3: 17187,
         4: 4746,
         5: 6337,
         6: 10298,
         7: 3694,
         8: 5058,
         9: 4281})
```

因此我们这里采用过采样的方法，过采样处理直接使用python中imblearn模块中的SMOTE
采样方法

```
from imblearn.over_sampling import SMOTE
```

使用方法如下：
```
train = np.load('../train/10type_sort_train_data_8192.npy')
val = np.load('../val/10type_sort_eval_data_8192.npy')

# 读取训练集和验证集的标签，测试集是没有标签的，需要你使用模型进行分类，并将结果进行提交
train_label = np.load('../train/10type_sort_train_label_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')

print('Stage2: data over_sampling')
smote = SMOTE(random_state=42, n_jobs=-1)
x_train, train_label = smote.fit_resample(train, train_label)

train_sp = get_fft_and_scaler(x_train, start=6892, end=7192)
val_sp = get_fft_and_scaler(val, start=6892, end=7192)
```

采样完后，数据样本的数量如下：

```
Counter({0: 17187,
         1: 17187,
         2: 17187,
         3: 17187,
         4: 17187,
         5: 17187,
         6: 17187,
         7: 17187,
         8: 17187,
         9: 17187})
```

可以发现，所有样本的数量，都被采样到了最多的那一类，即样本数量变多了，并且变均
衡了.

SMOTE原理就是，通过学习样本比较小的数据，通过算法对这些小样本数据进行补充。具
体可以参考：https://zhuanlan.zhihu.com/p/44055312

## 61-62提分技巧说明

在前面61分模型的基础上，我们主要使用了以下几点技巧：
- 主要采用了boost的学习思想
- snapeshot学习思想
- 模型软投票


### boost学习思想说明

boost主要思想就是，构建多个分类器件，并对上一个分类器分类错误的样本进行学习，
然后将这些分类器集成起来，完成分类任务。

针对这一思想，我们做了一些改变和以下尝试：
1. 构建多个DNN，每个DNN拟合上面一个DNN分类正确的数据
2. 构建一个DNN，不断学习自己分类正确的数据
3. 构建多个DNN，每个DNN拟合上一个DNN分类错误的数据
4. 构建一个DNN，不断学习自己分类错误的数据

经过试验发现，2方案即构建一个DNN，不断学习自己分类正确的数据，效果反而会比较好。相比与之前有以下进步:

1. 模型准确率越来越稳定，之前没有使用这个思想的时候，准确率总是在0.6左右跳跃，
  最好能到0.61，使用这个思想后，可以稳定在0.61左右。
2. 高分的模型越来越多，可以方便后面使用snapshot进行模型融合

snapshot+softvoting，3模型准确率可以达到0.623，完整代码如下:

- boost+snapshot代码部分
- softvoting代码部分

### boost+snapshot代码如下

```
import numpy as np
import pandas as pd
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
from imblearn.over_sampling import SMOTE

def get_fft_and_scaler(data, start=6892, end=7192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.dnn(x)
        return x

# 固定随机数种子，确保实验的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_boost_data(feature, preds_label, true_label):
    return feature[preds_label == true_label, :].detach().cpu(), true_label[preds_label == true_label].detach().cpu()

def model_train(train_loader, model, optimizer, criterion, labels):
    model.train()
    train_total_acc = 0
    train_loss = 0
    for feature, label in train_loader:
        feature = feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        preds = model(feature)

        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        # 存储boosting数据, 这里使用epoch=0时候的分类错误的，因为这样数据才会多一点，或包括大部分分类错误的
        if epoch == epochs-1:
            d1, d2 = get_boost_data(feature, preds.argmax(dim=1), label)
            boost_feature.append(d1)
            boost_label.append(d2)

        train_total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
        train_loss += loss.item()

        feature.cpu()
        label.cpu()

    print(
        f'Training loss: {train_loss/len(train_loader):.4f}',
        f'Training  acc: {train_total_acc/len(labels):.4f}',
         )

def predict(val_loader, model, criterion, labels):
    model.eval()
    val_total_acc = 0
    val_loss = 0
    for feature, label in val_loader:
        feature = feature.to(device)
        label = label.to(device)
        preds = model(feature)
        loss = criterion(preds, label)

        val_total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
        val_loss += loss.item()

        feature.cpu()
        label.cpu()

    print(
        f'Val loss: {val_loss/len(val_loader):.4f}',
        f'Val  acc:{val_total_acc/len(labels):.4f}'
    )
    return val_total_acc/len(labels)

print('Stage1: load data')
# 读取训练集，测试集和验证集

# 读取训练集和验证集的标签，测试集是没有标签的，需要你使用模型进行分类，并将结果进行提交
train = np.load('../train/10type_sort_train_data_8192.npy')
val = np.load('../val/10type_sort_eval_data_8192.npy')
train_label = np.load('../train/10type_sort_train_label_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')

print('Stage2: data over_sampling')
smote = SMOTE(random_state=42, n_jobs=-1)
x_train, train_label = smote.fit_resample(train, train_label)

train_sp = get_fft_and_scaler(x_train, start=6892, end=7192)
val_sp = get_fft_and_scaler(val, start=6892, end=7192)

# 将数据转换成pytorch的tensor
print('Stage3: transform numpy data to tensor')
batch_size = 128

train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(train_label).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()

# 使用Dataloader对数据进行封装
val_tensor = TensorDataset(val_tensor, y_val_tensor)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

lr = 0.0001
epochs = 30
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
eps = 1e-3

# 使用boost ing的想法，让神经网络学习错误的类别
# boosting 的想法：训练后面几次，分类错误的数据
# 重新定义一个新的分类器进行学习错误的数据, 保存这些模型的参数
# 然后使用模型融合
print('Stage4: start training')
# 这里设置boost 的num数，设置为多少就会训练多少个dnn模型
boost_epoch_num = 3
model = DNN().to(device)
for boost_num in range(boost_epoch_num):
    set_seed(42)
    # 分类器学习, 更新训练集
    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    boost_feature = []
    boost_label = []
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=eps)
    criterion = nn.CrossEntropyLoss()
    
    train_best = 0
    
    print('--'*8 + f'boost round: {boost_num}/{boost_epoch_num - 1}' + '--'*8)
    print(f'train shape: {train_tensor.shape}')
    print('--'*24)
    for epoch in range(epochs):
        print('='*20 + f' Epoch: {epoch} '+ '='*20)
        model_train(train_loader, model, optimizer, criterion=criterion, labels=y_train_tensor)
        acc = predict(val_loader, model, criterion=criterion, labels=val_label)
        if acc >= train_best:
            train_best = acc
            # 模型保存
            torch.save(model.state_dict(), f'./best_model{str(boost_num)}.point')
    # 开始boosting
    train_tensor = torch.cat(boost_feature, dim=0)
    y_train_tensor = torch.cat(boost_label, dim=0)

print('Stage5: model score')
for i in range(boost_epoch_num):
    model_name = f'./best_model{i}.point'
    # 重新初始化模型
    model = DNN().to(device)
    model.load_state_dict(torch.load(model_name))
    # 这里一定要开启验证模式
    model.eval()
    print('--'*24)
    print(f'Model name: {model_name}')
    # 这里只能用验证集来看准确率
    preds = model(torch.FloatTensor(val_sp).to(device)).argmax(dim=1).cpu().numpy()
    score = (preds == val_label).sum()/len(val_label)
    print(f'Score: {score}')
```

softvoting代码部分：
```
import numpy as np
import pandas as pd
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
import logging
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

# 固定随机数种子，确保实验的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]

# 搭建DNN模型
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val = np.load('../val/10type_sort_eval_data_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')
test = np.load('../test/10type_sort_test_data_8192.npy')

val_sp = get_fft_and_scaler(val, 6892, 7192)
test_sp = get_fft_and_scaler(test, 6892, 7192)

model_num = 3
preds_list = []
weight = [0.7, 0.6, 0.7]
for i in range(model_num):
    model = DNN().to(device)
    model.load_state_dict(torch.load(f'./best_model{i}.point'))
    model.eval()
    preds = model(torch.FloatTensor(test_sp).to(device)).detach().cpu()
    preds_list.append(preds * weight[i])

# 软投票
# pytorch 中使用这个方法让多个张量想加
ans = torch.sum(torch.stack(preds_list), dim=0).argmax(dim=1).numpy()

print('Stage5: make submmit')
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('boost_solution.csv', index=False)
```

### 模型改进和多DNN模型融合

上面boost思想中，使用的是最后一个epoch模型分类正确的样本，这里改变为，验证集最
高的分类正确的样本，因为这样可以找到训练集中，分布和测试集比较相似的数据。

代码如下：

```
import numpy as np
import pandas as pd
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
from imblearn.over_sampling import SMOTE

def get_fft_and_scaler(data, start=6892, end=7192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.dnn(x)
        return x

print('Stage1: load data')
# 读取训练集，测试集和验证集

# 读取训练集和验证集的标签，测试集是没有标签的，需要你使用模型进行分类，并将结果进行提交
train = np.load('../train/10type_sort_train_data_8192.npy')
val = np.load('../val/10type_sort_eval_data_8192.npy')
train_label = np.load('../train/10type_sort_train_label_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')

print('Stage2: data over_sampling')
smote = SMOTE(random_state=42, n_jobs=-1)
x_train, train_label = smote.fit_resample(train, train_label)

train_sp = get_fft_and_scaler(x_train, start=6892, end=7192)
val_sp = get_fft_and_scaler(val, start=6892, end=7192)

def train(train_loader, model, optimizer, criterion, labels):
    model.train()
    train_total_acc = 0
    train_loss = 0
    for feature, label in train_loader:
        feature = feature.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        preds = model(feature)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()

        train_total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
        train_loss += loss.item()

        feature.cpu()
        label.cpu()

    print(
        f'Training loss: {train_loss/len(train_loader):.4f}',
        f'Training  acc: {train_total_acc/len(labels):.4f}',
         )

def predict(val_loader, model, criterion, labels):
    model.eval()
    val_total_acc = 0
    val_loss = 0
    for feature, label in val_loader:
        feature = feature.to(device)
        label = label.to(device)
        preds = model(feature)
        loss = criterion(preds, label)

        val_total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
        val_loss += loss.item()

        feature.cpu()
        label.cpu()

    print(
        f'Val loss: {val_loss/len(val_loader):.4f}',
        f'Val  acc:{val_total_acc/len(labels):.4f}'
    )
    return val_total_acc/len(labels)

def get_boost_data(model, train_sp, train_label, chunks=128):
    model.eval()
    if not isinstance(train_sp, torch.Tensor):
        train_sp = torch.FloatTensor(train_sp)
    if not isinstance(train_label, torch.Tensor):
        train_label = torch.LongTensor(train_label)
    data = torch.chunk(train_sp, chunks=chunks, dim=0)
    start = 0
    end = 0
    for feature in data:
        feature = feature.to(device)
        end += feature.shape[0]
        label = train_label[start:end]
        preds = model(feature).argmax(1).cpu()
        tag = (preds == label)
        boost_feature.append(feature[tag])
        boost_label.append(label[tag])
        start = end
    #print(f'acc: {torch.cat(boost_label, dim=0).shape[0]/train_label.shape[0]}')

# 设计思路
# 首先进行第一轮boost学习，所以外层是很多boost for循环, 内层是epoch循环
# 如果第一轮中，有分数高于某个点，比如0.614的，则保存模型
# 使用比较好的模型，进行提取数据，作为下一轮的循环学习
train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(train_label).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()
# 使用Dataloader对数据进行封装
val_tensor = TensorDataset(val_tensor, y_val_tensor)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_best = 0
count = 0
boost_nums = 4
batch_size = 128
epochs = 30
model = DNN().to(device)
lr = 0.0001
eps = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=lr, eps=eps)
criterion = nn.CrossEntropyLoss()
set_seed(42)
for boost_num in range(boost_nums):
    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    boost_feature = []
    boost_label = []
    print('--'*8 + f'boost round: {boost_num}/{boost_nums - 1}' + '--'*8)
    print(f'train shape: {train_tensor.shape}')
    print('--'*24)
    for epoch in range(epochs):
        print('='*20 + f' Epoch: {epoch} '+ '='*20)
        train(train_loader, model, optimizer, criterion=criterion, labels=y_train_tensor)
        acc = predict(val_loader, model, criterion=criterion, labels=y_val_tensor)
        # 第0轮只保存最佳模型
        if boost_num == 0 and acc >= train_best:
            train_best = acc
            torch.save(model.state_dict(), f'./best_model{str(count)}.point')
        else:
            if acc >= 0.613:
                # 保存最佳模型
                count += 1
                torch.save(model.state_dict(), f'./best_model{str(count)}.point')
    # 获取下一个boost的数据
    temp_model = DNN().to(device)
    temp_model.load_state_dict(torch.load(f'./best_model{str(count)}.point'))
    get_boost_data(temp_model, train_tensor, y_train_tensor)
    train_tensor = torch.cat(boost_feature, dim=0)
    y_train_tensor = torch.cat(boost_label, dim=0)

# 这个10是自己定义的，要看你自己这边生成了多少模型
# 下面打印出每个模型的准确率
for i in range(10):
    model_name = f'./best_model{i}.point'
    # 重新初始化模型
    model = DNN().to(device)
    model.load_state_dict(torch.load(model_name))
    # 这里一定要开启验证模式
    model.eval()
    print('--'*24)
    print(f'Model name: {model_name}')
    # 这里只能用验证集来看准确率
    preds = model(torch.FloatTensor(val_sp).to(device)).argmax(dim=1).cpu().numpy()
    score = (preds == val_label).sum()/len(val_label)
    print(f'Score: {score}')
```

软投票代码：

```
from scipy import stats
import numpy as np
import pandas as pd
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
import logging
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

# 固定随机数种子，确保实验的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]

# 搭建DNN模型
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.dnn(x)
        return F.softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val = np.load('../val/10type_sort_eval_data_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')
test = np.load('../test/10type_sort_test_data_8192.npy')

val_sp = get_fft_and_scaler(val, 6892, 7192)
test_sp = get_fft_and_scaler(test, 6892, 7192)

model_num = 3
preds_list = []
for i in range(model_num):
    model = DNN().to(device)
    model.load_state_dict(torch.load(f'./best_model{i}.point'))
    model.eval()
    preds = model(torch.FloatTensor(val_sp).to(device)).argmax(dim=1).cpu().numpy()
    preds_list.append(preds)

# 使用scipy中的硬投票
preds = np.vstack(preds_list)
ans = stats.mode(preds).mode.flatten()
score = (ans == val_label).sum()/len(val_label)
print(score)

print('Stage5: make submmit')
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('boost_solution.csv', index=False)
```

## 思考与总结

在本次题目中，还可以对以下几点进行改进

1. 数据处理方面

>- 我们只是做了简单的fft变换，并没有对数据进行滤波等操作，可以尝试对信号进行处理
>后，再用神经网络进行训练
>- 关于fft的切片，我们也只是使用了其中的300个点，试验发现，以300个点为一个片
>  段，除了fft峰值那一部分，其他位置也能分类，但是准确率没那么高，有些位置也能
>  到40的准确率，可以查看下这些片段的分类混淆矩阵

2. 模型方面

>- 这里只是使用了简单的DNN，并没有尝试擅长处理时间序列的LSTM模型，以及一维卷积
>  核，可以尝试学习调节下这几类神经网络
> - 尝试使用多模型进行融合
