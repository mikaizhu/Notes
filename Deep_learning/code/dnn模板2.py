import numpy as np
import pandas as pd
from collections import Counter
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random

# 说明：这个代码有两个细节
# 1. 时间序列输入到神经网络中的时候一定要归一化
# 2. 输入到神经网络前，一定要batch norm
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
    def __init__(self, n_class=10):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(500),
            nn.Linear(500, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_class),
        )

    def forward(self, x):
        x = self.dnn(x)
        return F.softmax(x, dim=1)

train_path = './train'
val_path = './val'
test_path = './test'

# 读取训练集，测试集和验证集
train = np.load(train_path + '/' + '10type_sort_train_data_8192.npy')
test = np.load(test_path + '/' + '10type_sort_test_data_8192.npy')
val = np.load(val_path + '/' + '10type_sort_eval_data_8192.npy')

# 读取训练集和验证集的标签，测试集是没有标签的，需要你使用模型进行分类，并将结果进行提交
train_label = np.load(train_path + '/' + '10type_sort_train_label_8192.npy')
val_label = np.load(val_path + '/' + '10type_sort_eval_label_8192.npy')

train_sp = get_fft_and_scaler(train, start=6692, end=7192)
test_sp = get_fft_and_scaler(test, start=6692, end=7192)
val_sp = get_fft_and_scaler(val, start=6692, end=7192)

# 将数据转换成pytorch的tensor
batch_size = 128

train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(train_label).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()

# 使用Dataloader对数据进行封装
train_tensor = TensorDataset(train_tensor, y_train_tensor)
val_tensor = TensorDataset(val_tensor, y_val_tensor)


train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

n_class = 10
lr = 0.001
gamma = 0.9
step_size = 1
epochs = 80
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
model = DNN(n_class=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
    return val_loss

train_best = float('inf')
best_model = None

for epoch in range(epochs):
    print('='*20 + f' Epoch: {epoch} '+ '='*20)
    train(train_loader, model, optimizer, criterion=criterion, labels=train_label)
    loss = predict(val_loader, model, criterion=criterion, labels=val_label)
    if loss <= train_best:
        train_best = loss
        best_model = model

# 保存最佳模型
torch.save(best_model.state_dict(), './best_model.point')
# 加载模型权重
#model.load_state_dict(torch.load('model.point'))

