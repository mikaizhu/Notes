import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
from imblearn.over_sampling import SMOTE
import argparse

parser = argparse.ArgumentParser() # 首先实例化
parser.add_argument('--train_path', type=str)
parser.add_argument('--train_label_path', type=str)
parser.add_argument('--test_path', type=str)
parser.add_argument('--test_label_path', type=str)
parser.add_argument('--val_path', type=str)
parser.add_argument('--val_label_path', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--sp_start', type=int)
parser.add_argument('--sp_end', type=int)
args = parser.parse_args() # 解析参数

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:,start:end]

# 固定随机数种子，确保实验的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class SampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, 3, 3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 1024),
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
        x = x.view(x.shape[0], 1, -1)
        x = self.cnn(x)
        x = x.view(x.shape[0], x.size(1) * x.size(2))
        x = self.dnn(x)
        return F.softmax(x, dim=1)

# 读取数据
print('Stage1: load data')
train = np.load(args.train_path)
val = np.load(args.val_path)
train_label = np.load(args.train_label_path)
val_label = np.load(args.val_label_path)
test = np.load(args.test_path)
test_label = np.load(args.test_label_path)

# 样本均衡
print('Stage2: data over sampling')
smote = SMOTE(random_state=42, n_jobs=-1)
x_train, y_train = smote.fit_resample(train, train_label)

print(x_train.shape, y_train.shape)

# 注意这里要对x_train, 数据处理，做fft和切片
print('Stage3: data fft slice and scaler')
test_sp = get_fft_and_scaler(test, start=args.sp_start, end=args.sp_end)
train_sp = get_fft_and_scaler(x_train, start=args.sp_start, end=args.sp_end)
val_sp = get_fft_and_scaler(val, start=args.sp_start, end=args.sp_end)

batch_size = args.batch_size
train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(y_train).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()

# 使用Dataloader对数据进行封装
train_tensor = TensorDataset(train_tensor, y_train_tensor)
val_tensor = TensorDataset(val_tensor, y_val_tensor)

train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

lr = args.lr
gamma = 0.9
step_size = 1
epochs = args.epochs
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

model = SampleCNN().to(device)
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

print('Stage4: model training')
for epoch in range(epochs):
    print('='*20 + f' Epoch: {epoch} '+ '='*20)
    train(train_loader, model, optimizer, criterion=criterion, labels=y_train)
    loss = predict(val_loader, model, criterion=criterion, labels=val_label)
    if loss <= train_best:
        train_best = loss
        best_model = model

# 保存最佳模型
print('Stage5: model save')
torch.save(best_model.state_dict(), './cnn_best_model.point')
torch.save(model.state_dict(), './cnn_model.point')

print('Stage6: model eval')
model.eval()
best_model.eval()
ans = best_model(torch.FloatTensor(test_sp).to(device)).argmax(dim=1).detach().cpu().numpy()
score = (ans == test_label).sum()/len(test_label)
print(f'best model score: {score}')

ans = model(torch.FloatTensor(test_sp).to(device)).argmax(dim=1).detach().cpu().numpy()
score = (ans == test_label).sum()/len(test_label)

print(f'model score: {score}')
