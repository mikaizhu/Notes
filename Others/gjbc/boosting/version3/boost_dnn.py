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
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
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

def get_boost_data(feature, true_label, model):
        # 存储boosting数据, 这里使用epoch=0时候的分类错误的，因为这样数据才会多一点，或包括大部分分类错误的
        preds_label = model(feature).argmax(dim=1)
        f1 = feature[preds_label != true_label, :].detach().cpu()
        f2 = true_label[preds_label != true_label].detach().cpu()
        boost_feature.append(f1)
        boost_label.append(f2)

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

print('Stage1: load data')
# 读取训练集，测试集和验证集
train = np.load('../train/10type_sort_train_data_8192.npy')
val = np.load('../val/10type_sort_eval_data_8192.npy')

# 读取训练集和验证集的标签，测试集是没有标签的，需要你使用模型进行分类，并将结果进行提交
train_label = np.load('../train/10type_sort_train_label_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')

print('Stage2: data over_sampling')
smote = SMOTE(random_state=42, n_jobs=-1)
x_train, y_train = smote.fit_resample(train, train_label)

train_sp = get_fft_and_scaler(x_train, start=6892, end=7192)
val_sp = get_fft_and_scaler(val, start=6892, end=7192)

# 将数据转换成pytorch的tensor
print('Stage3: transform numpy data to tensor')
batch_size = 128

train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(y_train).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()

# 使用Dataloader对数据进行封装
val_tensor = TensorDataset(val_tensor, y_val_tensor)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

lr = 0.0001
gamma = 0.9
step_size = 1
epochs = 15
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# 使用boost ing的想法，让神经网络学习错误的类别
# boosting 的想法：训练后面几次，分类错误的数据
# 重新定义一个新的分类器进行学习错误的数据, 保存这些模型的参数
# 然后使用模型融合
print('Stage4: start training')
# 这里设置boost 的num数，设置为多少就会训练多少个dnn模型
boost_epoch_num = 4
model = DNN().to(device)
for boost_num in range(boost_epoch_num):
    # 分类器学习, 更新训练集
    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    #model = DNN().to(device)
    boost_feature = []
    boost_label = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_best = float('inf')
    best_model = None
    print('--'*8 + f'boost round: {boost_num}/{boost_epoch_num - 1}' + '--'*8)
    print(f'train shape: {train_tensor.shape}')
    print('--'*24)
    for epoch in range(epochs):
        print('='*20 + f' Epoch: {epoch} '+ '='*20)
        model_train(train_loader, model, optimizer, criterion=criterion, labels=y_train_tensor)
        loss = predict(val_loader, model, criterion=criterion, labels=val_label)
        if loss <= train_best:
            train_best = loss
            best_model = model
    # 模型保存
    torch.save(best_model.state_dict(), f'./best_model{str(boost_num)}.point')
    get_boost_data(train_tensor.to(device), y_train_tensor.to(device), model=best_model)
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
