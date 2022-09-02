import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

x_train, x_test, y_train, y_test = np.load('./30type__sort_junheng_pilot_train_data_10.npy'),\
np.load('./30type__sort_junheng_pilot_test_data_10.npy'),\
np.load('./30type__sort_junheng_pilot_train_label_10.npy'),\
np.load('./30type__sort_junheng_pilot_test_label_10.npy')

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(1080)
        self.fc1 = nn.Linear(1080, 4028)
        self.dropout = nn.Dropout(p=0.2)
        self.bn2 = nn.BatchNorm1d(4028)
        self.fc2 = nn.Linear(4028, 1080)
        self.bn3 = nn.BatchNorm1d(1080)
        self.out = nn.Linear(1080, 30)
    def forward(self, x):
        x = x.reshape(-1, 1080)
        x = self.dropout(x) # 在输入前dropout，相当于抛弃掉一些特征
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.out(x)
        return F.softmax(x)

x_train_tensor = torch.tensor(x_train).float()
y_train_tensor = torch.tensor(y_train).long()
x_test_tensor = torch.tensor(x_test).float()
y_test_tensor = torch.tensor(y_test).long()

train_tensor = TensorDataset(x_train_tensor, y_train_tensor)
test_tensor = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_tensor, shuffle=True, batch_size=batch_size)

batch_size = 32
lr = 0.00005
gamma = 0.9
step_size = 10
EPOCH = 30

model = DNN().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
lf = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in range(EPOCH):
    model.train()
    total_acc = 0
    total_num = 0
    
    for feature, label in train_loader:
        feature = feature.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        preds = model(feature)
        loss = lf(preds, label)
        loss.backward()
        optimizer.step()
        
        total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
        total_num += label.shape[0]
        
    print(f'epoch:{epoch:2} loss:{loss:4f}')
    print(f'train acc:{total_acc/total_num:4f}')
    
    model.eval()
    with torch.no_grad():
        total_acc = 0
        total_num = 0
        for feature, label in test_loader:
            feature = feature.cuda()
            label = label.cuda()
            total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
            total_num += label.shape[0]
            
        print('val_acc:{:4f}'.format(total_acc/total_num))
    
    scheduler.step()
