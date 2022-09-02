import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random

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
train = np.load('../train/10type_sort_train_data_8192.npy')
val = np.load('../val/10type_sort_eval_data_8192.npy')
train_label = np.load('../train/10type_sort_train_label_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')
test = np.load('../test/10type_sort_test_data_8192.npy')


# 注意这里要对x_train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_sp = get_fft_and_scaler(train, start=6892, end=7192)
val_sp = get_fft_and_scaler(val, start=6892, end=7192)
test_sp = get_fft_and_scaler(test, start=6892, end=7192)
model = SampleCNN().to(device)

lr = 0.0001
gamma = 1
step_size = 1
EPOCH = 1

optimizer = optim.Adam(model.parameters(), lr=lr)
lf = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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

train_best = 1000
for epoch in range(EPOCH):
    model.train()
    train_total_acc = 0

    for feature, label in tqdm(train_loader):
        feature = feature.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        preds = model(feature)
        loss = lf(preds, label)
        loss.backward()
        optimizer.step()

        train_total_acc += model(feature).argmax(dim=1).eq(label).sum().item()

        feature.cpu()
        label.cpu()

    model.eval()
    with torch.no_grad():
        val_total_acc = 0
        for feature, label in tqdm(val_loader):
            feature = feature.cuda()
            label = label.cuda()
            val_total_acc += model(feature).argmax(dim=1).eq(label).sum().item()

            feature.cpu()
            label.cpu()

    scheduler.step()
    print(f'epoch:{epoch:2} loss:{loss:4f} train_acc:{train_total_acc/len(train_label):4f}')
    print('val_acc:{:4f}'.format(val_total_acc/len(val_label)))

    # 保存最佳的训练模型
    if loss <= train_best:
        train_best = loss
        torch.save(model.state_dict(), './cnn_model.point')

model.eval()
ans = model(torch.FloatTensor(test_sp).unsqueeze(1).to(device)).argmax(dim=1).detach().cpu().numpy()

pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('cnn_solution.csv', index=False)
