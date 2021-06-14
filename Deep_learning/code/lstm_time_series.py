import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import pandas as pd


train_path = './train'
val_path = './val'
test_path = './test'

train = np.load(train_path + '/' + '10type_sort_train_data_8192.npy')
test = np.load(test_path + '/' + '10type_sort_test_data_8192.npy')
val = np.load(val_path + '/' + '10type_sort_eval_data_8192.npy')

train_label = np.load(train_path + '/' + '10type_sort_train_label_8192.npy')
val_label = np.load(val_path + '/' + '10type_sort_eval_label_8192.npy')

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:,start:end]

train_sp = get_fft_and_scaler(train)
test_sp = get_fft_and_scaler(test)
val_sp = get_fft_and_scaler(val)

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


sequence_length = 1
input_size = 3000
hidden_size = 512
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01
gamma = 0.9
step_size=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 官网默认的输入是seq_len, batch, input_size,这里将batch first=True放在第一个
        # seq len 相当于一句话的长度， input size相当于每个单词的向量长度
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化两个向量的值, 因为这里是两层的lstm
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播，这里只需要out即可
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 使用全连层进行分类
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 学习方式
# Train the model
total_step = len(train_loader)
train_acc, val_acc = [], []

for epoch in range(num_epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # reshape的维度变成了(batch size, 28, 28)
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == labels).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # val the model
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for images, labels in tqdm(val_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean()
            epoch_val_accuracy += acc / len(val_loader)

    scheduler.step()
    print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')
    print(f'val acc:{epoch_val_accuracy:.4f}')

    train_acc.append(epoch_accuracy)
    val_acc.append(epoch_val_accuracy)

def get_submmit(model, test, batch=128, dim=0):
    # 初始化结果提交向量
    # test为numpy类型的数据
    preds = np.zeros(test.shape[0])
    res = torch.FloatTensor(test)
    res = torch.chunk(res, chunks=batch, dim=dim)
    start = 0
    end = 0
    for images in tqdm(res):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        output = model(images).argmax(dim=1).detach().cpu().numpy()
        end += output.shape[0]
        preds[start:end] = output
        start = end

    return preds

model.eval()
ans = get_submmit(model, test_sp)
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('lstm_solution.csv')
