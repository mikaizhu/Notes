import gc
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal # 这样调用scipy中的signal模块
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import sys
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def read_data(DATA_PATH, n=12):
    features, labels = [], []
    for label, directory in enumerate(tqdm(os.listdir(DATA_PATH))):
        path = os.path.join(DATA_PATH, directory)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            data = np.fromfile(file_path, dtype=np.int16)
            data = np.asarray(data)[:302500].reshape([100, -1])[:, 1:].reshape([100, 14, 108, 2]) # 取出了100个ofdm符号, 因为是实数和虚数交替，所以最后一个维度是2
            data = np.delete(data, [3, 10], axis=1) # 剔除掉第3和第10列，参考信号
            data = data[:, :, :, 0] + 1j*data[:, :, :, 1] # 108 * 2 IQ信号， 实数和虚数相加
            data = data.reshape([-1, 108])
            data = np.fft.ifft(data, n=108) # 对每个信号做ifft
            length = data.shape[0] // n # 统计有多少个12符号，不足的话就删掉后面的
            features.extend([data[i*n:(i+1)*n, :].flatten() for i in range(length)]) # 抽取信号
            labels.extend([label]*length)
    return np.asarray(features), np.asarray(labels)
   
def get_stft(data, nfft=108, x=108, y=108, windows=20, oneside=True, overlap=8, dtype='complex64'):
    fs = 122.68e6
    result = np.zeros((data.shape[0], x, y), dtype=dtype)
    for idx, i in enumerate(tqdm(data)):
        f, t, zxx = signal.stft(
        i, 
        fs=fs,
        nfft=nfft,
        window=signal.get_window('hann', windows),
        noverlap=overlap,
        nperseg=windows,
        return_onesided=oneside) # 设置双边谱
        result[idx, :] = zxx[:, 1:].astype(dtype)
    return result
    
def get_image(stft, dtype='float32'):
    real = np.zeros_like(stft, dtype=dtype)
    imag = np.zeros_like(stft, dtype=dtype)
    angel = np.zeros_like(stft, dtype=dtype)
    for idx, i in enumerate(tqdm(stft)):
        temp_imag = i.imag
        real[idx, :] = i.real.astype(dtype)
        imag[idx, :] = i.imag.astype(dtype)
        angel[idx, :] = np.arctan(i.real/i.imag).astype(dtype)
    length = len(real)
    image = np.stack([real, imag, angel], axis=1) # 这里只要将维度设置为1即可
    del real, imag, angel; gc.collect()
    return image

root = 'original_data/'

data, labels = read_data(root)
stft = get_stft(data)
del data; gc.collect()

image = get_image(stft)
del stft; gc.collect()

class MyDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self, idx):
        img = self.file_list[idx]
        # img_transformed = self.transform(img) # 因为转换后嵌套是元组，所以这里取内部
        label = self.label_list[idx]
        
        return img, label

image_labels = torch.tensor(labels)

x_train, x_test, y_train, y_test = train_test_split(image, image_labels, test_size=0.3, shuffle=True)

# 图片尺寸为108，或者128
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(1152, 2048)
        self.out = nn.Linear(2048, 10)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.reshape(-1, 1152)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.out(x)
        out = F.softmax(x)
        return out

class MyDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self, idx):
        img = self.file_list[idx]
#         img_transformed = self.transform(img) # 因为转换后嵌套是元组，所以这里取内部
        label = self.label_list[idx]
        
        return img, label

model = CNN().cuda()
transform = transforms.Compose([
    transforms.ToTensor(),# 然后将图片转换成张量,注意会维度变换，具体看官网，所以这里不要使用transformer了
#     transforms.ToPILImage(),# 因为原先数据是矩阵，所以要先转换成图片
#     transforms.CenterCrop(224),
#     transforms.Resize(224),# 对图片进行裁剪，resize
#     transforms.ToTensor(),# 然后将图片转换成张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = MyDataset(x_train, y_train, transform=transform)
test_data = MyDataset(x_test, y_test, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

batch_size = 32
epochs = 100
lr = 0.0001
gamma = 0.9
step_size=5

device_count = torch.cuda.device_count()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if device_count > 1:
    model = nn.DataParallel(model,device_ids=range(device_count)) # multi-GPU
    model.to(device)

else:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 学习方式

# 训练模型
train_acc, test_acc = [], []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data.cuda()
        label = label.cuda()
        output = model(data)
    
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        
        data.cpu()
        label.cpu()
        
    with torch.no_grad():
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        for data, label in tqdm(test_loader):
            data = data.cuda()
            label = label.cuda()
            
            test_output = model(data)
            test_loss = criterion(test_output, label)
            
            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)
            data.cpu()
            label.cpu()
            
    print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')
    print(f'test loss:{epoch_test_loss:.4f}, test acc:{epoch_test_accuracy:.4f}')

    train_acc.append(epoch_accuracy)
    test_acc.append(epoch_test_accuracy)
