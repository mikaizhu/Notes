import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

train_path = '../train'
val_path = '../val'
test_path = '../test'

# 读取训练集，测试集和验证集 和 标签
train = np.load(train_path + '/' + '10type_sort_train_data_8192.npy')
test = np.load(test_path + '/' + '10type_sort_test_data_8192.npy')
val = np.load(val_path + '/' + '10type_sort_eval_data_8192.npy')

train_label = np.load(train_path + '/' + '10type_sort_train_label_8192.npy')
val_label = np.load(val_path + '/' + '10type_sort_eval_label_8192.npy')

# 定义函数，将原来的时间序列数据，转换成fft
# 并且是先归一化，然后再对数据进行切片
# 不能先切片再归一化
def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:,start:end]

# cnn和lstm的混合模型
# 利用一维的cnn卷积核来提取特征，用lstm来完成特征的分类
# 这里为了保留特征同时对特征提取，设置卷积核为奇数，步长为1，填充值为2k+1中的k
# 然后利用最大池化来降低数据维度，除了最大池化，还可以尝试其他的pooling结构
# cnn提取完特征后，每个数据样本会变成一个一维度的向量
# 然后将这个一维度的特征向量，输入到lstm中完成分类即可
# lstm原理：https://wmathor.com/index.php/archives/1397/
# cnn原理：https://wmathor.com/index.php/archives/1353/
# ps:自己可以在知乎上查下相关的教程
class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            #nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=2),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
        )
        # 官网默认的输入是seq_len, batch, input_size,这里将batch first=True放在第一个
        # seq len 相当于一句话的长度， input size相当于每个单词的向量长度
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        # 初始化两个向量的值, 因为这里是两层的lstm
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播，这里只需要out即可
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 使用全连层进行分类
        out = self.fc(out[:, -1, :])
        return out

# 调用切片函数
# 原始数据会切片以及fft和归一化
train_sp = get_fft_and_scaler(train, 6892, 7192)
test_sp = get_fft_and_scaler(test, 6892, 7192)
val_sp = get_fft_and_scaler(val, 6892, 7192)

# 将numpy数据转换成tensor，即pytorch的数据结构
train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(train_label).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()

# 使用Dataloader对数据进行封装
batch_size = 128
train_tensor = TensorDataset(train_tensor, y_train_tensor)
val_tensor = TensorDataset(val_tensor, y_val_tensor)

train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

#sequence_length = 128
# 参数说明
# input size：lstm模型中，每个向量的长度
# hidden size： lstm中 初始化的隐藏层向量长度
# num layers：因为lstm可以叠加，所以这里设置叠加多少层
# batch size: 因为不能一次性将所有数据都输入到模型中，所以只能将数据分成很多块输入到模型中
# learning rate：调整可以提高模型的准确率
# gamma 为学习率衰减, 每经过step size后，学习率就会变成lr = lr*gamma
input_size = 36
hidden_size = 512
num_layers = 2
num_classes = 10
batch_size = 128
num_epochs = 5
learning_rate = 0.001
gamma = 1
step_size=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = CRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数，告诉模型预测值与实际值还差多少
criterion = nn.CrossEntropyLoss()
# 定义优化器，告诉模型如何调整参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 学习方式
#Train the model
# 下面是开始训练模型
# 模型的训练步骤如下：
# 1. 前向传播，模型预测结果
# 2. 计算损失函数，看预测值与真实值差距
# 3. 根据损失函数，通过优化器更新神经网络模型参数
# 4. 再前向传播，查看差距，计算损失函数
# 5. 不断循环上面步骤
total_step = len(train_loader)
train_acc, val_acc = [], []

for epoch in range(num_epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
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
            images = images.to(device)
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
    for chunk_data in tqdm(res):
        chunk_data = chunk_data.to(device)
        output = model(chunk_data).argmax(dim=1).detach().cpu().numpy()
        end += output.shape[0]
        preds[start:end] = output
        start = end

    return preds.astype(int)

# 生成最后的提交结果
ans = get_submmit(model, test_sp)
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('stack_solution.csv', index=False)

