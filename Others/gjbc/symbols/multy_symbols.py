#!/home/zwl/miniconda3/envs/asr/bin/python3
from data import Data, Process, MyDataset, MyDataloader
from exp import Exp
from model import DnnNet
from train import Trainer
from utils import ModelSave, ModelLoad
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import logging

parser = argparse.ArgumentParser() # 首先实例化
parser.add_argument('--symbols', type=int, default=1)
args = parser.parse_args() # 解析参数

# 多符号有多种思路：
# 本思路是采用利用多个符号输入到神经网络中，然后将最后一层进行相加进行预测
# 还可以采用投票的方式进行预测

symbols = args.symbols
best_score = -float('inf')


def set_logger(file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    # 设置记录文件名和记录的形式
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)

    # 设置让控制台也能输出信息
    file_stream = logging.StreamHandler()
    file_stream.setFormatter(formatter)

    # 为记录器添加属性，第一行是让记录器只记录错误级别以上，第二行让记录器日志写入，第三行让控制台也能输出
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(file_stream)
    return logger

logger = set_logger('train.log')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

print('Stage1: data load and process')
data = Data()
x_train = data.load('../old_data/train/10type_sort_train_data_8192.npy')
y_train = data.load('../old_data/train/10type_sort_train_label_8192.npy')
x_test = data.load('../old_data/val/10type_sort_eval_data_8192.npy')
y_test = data.load('../old_data/val/10type_sort_eval_label_8192.npy')

x_train_data_list = []
y_train_data_list = []
x_test_data_list = []
y_test_data_list = []
for i in range(9):
    temp0 = x_train[y_train == i]
    temp1 = y_train[y_train == i]
    temp2 = x_test[y_test == i]
    temp3 = y_test[y_test == i]
    x_train_data_list.append(temp0)
    y_train_data_list.append(temp1)
    x_test_data_list.append(temp2)
    y_test_data_list.append(temp3)

for idx, i in enumerate(x_train_data_list):
    if i.shape[0] % symbols > 0:
        line = i.shape[0] % symbols
        i = np.delete(i, range(0, line), 0)
    temp = i.reshape(-1, 8192*symbols)
    x_train_data_list[idx] = temp
    y_train_data_list[idx] = np.full((len(temp), ), idx)

for idx, i in enumerate(x_test_data_list):
    if i.shape[0] % symbols > 0:
        line = i.shape[0] % symbols
        i = np.delete(i, range(0, line), 0)
    temp = i.reshape(-1, 8192*symbols)
    x_test_data_list[idx] = temp
    y_test_data_list[idx] = np.full((len(temp), ), idx)

x_train = np.concatenate(x_train_data_list, axis=0)
#self.x = self.x / np.expand_dims(np.max(np.abs(self.x), axis=1), axis=1)
x_train = x_train / np.expand_dims(np.max(np.abs(x_train), axis=1), axis=1)
y_train = np.concatenate(y_train_data_list, axis=0)
x_test = np.concatenate(x_test_data_list, axis=0)
x_test = x_test / np.expand_dims(np.max(np.abs(x_test), axis=1), axis=1)
y_test = np.concatenate(y_test_data_list, axis=0)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = DnnNet()
model.classify[0].out_features = 10

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_loader = MyDataloader(train_dataset, shuffle=True, drop_last=True, batch_size=128)
test_loader = MyDataloader(test_dataset, shuffle=True, drop_last=True, batch_size=128)

model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

print('Stage2: model training')
for epoch in range(100):
    model.train()
    train_loss = 0
    train_acc_num = 0
    train_num = 0
    for feature, label in train_loader:
        feature = feature.reshape(128, symbols, 8192).cuda()
        label = label.cuda()
        optimizer.zero_grad()
        preds = torch.zeros((128, 10)).cuda()
        for i in range(symbols):
            preds = model(feature[:, i, :])
            loss = criterion(preds, label)
            train_loss += loss/symbols
            train_acc_num += (preds.argmax(1) == label).sum().item()
            train_num += len(label)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_acc_num = 0
        test_num = 0
        for feature, label in test_loader:
            feature = feature.reshape(128, symbols, 8192).cuda()
            label = label.cuda()
            preds = torch.zeros((128, 10)).cuda()
            output = np.zeros((128, symbols))
            for i in range(symbols):
                preds = model(feature[:, i, :])
                loss = criterion(preds, label)
                test_loss += loss/symbols
                output[:, i] = preds.argmax(1).cpu().numpy()
                #test_acc_num += (preds.argmax(1) == label).sum().item()
            # start voting
            preds = torch.zeros(128, )
            output = output.astype(int)
            for idx, line in enumerate(output):
                preds[idx] = np.argmax(np.bincount(line))
            test_acc_num += (preds == label.cpu()).sum().item()
            test_num += len(label)

    train_acc = train_acc_num / train_num
    test_acc = test_acc_num / test_num
    best_score = max(best_score, test_acc)
    print(f'Epoch:{epoch:4} | Train Loss:{train_loss/len(train_loader):6.4f} | Train Acc:{train_acc:.4f}')
    print(f'           | Test Loss:{test_loss/len(test_loader):6.4f} | Test Acc:{test_acc:.4f}')
    scheduler.step(test_loss)

logger.info(f'symbols: {symbols}')
logger.info(f'best_score:{best_score}')
