#!/home/zwl/miniconda3/envs/asr/bin/python3
import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from data import Data, MyDataLoader, MyDataset
from trainer import Trainer
from utils import set_seed
from pathlib import Path

import logging
import logging.config
from utils import get_logging_config
import gc
import numpy as np
import random
from tqdm import tqdm

##########数据读取##########
# logger set
logging.config.dictConfig(get_logging_config(file_name='9_2data_read_data_as_30_phones_by_day_resnet_StandarScale.log'))
logger = logging.getLogger('logger')

set_seed(42)

train_config = {
    'batch_size':128,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}

test_config = {
    'batch_size':128,
    'shuffle':True,
    'drop_last':True,
    'pin_memory':True,
}

val_config = {
    'batch_size':128,
    'shuffle':False,
    'drop_last':False,
    'pin_memory':True,
}

print('Stage1: data load')
data = Data(logger)
x_train, x_test, y_train, y_test = data.read_9_2_data_as_30_phones_by_day()
# 如果只是单纯训练模型，则只要将下面注释即可，如果要未知源识别，则取消注释下面代码
x_train, x_test, x_val, y_train, y_test, y_val = data.recognization_data_process(x_train, x_test, y_train, y_test)
gc.collect()
print('load data successful')
x_train = data.process(x_train)
x_test = data.process(x_test)
x_val = data.process(x_val) #模型训练部分不需要val
num_classes = len(np.bincount(y_test))

##########特征提取##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = resnet18()
model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.maxpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
model.fc = nn.Linear(512, num_classes, bias=True)
model.load_state_dict(torch.load('./model/9_2data_read_data_as_30_phones_by_day_resnet_StandarScale.model'))
model = model.to(device)

print_base = {} # 存放所有的特征，键为类别，值为指纹
# 从训练集中挑选样本, 选取0.3作为样本
train_idx = random.sample(range(len(y_train)), int(len(y_train)*0.3))
# print(np.bincount(y_train)) # 查看标签分布
train_sample = x_train[train_idx, :]
train_dataset = MyDataset(train_sample, y_train[train_idx])
train_loader = MyDataLoader(train_dataset, shuffle=False, batch_size=128)

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def regroup(print_base, activation, label):
    for key, value in activation.items():
        activation[key] = value.reshape(-1, 512)
        for i in range(num_classes):
            idx = (label == i)
            print_base[i].extend(activation[key][idx, :])
    return print_base

num_classes = len(np.bincount(y_train))
print_base = {key:[] for key in range(num_classes)}

# 开始提取训练集的指纹特征，存储在print_base中
with torch.no_grad():
    model.eval()
    model.avgpool.register_forward_hook(get_activation(model.avgpool))
    for feature, label in train_loader:
        activation = {}
        feature = feature.reshape(-1, 2, 64, 64).to(device)
        label = label.to(device)
        model(feature)
        print_base = regroup(print_base, activation, label)

for key, value in print_base.items():
    value = torch.cat(value).reshape(-1, 512)
    print_base[key] = torch.sum(value, 0) / value.shape[0] # 对向量求和 并求平均

del x_train, y_train, x_test, y_test
gc.collect()
##########特征分类##########
val_dataset = MyDataset(x_val, y_val)
val_loader = MyDataLoader(val_dataset, **val_config)

val_feature = []
val_label = []
with torch.no_grad():
    model.eval()
    model.avgpool.register_forward_hook(get_activation(model.avgpool))
    for feature, label in val_loader:
        activation = {}
        feature = feature.reshape(-1, 2, 64, 64).to(device)
        label = label.to(device)
        model(feature)
        for key, value in activation.items():
            val_feature.append(value.reshape(-1, 512).cpu().numpy())
            val_label.extend(label.cpu().numpy())

val_feature = np.concatenate(val_feature)
val_label = np.array(val_label)
gc.collect()

val_feature = torch.tensor(val_feature).float()
val_label = torch.tensor(val_label).long()
# 开始为每个指纹计算余弦相似度

res = []
for source_tensor in tqdm(val_feature):
    for key, value in print_base.items():
        attention_score = torch.cosine_similarity(source_tensor, value, dim=0)
        res.append(attention_score)
res = np.array(res).reshape(-1, num_classes)

max_res = np.max(res, 1)
result = ((max_res > 0.95).astype(int) == val_label.numpy())
acc = sum(result) / len(result)
print(acc)
