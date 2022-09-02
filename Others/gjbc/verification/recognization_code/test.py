#!/home/zwl/miniconda3/envs/asr/bin/python3
import torch
import torch.nn as nn
import numpy as np
from pathlib import path
import sys
sys.path.append('..')
import data
from torchvision.models import resnet18
import logging
import logging.config
from utils import get_logging_config
import gc
from data import data, mydataloader, mydataset

# 测试文件，验证准确率读取是不是正确的

# logger set
logging.config.dictconfig(get_logging_config(file_name='test.log'))
logger = logging.getlogger('logger')

# 数据读取
train_config = {
    'batch_size':128,
    'shuffle':true,
    'drop_last':true,
    'pin_memory':true,
}

test_config = {
    'batch_size':128,
    'shuffle':true,
    'drop_last':true,
    'pin_memory':true,
}

print('stage1: data load')
data = data(logger)
x_train, x_test, y_train, y_test = data.read_9_2_data_as_30_phones_by_day('../../9_2data/')
un_label = [0, 1, 2, 3, 4, 5, 6, 7] # 设置未知源

for label in un_label:
    idx = (y_test == label)
    x_test = np.delete(x_test, idx, axis=0)
    y_test = np.delete(y_test, idx)
def get_label_map(label):
    true_label = label
    label = set(label)
    label_map = dict(zip(label, range(len(label))))
    true_label = list(map(lambda x:label_map.get(x), true_label))
    return np.array(true_label)

y_test = get_label_map(y_test) if un_label else y_test # 如果有未知源，就标签映射

class_num = len(np.bincount(y_test))
gc.collect()
print('load data successful')
x_test = data.process(x_test)
print('process data successful')

gc.collect()
test_dataset = mydataset(x_test, y_test)

test_loader = mydataloader(test_dataset, **test_config)

del x_train, y_train

gc.collect()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnet18()
model.conv1 = nn.conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=false)
model.maxpool = nn.adaptiveavgpool2d(output_size=(7, 7))
model.fc = nn.linear(in_features=512, out_features=class_num, bias=true)
# 数据测试
model.load_state_dict(torch.load('./9_2data_read_data_as_30_phones_by_day_resnet_standarscale.model'))
model = model.to(device)

with torch.no_grad():
    model.eval()
    acc_num = 0
    for feature, label in test_loader:
        feature = feature.reshape(-1, 2, 64, 64).to(device)
        label = label.to(device)
        acc_num += (model(feature).argmax(1) == label).sum()
logger.info(acc_num / len(y_test))
