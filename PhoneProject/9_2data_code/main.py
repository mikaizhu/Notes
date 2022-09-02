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

print('Stage1: data load')
data = Data(logger)
x_train, x_test, y_train, y_test = data.read_9_2_data_by_shutdown()
# 如果只是单纯训练模型，则只要将下面注释即可，如果要未知源识别，则取消注释下面代码
# x_train, x_test, x_val, y_train, y_test, y_val = data.recognization_data_process(x_train, x_test, y_train, y_test)
gc.collect()
print('load data successful')
x_train = data.process(x_train)
x_test = data.process(x_test)
#x_val = data.process(x_val) 模型训练部分不需要val
#del x_val, y_val
print('data process done')

gc.collect()
train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_loader = MyDataLoader(train_dataset, **train_config)
test_loader = MyDataLoader(test_dataset, **test_config)


model = resnet18()
model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.maxpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
model.fc = nn.Linear(512, len(np.bincount(y_test)), bias=True)

epochs = 10
lr = 1e-4
criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model' : model,
    'optimizer' : optimizer,
    'scheduler' : scheduler,
    'criterion' : criterion,
    'epochs' : epochs,
    'train_loader' : train_loader,
    'test_loader' : test_loader,
    'logger' : logger,
    'model_save_name' : str(Path(logger.handlers[0].baseFilename).stem) + '.model',
}

print('Stage2: model training')
trainer = Trainer(**config)
trainer.train()
