import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


parser = argparse.ArgumentParser() # 首先实例化
parser.add_argument('--test_path', type=str)
parser.add_argument('--test_label_path', type=str)
parser.add_argument('--sp_start', type=int)
parser.add_argument('--sp_end', type=int)
args = parser.parse_args() # 解析参数

# 固定随机数种子，确保实验的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

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
            nn.BatchNorm1d( 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.dnn(x)
        return F.softmax(x, dim=1)
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

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:,start:end]

logger.info('Stage1: load data')
test = np.load(args.test_path)
test_label = np.load(args.test_label_path)

logger.info('Stage2: data fft sp scaler')
test_sp = get_fft_and_scaler(test, start=args.sp_start, end=sp_end)

logger.info('Stage3: model stack')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = DNN().to(device)
model2 = SampleCNN().to(device)
model1.load_state_dict(torch.load('dnn_best_model.point'))
model2.load_state_dict(torch.load('cnn_best_model.point'))

logger.info('Stage4: model score')
model1.eval()
model2.eval()
preds1 = model1(torch.FloatTensor(test_sp).to(device))
preds2 = model2(torch.FloatTensor(test_sp).to(device))
logger.info(preds1.shape, preds2.shape)
ans = (preds1 * 0.6 + preds2 * 0.4).argmax(dim=1).detach().cpu().numpy()

score = (ans == test_label).sum()/len(test_label)
logger.info(f'model stack score: {score}')

logger.info('Stage5: make submmit')
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('stack_solution.csv', index=False)
