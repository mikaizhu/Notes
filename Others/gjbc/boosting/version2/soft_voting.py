from scipy import stats
import numpy as np
import pandas as pd
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import random
import logging
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

# 固定随机数种子，确保实验的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]

# 搭建DNN模型
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
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.dnn(x)
        return F.softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val = np.load('../val/10type_sort_eval_data_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')
test = np.load('../test/10type_sort_test_data_8192.npy')

val_sp = get_fft_and_scaler(val, 6892, 7192)
test_sp = get_fft_and_scaler(test, 6892, 7192)

model_num = 3
preds_list = []
weight = [0.4, 0.5, 0.3]
for i in range(model_num):
    model = DNN().to(device)
    model.load_state_dict(torch.load(f'./best_model{i}.point'))
    model.eval()
    preds = model(torch.FloatTensor(val_sp).to(device)).detach().cpu()
    preds_list.append(preds * weight[i])

# 软投票
# pytorch 中使用这个方法让多个张量想加
ans = torch.sum(torch.stack(preds_list), dim=0).argmax(dim=1).numpy()
score = (ans == val_label).sum()/len(val_label)
print(score)

print('Stage5: make submmit')
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('boost_solution.csv', index=False)
