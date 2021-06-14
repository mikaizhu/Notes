import numpy as np
import pandas as pd
# dnn模型构建
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from catboost import Pool, CatBoostClassifier

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

def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:, start:end]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Stage1: load data')
val = np.load('../val/10type_sort_eval_data_8192.npy')
val_label = np.load('../val/10type_sort_eval_label_8192.npy')
test = np.load('../test/10type_sort_test_data_8192.npy')
test_label = np.load('../test/10type_sort_test_label_8192.npy')

print('Stage2: get fft and scaler')
val_sp = get_fft_and_scaler(val, 6892, 7192)
test_sp = get_fft_and_scaler(test, start=6892, end=7192)


print('Stage3: model predict')
model0 = DNN().to(device)
model1 = DNN().to(device)
model2 = DNN().to(device)
model0.load_state_dict(torch.load('best_model0.point'))
model1.load_state_dict(torch.load('best_model1.point'))
model2.load_state_dict(torch.load('best_model2.point'))

model0.eval()
model1.eval()
model2.eval()
preds0 = model0(torch.FloatTensor(val_sp).to(device)).argmax(dim=1).cpu().numpy()
preds1 = model1(torch.FloatTensor(val_sp).to(device)).argmax(dim=1).cpu().numpy()
preds2 = model2(torch.FloatTensor(val_sp).to(device)).argmax(dim=1).cpu().numpy()

score0 = (preds0 == val_label).sum()/len(val_label)
score1 = (preds1 == val_label).sum()/len(val_label)
score2 = (preds2 == val_label).sum()/len(val_label)
print(f'score0: {score0}, score1: {score1}, score2: {score2}')

print('Stage4: Start stacking')
val_feature = np.stack([preds0, preds1, preds2], axis=1)

model = CatBoostClassifier(#loss_function="Logloss",
                           eval_metric="AUC",
                           task_type="GPU",
                           learning_rate=0.1,
                           iterations=5000,
                           l2_leaf_reg=50,
                           random_seed=43,
                           od_type="Iter",
                           depth=5,
                           early_stopping_rounds=1000,
                           border_count=64,
                           loss_function='MultiClass',
                           #has_time= True
                          )

model.fit(
val_feature, val_label
)
preds0 = model0(torch.FloatTensor(test_sp).to(device)).argmax(dim=1).cpu().numpy()
preds1 = model1(torch.FloatTensor(test_sp).to(device)).argmax(dim=1).cpu().numpy()
preds2 = model2(torch.FloatTensor(test_sp).to(device)).argmax(dim=1).cpu().numpy()
test_feature = np.stack([preds0, preds1, preds2], axis=1)
preds = model.predict(test_feature)
ans = preds.flatten()

print((ans == test_label).sum() / len(test_label))

print('Stage5: make submmit')
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('boost_solution.csv', index=False)

