import os
import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb

train_path = './train'
val_path = './val'
test_path = './test'

# 读取训练集，测试集和验证集
train = np.load(train_path + '/' + '10type_sort_train_data_8192.npy')
test = np.load(test_path + '/' + '10type_sort_test_data_8192.npy')
val = np.load(val_path + '/' + '10type_sort_eval_data_8192.npy')

# 读取训练集和验证集的标签，测试集是没有标签的，需要你使用模型进行分类，并将结果进行提交
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

xg_train = xgb.DMatrix(train_sp, label=train_label)
xg_val = xgb.DMatrix(val_sp, label=val_label)
xg_test = xgb.DMatrix(test_sp)


# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['nthread'] = -1
param['num_class'] = 10
param['tree_method']: 'gpu_hist'
param['eval_metric'] = 'mlogloss'
#watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 1


bst = xgb.train(param, xg_train, num_round)
# get prediction
pred = bst.predict(xg_val)
acc_rate = np.sum(pred == val_label) / val_label.shape[0]
print('Test acc using softmax = {}'.format(acc_rate))

ans = bst.predict(xg_test)
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('xgb_solution.csv')

