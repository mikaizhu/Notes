from catboost import Pool, CatBoostClassifier
import numpy as np
import pandas as pd
from collections import Counter

train_path = './train'
val_path = './val'
test_path = './test'

train = np.load(train_path + '/' + '10type_sort_train_data_8192.npy')
test = np.load(test_path + '/' + '10type_sort_test_data_8192.npy')
val = np.load(val_path + '/' + '10type_sort_eval_data_8192.npy')

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

model = CatBoostClassifier(#loss_function="Logloss",
                           eval_metric="AUC",
                           task_type="GPU",
                           learning_rate=0.1,
                           iterations=10,
                           l2_leaf_reg=50,
                           random_seed=43,
                           od_type="Iter",
                           depth=5,
                           early_stopping_rounds=15000,
                           border_count=64,
                           loss_function='MultiClass',
                           #has_time= True
                          )
model.fit(
    train_sp, train_label,
    eval_set=(val_sp, val_label),
)
preds = model.predict(val_sp)
print((preds.flatten() == val_label).sum() / len(val_label))
ans = model.predict(test_sp).flatten()
pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('catboost_solution.csv')
