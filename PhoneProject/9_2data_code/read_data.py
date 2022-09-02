from pathlib import Path
import numpy as np


def read_data(path='../9_2data/'):
    p = Path(path)
    phone_count = {}
    for phone in list(p.iterdir()):
        # 记录所有手机
        phone_name, rounds = phone.name.split('_')
        rounds = int(rounds)
        if phone_count.get(phone_name) is None:
            phone_count[phone_name] = {}
        if phone_count[phone_name].get(rounds) is None:
            phone_count[phone_name][rounds] = []
        for data_file in phone.iterdir():
            data = np.fromfile(str(data_file), np.int16)
            H = int(data.shape[0]/8192)
            data = data[:H*8192]
            phone_count[phone_name][rounds].append(data.reshape(-1, 8192))

    x_train, x_test = [], []
    train_label, test_label = [], []
    count = 0
    for phone, value in phone_count.items():
        for rounds, data in value.items():
            data = np.concatenate(data)
            length = data.shape[0]
            if rounds != 5:
                x_train.extend(data)
                train_label.extend([count] * length)
            else:
                x_test.extend(data)
                test_label.extend([count] * length)
        count += 1
    return np.array(x_train), np.array(x_test), np.array(train_label), np.array((test_label))
