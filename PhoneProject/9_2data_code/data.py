import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import gc
from datetime import datetime
import random

random.seed(42)
np.random.seed(42)

class Data:
    def __init__(self, logger):
        self.data = None
        self.logger = logger

    def load(self, path):
        self.data = np.load(path)
        return self.data

    @staticmethod
    def process(data):
        #length,  = len(data)
        H, W = data.shape
        #data = (data - np.mean(data, 1).reshape(length, -1)) / np.std(data, 1).reshape(length, 1)

        # 下面三行是对数据的标准化
        #mean_ = np.mean(data, 1).reshape(H, -1)
        #std_ = np.std(data, 1).reshape(H, 1)
        #data = ((data - mean_) / std_).reshape(-1, 2, 64, 64)

        # 下面几行是对数据的归一化
        # data = np.abs(data)
        data = data / np.expand_dims(np.max(np.abs(data), 1), axis=1)
        #data = np.fft.fft(data)
        #data = np.stack([np.real(data), np.imag(data)])
        gc.collect()
        return data

    def read_9_2_data_by_shutdown(self, path='../9_2data'):
        count = 0
        last_time = None
        res = {}
        path = Path('../9_2data/')
        for sub_path in list(path.iterdir()):
            parent_name = sub_path.name
            for dat_file in sorted(list(sub_path.iterdir())):
                dat_file = str(dat_file.stem)
                if dat_file.split('_')[1] != '0':
                    break
                else:
                    count += 1
                    last_time = dat_file.split('_')[-1]
            if count > 1:
                res[parent_name] = last_time
            count = 0
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        label_count = 0
        for sub_path in list(path.iterdir()):
            parent_name = sub_path.name
            if res.get(parent_name) == None:
                continue
            else:
                end_time = res.get(parent_name)
                end_time = datetime.strptime(end_time, '%Y%m%d%H%M%S')
                for dat_file in list(sub_path.iterdir()):
                    time = str(dat_file.stem).split('_')[-1]
                    time = datetime.strptime(time, '%Y%m%d%H%M%S')
                    data = np.fromfile(dat_file, np.int16)
                    L = len(data) // 8192
                    data = data[: L*8192].reshape(-1, 8192)
                    if (time < end_time):
                        x_train.extend(data)
                        y_train.extend([label_count]*L)
                    else:
                        x_test.extend(data)
                        y_test.extend([label_count]*L)
                label_count += 1
        x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
        self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
        self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
        return x_train, x_test, y_train, y_test

    def read_data_by_del_less_data_phone(self, root='../9_2data'):
        p = Path('../9_2data')
        phone_count = {}
        for phone in list(p.iterdir()):
            phone_name = phone.name
            if phone_count.get(phone_name) is None:
                phone_count[phone_name] = {}
            for data_file in phone.iterdir():
                time = data_file.stem.split('_')[-1][0:8]
                time = datetime.strptime(time, '%Y%m%d')
                if phone_count[phone_name].get(time) is None:
                    phone_count[phone_name][time] = []
                phone_count[phone_name][time].append(data_file)
        del_phone = []
        for phone, value in phone_count.items():
            if len(value.keys()) <= 2:
                del_phone.append(phone)
        for name in del_phone:
            del phone_count[name] # 删除测试时间小于一天的手机
        x_train, x_test, y_train, y_test = [], [], [], []
        count = 0
        for phone_name, value in phone_count.items():
            sort_list = sorted(value.keys())
            for train_time in sort_list:
                for file_path in value.get(train_time):
                    data = np.fromfile(file_path, np.int16)
                    L = len(data) // 8192
                    data = data[: L*8192].reshape(-1, 8192)
                    if train_time not in sort_list[-2:]:
                        x_train.append(data)
                        y_train.extend([count]*L)
                    else:
                        x_test.append(data)
                        y_test.extend([count]*L)
            count += 1
        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
        self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
        return x_train, x_test, y_train, y_test


    def read_9_2_data_as_30_phones_by_day(self, root='../9_2data', rate=0.5):
        '''
        按天数排序，前面时间做训练集，后面时间做测试集
        '''
        phone_count = {}
        p = Path(root)
        # 读取数据部分
        phone_count = {}
        for phone in list(p.iterdir()):
            phone_name = phone.name
            if phone_count.get(phone_name) is None:
                phone_count[phone_name] = {}
            for data_file in phone.iterdir():
                time = data_file.stem.split('_')[-1]
                time = datetime.strptime(time, '%Y%m%d%H%M%S')
                phone_count[phone_name][time] = data_file

        gc.collect()

        x_train, x_test = [], []
        y_train, y_test = [], []
        count = 0
        for phone_name, value in phone_count.items():
            sort_list = sorted(value.keys())
            length = len(sort_list)
            train_len = int(length * rate)
            for train_path in sort_list[:train_len]:
                file_path = value.get(train_path)
                data = np.fromfile(file_path, np.int16)
                L = len(data) // 8192
                data = data[:L*8192].reshape(-1, 8192)
                x_train.append(data)
                y_train.extend([count]*L)
            for test_path in sort_list[train_len:]:
                file_path = value.get(test_path)
                data = np.fromfile(file_path, np.int16)
                L = len(data) // 8192
                data = data[:L*8192].reshape(-1, 8192)
                x_test.append(data)
                y_test.extend([count]*L)
            count += 1

        gc.collect()
        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
        self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
        return x_train, x_test, y_train, y_test

    def recognization_data_process(self, x_train, x_test, y_train, y_test, un_label=None, rate=0.6):
        # 因为要训练模型，同时做未知源的时候，手机信号要没见过
        # 所以要将已知源手机一部分作为训练，一部分作为未知源
        # 函数输出train, test, val, train and test是模型训练需要，val是未知源识别
        un_label = [0, 1, 2, 3, 4, 5, 6, 7] # 设置未知源
        x_val = None
        y_val = []
        # 先将所有未知源加入到val 中
        for label in un_label:
            train_temp = x_train[y_train == label, :]
            test_temp = x_test[y_test == label, :]
            y_val.extend([0]*(len(train_temp) + len(test_temp)))
            if x_val is None:
                x_val = np.concatenate([train_temp, test_temp])
            else:
                x_val = np.concatenate([x_val, train_temp, test_temp])
            del train_temp, test_temp

            # 将train test 中的未知源删除
            x_train = np.delete(x_train, y_train == label, axis=0)
            y_train = np.delete(y_train, y_train == label)
            x_test = np.delete(x_test, y_test == label, axis=0)
            y_test = np.delete(y_test, y_test == label)
        # 再将测试集中一定量的已知源加入到val中，并打上标签为1
        idx = random.sample(range(len(y_test)), int(len(y_test)*rate))
        test_temp = x_test[idx, :]
        x_val = np.concatenate([x_val, test_temp])
        y_val.extend([1]*len(test_temp))

        # 将这些加入进去的手机删除
        #x_test = np.delete(x_test, idx, axis=0)
        #y_test = np.delete(y_test, idx)

        # 将训练集和测试集的标签重新从0开始编号
        def get_label_map(label):
            true_label = label
            label = set(label)
            label_map = dict(zip(label, range(len(label))))
            true_label = list(map(lambda x:label_map.get(x), true_label))
            return np.array(true_label)
        y_train = get_label_map(y_train)
        y_test = get_label_map(y_test)
        y_val = np.array(y_val)
        self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
        self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
        self.logger.info(f'y_val bincount:{np.bincount(y_val)}')

        return x_train, x_test, x_val, y_train, y_test, y_val

    def read_9_2_data_by_day(self, root='../9_2data'):
        # 读取数据部分
        root = Path(root)
        phone_count = {}
        for phone in list(root.iterdir()):
            phone_name, _ = phone.name.split('_')
            if phone_count.get(phone_name) is None:
                phone_count[phone_name] = {}
            for data_file in phone.iterdir():
                stem_time = data_file.stem.split('_')[-1][:8]
                # 00 - 12 点的看成前半部分
                # 12 - 00 的看成后半部分
                time1 = datetime.strptime(stem_time + "00", '%Y%m%d%H')
                time2 = datetime.strptime(stem_time + "12", '%Y%m%d%H')

                day = data_file.stem.split('_')[-1][:10]
                day = datetime.strptime(day, '%Y%m%d%H')

                if phone_count[phone_name].get(time1) is None:
                    phone_count[phone_name][time1] = []
                if phone_count[phone_name].get(time2) is None:
                    phone_count[phone_name][time2] = []
                data = np.fromfile(str(data_file), np.int16)
                H = int(data.shape[0]/8192)
                data = data[:H*8192]
                if time1 <= day < time2:
                    phone_count[phone_name][time1].append(data.reshape(-1, 8192))
                else:
                    phone_count[phone_name][time2].append(data.reshape(-1, 8192))
        gc.collect()
        # 整理数据部分
        x_train, x_test = [], []
        y_train, y_test = [], []
        count = 0
        for phone, value in phone_count.items():
            date_list = sorted(value.keys())
            length = len(date_list)
            if length <= 4:
                # 如果测试时间只有两天或者一天，那么取一半做训练集，一半做测试集即可
                left = length//2 + 1
            else:
                # 如果测试时间有三天或者以上，则前两天用来训练，后一天用来测试
                left = length - 2
            for time in date_list[:left]:
                if not value.get(time):
                    continue
                data = np.concatenate(value[time])
                l = data.shape[0]
                x_train.extend(data)
                y_train.extend([count] * l)
            for time in date_list[left:]:
                if not value.get(time):
                    continue
                data = np.concatenate(value[time])
                l = data.shape[0]
                x_test.extend(data)
                y_test.extend([count] * l)
            count += 1
        x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
        self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
        self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
        return x_train, x_test, y_train, y_test

    def read_9_2_data_by_rounds(self, root='../9_2data'):
        '''
        9月2日新数据的读取方式，读取方式为按轮数读取，
        数据一共5轮，前4轮做训练，最后一轮做为测试
        '''
        p = Path(root)
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
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        gc.collect()
        self.logger.info(f'y_test bincount :{np.bincount(test_label)}')
        self.logger.info(f'y_train bincount:{np.bincount(train_label)}')
        return x_train, x_test, train_label, test_label


    def read_data_by_random(self, root='./data_8.8/'):
        x_train, x_test, y_train, y_test = [], [], [], []
        test_id = [9, 10]#random.sample(range(0, 9), 2)

        root = Path(root)
        root_path = list(sorted(root.iterdir()))
        phone_list = []
        for i in root_path:
            phone_list.append(list(i.iterdir()))
        sub_phone_list = []
        for i in phone_list:
            for phone in i:
                if phone.name[-2:] not in ['09', '15']:
                    temp = sorted(list(phone.iterdir()), key=lambda x:int(x.name))
                    if len(temp) == 12:
                        sub_phone_list.append(temp[:-2])
                    else:
                        sub_phone_list.append(temp)
        phone_name = []
        label_id = []
        count = 0
        for i in sub_phone_list:
            for num in i:
                name = num.parent.name
                if name not in phone_name:
                    phone_name.append(name)
                x_train_temp = []
                x_test_temp = []
                for file in num.iterdir():
                    temp = np.fromfile(file, dtype=np.int16)
                    length = min(int(len(temp) / 8192), 60) # 有些长度不确定 小于60
                    temp = temp[:length*8192].reshape(-1, 8192)
                    if int(num.name) not in test_id:
                        x_train_temp.extend(temp)
                    else:
                        x_test_temp.extend(temp)
                y_train.extend([count]*len(x_train_temp))
                y_test.extend([count]*len(x_test_temp))
                x_train.extend(x_train_temp)
                x_test.extend(x_test_temp)
            label_id.append(count)
            count += 1

        self.logger.info(f'test phone: {test_id[0]}, {test_id[1]}')
        self.logger.info(f'x_train:{len(x_train)}, y_train:{len(y_train)}, x_test:{len(x_test)}, y_test:{len(y_test)}')
        self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
        self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

class Train_Test_Split:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # 按类别分类好手机信号
        self.x_train_data_list = []
        self.y_train_data_list = []
        self.x_test_data_list = []
        self.y_test_data_list = []

    def fit_transform(self, mode):
        """
        mode: 数据划分的方式
            1. train 训练模型提取embedding, 就是对普通手机操作，测试集中不能含有未知源
            2. test 验证有源无源识别, 测试集中必须含有已知源和未知源
        """
        # 将手机信号按训练集和测试集的类别分好
        for i in range(10):
            temp0 = self.x_train[self.y_train == i]
            temp1 = self.y_train[self.y_train == i]
            temp2 = self.x_test[self.y_test == i]
            temp3 = self.y_test[self.y_test == i]
            self.x_train_data_list.append(temp0)
            self.y_train_data_list.append(temp1)
            self.x_test_data_list.append(temp2)
            self.y_test_data_list.append(temp3)
        # 因为一共有10类手机，这里按照7类手机为已知源，3类手机为未知源
        # 训练集中只能含有已知源手机，测试集中已知源和未知源手机信号都要含有
        # 因为该数据训练集和测试集的时间差异比较大，所以这里按照下列方法来创建新的训练测试集
        # 1. 新的训练集只来自旧的训练集，新的测试集只来自旧的测试集
        # 2. 从训练集中随机抽取3类手机作为未知源，这三类未知源的信号，可以分到新的测试集中，从而新的训练集只含有已知源信号
        # 3. 新的测试集中含有已知源，也含有未知源
        unknown_ = [0, 1, 2] # 为了保证结果稳定，先固定手机类别
        #unknow_ = random.sample(range(10), 3) # 随机生成3个未知源手机
        known_ = [3, 4, 5, 6, 7, 8, 9]
        self.new_x_train_list = []
        self.new_y_train_list = []
        self.new_x_test_list = []
        self.new_y_test_list = []
        if mode == 'train':
            for i in range(10):
                if i in known_:
                    self.new_x_train_list.append(self.x_train_data_list[i])
                    self.new_y_train_list.append(self.y_train_data_list[i])
                    self.new_x_test_list.append(self.x_test_data_list[i])
                    self.new_y_test_list.append(self.y_test_data_list[i])
        else:
            for i in range(10):
                if i in unknown_:
                    temp0 = np.concatenate([self.x_train_data_list[i], self.x_test_data_list[i]], axis=0)
                    temp1 = np.concatenate([self.y_train_data_list[i], self.y_test_data_list[i]], axis=0)
                    self.new_x_test_list.append(temp0)
                    self.new_y_test_list.append(temp1)
                if i in known_:
                    self.new_x_train_list.append(self.x_train_data_list[i])
                    self.new_y_train_list.append(self.y_train_data_list[i])
        x_train = np.concatenate(self.new_x_train_list, axis=0)
        y_train = np.concatenate(self.new_y_train_list, axis=0)
        x_test = np.concatenate(self.new_x_test_list, axis=0)
        y_test = np.concatenate(self.new_y_test_list, axis=0)
        return x_train, x_test, y_train, y_test

class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

class MyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

class Preporcess:
    def __init__(self):
        pass

