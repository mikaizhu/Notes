import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import gc
from datetime import datetime

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
        mean_ = np.mean(data, 1).reshape(H, -1)
        std_ = np.std(data, 1).reshape(H, 1)
        data = ((data - mean_) / std_).reshape(-1, 2, 64, 64)
        gc.collect()
        #data = np.abs(data)
        #data = data / np.expand_dims(np.max(np.abs(data), 1), axis=1)
        #data =
        #data = np.fft.fft(data)
        #data = np.stack([np.real(data), np.imag(data)])
        return data
    def read_9_2_data_as_30_phones_by_day(self, root='../9_2data'):
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
            train_len = int(length * 0.8)
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

