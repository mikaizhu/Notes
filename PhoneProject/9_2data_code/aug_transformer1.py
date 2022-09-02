#!/home/zwl/miniconda3/envs/asr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import os
import logging
import scipy.io as io
from sklearn.metrics import confusion_matrix
from pathlib import Path


def read_data_by_random(root='../data_8.8/'):
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

    logger.info(f'test phone: {test_id[0]}, {test_id[1]}')
    logger.info(f'x_train:{len(x_train)}, y_train:{len(y_train)}, x_test:{len(x_test)}, y_test:{len(y_test)}')
    logger.info(f'y_test bincount :{np.bincount(y_test)}')
    logger.info(f'y_train bincount:{np.bincount(y_train)}')
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


# exp config 选择试验
#exp = 'old_origin_transformer'
#exp = 'old_abs_origin_transformer'
#exp = 'old_fft_origin_transformer'
#exp = 'old_abs_fft_origin_transformer'
exp = 'newdata_old_time_aug_origin_transformer'
#exp = 'old_time_abs_aug_origin_transformer'

# data config
#train_path = '../../old_data/train/10type_sort_train_data_8192.npy'
#val_path = '../../old_data/val/10type_sort_eval_data_8192.npy'
#train_label_path = '../../old_data/train/10type_sort_train_label_8192.npy'
#val_label_path = '../../old_data/val/10type_sort_eval_label_8192.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

window_len = 7808       # 信号切分的大小
signal_len = 8192
window_num = 64         # 切分信号的数量
stride_len = (signal_len - window_len)/window_num          # 切分信号的步长

def set_logger(file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    # 设置记录文件名和记录的形式
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)

    # 设置让控制台也能输出信息
    file_stream = logging.StreamHandler()
    file_stream.setFormatter(formatter)

    # 为记录器添加属性，第一行是让记录器只记录错误级别以上，第二行让记录器日志写入，第三行让控制台也能输出
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(file_stream)
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class MyDataset(Dataset):
    def __init__(self, x, y, data_process=True):
        #self.x, _, self.y, _ = np.load(x_path)
        #self.y = np.load(y_path)
        self.x = x
        self.y = y
        if data_process:
            self.data_process()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        feature = self.x[idx]
        label = self.y[idx]
        return [feature, label]


    def data_process(self):
        #self.x = np.abs(self.x)
        #self.x = np.fft.fft(self.x)
        #self.x = np.abs(self.x)#[:, 6892:7192]
        self.x = self.x / np.expand_dims(np.max(np.abs(self.x), axis=1), axis=1)

    def collate_fn(self, batch):
        # 自己定义一些batch的数据处理方式
        # 任何数据处理的工作，都可以在collate fn中进行操作
        '''
        dtype[optional]:
            torch.float
            torch.long
        '''
        batch = np.array(batch, dtype=object)
        # 这里一定要stack，不然会报错
        x = np.stack(batch[:, 0], axis=0)
        y = np.stack(batch[:, 1], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def dataloader(self, batch_size, shuffle=True, drop_last=False):
        return DataLoader(
            self, # 这里self就是实例本身
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
class Net(nn.Module):
    def __init__(self, d_model=64, num_layers=2, n_head=8, dropout=0.2):
        super().__init__()
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout).to(device)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout).to(device)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=num_layers)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=num_layers)
        self.decoder = nn.Sequential(
                nn.Linear(7808, 8),
                #nn.Softmax(dim=1),
            )
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = x[:, 0, ...]
        x2 = x[:, 1, ...]
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x1 = x1.reshape(batch_size, -1)
        x2 = x2.reshape(batch_size, -1)
        x = torch.cat([x1, x2], axis=1)
        x = self.decoder(x)
        return x

# below func is to debug collate_fn
# next(iter(train_loader))

class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 exp='fft_abs_dnn',
                 ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model = model.to(self.device)
        self.exp = exp # set your own experiment
        self.best_model = None
        self.best_score = -np.inf
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

    def train_step(self, dataloader):
        self.model.train()
        acc_num = 0
        total_loss = 0
        for feature, label in dataloader:
            feature = feature.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            predict = self.model(feature)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            acc_num += self.cal_acc(predict, label)

            feature.cpu()
            label.cpu()
        return total_loss, acc_num

    def eval_step(self, dataloader):
        self.model.eval()
        acc_num = 0
        total_loss = 0
        with torch.no_grad():
            for feature, label in dataloader:
                feature = feature.to(self.device)
                label = label.to(self.device)
                predict = self.model(feature)
                loss = self.criterion(predict, label)
                acc_num += self.cal_acc(predict, label)
                total_loss += loss.item()

                feature.cpu()
                label.cpu()
        return total_loss, acc_num

    def aug_train_step(self, dataloader):
        '''
        set your own augmentation method
        '''
        self.model.train()
        acc_num = 0
        train_num = 0
        total_loss = 0
        for feature, label in dataloader:
            feature = feature.to(self.device)
            label = label.to(self.device)
            for _ in range(window_num):
                self.optimizer.zero_grad()
                start = random.randint(0, signal_len - window_len - 1)
                aug_feature = feature[:, int(start) : int(start+window_len)]
                aug_feature = aug_feature.reshape(-1, 2, 61, 64)
                predict = model(aug_feature)
                loss = self.criterion(predict, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                acc_num += self.cal_acc(predict, label)
                train_num += feature.shape[0]

                feature.cpu()
                label.cpu()

        return total_loss, acc_num, train_num

    def aug_val_step(self, dataloader):
        self.model.eval()
        acc_num = 0
        val_num = 0
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for feature, label in dataloader:
                feature = feature.to(self.device)
                label = label.to(self.device)
                output = torch.zeros(feature.shape[0], window_num).to(self.device)
                start = 0
                val_num += feature.shape[0]
                for _ in range(window_num):
                    aug_feature = feature[:, int(start):int(start + window_len)]
                    start += stride_len
                    aug_feature = aug_feature.reshape(-1, 2, 61, 64)
                    predict = self.model(aug_feature)
                    loss = self.criterion(predict, label)
                    total_loss += loss.item()
                    predict = torch.argmax(predict, 1)
                    output[:, _] = predict.reshape(-1, )

                    feature.cpu()
                    label.cpu()
                output = output.cpu().numpy().astype(np.int16)
                rank = np.zeros(output.shape[0])
                for i in range(output.shape[0]):
                    rank[i] = np.argsort(np.bincount(output[i, :]))[-1]

                rank = torch.LongTensor(rank)
                acc_num += (rank.cpu() == label.cpu()).sum()
                all_preds.append(rank.cpu())
                all_labels.append(label.cpu())
                #rank = torch.LongTensor(np.expand_dims(rank, 1))
                #acc_num += self.cal_acc(rank.cpu(), label.cpu())

        return total_loss, acc_num, val_num, torch.cat(all_preds, 0), torch.cat(all_labels, 0)

    def start_train(self, epochs, train_loader, val_loader, patience=10, augmentation=True):
        best_val_loss = np.inf
        best_val_acc = -np.inf
        for epoch in range(epochs):
            if not augmentation:
                train_loss, train_acc_num = self.train_step(dataloader=train_loader)
                val_loss, val_acc_num = self.eval_step(dataloader=val_loader)
            else:
                train_loss, train_acc_num, train_num = self.aug_train_step(dataloader=train_loader)
                val_loss, val_acc_num, val_num, all_preds, all_labels = self.aug_val_step(dataloader=train_loader)

            # scheduler step
            self.scheduler.step(val_loss)

            # cal acc
            if not augmentation:
                train_acc = train_acc_num / len(train_loader.dataset)
                val_acc = val_acc_num / len(val_loader.dataset)
            else:
                train_acc = train_acc_num / train_num
                val_acc = val_acc_num / val_num

            # 这里loss不除以样本数，而除以loader数，是因为loss是按batch来进行计算的j
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # early stpping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _patience = patience
            else:
                _patience -= 1
            if not _patience:
                logger.info('Stopping early!')
                break

            # logging
            logger.info(
                f'Epoch: {epoch:^3} | '
                f'train loss: {train_loss:^6.4} | '
                f'train acc: {train_acc:^6.4} | '
                f'val loss: {val_loss:^6.4} | '
                f'val acc: {val_acc:^6.4}'
            )

            # get best score
            self.best_score = max(self.best_score, val_acc)

            # save exp
            self.train_acc.append(train_acc)
            self.train_loss.append(train_loss)
            self.val_acc.append(val_acc)
            self.val_loss.append(val_loss)

            # best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_model = self.model
                self.save_preds(all_preds, all_labels)

    def cal_acc(self, predict, label):
        return (predict.argmax(1) == label).sum().item()

    def save_preds(self, preds, labels):
        preds = preds.numpy()
        labels = labels.numpy()
        path = ''
        if not os.path.exists('../../exp'):
            os.mkdir('../../exp')
        if not os.path.exists('../../exp/' + self.exp):
            path += '../../exp/' + self.exp
            os.mkdir(path)
        else:
            path += '../../exp/' + self.exp
        mat = confusion_matrix(labels, preds)
        io.savemat(path + '/confusion_matrix.mat', {'name': mat})
        np.save(path + '/train_acc.npy', self.train_acc)
        np.save(path + '/preds.npy', preds)
        np.save(path + '/labels.npy', preds)

    def model_save(self):
        if not os.path.exists('../../checkpoint'):
            os.mkdir('../../checkpoint')
        file_name = '../../checkpoint/' + self.exp + '.model'
        torch.save(self.best_model.state_dict(), file_name)

    def save_exp(self):
        path = ''
        if not os.path.exists('../../exp'):
            os.mkdir('../../exp')
        if not os.path.exists('../../exp/' + self.exp):
            path += '../../exp/' + self.exp
            os.mkdir(path)
        else:
            path += '../../exp/' + self.exp
        np.save(path + '/train_acc.npy', self.train_acc)
        np.save(path + '/train_loss.npy', self.train_loss)
        np.save(path + '/val_acc.npy', self.val_acc)
        np.save(path + '/val_loss.npy', self.val_loss)

log_path = '../../log'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logger = set_logger(log_path + '/' + exp + '.log')

logger.info('Stage1: load and process data')
x_train, x_test, y_train, y_test = read_data_by_random()
train_dataset = MyDataset(x_train, y_train, data_process=True)
val_dataset = MyDataset(x_test, y_test, data_process=True)

batch_size = 128
train_loader = train_dataset.dataloader(batch_size=batch_size)
val_loader = val_dataset.dataloader(batch_size=batch_size)

model = Net()

# basic config
lr = 1e-4
patience = 15
epochs = 70

# 为不均衡样本设置权重
counts = np.bincount(train_dataset.y)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
class_weights_tensor = torch.Tensor(list(class_weights.values())).to(device)

# 设置L2正则化，减小过拟合
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

logger.info('Stage2: model training')
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device = device,
    exp=exp,
)

# start training
trainer.start_train(
    epochs=epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    patience=patience,
)

logger.info('Stage3: model save')
trainer.model_save()

logger.info('Stage4: save exp')
trainer.save_exp()

logger.info(f'{exp} best_score: {trainer.best_score:.4f}')
