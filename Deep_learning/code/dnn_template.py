#!/home/zwl/miniconda3/envs/asr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class MyDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.data_process()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        feature = self.x[idx]
        label = self.y[idx]
        return [feature, label]

    def data_process(self):
        self.x = np.fft.fft(self.x)
        self.x = np.abs(self.x)[:, 6892:7192]
        self.x /= np.expand_dims(np.max(self.x, axis=1), axis=1)

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
        #x = torch.abs(x)
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

train_path = './train/10type_sort_train_data_8192.npy'
val_path = './val/10type_sort_eval_data_8192.npy'
train_label_path = './train/10type_sort_train_label_8192.npy'
val_label_path = './val/10type_sort_eval_label_8192.npy'
print('Stage1: load and process data')
train_dataset = MyDataset(train_path, train_label_path)
val_dataset = MyDataset(val_path, val_label_path)

batch_size = 128
train_loader = train_dataset.dataloader(batch_size=batch_size)
val_loader = val_dataset.dataloader(batch_size=batch_size)

# below func is to debug collate_fn
# next(iter(train_loader))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(1)
        )
        #self.init_weight()

    def forward(self, x):
        return self.dnn(x)

    def init_weight(self):
        for name, layer in self.dnn.named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))


class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 device,):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model = model.to(self.device)
        self.best_model = None

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
            total_loss += loss
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
                total_loss += loss

                feature.cpu()
                label.cpu()
        return total_loss, acc_num

    def start_train(self, epochs, train_loader, val_loader, patience=10,):
        best_val_loss = np.inf
        best_val_acc = -np.inf
        for epoch in range(epochs):
            train_loss, train_acc_num = self.train_step(dataloader=train_loader)
            val_loss, val_acc_num = self.eval_step(dataloader=val_loader)

            # scheduler step
            self.scheduler.step(val_loss)

            # cal acc
            train_acc = train_acc_num / len(train_loader.dataset)
            val_acc = val_acc_num / len(val_loader.dataset)

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
                print('Stopping early!')
                break
            # logging
            print(
                f'Epoch: {epoch:^3} | '
                f'train loss: {train_loss:^6.4} | '
                f'train acc: {train_acc:^6.4} | '
                f'val loss: {val_loss:^6.4} | '
                f'val acc: {val_acc:^6.4}'
            )
            # best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_model = self.model

    def cal_acc(self, predict, label):
        return (predict.argmax(1) == label).sum().item()

    def predict(self):
        pass

    def model_save(self, file_name):
        torch.save(self.best_model.state_dict(), file_name)

model = Net()

# basic config
lr = 1e-4
patience = 8
epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
counts = np.bincount(train_dataset.y)

# 为不均衡样本设置权重
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
class_weights_tensor = torch.Tensor(list(class_weights.values())).to(device)

# 设置L2正则化，减小过拟合
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

print('Stage2: model training')
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device = device
)

# start training
trainer.start_train(
    epochs=epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    patience=patience,
)

print('Stage3: model save')
trainer.model_save('dnn.point')
