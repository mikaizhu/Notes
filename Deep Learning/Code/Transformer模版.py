#这里使用Fashion minist数据集，使用transformer进行特征提取
#使用linear层进行下游任务分类
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import math
from torchvision import datasets, transforms

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=28, max_len=28):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    # 这里feature size必须可以整除以nhead
    def __init__(self, feature_size=28, num_layers=3, nhead=7):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size).to(device)
        slf.nhead = nhead
        self.feature_size = feature_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size**2, 10)
        self.init_weight()

    def forward(self, x):
        if self.src_mask is None:
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = self.decoder(x.permute(1, 0, 2).reshape(-1, self.feature_size**2))
        return x

   # 将decoder 的偏置转换成0， 将权重进行标准化
    def init_weight(self):
        self.decoder.bias.data.zero_()

   # 将输入的序列进行mask操作
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
# 数据下载
# 对图片数据进行转换，因为Fashion minist是灰度图，所以使用下面的normalization方式
transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


# 数据下载与封装
train_set = datasets.FashionMNIST('./', download = True, train = True, transform = transform)
test_set = datasets.FashionMNIST('./', download = True, train = False, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer().to(device)

batch_size = 100
num_epochs = 30
learning_rate = 0.001
gamma = 0.9
step_size=1

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 学习方式
# Train the model
total_step = len(train_loader)
train_acc, test_acc = [], []
# seq_len 相当于 max len， 即每句话切分的最大长度
sequence_length = 28
# input_size 相当于embedding dim
input_size = 28
for epoch in range(num_epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # reshape的维度变成了(batch size, 28, 28)
        images = images.reshape(-1, sequence_length, input_size)
        images = images.permute(1, 0, 2).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # Test the model
    model.eval()
    with torch.no_grad():
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        for images, labels in tqdm(test_loader):
            images = images.reshape(-1, sequence_length, input_size)
            images = images.permute(1, 0, 2).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean()
            epoch_test_accuracy += acc / len(test_loader)

    scheduler.step()
    print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')
    print(f'test acc:{epoch_test_accuracy:.4f}')

    train_acc.append(epoch_accuracy)
    test_acc.append(epoch_test_accuracy)
