import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

# 代码说明
# 本代码使用的深度学习框架是pytorch，所以要导入相关的torch模块


train_path = '../train'
val_path = '../val'
test_path = '../test'

# 读取训练集，测试集和验证集
train = np.load(train_path + '/' + '10type_sort_train_data_8192.npy')
test = np.load(test_path + '/' + '10type_sort_test_data_8192.npy')
val = np.load(val_path + '/' + '10type_sort_eval_data_8192.npy')

train_label = np.load(train_path + '/' + '10type_sort_train_label_8192.npy')
val_label = np.load(val_path + '/' + '10type_sort_eval_label_8192.npy')


# 这个函数主要的功能是对时间数据做fft，因为原始数据只有8192个点
# 对每个数据样本做8192个fft，然后取他的模值
# 然后可以画出fft的图像出来，通过画图可以发现，图像是关于x轴对称的
# 所以不论是取左边还是右边一半的点，效果都是一样的。
# 这里测试的6892 - 7192 300个点的效果最好
def get_fft_and_scaler(data, start=5192, end=8192):
    data = np.fft.fft(data)
    data = np.abs(data)
    data = data/np.expand_dims(data.max(axis=1), axis=1)
    return data[:,start:end]


# 下面就是定义模型的代码了
# 本次使用的是Transformer模型，transformer模型的框架可以参考网站：https://wmathor.com/index.php/archives/1438/
# 首先是位置编码，因为每个样本是一个300维度的fft点，因为transformer本来就是处理自然语言的问题，自然语言中输入的是一个文
# 本，比如i love you，这句话有三个单词，我们可以事先对每个单词转换成向量，如：i用一个128维度的向量表示，那么i love you
# 这句话就可以用一个3*128维度的向量表示。
# 同理，在这个数据中，我们可以将每个样本，都看成一个单词，如i，每个单词都是一个300维度的向量。
# 然后对这个向量进行位置编码操作
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=300, max_len=1):
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

# 这里feature size必须可以整除以nhead
# 位置编码完后，就可以输入到transformer模型中了，其中src len参数为1表示只有一个单词
# d model 表示，每个单词都是一个300维度的向量
# nhead是transformer模型中的一种结构，叫多头注意力机制，可以让模型学习的更好
class Transformer(nn.Module):
    def __init__(self, src_len=1, d_model=300, num_layers=4, nhead=3):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model).to(device)
        self.nhead = nhead
        self.d_model = d_model
        self.src_len = src_len
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 这里encoder输出的维度还是[batch size, max_len, embedding dim]
        # 所以embedding dim维度应该是 max len * embedding dim
        self.decoder = nn.Linear(self.d_model*self.src_len, 10)
        self.init_weights()

    def forward(self, x):
        #if self.src_mask is None:
        #    mask = self._generate_square_subsequent_mask(len(x)).to(device)
        #    self.src_mask = mask
        # 输入x加上位置编码后，得到encoder的输入
        x = self.pos_encoder(x)
        # 这里mask不是要免除padding=0后，softmax的影响吗
        # encoder mask讲解参考下面链接
        # https://medium.com/data-scientists-playground/transformer-encoder-mask%E7%AF%87-dc2c3abfe2e
        # 这里不需要为encoder添加mask，首先每个长度都是一样
        # 什么时候需要？参考：https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
        # 大致意思是说当我们要做seq2seq任务时，encoder只能接受前面位置的mask，要对后面位置的seq进行mask。
        # 当输入的batch size 长度不一样时候，也需要mask
        x = self.transformer_encoder(x)#, self.src_mask)
        x = self.decoder(x.permute(1, 0, 2).reshape(-1, self.d_model*self.src_len))
        return x

    # 将decoder 的偏置转换成0， 将权重进行标准化
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    # 将输入的序列进行mask操作
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 调用切片函数，并对样本进行归一化
train_sp = get_fft_and_scaler(train, 6892, 7192)
test_sp = get_fft_and_scaler(test, 6892, 7192)
val_sp = get_fft_and_scaler(val, 6892, 7192)

# 上面切片后数据是numpy格式,这里转换成torch框架的数据格式
train_tensor = torch.tensor(train_sp).float()
y_train_tensor = torch.tensor(train_label).long()
val_tensor = torch.tensor(val_sp).float()
y_val_tensor = torch.tensor(val_label).long()

# 使用Dataloader对数据进行封装, 因为不能直接把数据全部一次性使用完
# 主要是因为电脑的配置，所以这里要封装成dataloader，方便模型训练
batch_size = 128
train_tensor = TensorDataset(train_tensor, y_train_tensor)
val_tensor = TensorDataset(val_tensor, y_val_tensor)

train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_tensor, shuffle=False, batch_size=batch_size)

# 模型的参数控制
# num classes = 10 是因为这是一个10分类的模型
num_classes = 10
batch_size = 128
num_epochs = 11
# 学习率参数，调节这个可以让模型学习的更好
learning_rate = 0.0001
gamma = 0.9
step_size=1
# 选择是让模型在cpu上跑还是gpu上跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer().to(device)

# 定义损失函数和优化器
# 损失函数可以告诉模型预测值与真实值还差多远
# 优化器可以告诉模型要怎么修改参数，减小真实值和预测值的差距
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 学习方式
#Train the model
total_step = len(train_loader)
train_acc, val_acc = [], []

# 开始训练模型
for epoch in range(num_epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == labels).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # val the model
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean()
            epoch_val_accuracy += acc / len(val_loader)

    scheduler.step()
    print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')
    print(f'val acc:{epoch_val_accuracy:.4f}')

    train_acc.append(epoch_accuracy)
    val_acc.append(epoch_val_accuracy)

# 生成提交结果
def get_submmit(model, test, batch=128, dim=0):
    # 初始化结果提交向量
    # test为numpy类型的数据
    preds = np.zeros(test.shape[0])
    res = torch.FloatTensor(test)
    res = torch.chunk(res, chunks=batch, dim=dim)
    start = 0
    end = 0
    for chunk_data in tqdm(res):
        chunk_data = chunk_data.to(device)
        output = model(chunk_data).argmax(dim=1).detach().cpu().numpy()
        end += output.shape[0]
        preds[start:end] = output
        start = end

    return preds.astype(int)
ans = get_submmit(model, test_sp)
#ans = model(torch.FloatTensor(test_sp).to(device)).argmax(1).cpu().numpy()

pd.DataFrame({'Id':range(len(ans)), 'Category':ans}).to_csv('transformer_solution.csv', index=False)


