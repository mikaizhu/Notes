## pytorch中的resnet使用

在pytorch中可以直接使用resnet50，18等网络

```
from torchvision import models

model = models.resnet50()
```

## 数据处理

```
torch.stack([
    torch.tensor(real[0]),
    torch.tensor(imag[0]),
    torch.tensor(angel[0]),
]).shape

# torch.Size([3, 256, 257])
```

因为这里是手机信号，经过stft变换后，每个一维信号会变成一个二维复数矩阵。可以取出信号的：实部，虚部，角度三个维度，作为三个channel。然后拼接成RGB类似的3维tensor。

```
from torchvision import models
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),# 因为原先数据是矩阵，所以要先转换成图片
    transforms.CenterCrop(256),
    transforms.Resize(256),# 对图片进行裁剪，resize
    transforms.ToTensor(),# 然后将图片转换成张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

因为数据原本就不是图片，所以这里要先变化成图片。使用transform变换即可

```
transforms.ToPILImage()
```

然后将图片裁剪为256尺寸

```
transforms.Resize(256)
```

由于最后还是只能处理数据，所以又要将图片转换成tensor

```
transforms.ToTensor()
```

官网说图片必须进行如下的标准化

```
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**然后是构造自己的数据类别，相当于Dataset**

```
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self, idx):
        img = self.file_list[idx]
        img_transformed = self.transform(img) # 因为转换后嵌套是元组，所以这里取内部
        
        label = self.label_list[idx]
        
        return img_transformed, label

x_train, x_test, y_train, y_test = train_test_split(image, image_labels, test_size=0.3, shuffle=True)        
train_data = MyDataset(x_train, y_train, transform=transform)
test_data = MyDataset(x_test, y_test, transform=transform)

#Creating data loaders for test and training sets
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, 
shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
```

- 这里需要重写数据集部分
- 然后放入到data loader中

处理好后，原来是3\*256\*257的图片就变成256尺寸了，接下来就可以使用残差网络了

## 使用网络进行训练

```
from torchvision import models

model = models.resnet50()
```

```
# 参数设置
batch_size = 64
epochs = 20
lr = 0.001
gamma = 0.7
seed = 42
```

**因为是图像分类问题，所以这里要对神经网络最后一层进行微调**

```
numFit = model.fc.in_features
model.fc = nn.Linear(numFit, 13)
```

```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 学习方式
```

**最后就可以开始进行训练了**

```
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data.cuda()
        label = label.cuda()
        output = model(data)
    
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        
        data.cpu()
        label.cpu()
        
    with torch.no_grad():
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        for data, label in tqdm(test_loader):
            data = data.cuda()
            label = label.cuda()
            
            test_output = model(data)
            test_loss = criterion(test_output, label)
            
            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)
            
    print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')
    print(f'test loss:{epoch_test_loss:.4f}, test acc:{epoch_test_accuracy:.4f}')
```

