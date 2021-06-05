假如现在我构造了一个图片数据，维度分别为 $(B,C,H,W)$ ,实际在读取数据的时候，并不是图片，而是一个张量矩阵。

## 问题1:

假如现在图像矩阵是浮点型矩阵，尺寸为$(32,3,260,260)$, 现在要将图片裁剪为256\*256的样式

```
transform = transforms.Compose([
    transforms.ToPILImage(),# 因为原先数据是矩阵，所以要先转换成图片
    transforms.CenterCrop(256),
    transforms.Resize(256),# 对图片进行裁剪，resize
    transforms.ToTensor(),# 然后将图片转换成张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = MyDataset(x_train, y_train, transform=transform)
test_data = MyDataset(x_test, y_test, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 使用代码后报错如下
```

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goy8cd7e9oj318406y0tr.jpg)

对代码进行修改，因为数据是numpy矩阵，所以可以先转换成tensor的形式

```
transform = transforms.Compose([
    transforms.ToTensor(),# 然后将图片转换成张量, 加上这个代码即可
    transforms.ToPILImage(),# 因为原先数据是矩阵，所以要先转换成图片
    transforms.CenterCrop(256),
    transforms.Resize(256),# 对图片进行裁剪，resize
    transforms.ToTensor(),# 然后将图片转换成张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = MyDataset(x_train, y_train, transform=transform)
test_data = MyDataset(x_test, y_test, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
```

## 问题2

一般算法的输入图片尺寸为224或者256，如果使用其他尺寸的图片，则会报错。

假如现在图片尺寸为260

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goy9g10j0qj31i408k408.jpg)