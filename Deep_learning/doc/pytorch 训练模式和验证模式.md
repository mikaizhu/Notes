# pytorch 提供了两种模式

**train 模式 和 eval模式**

- 使用方法：之前定义好了模型，不用在里面定义train方法，也不用在里面自己定义eval方法。
- 直接使用继承的即可

```
for epoch in range(EPOCH):
    model.train()
    for feature, label in train_loader:
        feature = feature.float().cuda()
        label = label.cuda()
        optimizer.zero_grad()
        preds = model(feature)
        loss = lf(preds, label)
        loss.backward()
        optimizer.step()
    print(f'epoch:{epoch:2} loss:{loss:4f}')

    model.eval()
    with torch.no_grad():
        total_acc = 0
        total_num = 0
        for feature, label in test_loader:
            feature = feature.float().cuda()
            label = label.cuda()
            total_acc += model(feature).argmax(dim=1).eq(label).sum().item()
            total_num += label.shape[0]
        print('val_acc:{:4f}'.format(total_acc/total_num))
```

两个模式的区别，主要针对dropout和batchnorm两个方法

- train模式dropout会随机丢弃元素，eval模式，会让所有神经元都可以通过数据
- train模式下batch norm会自动计算均值和方差，eval模式下会使用训练时候的均值和方差

