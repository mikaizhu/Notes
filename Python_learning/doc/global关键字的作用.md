# global 关键字有什么用呢？

先来看看下面代码

```
n = 1
def func():
	n = n + 1
```

报错如下：

```
UnboundLocalError： local variable 'xxx' referenced before assignment
```

这是因为在函数内部对变量赋值进行修改后，该变量就会被Python解释器认为是局部变量而非全局变量，当程序执行到n+=1的时候，因为这条语句是给n赋值，所以n成为了局部变量，那么在执行print n的时候，因为n这个局部变量还没有定义，自然就会抛出这样的错误。

所以做如下修改即可：

```
n = 1
def func():
	global n
	n = n + 1
```

实际场景：

因为我要调节优化器的参数。所以出现了上面的情况

```
lr = 0.01
optimizer = optim.Adam(network.parameters(), lr=lr)

def epoch_train(data_loader):
    global optimizer
    global lr
    for epoch in range(EPOCH):
        total_loss = 0
        total_acc_num = 0
        time1 = time.time()
        for feature1, label1 in data_loader:
            feature1 = feature1.to(device)
            label1 = label1.to(device)

            preds = network(feature1)

            loss = F.mse_loss(preds, label1)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss
            total_acc_num += get_correct_num(preds, label1)
        time2 = time.time()
        get_epoch_time(epoch, n=20, time1=time1, time2=time2)
        print(f'epoch:{epoch+1}, acc:{total_acc_num/(data_size*1024):.5f}, lr:{lr:.8f}, loss:{total_loss:.2f}')
        optimizer, lr = optimizer_lr_modify(optimizer, epoch, lr, rate=0.9, n=40)
```

