在遇到很多图像神经网络问题中，我们遇到的图片尺寸通常是$(3, 256, 256)$ 或者 $(3, 224, 224)$ 这样尺寸的。

1. 当我们想自定义输入神经网络的图片尺寸该怎么处理呢？

参考：https://discuss.pytorch.org/t/transfer-learning-usage-with-different-input-size/20744

**总的来说就是将池化层转换成自适应池化层**

2. 什么是自适应池化层？

参考：https://www.zhihu.com/question/282046628/answer/767681310

在实际应用中，我们往往只知道输入数据和输出数据的大小，而不知道卷积核大小以及卷积步长大小，自适应就是自己会学习。

**如何自适应？**

将中间的maxpooling改成adaptivemaxpooling就好了：

```
for name, layer in model.named_modules():
    if isinstance(layer, nn.MaxPool2d):
        model.maxpool = nn.AdaptiveAvgPool2d((7, 7))
```

修改最后一层的分类：

```
n_class = 10
numFit = model.fc.in_features
model.fc = nn.Linear(numFit, n_class) # 直接修改最后一层
```

