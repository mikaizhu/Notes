**这里介绍下cross entropy的使用方法**

cross entropy一般是用在分类算法中，但之前使用一直在报错


```
func_loss = nn.CrossEntropyLoss()
loss = func_loss(preds, labels.view(-1).long())
```

使用说明：

preds:应该是概率的形式
```
[
[0.2, 0.3, 0.5],
[0.3, 0.2, 0.5],
]
```
其中preds的shape为2\*3，2表示有多少个样本，3表示多少个类别。

只要用概率分布的形式输入进去即可。然后标签就是真实的类别。
