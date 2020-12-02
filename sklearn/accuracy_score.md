计算函数准确率，不需要自己编写代码

```
from sklearn.metrics import accuracy_score
```

使用方式：

```
acc = accuracy_score(preds, labels)
```

**preds和labels都要是列表的形式，或者是可迭代对象**

