# 计算准确度

当然，我们可以自己写个for循环来计算准确度，也可以使用sklearn中集成的函数来计算准确度



```
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_preds, normalize=True)
```

