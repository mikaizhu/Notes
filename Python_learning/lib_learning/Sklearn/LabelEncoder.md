通常在处理分类任务的时候，比如判断数据类别是猫还是狗。我们通常要映射为0-1

如果处理的数据类别本来就是数字，但是原来的类别是22，24，25。这实际上是3分类问题。所以要转换成0-2。

**在处理上述场景的问题的时候，我们可以直接使用sklearn中的LabelEncoder函数**

```
from sklearn.preprocessing import LabelEncoder
```

使用方法：

```
coder = LabelEncoder()
y = coder.fit_transform(y)
```

**其中y数据格式是列表或者numpy数据**

**注意：既然有进行编码，那么肯定有对标签进行解码，这些在函数中都是有的，百度即可**

