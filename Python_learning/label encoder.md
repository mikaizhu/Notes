# label encoder

在进行分类任务处理时，有时候标签并不是数值的，这时候就需要用到编码

## 问题说明

假设现在有一个三分类问题，标签如下

```
labels1 = ['a', 'b', 'c']
labels2 = [10, 20, 30]
```

- 对于第一个标签，由于算法只能处理数值的标签，所以要转换成0-2
- 对于第二个标签，虽然是数值类型的数据，但是范围超过了2，所以要映射到0-2

## 开始编码

**方法1:使用sklearn中的模块**

```
from sklearn.preprocessing import LabelEncoder

coder = LabelEncoder()
labels1 = coder.fit_transform(labels1)# 开始转换
coder.classes_ # 查看有多少类
labels1 = coder.invers_transform(labels1)# 逆向转换
```



**方法2：使用numpy或者pandas**

如果数据本身的接口是numpy或者pandas，那么可以使用内置的函数，可以避免导入其他模块

- numpy中使用下面方式

```
map_dict = {4:0, 9:1, 13:2, 14:3, 15:4, 16:5, 17:6, 22:7, 24:8, 25:9}
labels = np.vectorize(map_dict.get)(labels)
```

- pandas中使用map函数即可