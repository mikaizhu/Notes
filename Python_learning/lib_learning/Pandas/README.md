**Pandas 参考教程**

http://joyfulpandas.datawhale.club/Content/ch2.html

# 判断某一元素是否在某一列中

使用isin方法, 最后会返回一个DF

```
import pandas as pd
 
df1 = pd.DataFrame([[1, 1000, 23241], [1111, 2, 4], [5, 23, 25]], columns=['a', 'b', 'c'])
list1 = [1, 5]
df2 = df1[df1['a'].isin(list1)]
df3 = df1[~df1['a'].isin(list1)]
print(df2)
print(df3)
```

