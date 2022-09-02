# stack , unstack

这里介绍下两个的用法

先介绍下两种表示方法

- 数据的树状结构
- 数据的表格结构

stack就是将表格数据转换成树状结构， unstack就是将树状结构数据，转换成表格的形式

```
d = {
    'item':['item0', 'item1'],
    'ctype':['gold', 'gold'],
    'usd':[1, 3],
    'eu':[1, 3],
}
df = pd.DataFrame(d)
```

```
	item	ctype	usd	eu
ix				
0	item0	gold	1	1
1	item1	gold	3	3
```

stack之后：

```
df.stack(level=-1, dropna=True) # 这两个参数默认就是这样的
```

```
ix       
0   item     item0
    ctype     gold
    usd          1
    eu           1
1   item     item1
    ctype     gold
    usd          3
    eu           3
dtype: object
```

```
ix       
0   item     item0
    ctype     gold
    usd          1
    eu           1
1   item     item1
    ctype     gold
    usd          3
    eu           3
dtype: object
```

**lever=-1表示第一行，也就是item， ctype这些特征**

- unstack就是将数据变回来

```
stack.unstack()
```

```
item	ctype	usd	eu
ix				
0	item0	gold	1	1
1	item1	gold	3	3
```



**当有多层索引的时候，可以指定lever**

请参考：https://blog.csdn.net/wsp_1138886114/article/details/80560351

![image.png](http://ww1.sinaimg.cn/large/005KJzqrly1gl99r48ujjj31a20moqi4.jpg)