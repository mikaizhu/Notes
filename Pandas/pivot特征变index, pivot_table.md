

# pivot

首先来创建数据

```
d = {
    'item':['item0', 'item1'],
    'ctype':['gold', 'gold'],
    'usd':[1, 3],
    'eu':[1, 3],
}
df = pd.DataFrame(d)

df.loc[2] = ['item0', 'Gold', 3, 3]
df.loc[3] = ['item1', 'Silver', 4, 4]
```

```
# 结果如下
	item	ctype	usd	eu
ix				
0	item0	gold	1	1
1	item1	gold	3	3
2	item0	Gold	3	3
3	item1	Silver	4	4
```

**在这个数据中，默认的是index从0-3，如果我们想将某列特征变成index，某列特征变成columns，进行观察，就可以使用pivot**

```
df.pivot(index='item', columns='ctype')
```

**结果如下：**

```
						usd										eu
ctype	Gold	Silver	gold	Gold	Silver	gold
item						
item0	3.0	NaN	1.0	3.0	NaN	1.0
item1	NaN	4.0	3.0	NaN	4.0	3.0
```

**咋一看，有点不知道这个表是什么意思，index是item，columns是ctype，那表格里面的数值是啥？**

**将代码补全，上面代码没写完整**

```
df.pivot(index='item', columns='ctype', values='usd')
```

**结果如下：**

```
ctype	Gold	Silver	gold
item			
item0	3.0	NaN	1.0
item1	NaN	4.0	3.0
```

**可以发现，value就是，取usd中，原先item=item0， Gold中的那个值**

# pivot_table

和上面一样，就是增加了自定义函数

```
pivot_table(index=None, column=None, values=None, aggfunc='mean')
```

我们先看看一个新的df

```
item	ctype	usd	eu
ix				
0	item0	gold	1	1
1	item1	gold	3	3
2	item0	Silver	4	4
3	item1	Silver	4	4
4	item0	gold	4	4
```

```
df.pivot(index='item', columns='ctype', values='usd')
```

**出现错误！！！！**

```
ValueError: Index contains duplicate entries, cannot reshape
```

**因为我指定的是新index是item， 新列是ctype。如果使用pivot，则会出现重复的，**

所以新的value，是1还是4呢？

```
0	item0	gold	1	1
4	item0	gold	4	4
```

**这时候就要使用pivot_table, 上面两个取平均，就是2.5**

```
ctype	Silver	gold
item		
item0	4.0	2.5
item1	4.0	3.0
```

**所以如果指定的列出现重复，那么就需要使用pivot_table**