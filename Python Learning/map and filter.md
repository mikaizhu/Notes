# 这里介绍两个函数

- map
- filter

# map的基本用法

```
map(function, iters)
```

- 第一个位置传入的是函数，第二个位置传入的是可迭代对象

**函数功能：**

将可迭代对象的每个元素传入function中，执行完后生成一个新的可迭代对象

```
l = [1, 2, 3, 4]
list(map(lambda x:x*2, l))

# [2, 4, 6, 8]
```

# filter基本用法

```
filter(function, iters)
```

- 第一个位置传入的是函数，第二个位置传入的是可迭代对象

**函数功能：**

将可迭代对象的每个元素传入function中，执行完后生成一个新的可迭代对象

```
l = [1, 2, 3, 4]
list(map(lambda x:x != 2), l)

# [1, 3, 4]
```

