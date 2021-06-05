# 介绍下itertools模块的用法

主要介绍itertools的排列组合：

**里面主要包含两个模块**

- product
- combinations

## product

```
from itertools import product
from itertools import combinations
```

```
l1 = [1,2,3]
l2 = [3,4]
l3 = [3,4]
for i in product(l1, l2):
  ...:     print(i)
```

```
(1, 3)
(1, 4)
(2, 3)
(2, 4)
(3, 3)
(3, 4)
```

还有三个数的排列组合：

```
for i in product(l1, l2, l3):
   ...:     print(i)
```

```
(1, 3, 3)
(1, 3, 4)
(1, 4, 3)
(1, 4, 4)
(2, 3, 3)
(2, 3, 4)
(2, 4, 3)
(2, 4, 4)
(3, 3, 3)
(3, 3, 4)
(3, 4, 3)
(3, 4, 4)
```

**注意，product是不带重复的**

## combinations

```
l1 = [3,4,5,1]
for i in combinations(l1, 2):
   ...:     print(i)
```

```
(3, 4)
(3, 5)
(3, 1)
(4, 5)
(4, 1)
(5, 1)
```

用法就是：第一个参数传入可迭代对象，第二个参数传入数字nums。

然后生成长度为nums的所有组合情况

## combinations_with_replacement

和combinations一样，但是带重复，就是指，自己可以和自己组合

```
from itertools import combinations_with_replacement
for i in combinations_with_replacement(l1, 2):
   ...:     print(i)
```

```
(3, 3)
(3, 4)
(3, 5)
(3, 1)
(4, 4)
(4, 5)
(4, 1)
(5, 5)
(5, 1)
(1, 1)
```

