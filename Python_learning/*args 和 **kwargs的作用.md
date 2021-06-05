## 先来回忆下*args和**kwargs的用法

- 当定义一个函数，不知道要传入多少参数的时候，使用*args

输入：

```
def test(*name):
    print(type(name))
    print(name)

test(1,2,'a')
```

输出：

```
<class 'tuple'>
(1, 2, 'a')
```

- 如果想向函数中传入一些属性，可以使用**kwargs

输入：

```
def tes(**name):
    print(type(name))
    print(name)
    
tes(fruit = 'apple',veg = 'cabbage')
```

输出：

```
<class 'dict'>
{'fruit': 'apple', 'veg': 'cabbage'}
```

- 当两者混合使用的时候，**会按顺序获得值**

**注意：args必须在kwargs的前面**

输入：

```
def t(a, b, *args, **kwargs):
    print(f'a:{a}')
    print(f'b:{b}')
    print(f'args{args}')
    print(f'kwargs{kwargs}')

t(1, 'xiao', 3, 4, 5, c = 2, d = 3)
```

输出：

```
a:1
b:xiao
args(3, 4, 5)
kwargs{'c': 2, 'd': 3}
```

**下面两个地方会报错，是为什么呢？**


```
import time

def cac(func):
    def inner(*args, **kwargs):
        time1 = time.time()
        print(
            f'args:{args}'
            
            # f'*args:{*args}' # 会报错
            
            f' kwargs:{kwargs}'

            # f'**kwargs:{**kwargs}' # 会报错
        )
        func(*args, **kwargs) # 是因为args是元组，kwargs是字典
        time2 = time.time()
        print(time2 - time1)
    return inner

@cac
def a(key, value, c=1, b=2):
    pass
```

```
是因为*和**只用在函数中，函数要接收元组或者字典时候用到
```