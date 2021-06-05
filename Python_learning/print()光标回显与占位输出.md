# print光标回显

通常显示会一行一行显示，使用print的光标回显

```
print(i, end='\r')
```

例子：

```
import time

for i in range(10):
    time.sleep(1)
    print(f'time:{i}/10', end='\r')
```

# print的占位输出

```
a = 'abc'
d = 'ab'
print(f'{a:4}|{d:4}')
```

这样每个输出都会占4位，好处是方便对齐