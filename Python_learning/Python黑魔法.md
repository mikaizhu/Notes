[TOC]



# argument和parameter的区别

argument是自变量的意思，也就是实参

parameter是形参

# /usr/bin/env python有什么用？

很多程序脚本开始都会加入

```
#!/usr/bin/python
```

通常我们要运行某个python代码

```
python3 code.py
```

当我们在上面添加这个代码的时候，意思就是告诉我们用哪个python来执行这个文件

不加这个代码头的话，每次执行这个脚本都要 

```
python xx.py
```

加了以后，可以直接执行

```
xx.py
```

怎么操作呢？

```
# 这两行代码就好了
chmod +x x.py
./x.py

-rwxr-xr-x  1 mikizhu  staff   27 10 22 16:27 x.py
```

我们要给这个文件可执行的权限，正常创建的文件是没有x这个权限的

```
-rw-r--r--  1 mikizhu  staff   11 10 22 16:29 xx.py
```

# dict()和{}

两种初始化字典的方式

使用{}比dict要更快



