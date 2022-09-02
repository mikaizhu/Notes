<!--ts-->
* [pathlib 介绍](#pathlib-介绍)
* [官方教程](#官方教程)
* [路径管理](#路径管理)
   * [导入模块](#导入模块)
   * [获取当前路径](#获取当前路径)
   * [获取home路径](#获取home路径)
   * [获取绝对路径](#获取绝对路径)
* [文件管理](#文件管理)
   * [查看当前路径文件](#查看当前路径文件)
   * [删除文件](#删除文件)
* [pathlib 与其他模块配合使用情况](#pathlib-与其他模块配合使用情况)
* [参考资料](#参考资料)

<!-- Added by: zwl, at: 2021年 7月21日 星期三 09时38分37秒 CST -->

<!--te-->
# pathlib 介绍
Pathlib 的优点是什么？
1. 老的路径操作函数管理比较混乱，有的是导入 os, 有的又是在 os.path 当中，而新的用法统一可以用 pathlib 管理。
2. 老用法在处理不同操作系统 win，mac 以及 linux 之间很吃力。换了操作系统常常要改代码，还经常需要进行一些额外操作。
3. 老用法主要是函数形式，返回的数据类型通常是字符串。但是路径和字符串并不等价，所以在使用 os 操作路径的时候常常还要引入其他类库协助操作。新用法是面向对象，处理起来更灵活方便。
4. pathlib 简化了很多操作，用起来更轻松。

Pathlib框架：

- 路径管理
- 文件管理

# 官方教程

https://docs.python.org/3/library/pathlib.html

并且网页最下方有os函数与pathlib函数对比表

# 路径管理

路径的基础知识：

1. .name 文件名，包含后缀名，如果是目录则获取目录名
2. .stem 文件名，不包含后缀
3. .suffix 后缀，比如 .txt, .png
4. .parent 父级目录，相当于 cd ..
5. .anchor 锚，目录前面的部分 C:\ 或者 /

## 导入模块

```
from pathlib import Path as p
```

## 获取当前路径

```
# 此时返回的是一个实例对象
path = p.cwd()

# 如果想获得字符串
str_path = str(p.cwd())
```

## 获取home路径

```
path = p.home()
```

## 获取绝对路径

```
path = p('.')
path.resolve()
```

# 文件管理

## 查看当前路径文件

第一种写法：

```
path = p('.')
list(path.glob('*')) # glob里面可以使用python的正则表达

list(path.glob('*.txt'))
```

第二种写法：


```
path = p('.')
list(path.iterdir())
```


## 删除文件

```
path = p('.')
for file in list(path.glob('*.pt')):
    file.unlink()
```

# pathlib 与其他模块配合使用情况

通常我们比如创建文件或者通过日志logging，都会通过其他模块的函数来生成创建文件
，现在检查这些接口的适配情况.

发现logging模块并不能很好适配pathlib，必须将路径转换成字符串

```
path = '.'
path = p(path)
logger = set_logger(str(p))
```

# 参考资料

https://zhuanlan.zhihu.com/p/139783331
