

[TOC]

# python argparse模块介绍

## 实现sys的命令行读取参数的功能

**首先是实现类似sys模块的功能：**

```
import argparse

parser = argparse.ArgumentParser() # 首先实例化
parser.add_argument('x', help='axis x', type=int) # 添加参数
parser.add_argument('y', help='axis y', type=int) # 这样可以将参数解析为int类型
args = parser.parse_args() # 解析参数
print(args.x, args.y) # 调用参数
```

设置type为字符串

**然后在命令行输入：**

```
python3 test.py 3 4
```

**想查看各个参数的意思：**

```
python3 test.py -h
```

**然后会输出：**

```
usage: test2.py [-h] x y

positional arguments:
  x           axis x
  y           axis y

optional arguments:
  -h, --help  show this help message and exit
```

## 实现可以指定参数的功能

但上面还是有时候容易搞混，我希望实现下面这样的功能：

```
python3 test.py -x 3 -y 4
python3 test.py -y 4 -x 3
```

这样交换位置也可以

**只要在前面加个`-`就好了**

```
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', help='axis x', type=int)
parser.add_argument('-y', help='axis y', type=int)
args = parser.parse_args()
print(args.x, args.y)
```

```
(base) mikizhu@MikiZhudeMacBook-Pro python % python3 test2.py -x 3 -y 4
3 4
(base) mikizhu@MikiZhudeMacBook-Pro python % python3 test2.py -y 4 -x 3
3 4
```

**但感觉这样看参数，还是不详细：**

```
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--x', help='axis x', type=int)
parser.add_argument('--y', help='axis y', type=int)
args = parser.parse_args()
print(args.x, args.y)
```

注意，如果前面是双杠，那么解析的时候也要用双杠，不然会报错：

```
(base) mikizhu@MikiZhudeMacBook-Pro python % python3 test2.py -y 4 -x 3  
usage: test2.py [-h] [--x X] [--y Y]
test2.py: error: unrecognized arguments: -y 4 -x 3
```

**我们来看看这个效果怎么样， 这样表达效果确实很明显，自己也看得懂：**

**注意一定要加--**

```
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--axis_x', help='axis x', type=int)
parser.add_argument('-y', '--axis_y', help='axis y', type=int)
args = parser.parse_args()
print(args.x, args.y)
```

当我在命令行输入：

```
python3 test.py -x 3 -y 4
```

结果报错了：

```
(base) mikizhu@MikiZhudeMacBook-Pro python % python3 test2.py -y 4 -x 3
Traceback (most recent call last):
  File "test2.py", line 7, in <module>
    print(args.x, args.y)
AttributeError: 'Namespace' object has no attribute 'x'
```

**当我将原函数改成：**

```
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--axis_x', help='axis x', type=int)
parser.add_argument('-y', '--axis_y', help='axis y', type=int)
args = parser.parse_args()
print(args.axis_x, args.axis_y) # 这时候一定要使用双杠的名字
```

然后就可以正常解析了：

```
python3 test.py -x 3 -y 4
```

```
(base) mikizhu@MikiZhudeMacBook-Pro python % python3 test2.py -y 4 -x 3
3 4
(base) mikizhu@MikiZhudeMacBook-Pro python % python3 test2.py --axis_x 3 --axis_y 4
3 4
```

**所以：单杠一般用在双杠名字的简写，就像--help命令，也可以使用-h**

## 参考链接

官网 https://docs.python.org/3/howto/argparse.htmlhttps://docs.python.org/3/howto/argparse.html
