<!--ts-->
* [参考资料推荐](#参考资料推荐)
* [python 日志模块使用](#python-日志模块使用)
* [python 回调函数(callback)](#python-回调函数callback)

<!-- Added by: zwl, at: 2021年 7月 4日 星期日 15时23分35秒 CST -->

<!--te-->
# 参考资料推荐

- https://pyzh.readthedocs.io/en/latest/
- python魔法方法指南：https://pyzh.readthedocs.io/en/latest/python-magic-methods-guide.html
- https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/02_Python.ipynb
- [知识点有例子，适合学习](https://zhuanlan.zhihu.com/p/137302250) 

# python debug

如果是使用nvim来进行脚本编写，使用ipdb模块进行debug

# python 日志模块使用

如果想和Linux脚本使用终端输出，使用print不行，只能用logger文件，如果想简单使用
logger，使用下面代码即可

```
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s', level=logging.INFO)
```

or

```
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
```

然后将print修改成logger.info:

```
:%s/print/logger.info
```

# python 回调函数(callback)

装饰器，回调函数的作用具体可以参考：https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/02_Python.ipynb

装饰器允许我们在函数开始前和开始后执行一些功能，而不用修改原先的函数，那么回调
函数就允许我们在调用函数的时候增加一些功能.

将装饰器和回调函数结合起来，就可以组成强大的脚本函数

什么是回调函数？

> 你到一个商店买东西，刚好你要的东西没有货，于是你在店员那里留下了你的电话，过了几天店里有货了，店员就打了你的电话，然后你接到电话后就到店里去取了货。在这个例子里，你的电话号码就叫回调函数，你把电话留给店员就叫登记回调函数，店里后来有货了叫做触发了回调关联的事件，店员给你打电话叫做调用回调函数，你到店里去取货叫做响应回调事件。

> 如果你把函数的指针（地址）作为一个参数传递给另外一个函数，当这个指针被用来调用其所指向的函数时，我们就说这是一个回调函数。**回调函数不是由该函数的实现方法直接调用，而是在特定的事件或条件发生时由另外的一方调用的，用于对该事件或条件进行响应。**

回调函数的例子：https://zhuanlan.zhihu.com/p/137302250

```
# 回调函数calculate
def calculate(x,y,func):
    return func(x,y)

# 需要定制的功能
def max(x,y):
    while x>y:
        print(x)
    else:
        print(y)

def sum(x,y):
    print(x+y)

if __name__ = "_main_":
    # 需要用到的时候唤醒
    result_max = calculate(5,7,max)
    print(result_max)
    result_sum = calculate(5,7,sum)
    print(result_sum)
```

# call方法

- 参考：https://blog.csdn.net/Yaokai_AssultMaster/article/details/70256621

init就是初始化一个实例对象，call方法就是可以直接调用这个实例对象

```
class X(object):
	def __init__(self, a, b, range):
		self.a = a
		self.b = b
		self.range = range
	def __call__(self, a, b):
		self.a = a
		self.b = b
		print('__call__ with （{}, {}）'.format(self.a, self.b))
	def __del__(self, a, b, range):
		del self.a
		del self.b
		del self.range

>>> xInstance = X(1, 2, 3)
>>> xInstance(1,2)
__call__ with (1, 2)
```

# class 方法中return self有什么作用？

有时候我们会发现有些类方法中会返回self，这样的好处是什么呢？

```
class person:
    def __init__(self, name):
        self.step = 0
        self.name = name
    
    def set_name(self, name):
        self.name = name
    
    def move(self, step):
        self.step += step
        print(f'I have moved {self.step} steps')
        return self
```

```
man = person('Dav')
man.move(3).move(3) # return self方便我们这样连续.调用内部函数
```

好处就是允许我们链式调用

# stdin stdout


