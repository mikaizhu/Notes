[TOC]
# 参考文档地址

详细请看[GitHub](https://github.com/Miki123-gif/interview_python)
# 函数的参数传递机制
看两个例子:
```
a = 1
def fun(a):
    a = 2
fun(a)
print a  # 1
```

```
a = []
def fun(a):
    a.append(1)
fun(a)
print a  # [1]
```

通过`id`来看引用`a`的内存地址可以比较理解：

```
a = 1
def fun(a):
    print "func_in",id(a)   # func_in 41322472
    a = 2
    print "re-point",id(a), id(2)   # re-point 41322448 41322448
print "func_out",id(a), id(1)  # func_out 41322472 41322472
fun(a)
print a  # 1
```
而第2个例子a引用保存的内存值就不会发生变化：

```
a = []
def fun(a):
    print "func_in",id(a)  # func_in 53629256
    a.append(1)
print "func_out",id(a)     # func_out 53629256
fun(a)
print a  # [1]
```

1. 函数内外都用的同一个标签a，先给1赋一个内存地址，让a标签指向这个内存
2. 传入到函数中的a中函数的a标签指向了1，当重新赋值时，就会指向2的地址
3. 函数内部调用完后，地址就会自动释放，a又指向了1的地址

**重点：**

**为什么1中传入数字到函数中，函数外并没改变a标签的值，而当传入的是一个列表，并向列表中添加数值时，函数外的a列表却发生变化？**

因为传入的对象不同，python中的对象有：
1. 可修改对象：包括字典dict，列表list，集合set
2. 不可修改对象：数字numbers，元组tuple，字符串string

**当我们给函数中传入不可修改对象时**，形参a会自动复制一个新的变量，和外面的a一点关系都没了
```
a = 1
def fun(a):
    # 形参a会自动复制一个新的变量，和外面的a一点关系都没了
    a = 2
    # 调用完后自动释放，所以外面引用不到里面的
fun(a)
print a  # 1
```
**当我们传入可修改的对象时**，python就会有个指针，一直指向这个可修改的对象，在函数里面，形参依旧可以修改该对象的值

**此问题的关键就是看传入函数的是可修改的对象还是不可修改的对象**

# 类方法中cls和self的区别
[参考自CSDN](https://blog.csdn.net/daijiguo/article/details/78499422)

- **cls是类本身，self是实例本身**

如果用了staticmethod，那么就可以无视这个self，将这个方法当成一个普通的函数使用。

```
>>> class A(object):
        def foo1(self):
            print "Hello",self
        @staticmethod
        def foo2():
            print "hello"
        @classmethod
        def foo3(cls):
            print "hello",cls


>>> a = A()

>>> a.foo1()          #最常见的调用方式，但与下面的方式相同
Hello <__main__.A object at 0x9f6abec>

>>> A.foo1(a)         #这里传入实例a，相当于普通方法的self
Hello <__main__.A object at 0x9f6abec>

>>> A.foo2()          #这里，由于静态方法没有参数，故可以不传东西
hello

>>> A.foo3()          #这里，由于是类方法，因此，它的第一个参数为类本身。
hello <class '__main__.A'>

>>> A                 #可以看到，直接输入A，与上面那种调用返回同样的信息。
<class '__main__.A'>
```

# 类变量和实例变量

**先说明下什么是类变量，什么是实例变量：**

注意类开头字母要大写，普通函数开头要小写，这是约定
```
class Test：
    number = 0
    def __init__(self):
        self.number = 0
```
- number就是类变量，self.number就是实例变量

**问题1：**
两者是如何使用的？

下例中，num_of_instance 就是类变量，用于跟踪存在着多少个Test 的实例。
```
class Test(object):  
    num_of_instance = 0  
    def __init__(self, name):  
        self.name = name  
        Test.num_of_instance += 1  
  
if __name__ == '__main__':  
    print Test.num_of_instance   # 0
    t1 = Test('jack')  
    print Test.num_of_instance   # 1
    t2 = Test('lucy')  
    print t1.name , t1.num_of_instance  # jack 2
    print t2.name , t2.num_of_instance  # lucy 2
```
**问题2：**

```
class Person:
    name="aaa"

p1=Person()
p2=Person()
p1.name="bbb"
print p1.name  # bbb
print p2.name  # aaa
print Person.name  # aaa
```

**一开始p1.name指向类变量中的name，当我们对实例p1.name修改时，p1.name就不再指向name类变量了，而变成了实例变量，所以我们再次输出类变量name的值时，发现并没有改变**


```
class Test:
    a = []
# 实例化
p1 = Test()
p2 = Test()
p1.a.append(1)
print(Test.a)
p2.a.append(2)
print(Test.a)
print(p1.a)

[1]
[1, 2]
[1, 2]
```
**重点在于看传入的是可修改的还是不可修改的，如果是不可修改的类对象，那就不会改变类变量的值，如果是可修改类对象，则会有个指针指向列表等对象**


# python 的自省

就是python能够获得对象的属性，如dir可以查看函数的方法，isinstance可以查看两个属性是否一样


```
a = list('123') # ['1', '2', '3']
b = {1:2,3:4}
isinstance(a, dict) # False
dir(isinstance) # ['__call__', '__class__', '__delattr__
```

# 字典推导式

语法：
```
d = {key: value for (key, value) in iterable}
```
例如：

```
dict1 = {1:2,3:4,'a':'b'}
print(dict1.items()) # dict_items([(1, 2), (3, 4), ('a', 'b')])

a = {key: value for key, value in dict1.items() if isinstance(key,str)}
print(a) # {'a': 'b'}
```

# 单下划线和双下划线

- 双下划线__init__(self)指的是魔法方法，类中会自己调用的一种方法，如果类中有__add__()方法，则说明，当我们使用+号时，就会调用这个类中的魔法方法
- self.__superprivate私有变量，外界不能直接访问该属性名，是python的匿名机制，不过换一种方式就可以访问，通过对象名._类名__xxx这样的方式可以访问.
- self._semiprivate，假如我们完成一个模块，当别人导入这个模块时，是无法查看这个属性的，因为根本就没法导入，不能用from module import * 导入
```
>>> class MyClass():
...     def __init__(self):
...             self.__superprivate = "Hello"
...             self._semiprivate = ", world!"
...
>>> mc = MyClass()
>>> print mc.__superprivate
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: myClass instance has no attribute '__superprivate'
>>> print mc._semiprivate
, world!
>>> print mc.__dict__
{'_MyClass__superprivate': 'Hello', '_semiprivate': ', world!'}
```

# 字符串格式化format

- 如果使用%输出想打印一个元组，则会抛出TypeErro错误，所以输出最好直接使用format

```
a = (1,2,3)
print(f'{a}')
print('a:%s'%a) # TypeError
```

# 迭代器和生成器

参考笔记：
https://blog.csdn.net/mieleizhi0522/article/details/82142856

yield相当于函数中的return，yield后面的都不会运行，并且会返回一个值，不同点在于yield配合next，可以实现中断功能


```
def myGener():
    for i in range(10):
        yield i
        print('-'*20)
        print('hello')

g = myGener()      
print(type(g))# <class 'generator'>  
print(next(g))# 0
print('hello')
print(next(g))# hello， 1
print(next(g))# hello， 2
```


- 当函数中有yield时，我们就生成了一个生成器对象，就像类一样，要先进行实例化,此时函数并不会全部直接运行，而是先生成一个迭代器，当我们需要调用时，才会进行运行，大大节省了内存
- yield相当于一个中断，遇到next会往下运行


```
def myGener():
    for i in range(10):
        yield i

myGener()        
print(next(myGener()))
print('hello')
print(next(myGener()))
print(next(myGener()))

# 这里输出全为0，并没有像上面那样，就是因为没有实例化
0
hello
0
0

```
**当列表推导式的`[]`变成`()`，生成的还是列表吗？**

此时生成的是生成器，不再是列表了，区别在于列表推导式会一次性将整个列表生成，当数据很大时，会消耗大量的内存，生成器只有当我们需要的时候，才会生成

当我们只需要少量的数据时，这里依旧可以使用next获取
**生成器貌似都能配合for循环和next生成数据**
```
g = (x for x in range(10))
for i in range(5):
    print(next(g))
    
# 输出
0
1
2
3
4
```
# *args and **kwargs

- 当定义一个函数，不知道要传入多少参数时，使用***arg**


```
def test(*name):
    print(type(name))
    print(name)
    
test(1,2,'a')


<class 'tuple'>
(1, 2, 'a')
```
- 如果想向函数中传入一些属性，则使用**arg，允许你使用没有事先定义的参数名

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
- 混合使用

**参数a会先获得值，剩下的传给arg**
```
def test(a,*arg):
    print(a)
    print(arg)

test('a',12,23)
```
**也可以这样混合使用**

***args和\*\*kwargs可以同时在函数的定义中,但是\*args必须在\*\*kwargs前面.**

```
def test(a,*arg,**name):
    print(a)
    print(arg)
    print(name)

test('a',12,23,title = 'zhu',d ='c')
# 输出
a
(12, 23)
{'title': 'zhu', 'd': 'c'}
```

# 修饰器
修饰器就是在不改变原来函数的接口上，给函数添加一些新的功能

**我们先来理解一个概念，请看下面代码：**

```
import time

def calculate():
    star = time.time()
    time.sleep(2)
    end = time.time()
    print(end - star)

another = calculate

another() # 2.0003886222839355
```
> 我们讲函数名calculate传给了another，然后再调用another函数，一样可以打印

**当我们删掉calculate函数时，another函数还能继续使用**
```
import time

def calculate():
    star = time.time()
    time.sleep(2)
    end = time.time()
    print(end - star)

another = calculate

print(id(another)) # 1656403252136

print(id(calculate)) # 1656403252136

del calculate # 删除原先的函数名
try:
    calculate()
except:
    print('func name not exist')

another()


func name not exist
2.0008578300476074
```

**现在，假设我们要给每一个函数，都额外添加一个计算函数运行时间的功能，使用修饰器来完成这个任务**

1. 写一个计算时间的修饰器
2. 给每个函数都装饰上这个修饰器

**假如现在有一个func函数，我们要计算这个函数的运行时间，一般做法如下：**
```
import time

def calculate():
    star = time.time()
    func()
    end = time.time()
    print(end - star)

def func():
    print('testing...')
    time.sleep(2)

calculate()
```
**但如果函数很多，每个都这样写，就太麻烦了**

额...利用刚才的知识，我想到一个方法，虽然也比较简单，但是致命的是里面利用了循环，这会大大增加运算的复杂度，而且**函数的调用接口变换了，我们希望再使用func1的时候，就打印时间出来**
```
import time

def calculate(*args):
    for each in args:
        star = time.time()
        each()
        end = time.time()
        print(end - star)

def func1():
    print('testing 1 ...')
    time.sleep(2)

def func2():
    print('testing 2 ...')
    time.sleep(2)
    
calculate(func1, func2)

testing 1 ...
2.0000758171081543
testing 2 ...
2.0015196800231934

```
**开始用修饰器，实现上面功能**

**初步实现，此时不改变接口**

1. 将要添加的功能，写成闭包的形式，添加函数func1
2. 重新给func1指向内部的inner函数，看下面代码
```
import time

def calculate(func): # 这里传入要计算时间的方程
    def inner(): # 写成闭包形式
        star = time.time()
        func() # func相当于是个标签，表示传入的函数
        end = time.time()
        print(end - star)
    return inner # 返回inner函数标签，因为外面想直接使用inner是不行的
    
def func1():
    print('testing 1 ...')
    time.sleep(2)

def func2():
    print('testing 2 ...')
    time.sleep(2)
    
print(id(func1))     # 1656403252856
func1 = calculate(func1) # 相当于func1 = inner，修改下标签的指向
# 这样外部函数func1，既可以使用自己原来的功能，又计算了时间
print(id(func1)) # 1656403254152
func1()

testing 1 ...
2.000030517578125
```

如果每次都要执行一次`func1 = calculate(func1)`，就有些复杂了，python提供了一个语法糖@，**进行下面修改即可**

```
import time

# 修饰器，一定要写在要使用该修饰器的函数前面
def calculate(func):
    def inner():
        star = time.time()
        func()
        end = time.time()
        print(end - star)
    return inner

@calculate   # 语法糖，相当于指令func1 = calculate(func1) 
def func1():
    print('testing 1 ...')
    time.sleep(2)
    
@calculate # 相当于指令func2 = calculate(func2)
def func2():
    print('testing 2 ...')
    time.sleep(2)
       
# func1 = calculate(func1)
func1()

testing 1 ...
2.000004768371582

```
# 鸭子类型

当一只鸟走起来像鸭子，叫起来也像鸭子，我们就认为这只鸟是鸭子

鸭子类型就是，我们不关注对象的类型，只关心对象的方法

**通过鸭子类型的思想进行设计：当我们想自定义个类方法，我们想让他像容器一样，可以存放东西，我们就必须定义一个方法，让这个类长得想一个容器，可以在类中定义__iter__魔法方法，只要有这个方法，他就叫起来像鸭子了**

**list的extend方法，不仅可以传入列表，而且可以传入字符串和元组等任何可以迭代的容器，因为容器中都有一个共同的属性，iterable，extend函数就是会调用这些容器中共同的方法**
```
a = [1,2,3]
b = [4,5,6]
c = (1,2)
d = '123'
a.extend(b)
print(a)
a.extend(c)
print(a)
a.extend(d)
print(a)

[1, 2, 3, 4, 5, 6]
[1, 2, 3, 4, 5, 6, 1, 2]
[1, 2, 3, 4, 5, 6, 1, 2, '1', '2', '3']

```

**再来看一个例子：**
**这里有三个需要注意的地方**


**1. 类中想调用自己类中的方法，则使用self.func()**

**2. 想直接调用类中的方法，则需要实例化再调用，直接Cat().say(),其中Cat()加括号就是实例化**

**3. python中标签可以指向任何对象，不管是类名还是变量名**

例子中类都有say方法，不管他是什么类对象，我只需要调用里面共有的say方法

```
class Dog:
    def say(self):
        print('喵~')
    # print('汪汪')
    # 如果我们想让狗变得像猫，可以定义它的叫声像猫，走路像猫
        
    
    def run(self):
        self.say()
        print('i am running')

class Cat:
    def say(self):
        print('喵~')
    
animal_list = [Dog, Cat]

for animal in animal_list:
    animal().say()
```

**总结：鸭子类型是编写类方法的核心思想，我们想让类方法具有容器功能，只需要让它具备容器的特征，需要让这个容器可迭代，则只需要让它具备可迭代的功能即可**


