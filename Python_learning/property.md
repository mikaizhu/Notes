# property的使用

先来看看代码：

```
class Employee:
    def __init__(self, first, last):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'

    def fullname(self):
        return f'{self.first} {self.last}'


emp1 = Employee('John', 'Smith')
print(emp1.first)
print(emp1.email)
print(emp1.fullname())
```

```
John
John.Smith@email.com
John Smith
```

当我们作出如下的改变的时候：

```
emp1 = Employee('John', 'Smith')
emp1.first = 'Jim'

print(emp1.first)
print(emp1.email)
print(emp1.fullname())
```

输出：

```
Jim
John.Smith@email.com
Jim Smith
```

我们改变了firstname，但是email的firstname并没有发生改变。但我们希望这些参数同时变化怎么办

**可以使用property装饰器，property装饰器可以让我们更好的管理类的属性**

因为上面我们改变firstname的时候，我们希望email也改变

可以修改代码如下：

```
class Employee:
    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def email(self): # 将email封装成函数，并使用描述符property
        return f'{self.first}.{self.last}@email.com'

    def fullname(self):
        return f'{self.first} {self.last}'


emp1 = Employee('John', 'Smith')
emp1.first = 'Jim'

print(emp1.first)
print(emp1.email) # 这样我们就可以通过属性来访问这个函数
print(emp1.fullname())
```

**如果我们想改变fullname怎么办？**

```
emp1 = Employee('John', 'Smith')
emp1.first = 'Jim'

emp1.fullname = 'Corey Schafer'


print(emp1.first)
print(emp1.email)
print(emp1.fullname)
```

这样直接运行会报错

**因为描述符实现了del， set，get三个魔法方法**

```
    @property
    def fullname(self):
        return f'{self.first} {self.last}'

    @fullname.setter
    def fullname(self, name):
        first, last = name.split()
        self.first = first
        self.last = last        
```

```
emp1 = Employee('John', 'Smith')
emp1.first = 'Jim'

emp1.fullname = 'Corey Schafer'
```

输出，这样就能正常执行了：

```
Corey
Corey.Schafer@email.com
Corey Schafer
```

**同理，还可以实现del**

```
    @property
    def fullname(self):
        return f'{self.first} {self.last}'

    @fullname.setter
    def fullname(self, name):
        first, last = name.split()
        self.first = first
        self.last = last

    @fullname.deleter
    def fullname(self):
        print(f'del name')
        self.first = None
        self.last = None
```

```
emp1 = Employee('John', 'Smith')
emp1.first = 'Jim'

emp1.fullname = 'Corey Schafer'

del emp1.fullname


print(emp1.first)
print(emp1.email)
print(emp1.fullname)
```

输出：

```
del name
None
None.None@email.com
None None
```



# 参考网址

- https://www.youtube.com/watch?v=jCzT9XFZ5bw