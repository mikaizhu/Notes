## classmethod 和 staticmethod 有什么用？

先来看个例子:

- 平时实例传入的是self，也就是通常实例化后，调用该方法，self指的是实例本身
- 采用classmethod方法，传入的是cls，也就是类本身，就是通过类直接调用该方法

```
class Employee:

    num_of_emps = 0
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

        Employee.num_of_emps += 1

    def fullname(self):
        return f'{self.first} {self.last}'

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount


emp1 = Employee('Zhu', 'weilin', 50000)

emp2 = Employee('Xiao', 'ming', 60000)

print(Employee.raise_amt)
print(emp1.raise_amt)
print(emp2.raise_amt)
```

```
1.04
1.04
1.04
```

接下来，我们进行一些尝试，来看看classmethod的作用

```
Employee.raise_amt = 2
```

```
2
2
2
```

```
Employee.set_raise_amt(2) # 和上面的效果是一样的
```

现在，假设我们要添加很多新的员工，但是这些员工都不是上面的形式，员工结构如下：

```
emp_str1 = 'John-Doe-70000'
emp_str2 = 'Steve-Smith-30000'
emp_str3 = 'Jane-Doe-90000'
```

我们可以这样添加一个新的员工

```
class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

        Employee.num_of_emps += 1

    def fullname(self):
        return f'{self.first} {self.last}'

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)


emp_str1 = 'John-Doe-70000'

first, last, pay = emp_str1.split('-')

new_emp1 = Employee(first, last, pay)

print(new_emp1.email)
print(new_emp1.pay)
```

每次都要这样？我们可以写个函数，但这样太不统一了，所以我们使用classmethod封装进类中。

```
class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return f'{self.first} {self.last}'

    @classmethod
    def from_string(cls, string):
        first, last, pay = emp_str1.split('-')
        return cls(first, last, pay) # cls其实就是Employee,这里必须返回实例对象


emp_str1 = 'John-Doe-70000'
emp_str2 = 'Steve-Smith-30000'

new_emp1 = Employee.from_string(emp_str1) # 直接这样创建就好了
new_emp2 = Employee.from_string(emp_str2)

print(new_emp1.email)
print(new_emp1.pay)
```

**上面介绍了classmethod，我们再来理一下思路：**

- 普通的类函数，self传入的是实例对象，也就是类实例化后调用，比如a = A(), a.func1()
- 加了classmethod的类函数，传入的是cls，也就是类，不需要实例化，如：a = A.func1()，a就是实例对象

**下面介绍staticmethod：**

staticmethod不需要传入任何的参数，就像外部的def函数一样，不需要传入self或者cls

- 判断某一天是不是工作日

```
import datetime


class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return f'{self.first} {self.last}'

    @staticmethod # 定义了以后，这个函数就不需要传入self了
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


my_date = datetime.date(2016, 7, 10)
print(Employee.is_workday(my_date)) # 直接通过类调用
```

**staticmethod就是让一个函数专门服务于这个类**