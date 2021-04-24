python中max和min函数的key操作：

用法：

```
a = dict(((1,3),(0,-1),(3,21)))
m = max(a, key=a.get)
```

- max函数有两个参数：一个可迭代对象(a)和一个可选的“key”函数。 Key功能将用于评估a中最大的项目的值。

```
>>> a = dict(((1,3),(0,-1),(3,21)))
>>> for x in a:
...     print x #output the value of each item in our iteration
... 
0
1
3
```

```
>>> a.get(0)
-1
>>> a.get(1)
3
>>> a.get(3)
21
```

我们还可以自己定义一个函数：

```
>>> b=[2, 3, 5, 6, 4]
>>> max(b)
6
>>> def inverse(x):
...     return 1.0 / x
... 
>>> max(b, key=inverse)
2
```

**总结：我们看到max(a，key = a.get)将返回一个a.get(item)的值最大的项。即对应于最大值的键。**

参考：https://blog.csdn.net/knidly/article/details/85130363