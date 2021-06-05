参考：https://zhuanlan.zhihu.com/p/75206669

**建议好好看看这篇文章**

一般情况下 nn.Sequential 的用法是来组成卷积块 (block)，然后像拼积木一样把不同的 block 拼成整个网络，让代码更简洁，更加结构化。



有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们，而不是一行一行地写，比如：

```
layers = [nn.Linear(10, 10) for i in range(5)]
```

