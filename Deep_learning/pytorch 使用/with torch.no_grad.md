## With torch.no_grad()使用说明

参考教程：https://blog.csdn.net/weixin_46559271/article/details/105658654

总的来说，作用就是可以避免变量计算梯度

**实用场景：**

- 冻住某些神经元，不进行梯度传播

查看一个变量是不是要计算梯度：

```
b.requries_grad()
b.grad_fn
```

其他实用场景：

- 可以使用修饰器

```
@torch.no_grad()
def eval():
	...
```

- 使用函数形式

```
def eval():
	torch.set_grad_enabled(False)
	...	# your test code
	torch.set_grad_enabled(True)
```

