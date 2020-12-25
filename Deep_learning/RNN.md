# pytorch 实现RNN

## 什么是RNN？

通常的神经网络有个缺点，就是输出和输入是没有关系的，即只要确定输入x，然后得到输出y就没了。

然而现实生活中很多数据之间都是呈现一定的关联性的，比如对未来几天的天气进行预测，我们根据常识可以知道。昨天的天气对今天的天气是有影响的。

在语言翻译中，一句话的上下是有关联的。所以使用平常的神经网络肯定不好处理这类问题，这就导致了RNN和更复杂的模型产生。

### 传统神经网络的痛点

> 来自李宏毅老师的课程ppt

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzuy1eyvoj310c0n24f4.jpg" alt="image.png" style="zoom:33%;" />



**上面这个图中，假设问题是现在我们要根据一个句子判断地点和时间。**

**台北前分别是arrive 和 leave。如果是传统的神经网络。是不会考虑到是离开台北还是到达台北的**

因此rnn神经网络就出现了～

## rnn基础

**我们需要神经网络在看过Taipei这个单词的时候，已经看过前面的是arrive还是leave**

> rnn示意图

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzv3k0800j30z40pi4e4.jpg" alt="image.png" style="zoom:33%;" />

- 简单的rnn是有个记忆单元的。就是希望神经网络看过什么东西
- x1，x2是输入。与一个权重相乘后，就会得到一个输出。然后将这个输出存储到记忆单元中
- 下一次的输入，会将记忆单元的值进行相加，然后输出

**就比如我先输入leave单词，向量化后变成x1=1， x2=1，然后输入到神经网络中**

**因为我们输入的是一个序列，所以这个单元x1，x2会反复利用**

> 单元被重复利用

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzva3kjmbj30y20o47m0.jpg" alt="image.png" style="zoom:33%;" />

- 可以看到上图中，记忆单元作为输出，和下一次的输入进行了相加
- 上图中，同一个network，被重复利用了三次而已

> rnn也可以往深层叠加

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzvcs9nlij310c0lotmx.jpg" alt="image.png" style="zoom:33%;" />

> 还有其他种类rnn

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzvdrcl7ij30xk0kswsx.jpg" alt="image.png" style="zoom:33%;" />



- jordan network是把最后结果作为输出，而不是隐藏层作为输出
- 效果比之前的rnn要好，因为最后一层是有标签的，所以更靠谱一些

> bidirectional rnn 双向rnn

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzvhiono7j310c0madt6.jpg" alt="image.png" style="zoom:33%;" />

- 在之前单项的rnn基础上，我们不仅只训练一个单项的rnn
- 还会训练一个逆向的rnn，然后将两个最后输出相加得到输出

## lstm（long sort term memory）

> lstm由下面几个单元组成



<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzvkcnyqkj312g0so4nl.jpg" alt="image.png" style="zoom:33%;" />

- 输入门output gat
- 输出门input gat
- 遗忘门forget gate
- 记忆单元memory cell

**输入门控制输入的参数，是否会进入到记忆单元中**

**输出门决定，记忆单元的输出，是否会被最后输出**

**遗忘门决定是不是要清空记忆单元中的值**

**记忆单元用来存储数据，由遗忘门决定是不是要输出和输入相加**

> Lstm 4 inputs, 1outputs

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzvq31yeuj312e0s61kx.jpg" alt="image.png" style="zoom:33%;" />

- lstm的4个输入，1个输出，别忘记了输入信号～



> 示意图

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzvt21zglj31240s4ncz.jpg" alt="image.png" style="zoom:33%;" />



- 遗忘门控制记忆单元是不是要清零

> lstm运算流程

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201225101654219.png" alt="image-20201225101654219" style="zoom:33%;" />

- 可以看到，上图中输入是x1，x2， x3，三个单元。
- 但是这三个单元都乘上了不同的权重和加上不同偏置
- 然后分别作为四个输入
- 然后神经网络会训练这些参数

> lstm和rnn的关系

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzw09bdj1j31200ow12e.jpg" alt="image.png" style="zoom:33%;" />



- 假设现在是xt时间。信号xt会乘上一个矩阵，变换成vector，vector的每个维度，都会作为lstm的输入

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzw1n3prwj311o0oqdqp.jpg" alt="image.png" style="zoom:33%;" />

- 然后xt乘以多个矩阵，得到zf，zi**等向量vector**，分别作为三个控制门的输入

## 训练rnn

> 简单rnn的训练

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20201225103004624.png" alt="image-20201225103004624" style="zoom:33%;" />

- 训练的时候也是要有标签的，然后用反向传播

> 然而rnn很难训练

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzwhlbe3wj312u0qu7rj.jpg" alt="image.png" style="zoom:33%;" />

- 如何查看rnn梯度的变化？
- 我们只要改变输入，看输入改变多少即可。
- rnn是累积效应，很小变化，会产生很大的输出
- **即记忆单元因为是一直在用，记忆单元的影响，累积下来，要么产生很大的影响，要么不产生影响**

## S2S (sequence2sequence)

> 序列到序列的问题

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzx12jbn7j31020pqkdq.jpg" alt="image.png" style="zoom:33%;" />

**假如现在输入的是机器学习这两个单词，输入假如先试machine，然后rnn会记录到神经网络中，然后看到learning，就知道看完了，最后一个rnn单元，会输出产生‘机’这个字，然后将‘机’作为输入，和记忆单元一起运算，作为输出。如此往复下去...**

**如何知道要如何停止呢？**

> 训练的时候，输出停止符号即可

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzx5h7bioj311k0qi7ub.jpg" alt="image.png" style="zoom:33%;" />

> auto encoder

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1glzxdqq11dj31100rq7u8.jpg" alt="image.png" style="zoom:33%;" />

- Encoder 就是将语音信号输入，然后学习语音信号的上下相关性，得到记忆矩阵。记忆矩阵作为最后一层的输出
- 然后输出作为decoder的输入，得到第一个y1，然后y1作为输入，和记忆单元作用，如此往复
- 我们希望y1和x1越接近越好



参考教程：

https://www.jiqizhixin.com/articles/2018-12-14-4

https://www.youtube.com/watch?app=desktop&v=xCGidAeyS4M&t=7s


这里说明几个关键的信息。

假设输入为X=(x1, x2, x3, x4)，每个x是一个单词的词向量。

为了建模序列问题，RNN引入了隐状态h（hidden state）的概念，h可以对序列形的数据提取特征，接着再转换为输出。先从h1的计算开始看：

$$
h_1=f(Ux_1+Wh_0+b)
$$

h2的计算和h1类似。要注意的是，在计算时，每一步使用的参数U、W、b都是一样的，也就是说每个步骤的参数都是共享的，这是RNN的重要特点，一定要牢记。

注意：如果以后在RNN中看到很多箭头，那么箭头就是进行一次矩阵变换，即带入到f表达式中。

先来看看官网的案例：

```
import torch.nn as nn
rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
```
output:
```
output.shape
torch.Size([5, 3, 20])
```

可以看出，如果是使用普通的rnn，那么输入和输出的类型是一样的。

这里介绍几个参数：

官网地址：https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

input_size – The number of expected features in the input x

hidden_size – The number of features in the hidden state h

num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1

nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'

bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False

dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0

bidirectional – If True, becomes a bidirectional RNN. Default: False

