# pytorch 实现RNN

## 什么是RNN？

通常的神经网络有个缺点，就是输出和输入是没有关系的，即只要确定输入x，然后得到输出y就没了。

然而现实生活中很多数据之间都是呈现一定的关联性的，比如对未来几天的天气进行预测，我们根据常识可以知道。昨天的天气对今天的天气是有影响的。

在语言翻译中，一句话的上下是有关联的。所以使用平常的神经网络肯定不好处理这类问题，这就导致了RNN和更复杂的模型产生。

参考教程：

https://www.jiqizhixin.com/articles/2018-12-14-4


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

