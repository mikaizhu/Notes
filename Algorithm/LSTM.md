# RNN讲解

TODO:

- [ ] 什么是双向的rnn?
- [ ] rnn中代码如何实现递归
- [ ] rnn是如何训练的

参考：https://easyai.tech/ai-definition/rnn/

RNN 其实就是DNN的一个变形，也就是输入是一个序列，输出也是一个序列, DNN的结构其
实就是输入一个序列，然后经过一层隐藏层，变成另一个序列,其实就是维度变换。而RNN
相对于DNN就是多了个hidden state，hidden state 会记录之前的输入的信息，最近的输
入影响越大，所以RNN的缺点就是不能记忆很久之前的信息。

RNN的更新过程：假如现在有个句子，我喜欢你, 先对这个句子进行分词。['我','喜欢','你'], 然后将每个词都变成固定长度的向量。先将‘我’向量输入到rnn中，会得到一个输出，同时rnn会更新其中的hidden state，然后将‘喜欢’向量输入到rnn中，得到一个输出，然后将‘你’向量输入到rnn中，rnn根据之前的hidden state，得到一个输出。
这个输出向量，包括了之前句子的所有信息。

RNN的输出：类似dnn的输出向量以及hidden state

因此，我们要做什么下游任务，只要根据最后一个句子的输出即可。

所以普通RNN的缺点就是要等前面的句子一个个输入

RNN 代码实现：

参考：https://github.com/fancyerii/deep_learning_theory_and_practice/blob/master/src/ch4/char-rnn-classifier.ipynb

参考：https://github.com/python-engineer/pytorch-examples/tree/master/rnn-name-classification

参考视频：https://www.youtube.com/watch?v=WEV61GmmPrk

![](https://github.com/fancyerii/deep_learning_theory_and_practice/raw/038f2391780ba6aeb3f2be8f4945c5c7abcd21ab/src/ch4/network.png) 


```
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
```

- variable 只是封装为tensor对象，意味着这个对象需要梯度更新
- 从上面代码中可以发现，rnn实现其实就是利用到了线性层
- cat dim=1表示[1, 1], [2, 2] 会拼接成[1, 1, 2, 2]
- 最后神经网络输出的有两个，还有一个隐藏层, output是定长的向量，所以注意RNN可
  以将输入转换成向量，然后用在分类中。

# LSTM讲解

参考：

- https://zhuanlan.zhihu.com/p/32085405
- https://easyai.tech/ai-definition/lstm/

LSTM的输入是什么输出是什么？

和rnn一样，将输入变成一个序列而已
我们需要的只是输出x

- lstm处理灰度图片：假设图片的长度为28\*28,可以将一个图片看成一个长度为28的文本,里面每个单词的向量长度也是28

例子：

![](https://i2.wp.com/img-blog.csdnimg.cn/20200610114224311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Nzc3NTUw,size_16,color_FFFFFF,t_70) 

```
例子
举个例子，比如一批训练64句话，每句话20个单词，每个词向量长度为200，隐藏层单元个数为128那么训练一批句子，输入的张量维度是[6420200]，htC的维度是[128]，那么LSTM单元参数矩阵的维度是[128+200,4*128],
在时刻1，把64句话的第一个单词作为输入，即输入一个[64,200]的矩阵，由于会和ht进行concat,输入矩阵变成了64200+1281输入矩阵会和参数矩阵[200+1284*1281相乘，输出为[64.4*1281，也就是每个黄框的输出为[64128]，黄框之间会进行一些操作，但不改变维度，输出依旧是[64,128]，即每个句子经过LSTM单元后，输出的维度是128，所以上一章节的每个LSTM输出的都是向量，包括
 C_t,h_t，它们的长度都是当前LSTM单元的hidden_size得到了解释。那么我们就知道cell_output的维度为[64,128]
之后的时刻重复刚才同样的操作	那么outputs的维度是[20,64,128].
 softmax相当于全连接层，将outputs映射到vocabsize个单词上，进行交叉熵误差计算。
然后根据误差更新LSTM参数矩阵和全连接层的参数。
```

即输入是[64, 20, 200], 然后输出变成了[64, 128]
