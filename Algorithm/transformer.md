[TOC]

# Attention算法介绍

参考资料：

- [论文 Attension is all you need](https://arxiv.org/pdf/1706.03762.pdf) 
- [图解Attension](https://wmathor.com/index.php/archives/1450/) 
- [Attension 公式讲解](https://wmathor.com/index.php/archives/1432/) 
- [Reference](https://wmathor.com/index.php/archives/1432/) 

- [zhihu Reference1](https://zhuanlan.zhihu.com/p/47063917) 
- [zhihu Reference2](https://zhuanlan.zhihu.com/p/47282410) 

- [李宏毅attention视频讲解](https://www.youtube.com/watch?v=ugWDIIOHtPA) 
- [李宏毅self attention 1视频 讲解](https://www.youtube.com/watch?v=hYdO9CscNes&t=2s) 
- [李宏毅self attention 2视频 讲解](https://www.youtube.com/watch?v=gmsMY5kc-zw) 

思路：

- 先介绍下传统的seq2seq模型的缺点
- attention的设计初衷
- 介绍attention
- 介绍multi head attention
- 介绍self attention
- 介绍下attention机制的模型大小如何计算


## 什么是seq2seq模型

参考：https://easyai.tech/ai-definition/encoder-decoder-seq2seq/

encoder-decoder 是一种模型，encoder将一个句子，或者一张图片，转换成固定长度的
向量，然后将这个向量输入到decoder中，变换成一个新的图片或者句子。

Q1: encoder and decoder的输入的长度和输出的长度都是不固定的，但中间转换的向
量长度是固定的。这就是传统的seq2seq的缺陷。

Q2: encoder-decoder是由什么组成的？

只要符合上面这种形式的结构，都可以统称为encoder and decoder 的模型，encoder和
decoder可以由不同的形式组成，主要现在有RNN and LSTM

Q3: 什么是seq2seq

只要满足输入是一个不定长的序列，输出也是一个不定长的序列，即输入输出序列的长度
是可变的，这种模型的结构就是seq2seq, 

Q4: seq2seq和encoder-decoder有什么关系

seq2seq只是一种目的，即输入是一个序列，输
出是另一个序列

而encoder 和decoder是一种方法，即实现seq2seq的一种方法，一种手段.

Q5: encoder-decoder 有哪些主要应用？

- 文字到文字：通常的翻译任务
- 语音到文字：asr
- 图片到文字：图片描述任务

Q6：encoder-decoder这个模型有什么缺陷？

encoder-decoder的结构是将输入转换成固定长的向量，然后输入到decoder中，这就像压
缩和解压的过程，输入的所有信息都由这个固定的向量表示，当句子太长的时候，句子的信息会出现丢失

Q7: attention机制有什么用？

attention就是来解决这个信息丢失的问题

Ａttention 模型的特点是 Eecoder 不再将整个输入序列编码为固定长度的「中间向量 Ｃ」 ，而是编码成一个向量的序列。

![](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-10-28-attention.png) 



## 什么是Attention

参考：https://easyai.tech/ai-definition/attention/


# Transformer使用

Transformer中的position encoding：https://wmathor.com/index.php/archives/1453/

Transformer 详解：https://wmathor.com/index.php/archives/1438/

Transformer的pytorch实现：https://wmathor.com/index.php/archives/1455/

代码的GitHub链接：https://github.com/wmathor/nlp-tutorial

bert详细解析：https://wmathor.com/index.php/archives/1456/

bert的pytorch实现：https://blog.csdn.net/qq_37236745/article/details/108845470

# 问题记录

接下来，为了理解模型，我们假设现有数据: 我喜欢你，分词后得到['我','喜欢','你
'], 将每个单词映射成一个向量所以这句话就变成3\*28的向量。因为每句话的长短不一
样，所以这里设置最大长度为28因此我喜欢你这句话，就会变成28\*28的向量。

Q:Transformer 是怎么学习时间序列的？因为RNN有按时间序列输入，即有顺序关系。

A:Transformer可以同时对所有时间步长计算, 所以计算速率很快，我们需要提供
positional encoding 即位置编码，这样才能让transformer了解位置的先后关系。

# 理论知识讲解


## Position encoding

transformer 中有一个位置编码

由于 Transformer 模型**没有**循环神经网络的迭代操作，所以我们必须提供每个字的位置信息给 Transformer，这样它才能识别出语言中的顺序关系

现在定义一个**位置嵌入**的概念，也就是 Positional Encoding，位置嵌入的维度为 `[max_sequence_length, embedding_dimension]`, 位置嵌入的维度与词向量的维度是相同的，都是 `embedding_dimension`。`max_sequence_length` 属于超参数，指的是限定每个句子最长由多少个词构成，长度不够会进行补0。

注意，我们一般以**字**为单位训练 Transformer 模型。首先初始化字编码的大小为 `[vocab_size, embedding_dimension]`，`vocab_size` 为字库中所有字的数量，`embedding_dimension` 为字向量的维度，对应到 PyTorch 中，其实就是 `nn.Embedding(vocab_size, embedding_dimension)`

> anyway， 我们现在将文本转换成了28\*28长度的向量, 也就是一句话28个词，每个词
> 转换成了28维度的向量，利用位置编码后，每个词的位置编码是1\*28，所以28个词的
> 位置编码是28\*28，然后将位置编码和文本向量相加，就完成了

## self attention

位置编码好后，文本还是变成了28\*28长度的矩阵，这时我们定义三个矩阵W_Q，W_K，W_V, 与
文本矩阵相乘，利用矩阵乘法运算原理，文本矩阵中，每一个字向量(一行)，都会和三个
矩阵相乘。然后得到三个矩阵，Q，K，V。分别叫做query, key, value。

Q: 这三个矩阵有什么用？

参考：https://zhuanlan.zhihu.com/p/43493999

为了获得注意力权重，即为了抓住和记忆重点。即每个字向量的权重是不一样的.

```
举个例子。比如在预测“我妈今天做的这顿饭真好吃”的情感时，如果只预测正向还是负向，那真正影响结果的只有“真好吃”这三个字，前面说的“我妈今天做的这顿饭”基本没什么用，如果是直接对token embedding进行平均去求句子表示会引入不少噪声。所以引入attention机制，让我们可以根据任务目标赋予输入token不同的权重，理想情况下前半句的权重都在0.0及，后三个字则是“0.3, 0.3, 0.3”，在计算句子表示时就变成了：

最终表示 = 0.01x我+0.01x妈+0.01x今+0.01x天+0.01x做+0.01x的+0.01x这+0.01x顿+0.02x饭+0.3x真+0.3x好+0.3x吃
```

Q:如何获得每个字向量的权重？

使用Q点乘K, Q矩阵每行都是字的查询向量，用字查询向量，乘以所有字的key向量，这样
能得到这个字查询向量的权重, 即该字与其他字的权重.eg：2，4，4. 4表示第一个字与
第二个字的权重.

```
            [0, 4, 2]
[1, 0, 2] x [1, 4, 3] = [2, 4, 4]
            [1, 0, 1]

# 然后通过softmax
softmax([2, 4, 4]) = [0.0, 0.5, 0.5]
```

softmax后的权重，然后乘以value V矩阵

```
0.0 * [1, 2, 3] = [0.0, 0.0, 0.0]
0.5 * [2, 8, 0] = [1.0, 4.0, 0.0]
0.5 * [2, 6, 3] = [1.0, 3.0, 1.5]
```

最后将这些权重化后的值向量求和，得到第一个字的输出

```
  [0.0, 0.0, 0.0]
+ [1.0, 4.0, 0.0]
+ [1.0, 3.0, 1.5]
-----------------
= [2.0, 7.0, 1.5]
```

对其它的输入向量也执行相同的操作，即可得到通过 self-attention 后的所有输出

## Padding

由于训练的时候，不可能是一句话一句话训练，而是使用minibatch，这就导致每句话的
长度肯定不一样。所以要用0进行填充，注意，我们是先构建字典，键是字，值是idx，将
每句话转换成idx后，输入到embedding中，得到每个字的向量。

Q:什么是mask？

参考：https://wmathor.com/index.php/archives/1438/

mask其实就是加一个很大的负偏差，因为在softmax中，如果e^x次方为0时是1，而
padding=0的部分应该不参与运算，所以应该加个负无穷。

## 残差连接

就是将embedding的向量和self attention的向量相加。

## layer normalization

layer normalization是对同一个维度，不同batch，进行标准化，假设输入是[batch size
, embedding dim], 设为[28, 28], batch 为28.那么norm的方式是对[28, 1]进行norm。

而batch norm是对[1, 28]进行norm。

## feedforward

其实就是经过2个线性层, nn.Linear

首先将attention输出的向量，输入一层线性层，然后再通过一层。

注意：attention输出的是一个向量了，即[batch size, embedding dim]

## 回顾Encoder

**Encoder 的部分到这里就结束了，回忆下Encoder的输入和输出**


![](https://s1.ax1x.com/2020/04/25/JyCLlQ.png#shadow) 

- 输入batch size, seq len. seq是 字典中的字id，不够长的要补0
- 经过embedding 得到每个id的向量，变成[batch size, seq len, embed size]
- 进行position embedding 后，位置编码的向量为[batch size, seq len, embed
  size], 因为尺寸一样，所以可以直接和embedding相加
- 接下来输入到attention中，假设['i', 'love', 'you'], max len 长度为3，
  embedding dim 为 4， 那么这个句子变成[3, 4] 的向量, 将这个向量分别与三个
矩阵相乘Wq, Wk, Wv 进行线性变换(三个为4\*3)每个字转变成长度为3的向量, 因为每个字向量长度都是3，所以可以拼接成矩阵, 得到三个矩阵，Q，K，V，乘完后Q矩阵为[3, 28], 为了获得每个字
的权重，将Q矩阵第一个字的query向量[1, 3],和K矩阵相乘，得到1\*3的向量, 这个就
是权重，然后输入到softmax中，再与value V矩阵相乘, 得到这个字的向量[1, 3], 所以
经过self attention后输出还是[batch size, seq len, embed size], 只是embedding
dim维度在变化学习而已.


## Decoder 结构

这里博客没说太清楚，可以

结合视频: youtube.com/watch?v=ugWDIIOHtPA&t=1598s

结合文章：https://wmathor.com/index.php/archives/1438/

![](https://z3.ax1x.com/2021/04/20/c7wyKU.png#shadow) 

注意 outputs 输入的是ground truth，也就是真实的标签。因为attention可以看到所有
的字，这不合理，我们是一个个产生的，所以需要mask操作，这里可以看文章，很详细。

# 代码讲解

参考：https://wmathor.com/index.php/archives/1455/

参考视频：https://www.bilibili.com/video/BV1mk4y1q7eK?p=2

官网代码教程：https://pytorch.org/tutorials/beginner/transformer_tutorial.html

主要讲解的是翻译任务，即将德语翻译成英文.

**最开始的代码如下**：

- 该代码实现的是从德语翻译成英语
- 注意要分别构建德语和英语的字典，即构建两个字典
- encoder 和 decoder 都有输入，encoder输入的是待翻译的句子，decoder输入的是真
  实的标签，只不过decoder的输入要mask，因为看不到未来

```
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# S: decoding的起始输入符号
# E: decoding符号的输出终止符号
# P: 填充符号，因为有些句子里的单词数量长短不一，为了使他们一样，要进行填充，用P符号代替
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}

# 构建一个序号到单词的字典
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length
```

**然后构造数据集**：

```
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    # 将单词构造成idx，同时标点符号也要分开
    for i in range(len(sentences)):
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

# 构造好数据集是为了方便装入Data Loader中
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
```

**transformer主要由三个部分组成**

- encoder layer
- decoder layer
- 以及最后的线性层

**先来看看encoder 部分， encoder 由很多encoder layer组成**

```
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
```

```
# encoder layer拼接成一个完整的encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```

使用 `nn.ModuleList()` 里面的参数是列表，列表里面存了 `n_layers` 个 Encoder Layer

由于我们控制好了 Encoder Layer 的输入和输出维度相同，所以可以直接用个 for 循环以嵌套的方式，将上一次 Encoder Layer 的输出作为下一次 Encoder Layer 的输入

> 注意，module lists使用时为了方便制作维度和参数相同的模型，简化代码

**同样，decoder部分也是类似，由很多decoder layer组成，所以全局的transformer结构如下：**

```
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```

**模型，损失函数和优化器**

```
model = Transformer().cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
```

这里的损失函数里面我设置了一个参数 `ignore_index=0`，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
