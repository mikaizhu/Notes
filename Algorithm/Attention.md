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


