# 2020腾讯广告大赛第一名代码理解

- 预训练教程 https://huggingface.co/blog/how-to-train
- 分类教程 https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

## nn.embedding 

**使用方法**：https://www.jianshu.com/p/63e7acc5e890

其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。

输入为一个编号列表，输出为对应的符号嵌入向量列表。

**详细过程请看上面的链接**

```
参数说明：
num_embeddings (python:int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
embedding_dim (python:int) – 嵌入向量的维度，即用多少维来表示一个符号。
padding_idx (python:int, optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
```

**简单介绍下流程：**

假设一个mini-batch如下所示：

```json
['I am a boy.','How are you?','I am very lucky.']
```

显然，这个mini-batch有3个句子，即batch_size=3

第一步首先要做的是：将句子标准化，所谓标准化，指的是：大写转小写，标点分离，这部分很简单就略过。经处理后，mini-batch变为：

```json
[['i','am','a','boy','.'],['how','are','you','?'],['i','am','very','lucky','.']]
```

记录每个句子的长度

```
lens = [5,5,4]
```

之后，为了能够处理，将batch的单词表示转为在词典中的index序号，这就是word2id的作用。转换过程很简单，假设转换之后的结果如下所示

```undefined
batch = [[3,6,5,6,7],[6,4,7,9,5]，[4,5,8,7]]
```

同时，每个句子结尾要加EOS，假设EOS在词典中的index是1。

```undefined
batch = [[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1]]
```

那么长度要更新：

```undefined
lens = [6,6,5]
```

很显然，这个mini-batch中的句子长度**不一致！**所以为了规整的处理，对长度不足的句子，进行填充。填充PAD假设序号是2，填充之后为：

```undefined
batch = [[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,2]]
```

上面batch有3个样例，RNN的每一步要输入每个样例的一个单词，一次输入batch_size个样例，所以batch要按list外层是时间步数(即序列长度)，list内层是batch_size排列。即batch的维度应该是：

```json
[seq_len,batch_size]
[seq_len,batch_size]
[seq_len,batch_size]
```

怎么变换呢？变换方法可以是：使用itertools模块的zip_longest函数。而且，使用这个函数，连填充这一步都可以省略，因为这个函数可以实现填充！

```cpp
batch = list(itertools.zip_longest(batch,fillvalue=PAD))
# fillvalue就是要填充的值，强制转成list
```

```undefined
batch = [[3,6,4],[6,4,5],[5,7,8],[6,9,7],[7,5,1],[1,1,2]]
```

- 这里说明下变换
- 将每个句子的第一个，组成在一起，就变成364这样

```
batch = [[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,2]]
```

好了，现在使用建立了的embedding直接通过batch取词向量了，如：

```undefined
embed_batch = embed (batch)
```

