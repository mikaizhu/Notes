参考1：https://www.jianshu.com/p/63e7acc5e890

参考1讲解了embedding的作用，即输入单词id，然后吐出这个单词的嵌入向量。但是没有说内部是怎么训练的。

参考2: embedding 有什么用？https://blog.csdn.net/tommorrow12/article/details/80896331


nn.embedding 有什么用？

建立一个查找表，将每个单词嵌入为一个向量，越相近的单词，向量的相似度就会越接近。

每个单词的权重在训练过程中会自动学习

我们也可以自己导入训练好的权重。

```
from_pretrain(, freeze=True)
```

转换代码如下：

```
word_to_idx = {'hello':0, 'world': 1}
embeds = nn.Embedding(2, 5)
hello_idx = torch.tensor(word_to_idx['hello']).long()
hello_embed = embeds(hello_embed)
```

