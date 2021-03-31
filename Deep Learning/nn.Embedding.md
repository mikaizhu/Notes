参考1：https://www.jianshu.com/p/63e7acc5e890

参考1讲解了embedding的作用，即输入单词id，然后吐出这个单词的嵌入向量。但是没有说内部是怎么训练的。

nn.embedding 有什么用？

建立一个查找表，将每个单词嵌入为一个向量，越相近的单词，向量的相似度就会越接近。

每个单词的权重在训练过程中会自动学习

我们也可以自己导入训练好的权重。

```
from_pretrain(, freeze=True)
```

