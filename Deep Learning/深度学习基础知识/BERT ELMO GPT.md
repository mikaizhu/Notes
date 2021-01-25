## ELMO

参考视频：https://www.youtube.com/watch?v=UYPa347-DdE&t=26s

> Embedding from Language Model

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0aoyj2zqj311o0s41kx.jpg" alt="image.png" style="zoom:33%;" />

- 为了解决不同的词语在不同的句子中意思不一样，所以发明了ELMO
- BOS begging of sentence

- 但这个是单向的，即输入潮水，然后预测退了

- 缺点是只能根据上文的信息预测下文的信息

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0ax2sfs0j312a0qw1im.jpg" alt="image.png" style="zoom:33%;" />

- 因此产生双向的训练

- 蓝色和黄色都是不同的embedding向量

## BERT

> BERT

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0baxnrhxj31140s6kiw.jpg" alt="image.png" style="zoom:33%;" />

- Bert 其实就是transformer的变形
- bert训练不需要标签，因为只要训练encoder部分就可以了
- bert在做的其实就是输入句子，然后得到句子的embedding向量

**bert的训练：**

**训练方法1**

**mask单词**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0brv3c23j31080q64db.jpg" alt="image.png" style="zoom:33%;" />

- 可以mask掉一部分单词，大约15%
- 输入为单词，然后输入bert后会变成embedding的向量。
- 然后将mask掉的单词，输入到一个分类能力很弱的线性分类器中
- 让线性分类器分类出那个被mask掉的单词

**训练方法2:**

**预测下一个句子**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0bzg2esrj31300rqh2d.jpg" alt="image.png" style="zoom:33%;" />

- 这里引入两个新的token CLS SEP
- CLS指的是分类开始的地方，SEP指的是两个句子的边界
- 然后将向量输入到线性分类器中，分类器会输出两个句子是不是要拼接在一起，即True or False

**BERT在实际上用的时候，是两个训练方式都在同时进行**

**用BERT进行分类：**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0c6tw62zj311e0pktnq.jpg" alt="image.png" style="zoom:33%;" />

- 在实际应用中，bert可以完成分类等任务，完成分类的时候，要在sentence前加CLS标记符号
- 然后将CLS这个位置输出的output embedding输入到线性分类器中，完成分类
- 我们可以直接用训练好的bert输出向量，然后用向量完成其他任务
- 但原来论文中，作者是直接将BERT和分类任务一起训练的
- 即BERT，是在上面过过程中训练好的，只要微调即可，线性分类器是要从头开始训练的。