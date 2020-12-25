## self attention

参考视频：youtube.com/watch?v=ugWDIIOHtPA&t=171s

> 平行化处理问题

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm04pvp48pj312q0ri1k4.jpg" alt="image.png" style="zoom:33%;" />

- 单向或者双向的rnn，如果现在要得到b4的向量，我们必须先等rnn看完了a1，a2，a3，a4之后才能生成，所以就不能并行计算
- 我们也可以利用cnn来检查文本，第一层虽然只可以看少量信息，第二层就可以看b1，b2向量，而b1，b2已经看了a1，a2，a3.这就导致第二层一个卷积，相当于看了a1，a2，a3所有的，所以越高层，越能看到全局的信息。

> Self attention

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm04xtdlixj312k0sk1kx.jpg" alt="image.png" style="zoom:33%;" />

- Self attention 可以让我们同时计算出b1，b2...这些向量，不用等待
- 而且效果和之前的都一样

> 原理

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm050ql27lj31140sqb12.jpg" alt="image.png" style="zoom:33%;" />

- x1和x2...是输入的向量，然后会分别乘以一个W，变成另一个向量a1，a2，a3..
- ai会乘以三个不同的矩阵，然后得到三个向量query，key和v

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm055fhw3nj312y0rokd9.jpg" alt="image.png" style="zoom:33%;" />

- 接下来拿q，与每个向量的k进行内积运算，得到一个标量alpha

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm057b91iij311g0sah3p.jpg" alt="image.png" style="zoom:33%;" />



- 然后将输出softmax，得到alpha hat



<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm058plfmbj31180t87pa.jpg" alt="image.png" style="zoom:33%;" />

- 得到alpha hat之后，然后与v相乘，然后再求和，就得到b向量输出
- 可以看到b1里面含有x1，x2...的所有信息
- 所以再远的信息，都可以用alpha hat控制。

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm05b3ajxgj311s0sk7rv.jpg" alt="image.png" style="zoom:33%;" />

- 以此类推

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09a2g1ftj311u0si4po.jpg" alt="image.png" style="zoom:33%;" />

- 通俗说下
- 将原来的rnn换成self attention以后，输入是一类向量，输出是另一类向量
- 只不过这些向量可以平行运算出来

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09d3unvbj312c0rye52.jpg" alt="image.png" style="zoom:33%;" />

- 可以看到self attention 内部就是一系列的矩阵运算。
- 可以将这些向量拼接成矩阵，然后进行平行的运算

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09fyhx8xj31260t01f0.jpg" alt="image.png" style="zoom:33%;" />

- 可以看到里面的运算都是平行的运算

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09hkkn5oj311y0si4jy.jpg" alt="image.png" style="zoom:33%;" />

- 然后得到输出

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09ivm7n2j312u0sg4iv.jpg" alt="image.png" style="zoom:33%;" />

- 假设输入的特征是向量I，a1，a2，a3，a4.
- 向量I分别和三个不同的矩阵相乘，得到三个向量，QKV
- 向量可以拼接成矩阵QKV，然后K和Q相乘得到A，然后再softmax得到Ahat。
- Ahat与V向量相乘，得到output向量

> Multi head self attention

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09o3vvu8j311m0t8ww6.jpg" alt="image.png" style="zoom:33%;" />

- multi head self attention 其实就是每个qkv都分裂成了两个小的向量

- 这样做的原因是q可以做不同的事，q1可以看近的问题，q2可以看远的问题

## transformer

> transformer的结构示意图

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm09x3cbnfj310s0t0auz.jpg" alt="image.png" style="zoom:33%;" />

- 假设现在是一个翻译的问题

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm0a0jyyw3j312k0sa7wh.jpg" alt="image.png" style="zoom:33%;" />

**流程如下：**

- 对输入的单词embedding，然后加上位置信息输入到self attention中
- self attention会重复N次
- 右边的output是输入，比如机器学习翻译是machine learning。
- 那么第一个输入是EOS，然后输入是machine
- layer norm和batchnorm的区别：layer norm在行列上都是均值为0，方差为1的分布。batch norm只有在行上是0-1分布

