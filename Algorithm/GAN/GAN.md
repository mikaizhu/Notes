## GAN 的概念

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1gliwd2ld2ej30wg0l4nlw.jpg" alt="image.png" style="zoom: 50%;" />



假设我们现在的任务是要生成人脸，那么生成的数据分布是非常高维度的。而且有些数据分布的生成的图片效果是比较好的，其他的都是比较差的。

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhi7awtudj30xi0oik7w.jpg" alt="image.png" style="zoom:50%;" />

GAN的generator就是要学习这个数据点的分布$P_{data}(x)$,接下来是极大似然函数的公式$P_G(x;\theta)$,也就是说我们希望通过固定的x样本点，（极大似然函数就是已知数据点的分布，反过来寻找到使这些样本点出现概率最大时候到theta, [reference](https://zhuanlan.zhihu.com/p/26614750),既然是最值问题，就可以求导或者梯度下降解决--神经网络）找到这样的参数theta，使得上面两个公式，$P_{data}(x)$ and $P_G(x;\theta)$ 越接近越好,由于总的数据点有很多个，我们不可能全部去生成，我们会从样本点中随机sample出一部分点m个样本，也就是说，其实generator其实拟合的是（P是theta的函数，所以需要寻找一组theta）
$$
P_G(x_1,x_2,\dots,x_m|\theta)
$$
根据极大似然估计可以证明，这些样本点的拟合出theta参数，可以很好描述总的样本点的分布，由于样本点都是 [**独立同分布的**](https://zhuanlan.zhihu.com/p/52530189)（每个数据点之间互不影响，且属于同一分布），所以可以单独相乘
$$
L=\prod_{i=1}^{m}P_G(x^i;\theta)
$$
我们希望找到这样的theta，能够使得每个样本点出现的概率最大，也就是使整个公式的概率最大。
$$
\theta^*=arg\ \underset\theta{max}\prod_{i=1}^{m}P_G(x^i;\theta)
$$
即变成了一个求最大值的问题，即找到一组参数theta，使得整个概率分布最大

也就是说，我们希望$P_G$的概率分布，能够很好的描述$P_{data}$的概率分布，也就是总样本的概率分布

**最大值这个极大似然函数的过程，就等价于最小化这个KL diversion的问题**

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhlusinomj31so17qb29.jpg" alt="image.png" style="zoom: 33%;" />

**什么是KL divergence呢？因为总的数据分布和样本的分布还是存在差距的，所以要使用KL散度公式来描述这个差距，这个差距越小越好。所以将问题从一个argmax的问题变成了argmin的问题，就是问题中也就是PG我们是假设数据是正太分布的，但是通常来说，如果用正太分布来拟合一个不是正太分布的数据，肯定会出现偏差，因为总的数据点，更平常来说不一定是正太分布**

**现在，G就是一个神经网络，由神经网络学习上面的规律，也就是将一个不是正太分布的数据，输入到神经网络中，学习到生成的数据，是高维度的不是正太的结果**

即：虽然总数据不是正太分布，但输入的样本点是正太分布。现在我们就是要找到一个公式，输入正太分布的数据，来拟合高维度不是正太分布的总数据。因为神经网络，如果层数越深，就越可以拟合所有函数。这个问题中的函数，我们使用神经网络G来进行学习。
$$
P_G(x)=G(x)
$$
然后会变成下面公式：
$$
G^*=arg\ \underset G{min} Div(P_G,P_{data})
$$
**这个公式的意思就是：找到一个生成器G，对样本点的PG，与总的Pdata的diversion，能够越小越好**

 所以，问题就在于，GAN网络，是如何处理这个min的问题呢？

要处理这个问题，首先要知道如何度量这两个的diversion

- 首先，我们只要从总数据中sample出一些样本就好了，就完成了Pdata
- 然后我们只要生成一些随机的正太分布的vector，再放入到G中生成一些图片，就生成了PG
- 最后，我们要比较PG和Pdata的diversion，如何衡量呢？

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhmljeuyuj31pq16ihdu.jpg" alt="image.png" style="zoom:33%;" />

**我们可以使用discriminator来完成这个比较diversion的任务**

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhmq6x32jj31ti188u0x.jpg" alt="image.png" style="zoom:33%;" />

看到上面公式 $V(G,D)$中我们是固定住generator的参数的，也就是固定generator的神经网络参数，只调节discriminator的参数，那么discriminator的object function，就可以用上面公式表示。

我们希望，$E_{x~P_{data}}[logD(x)]$ ,这个公式的意思就是，样本如果是从Pdata中生成的，那么希望得很高的分，如果是generator生成的图片，就得比较低的分。

所以，discriminator的方程可以写成：
$$
D^*=arg\ \underset D{max} V(D, G)
$$
<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhn13gh8ij31280r61cw.jpg" alt="image.png" style="zoom:33%;" />

**如果G生成的数据点分布，和Pdata的数据分布，靠的非常近，那么discriminator就很难区分两者的差异性。所以diversion的分会比较低。如果两个数据点隔的很开，那么discriminator就很容易区分，diversion的分会很高。**

注意到原来的公式：
$$
G^*=arg\ \underset G{min} Div(P_G,P_{data})\\
D^*=arg\ \underset D{max} V(D, G)
$$

**然后其中的div是用的discriminator来解决的。所以方程可以变成下面公式，将问题变成argminmax的问题**

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhnep517lj312q0ps7io.jpg" alt="image.png" style="zoom:33%;" />

**这个图是什么意思呢？也就是我们先固定G，在图中也就是G1，G2，G3。然后输入到D中，可以得到不同的diversion**,为什么是从里到外解呢？就像求解积分一样，先解决内部积分，再解决外部积分

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhnotk7pmj310c0ni4cy.jpg" alt="image.png" style="zoom:33%;" />

每个diversion都有最大值，我们就是要找个G，使得这个最大值最小。

**那么GAN网络是怎么解决这个minmax问题呢？**

- 其实就是在

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhnsez176j31420ny7l7.jpg" alt="image.png" style="zoom:33%;" />

算法流程：

- 对照上面的step1，首先是fix住G，然后更新D
  $$
  \{x^1,x^2,\dots,x^m\}是样本中的点\\
  \{z^1,z^2,\dots,z^m\}是随机生成的点\\
  \{\widetilde x^1, \widetilde x^2, \dots, \widetilde x^m\}是生成的点，也可以看成是图片
  $$

- 然后可以更新V，即validation参数，只不过这里我们使用的梯度下降方法。迭代多次。
- 最后更新G，只迭代一次，这次是minimize这个function，G的任务是为了减小差异。


<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhovx0bj6j31ti1d8qv6.jpg" alt="image.png" style="zoom:33%;" />

**这里有个问题，就是为什么Discriminator要迭代多次，但是Generator只迭代一次呢？因为如果Generator迭代很多次，那么就会提升的很快，提升很快，Discriminator就不能很好地找到最大的diversion，discriminator 迭代多次是为了增加diversion，刚开始的时候拟合的diversion 可能不是很高。迭代多次可以加大diversion**

- 下面这个图中，蓝色的点是generator生成的点的分布。绿色的点是真实数据的点分布
- discriminator会学习，给蓝色的点很低的分数，给绿色的点很高的分数。
- generator就会学习减小这个diversion，然后会慢慢向绿色的点移动
- 最后迭代多次以后，蓝色的点就会不断向绿色的点靠拢

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1glhpie0e90j311e0qytrl.jpg" alt="image.png" style="zoom:33%;" />