## 生成任务

参考视频：https://www.youtube.com/watch?v=f1KUUz7v8g4&t=1980s

> 生成句子

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm01miwzv3j31200pwtv3.jpg" alt="image.png" style="zoom:33%;" />

- 先输入一个起始符号，然后会生成第一个汉子，然后利用rnn的结构，会得到已经生成第一个字后，第二个字的条件概率分布。然后形成一篇文章

> 生成图片

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm01rz4006j310w0qwqse.jpg" alt="image.png" style="zoom:33%;" />

- 同样也可以生成图片
- 就是吧一个图片，拆成一个个像素点，然后得到类似文章的分布。

> 机器翻译

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm024usx26j31080q8ki3.jpg" alt="image.png" style="zoom:33%;" />

- 先把汉子输入到rnn，然后最后输出一个vector

- 然后把vector输入到另一个rnn中，然后进行翻译
- 然后左边就是encoder， 右边叫decoder

## Attention

> attention 示意图

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm029s5haxj30z40q41du.jpg" alt="image.png" style="zoom:33%;" />

- 在以前的任务中，rnn都是在看完整个句子的时候再完成输出
- 然而我们知道机器翻译成machine，学习翻译成learning。
- 人在学习过程中往往都有个关注点

> attention 原理

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm02vuqukkj310i0qknjq.jpg" alt="image.png" style="zoom:33%;" />

- attention中多引入了一个query向量
- query表示查询向量。就是能和输入的vector进行匹配

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm03251d58j30zu0qa1ci.jpg" alt="image.png" style="zoom:33%;" />

- 先随机生成z0，然后将z0与所有的h向量进行运算，输入到一个match模块中
- match模块可以是神经网络，也可以是矩阵运算。
- 经过放大后，attention会注意到机器这两个向量上，可以看到只有alpha有数值

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gm037hbojsj30yc0pi1bq.jpg" alt="image.png" style="zoom:33%;" />

- 然后不断将z*进行输入

