# CV课程作业

## 基本介绍

本次课程ppt讲解使用的是FashionMinist数据集，相当于给图片进行分类，使用的算法有
：
- DNN
- LSTM
- Resnet
- Transformer

## 文件说明

[fashion.ipynb](./fashion.ipynb): 四种算法实现的代码细节

## PPT内容设置

首先介绍下数据集，然后尝试回答以下问题就OK了

1. dnn为什么可以对图像进行分类？

> 图片存储的本质还是使用的矩阵，因为数据是1通道的灰度图片, 所以可以将一个图片看
> 成一个向量，用一个向量表示一张图片，放入dnn中进行学习分类

2. dnn的原理是什么？

> dnn 的本质其实就是特征变换，dnn的输入是一个向量，经过某一层神经元后，转变成
> 另一个向量. 经过神经元相当于向量乘以一个矩阵，然后转换成另一个向量
> dnn 有个作用就是降低维度。相当于PAC，什么是降低数据维度？就是有些特征对结果
> 影响不是很大，目的是去除一些无关变量，找出一些主要的特征。降维有什么好处？首
> 先可以加速模型的训练，因为特征变少了，需要的参数也对应变少了。其次可以防止模
> 型过拟合，因为过拟合是模型学到了一些无关特征，而降维就是去除一些无关特征

3. cnn的图像分类原理? cnn是怎么抽取图像特征的？
4. resnet模型简单介绍?
5. 为什么transformer可以对图像进行分类？
6. transformer的原理是什么？

TODO:

- [ ] 这里写下ppt内容都要介绍什么
- [ ] 加入一些训练的结果

  下面讲解已经被摒弃, 但是还可以学习
  ----
说明：这次作业使用的项目为天池比赛

赛题：街景字符识别

赛题地址：https://tianchi.aliyun.com/competition/entrance/531795/introduction

TODO:
- [ ] 查看不同算法，写成一篇论文
- [ ] ppt的制作

参考阅读：

官方baseline:https://tianchi.aliyun.com/notebook-ai/detail?postId=108342

opencv入门教程:https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.24.2ce879detUyz2C&postId=95354

基础版教程:https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.30.2ce879detUyz2C&postId=188964

进阶版教程:https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.27.2ce879detUyz2C&postId=207291

baseline版，推荐按顺序阅读下面文章：

1. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.2ce823e6FF4FLX&postId=108659
2. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.21.2ce823e6FF4FLX&postId=108150
3. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.9.2ce823e6FF4FLX&postId=108711
4. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.2ce823e6FF4FLX&postId=108780
5. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.2ce823e6FF4FLX&postId=108656

其他高分文章：
- https://github.com/MHX1203/DataWhale--
- 上面github的技术博客:https://flashgene.com/archives/122108.html?spm=5176.21852664.0.0.4e097a98SZxo38
