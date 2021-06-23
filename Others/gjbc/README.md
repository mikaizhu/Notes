<!--ts-->
* [思路说明](#思路说明)
   * [stacking boosting and snapshot](#stacking-boosting-and-snapshot)
* [余弦相似度分析](#余弦相似度分析)
* [将时间序列转换成图片](#将时间序列转换成图片)
* [数据增强 and 网络调节](#数据增强-and-网络调节)
* [思考](#思考)

<!-- Added by: zwl, at: 2021年 6月23日 星期三 22时05分56秒 CST -->

<!--te-->
# 思路说明

## stacking boosting and snapshot

上分主要思路说明：

使用的stacking和boosting的思想, boosting代码有待改进
- version1代码出现了标签泄漏，按道理是不能使用val的标签的，即使是stacking也不行
- version2代码也出现错误，每利用到boosting，即dnn总是在学习正确的标签
- version3 解决了上面两个错误，但是准确率有待提高

在上面方法中，version2利用了Snap shot方法，主要思路是：

- 训练单个模型，保存一些效果比较好的点
- 将这些效果比较好的模型进行融合，提升准确率

1. fft，取模，切片，单模dnn 57
2. 样本均衡 + soft voting 60.3

~~3. 样本均衡 + boosting + stacking 63.4~~

3. boosting 文件中的version2 可以达到62.2准确率, 这个思路相当于snapshot


TODO:

- [x] 接下来尝试多次投票机制
- [x] 绘制下混淆矩阵，看下预测的效果差异，看下可不可以尝试模型融合
- [ ] 尝试下TTA方法

参考：

- [天池心跳预测比赛第一名](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.3cf267f7aXHfU6&postId=231585)
  - 学习下样本不均衡处理
  - 学习下投票机制:https://blog.csdn.net/qq_39478403/article/details/115533295
- [集成学习TTA](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.2ce823e6FF4FLX&postId=108656)
  - TTA 是对一个样本预测三次，然后取平均值，可以看下代码实现

# 余弦相似度分析

参考代码：[similarity cos](./similarity_cos/eda.ipynb) 

新思路：利用余弦相似度进行匹配，思路流程如下：
- 先有监督训练模型
- 构建一个类别字典，字典存储每个类别的向量
- 利用模型，提取测试集每个样本的特征向量
- 输入向量，与所有向量进行余弦相似度匹配，完成分类。

参考：
- pytorch 中计算余弦相似度的方法：https://blog.csdn.net/tszupup/article/details/100711874
- pytorch 中提取中间层的方法：https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/16

测试结果：相比于单模分数还降低了2个百分点.

# 将时间序列转换成图片

博客讲解：https://blog.csdn.net/weixin_39679367/article/details/86416439

模块使用：
- https://github.com/johannfaouzi/pyts
- 这个模块中集成了很多时间序列分类的算法

# 数据增强 and 网络调节

比赛参考：[新网银行金融科技挑战赛 AI算法赛道](https://www.heywhale.com/home/competition/5ece30cc73a1b3002c9f1bf5/content/6) 

参考思路and代码：

- [Top 1](https://github.com/miziha-zp/xw2020-top1) 
- [Top 2](https://github.com/China-ChallengeHub/Cellphone-Behavior) 
- [Top 14](https://github.com/MichaelYin1994/kesci-mobile-behavior) 
- [深度学习在时间序列上应用的论文总结](https://zhuanlan.zhihu.com/p/83130649) 

# 思考

- [ ] 因为只使用了片段6892-7192其他片段并没有利用，查看其他片段的区分度
- [ ] 如果使用标准化，不用最大最小值归一化效果会怎么样？
