上分主要思路说明：

使用的stacking和boosting的思想, boosting代码有待改进
- version1代码出现了标签泄漏，按道理是不能使用val的标签的，即使是stacking也不行
- version2代码也出现错误，每利用到boosting，即dnn总是在学习正确的标签
- version3 解决了上面两个错误，但是准确率有待提高

1. fft，取模，切片，单模dnn 57
2. 样本均衡 + soft voting 60.3
~~3. 样本均衡 + boosting + stacking 63.4~~
3. boosting 文件中的version2 可以达到62.2准确率

TODO:

- [x] 接下来尝试多次投票机制
- [x] 绘制下混淆矩阵，看下预测的效果差异，看下可不可以尝试模型融合

参考：

- [天池心跳预测比赛第一名](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.3cf267f7aXHfU6&postId=231585)
  - 学习下样本不均衡处理
  - 学习下投票机制:https://blog.csdn.net/qq_39478403/article/details/115533295
- [集成学习TTA](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.2ce823e6FF4FLX&postId=108656)
  - TTA 是对一个样本预测三次，然后取平均值，可以看下代码实现
