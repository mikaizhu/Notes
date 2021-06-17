<!--ts-->
   * [关于nn.Linear()的一点理解：](#关于nnlinear的一点理解)

<!-- Added by: mikizhu, at: 2021年 6月 3日 星期四 16时23分15秒 CST -->

<!--te-->
# 关于nn.Linear()的一点理解：

这里假设nn.Linear(3, 4), 表示输入的神经元个数为3， 输出的神经元个数为4，
后面4个神经元，每个都会被前面三个所连接。因此权重矩阵为(3, 4), 偏置的长度为4

因此nn.Linear就相当于矩阵变换,初始化一个权重矩阵，然后与输入向量相乘，输入为1\*3, 输出为1\*4

<a href="https://sm.ms/image/nIbV7M2JcyU6oBP" target="_blank"><img src="https://i.loli.net/2021/06/02/nIbV7M2JcyU6oBP.png" ></a>

# 计算机视觉

## 知识点记录

### 保持输出不变

我们在卷积神经网络中使用奇数高宽的核，比如3×3，5×5的卷积核，对于高度（或宽度）为大小为2k+1的核，令步幅为1，在高（或宽）两侧选择大小为k的填充，便可保持输入与输出尺寸相同。

### 卷积核与全连层

卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出
则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接
层的输出变换为四维，1×1卷积层可以看成全连接层，其中空间维度（高和宽）上的每个
元素相当于样本，通道相当于特征。因此，NiN使用1×1卷积层来替代全连接层，从而使空
间信息能够自然传递到后面的层中去。


## 参考教程
参考: https://tianchi.aliyun.com/course/337/3988

# 自然语言处理


# 教程推荐

**这里有个教程挺好的：**

https://www.jianshu.com/u/898c7641f6ea

**官网教程查询：**

https://pytorch.org/docs/stable/index.html

**pytorch中文文档：**

https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
