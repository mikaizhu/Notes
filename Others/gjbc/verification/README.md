# 手机未知源识别

目标：判断当前信号是属于已知源，还是未知源

训练流程：
1. 搜集数据，划分好训练集和测试集
2. 搭建分类模型
3. 训练模型，完成分类

步骤细化：
1. 数据集的划分(已完成)

Q1. 因为信号包括已知手机，和未知手机信号，如何划分训练集和测试集呢？

假如现在一共有13个类别的手机，10个已知，3个未知。那么训练集中，只能含有已知手
机，测试集中，既要含有已知，也要含有未知手机


代码思路分成两个部分：

因为总有新的数据出来，而且原始数据为dat，为了方便，所以代码分成两个部分

1. 将数据划分为x_train, x_test, y_train, y_test
2. 从上面数据中，分出已知源和未知源数据, x_train, x_test, x_val, y_train, y_test, y_val
3. 训练模型使用x_train, x_test，y_train, y_test，此时只含有已知源类别，并且需
   要将标签重置从0开始
4. x_val, y_val 既含有已知源，也含有未知源，这里需要将x_val从x_test中分离吗？
   不太需要，可以一样的，因为验证集本来模型就没见过，为了保证样本均衡，训练样
   本就会减少，所以可以直接使用验证集

代码说明：
- rate只会影响从多少test数据中添加已知源到验证中, 如果数据不均衡，设置为1也没
  问题，因为test数据集，本身也没见过

```
def recognization_data_process(self, x_train, x_test, y_train, y_test, un_label=None, rate=0.6):
    # 因为要训练模型，同时做未知源的时候，手机信号要没见过
    # 所以要将已知源手机一部分作为训练，一部分作为未知源
    # 函数输出train, test, val, train and test是模型训练需要，val是未知源识别
    un_label = [0, 1, 2, 3, 4, 5, 6, 7] # 设置未知源
    x_val = None
    y_val = []
    # 先将所有未知源加入到val 中
    for label in un_label:
        train_temp = x_train[y_train == label, :]
        test_temp = x_test[y_test == label, :]
        y_val.extend([0]*(len(train_temp) + len(test_temp)))
        if x_val is None:
            x_val = np.concatenate([train_temp, test_temp])
        else:
            x_val = np.concatenate([x_val, train_temp, test_temp])
        del train_temp, test_temp

        # 将train test 中的未知源删除
        x_train = np.delete(x_train, y_train == label, axis=0)
        y_train = np.delete(y_train, y_train == label)
        x_test = np.delete(x_test, y_test == label, axis=0)
        y_test = np.delete(y_test, y_test == label)
    # 再将测试集中一定量的已知源加入到val中，并打上标签为1
    idx = random.sample(range(len(y_test)), int(len(y_test)*rate))
    test_temp = x_test[idx, :]
    x_val = np.concatenate([x_val, test_temp])
    y_val.extend([1]*len(test_temp))

    # 将这些加入进去的手机删除
    # 这里不删除也可以
    #x_test = np.delete(x_test, idx, axis=0)
    #y_test = np.delete(y_test, idx)

    # 将训练集和测试集的标签重新从0开始编号
    def get_label_map(label):
        true_label = label
        label = set(label)
        label_map = dict(zip(label, range(len(label))))
        true_label = list(map(lambda x:label_map.get(x), true_label))
        return np.array(true_label)
    y_train = get_label_map(y_train)
    y_test = get_label_map(y_test)
    y_val = np.array(y_val)
    self.logger.info(f'y_test bincount :{np.bincount(y_test)}')
    self.logger.info(f'y_train bincount:{np.bincount(y_train)}')
    self.logger.info(f'y_val bincount:{np.bincount(y_val)}')

    return x_train, x_test, x_val, y_train, y_test, y_val
```

2. 模型搭建

Q1. 模型的框架如何确定？是先对信号做embedding， 然后再计算余弦相似度？

一般是使用resnet18即可，因为训练集全是已知源，这里使用分类模型即可，然后使用模
型的倒数第二层，作为该手机类别的embedding向量即可，(因为resnet最后第二层是2048维度，通常我们会降低一个维度即再添加一个线性层，根据手机的数量，映射到低维度如128， 256这样长度)所以embedding向量的好坏，非
常依赖于模型的分类效果。

为了提取这个手机类别的特征，我们将这个手机的所有特征提取出来，比如说现在1类手
机有100个信号，那么我们现在将这100个信号同时输入到网络中，提取到100个embedding
向量，然后将这100个embedding向量求平均，作为这个手机的embedding向量。然后存储
到数据库中。

特征向量存储完毕后，假如现在有一个手机信号，这个信号可能是已知源，也可能是未知
源，我们同样将这个信号输入到resnet中，同样的方式提取embedding特征，然后与数据
库中的向量做余弦相似度匹配，其中余弦相似度如果越大，那么则表示这两个向量越匹配
，如果相似度很低，那么说明信号不在数据库中。(注意这里要设置一个阈值，余弦相似
度小于阈值的话， 则表示不在数据库中)

代码实现：

```
1. 首先resnet18倒数第二层有512维度的向量，所以要取出来

前面手机有30个类别，使用0-7作为未知源，所以训练模型的时候，只用了其他手机类型
训练模型，未知源识别中，30类手机都存在。训练好模型后，保存训练模型的参数。

此前注意固定随机数种子，再次读取数据，确保数据划分和之前是一样的。因为这时候要
提取已知源手机的特征向量，这里我觉得提取多个向量存起来，不要取平均，多个特征方
便投票。
```

3. 模型训练

Q1. 模型的损失函数如何确定？如何保证embedding的向量能很好表示这个手机类别呢？

这个就按之前的分类即可，普通的损失函数即可，如果有什么其他的好的，可以改进


