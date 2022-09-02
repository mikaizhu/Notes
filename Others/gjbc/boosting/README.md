# 说明

使用的stacking和boosting的思想
- version1代码出现了标签泄漏，按道理是不能使用val的标签的，即使是stacking也不行
- version2代码也出现错误，每利用到boosting，即dnn总是在学习正确的标签
- version3 解决了上面两个错误，但是准确率有待提高

TODO:
- [ ] version3 代码准确率提高
- [ ] 因为错误的数据总是越来越少，所以可以尝试增加样本？同时提高迭代次数

## version1文件说明


boosting 思想：

- 首先通过样本均衡
- dnn学习一轮所有数据
- 找出第一轮dnn学习错误的数据
- 利用错误的数据作为训练集，重新初始化dnn进行训练
- 如此反复
- 然后就会学习到很多dnn模型，这些dnn模型的分布都不一样

version1存在的问题：

- boosting 存在一些bug, 即学错误的样本，按道理是越来越少

```
def get_boost_data(feature, true_label, model):
        # 存储boosting数据, 这里使用epoch=0时候的分类错误的，因为这样数据才会多一点，或包括大部分分类错误的
        preds_label = model(feature).argmax(dim=1)
        f1 = feature[preds_label != true_label, :].detach().cpu()
        f2 = true_label[preds_label != true_label].detach().cpu()
        boost_feature.append(f1)
        boost_label.append(f2)
```

stacking:

- 装载数据集，包括验证集和测试集，其中验证集是有标签的，测试集是无标签的
- 第一层使用上面训练得到的dnn模型, 进行预测，假设这里有三个模型，那么会有
  preds1，preds2，preds3，以及验证集标签，val_label
- 使用catboost中的fit方法，将preds1，preds2，preds3 作为训练集，val_label作为
  标签进行训练
- 对测试集进行预测，得到preds1， preds2，preds3,作为测试集的特征
- 使用catboost中的predict方法, 对测试集进行预测，生成结果后提交

# 
