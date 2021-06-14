<!--ts-->
* [说明](#说明)
* [run.sh脚本](#runsh脚本)
* [样本均衡问题](#样本均衡问题)
* [模型融合讲解](#模型融合讲解)

<!-- Added by: mikizhu, at: 2021年 6月13日 星期日 00时13分06秒 CST -->

<!--te-->
# 说明

使用原始数据，仅做切片处理，300个点效果最好，dnn单模能到57，cnn单模能到53.

样本均衡后，dnn单模能到59， cnn单模型能到54

模型融合后，线上能突破60

# run.sh脚本

注意args python模块的使用方式：记得在python文件中加`--`

脚本使用说明：

```
chmod +x run.sh
./run.sh
```

重定向输出说明：使用logger函数即可，使用方法请查看py文件和sh文件

TODO:
- [x] 解决重定向输出问题，py文件中的print输出不能重定向

# 样本均衡问题

样本均衡使用的是过采样，即会拓展数据集，让每个类别数量都一样，这里SMOTE的效果
最好。

SMOTE原理：https://blog.csdn.net/qq_39478403/article/details/115533295

使用方法如下：

```
# 安装模块
from imblearn.over_sampling import SMOTE

# 开始过采样
smote = SMOTE(random_state=42, n_jobs=-1)
x_train, y_train = smote.fit_resample(train, train_label)
print(x_train.shape, y_train.shape)
```

# 模型融合讲解

模型融合使用的软投票：就是不同模型输出的概率乘以权重，然后将概率相加

注意以下几点：
- 建议保存模型的参数
- 测试的时候，要开启eval模式
- F.softmax 要设置dim=1

模型的保存和加载:
```
model1 = DNN().to(device)
model2 = SampleCNN().to(device)
model1.load_state_dict(torch.load('dnn_best_model.point'))
model2.load_state_dict(torch.load('cnn_best_model.point'))
```


模型融合过程:
```
model1.eval()
model2.eval()
preds1 = model1(torch.FloatTensor(test_sp).to(device))
preds2 = model2(torch.FloatTensor(test_sp).to(device))
print(preds1.shape, preds2.shape)
ans = (preds1 * 0.6 + preds2 * 0.4).argmax(dim=1).detach().cpu().numpy()
```



