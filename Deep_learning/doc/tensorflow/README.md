<!--ts-->
* [参考教程](#参考教程)
* [tensorflow gpu 版本安装](#tensorflow-gpu-版本安装)
* [tensorflow学习](#tensorflow学习)
   * [查看gpu是否可用](#查看gpu是否可用)
   * [基础模型搭建](#基础模型搭建)
   * [DNN搭建](#dnn搭建)
   * [修改学习率](#修改学习率)
   * [Dropout层](#dropout层)
   * [函数式API搭建网络](#函数式api搭建网络)

<!-- Added by: zwl, at: 2021年 6月22日 星期二 20时45分29秒 CST -->

<!--te-->
# 参考教程

- 文字教程:https://tianchi.aliyun.com/course/779/13663
- 视频教程:https://www.bilibili.com/video/BV1Zt411T7zE?p=2

# tensorflow gpu 版本安装

建议新创建新环境安装

```
conda create -n tf python==3.8
```

```
conda install -c anaconda tensorflow-gpu
```

# tensorflow学习

tensorflow输入的数据格式:numpy即可，不需要转成tf的格式

tensorflow中，网络的搭建，有两种格式：

- 除了线性的sequential模式
- 还有一种函数式api的格式, 这种格式方便自己diy网络结构

## 查看gpu是否可用

```
tf.config.list_physical_devices('GPU')
```

## 基础模型搭建

```
import tensorflow as tf

# 获得数据
x = data.education
y = data.income

# 初始化模块, 一般使用sequential方法添加模块, 并且往模型中添加网络层, 一般使用
layers中的各种网络层添加
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, 1))

# 打印模型参数
print(model.summary())

# 使用compile定义模型的损失函数和优化器
model.compile(optimizer='adam', loss='mse')

# 拟合数据，并定义学习的次数, 这里的数据都是numpy数据
# 或者直接使用validation_data来进行测试集评估，输出准确率
model.fit(x, y, epochs=300, validation_data=(test, test_label))

# 使用这个函数对测试集评估
#model.evaluate()

# 预测数据
model.predict(x)
```

## DNN搭建

```
import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(10, 10, activation='relu'), tf.keras.layers.Dense(10, 1)])
print(model.summary())

# 使用compile定义模型的损失函数和优化器, metrics表示训练的时候还会输出准确率
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

# 拟合数据，并定义学习的次数, 这里的数据都是numpy数据, history是一个字典，里面
有acc，loss，epoch等值
history = model.fit(x, y, epochs=300)

# 预测数据
model.predict(x)
```

## 修改学习率

```
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
loss='categorical_crossentropy',
metrics=['acc']
)
```

## Dropout层

注意，dropout一般在激活函数之后

```
model = tf.keras.Sequential([
tf.keras.layers.Dense(10, 10, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, 1),
])
```

## 函数式API搭建网络

函数式API单输入:

```
# 神经网络搭建

# 这里要定义函数的输入接口，设置输入的数据格式，不用设置batch, 默认第一个就是batch
inputs = keras.Input(shape=(28, 28))

# 可以把.Flatten()看成一个函数，然后在最后加上()即可调用函数
x = keras.layers.Flatten()(input)

# 这里只要定义网络的输出即可
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

# 定义模型
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse')

# 拟合数据，并定义学习的次数, 这里的数据都是numpy数据
# 或者直接使用validation_data来进行测试集评估，输出准确率
model.fit(x, y, epochs=300, validation_data=(test, test_label))
```

函数式API多输入:

```
input1 = keras.Input(shape=(28, 28))
input2 = keras.Input(shape=(28, 28))
x1 = keras.layers.Flatten()(input1)
x2 = keras.layers.Flatten()(input2)
x = keras.layers.concatenate([x1, x2])
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=[input1, input2], outputs=outputs)
```


