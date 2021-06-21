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

## 查看gpu是否可用

```
tf.config.list_physical_devices('GPU')
```


