tty又称作virtual terminal

详细说明请自行百度

介绍下如何在terminal中使用命令行切换tty

1. 先安装插件

```
sudo apt-get install kbd
```

2. 使用kbd命令切换tty， 安装完成即可通过chvt命令来切换tty，比如切换到tty2。

```
sudo chvt 2
```

