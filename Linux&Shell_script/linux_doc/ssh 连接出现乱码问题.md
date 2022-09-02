# ssh连接出现乱码问题

一般是终端和服务器的字符集不匹配问题

1. 在终端下输入这个

```
vim ~/.zshrc
```

2. 在文件内容末端添加

```
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

3. 输入下面代码

```
source ~/.zshrc
```

## 参考链接

https://zhidao.baidu.com/question/1049589561455974019.html

