<!--ts-->
* [README](#readme)
* [给github添加电脑ssh公钥匙](#给github添加电脑ssh公钥匙)
* [让git下载更快](#让git下载更快)
* [github markdown 语法](#github-markdown-语法)

<!-- Added by: mikizhu, at: 2021年 6月 3日 星期四 13时28分32秒 CST -->

<!--te-->

目录生成代码：[toc.sh](./toc.sh)
github markdown: [github markdown](https://docs.github.com/cn/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

# README

  顶点

# 给github添加电脑ssh公钥匙
```
cd ~/.ssh
ssh-keygen -t rsa -C "747876457@qq.com"
cat id_rsa.pub
```
登陆到github，找到setting 点击add ssh keygen

# 让git下载更快
```
git config --global http.postBuffer 524288000
```

# github markdown 语法

参考：https://github.com/guodongxiaren/README#%E5%9B%BE%E7%89%87%E9%93%BE%E6%8E%A5

这里介绍下几个比较好用的语法：

1. 锚点

效果就是点击就会回到某个地方, 因此可以使用这个方法设置目录

[return top](#readme) 


