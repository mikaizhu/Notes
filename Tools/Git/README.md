<!--ts-->
* [README](#readme)
* [git教程](#git教程)
* [给github添加电脑ssh公钥匙](#给github添加电脑ssh公钥匙)
* [git分支知识点](#git分支知识点)
   * [git创建分支](#git创建分支)
   * [git查看分支](#git查看分支)
   * [git切换分支](#git切换分支)
   * [git合并分支](#git合并分支)
   * [git删除分支](#git删除分支)
   * [git解决冲突](#git解决冲突)
   * [git rebase作用](#git-rebase作用)
* [让git下载更快](#让git下载更快)
* [github markdown 语法](#github-markdown-语法)
* [git 将文件修改为小写字母](#git-将文件修改为小写字母)

<!-- Added by: zwl, at: 2021年 6月22日 星期二 16时22分38秒 CST -->

<!--te-->

目录生成代码：[toc.sh](./toc.sh)
github markdown: [github markdown](https://docs.github.com/cn/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

# README

  顶点
# git教程

https://www.liaoxuefeng.com/wiki/896043488029600/900003767775424

# 给github添加电脑ssh公钥匙
```
cd ~/.ssh
ssh-keygen -t rsa -C "747876457@qq.com"
cat id_rsa.pub
```
登陆到github，找到setting 点击add ssh keygen

# git分支知识点

## git创建分支

参考：https://docs.github.com/cn/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches#working-with-branches

git分支主要就是为了不影响master分支，可以方便我们实现新的想法，只有管理员才有
权限合并分支。

如果分支合并到master中了，就可以将该分支删除。

```
# 创建分支
git brach temp
```

## git查看分支

```
git branch
```

然后按q退出查看

## git切换分支

创建并切到新分支

```
git checkout -b temp
```

单独切换分支

```
git checkout temp
```

## git合并分支

git 是如何合并分支的请查看：https://www.liaoxuefeng.com/wiki/896043488029600/900003767775424

总的来说：

- 我们每次提交，git都会把提交看成一个时间线，然后会有head指针，master指针。

- master 指针指向最近的更新，head指针指向master指针

- 假如现在创建了一个temp分支，master分支就不会再往前更新了, head 指针指向了
  temp

- 合并就是将master指针指向temp指针

那么有多个分支的时候，git的合并原理是什么呢？

- 参考：https://zhuanlan.zhihu.com/p/149287658

代码：

```
git merge temp
```

## git删除分支

```
git branch -d temp
```

## git解决冲突

参考：

- https://www.liaoxuefeng.com/wiki/896043488029600/900004111093344

当我们在分支和master上对同一个文件进行了修改，merge的时候会将所有文件合并，最
后会只剩下一个文件，但是这个文件是两个文件合并起来的。会用符号标记出哪里出了问
题，需要我们手动删掉.

## git rebase作用

参考：
- http://jartto.wang/2018/12/11/git-rebase/
- https://www.liaoxuefeng.com/wiki/896043488029600/1216289527823648

作用1: 合并无用的提交

作用2: 合并分支


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

# git 将文件修改为小写字母

```
# 使用管道命令查看
git ls-files | gerp ABC

# 使用git mv对文件名进行修改
git mv ABC temp && git mv temp abc
```

