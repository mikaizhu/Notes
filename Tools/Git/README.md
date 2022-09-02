<!--ts-->
* [README](#readme)
* [git教程](#git教程)
* [给github添加电脑ssh公钥匙](#给github添加电脑ssh公钥匙)
* [git 版本控制](#git-版本控制)
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
* [最新版本的git教程更新7月1号](#最新版本的git教程更新7月1号)
   * [git配置(必须)](#git配置必须)
   * [git理论学习](#git理论学习)
   * [git设置忽略项](#git设置忽略项)
   * [git ssh 免密码登陆](#git-ssh-免密码登陆)
   * [git 分支管理，多人协作](#git-分支管理多人协作)
* [问题汇总](#问题汇总)

<!-- Added by: zwl, at: 2021年11月26日 星期五 10时47分46秒 CST -->

<!--te-->

目录生成代码：[toc.sh](./toc.sh)
github markdown: [github markdown](https://docs.github.com/cn/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

# README
  顶点
# git教程

- https://www.liaoxuefeng.com/wiki/896043488029600/900003767775424
- git 命令可视化学习: https://oschina.gitee.io/learn-git-branching/

# 给github添加电脑ssh公钥匙
```
cd ~/.ssh
ssh-keygen -t rsa -C "747876457@qq.com"
cat id_rsa.pub
```
登陆到github，找到setting 点击add ssh keygen

# git 版本控制

reference:https://www.liaoxuefeng.com/wiki/896043488029600/897013573512192

```
# 使用log命令查看hash值
git log
git reflog
# 使用下面两个命令版本回退
# 可以使用hash值回退也可以使用head指针回退
# head 指针为当前版本的位置
git reset --hard HEAD~1
git reset --hard hash
```

# git分支知识点

## git创建分支

参考：
- https://docs.github.com/cn/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches#working-with-branches
- https://www.liaoxuefeng.com/wiki/896043488029600/900003767775424(推荐)

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
git switch -c dev
```

单独切换分支

```
git checkout temp
git switch dev
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

- 使用no ff模式, 参考https://www.liaoxuefeng.com/wiki/896043488029600/900005860592480

```
git switch master # 先切换到master分支
git merge temp
git merge --no-ff -m "merge with no-ff" dev
```

如果merge后有出现问题，即conflict，那么git会将这些有冲突的地方合并并标记起来，
需要我们手动选择后，再进行commit

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

# 最新版本的git教程更新7月1号

## git配置(必须)

查看命令的配置
```
git config -l
```

查看git的身份邮箱设置，说明：这个配置是必须配置的，要让服务器知道你是谁
```
git config --global --list
```
配置好后其实是有个本地配置文件的, 可以在自己的电脑上找到, 配置文件为gitconfig
```
git config --global user.name 'mikizhu'
git config --global user.email '747876457@qq.com'
```

## git理论学习

git区域说明：

- 工作区 workspace：平时本地存放代码项目的地方
- stage 暂存区：顾名思义，文件在里面只是暂存的，用于临时存放改动，事实上他只是
  一个文件，保存你即将提交的文件列表信息
- repository 仓库区：存放所有数据的位置， 里面有你提交的所有版本的数据， 其中
  head文件指向最新放入仓库的版本
- remote 远程仓库：托管代码的服务器，如github

git 工作流程：
- 本地修改好文件，此时代码还只是在本地工作区，没有被git跟踪
- 然后将git存放到暂存区，即git add .
- 将暂存区的文件存提交commit 到repository仓库区
- 最后push到远程服务器上

工作流程实操：

- 初始化

```
git init # 会生成 .git文件
```

- 查看文件追踪, 查看哪些本地文件没有被追踪

```
git status [filename] # filename 可选，看某个制定文件
```

- `git add .`
- `git coommit -m ''`


## git设置忽略项

我们希望有些文件不被提交上去，希望被忽略，因此可以在gitignore文件内添加需要忽
略的文件

```
cd my_git_repository
touch .gitignore
vi .gitignore

###gitignore语法说明###
- # 开头为注释
- *.txt # 忽略所有以.txt为结尾的文件
- !lib.txt # 除了lib.txt文件，都会被忽略
- /temp # 忽略temp目录之外的所有目录
- temp/ # 忽略temp目录下的所有文件, 这里注意斜杠在前面表示上级，斜杠在后面表示
  下级
- temp/*.txt # 忽略temp目录下所有txt文件，但不包括temp/sub1/*.txt
```

## git ssh 免密码登陆

由于git是远程仓库，所以要设置ssh免密码登陆

本质就是在本地上生成本地电脑的密钥，然后添加到github上即可

## git 分支管理，多人协作

git 分支让我们可以对不同的分支进行开发，但是会导致下面一些问题:

- 同事A在对dev1分支开发
- 同事B在对dev2分支开发，但同事B可能会修改同事A的代码
- 然后合并会生成冲突

```
# 创建一个新的分支，并切换到这个分支
git checkout -b [branch]

# 只创建一个新的分支，不跳转
git branch dev # create new branch

# 删除分支
git branch -d [branch-name]

# 删除远程分支
git push origin --delete [branch-name]
```

请使用这个网站在线学习：https://oschina.gitee.io/learn-git-branching/

输入下面代码：
```
git checkout master
git commit
git commit

git checkout -b dev
git commit 
git commit
git commit

git checkout master
git merge --no-ff dev
# 使用git log --graph 可以直观用徒刑表示输出提交日志
```

# 问题汇总

[git_head_游离问题.md](doc/git_head_游离问题.md)

[代码折叠并显示.md](doc/代码折叠并显示.md)

[github目录生成.md](doc/github目录生成.md)

[git没有出现绿点.md](doc/git没有出现绿点.md)

[git_in_hand.md](doc/git_in_hand.md)

[git查看和修改文件.md](doc/git查看和修改文件.md)

[git 恢复文件.md](doc/git 恢复文件.md)

[git push 失败.md](doc/git push 失败.md)

