什么是游离问题？

比如我误操作了，然后使用git checkout 回到原来的版本

然后再次git push，本来以为github会显示原来的版本，但是并没有，主要是因为head还
在原来的位置。

解决办法参考:https://blog.csdn.net/u011240877/article/details/76273335

有个问题就是，如何解决git 冲突呢？

head一般它指向当前工作目录所在分支的最新提交。

HEAD 处于游离状态时，我们可以很方便地在历史版本之间互相切换，比如需要回到某次提交，直接 checkout 对应的 commit id 或者 tag 名即可。

也就是说我们的提交是无法可见保存的，一旦切到别的分支，游离状态以后的提交就不可追溯了。

解决办法就是新建一个分支保存游离状态后的提交：

```
# 创建一个新的分支
git branch temp

# 先切换到那个分支，然后将游离状态的历史版本push到这个分支
git checkout temp
git push origin temp

# 回到主分支，然后将刚才的分支merge即可
git checkout master
git merge temp

# 使用git status 查看merge的结果
# 如果有冲突就解决，然后提交主分支
git push origin master

# 删除刚才创建的分支
git branch -d temp
```

如何解决冲突？

参考：https://www.liaoxuefeng.com/wiki/896043488029600/900004111093344


