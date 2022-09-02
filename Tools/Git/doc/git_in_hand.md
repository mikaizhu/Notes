现在遇到的场景如下：

现在有两个分支，一个是master，另一个是dev，其中master分支里面有file1文件，dev
分支里面有file1，file2文件，现在在dev分支下commit后，然后切换到master分支下，
对dev分支进行merge，但是最后master分支下还是只有file1文件，没有file2文件


# 探索分支

测试代码如下：

```
mkdir t && cd t
echo 'version1' >readme.md
git add .
git commit -m 'update'
git checkout -b dev
echo 'version1' >readme.md
echo 'version1' >test.md
git add .
git commit -m 'update'
git checkout master
git merge --no-ff dev
ls
```
发现master目录下并没有text文件

```
git checkout dev
echo 'version2' >>readme.md
git add .
git commit -m 'update'
git checkout master
git merge --no-ff dev
```

只对同名字的readme.md进行修改后，发现会同步，但是不会同步新出现的文件夹text.md

```
git checkout master
echo 'version3' >>readme.md
git add .
git commit -m 'update'
git checkout dev
git merge master
```

调用完上面的代码后，然后会发现，dev分支中的readme.md会同步，然后dev分支多出的
text.md文件被删除了

```
git checkout dev
vi readme.md # 删除到只剩下version 1
echo 'version 1' >text.md
git add .
git commit -m 'update'
git checkout master
git merge --no-ff dev
```

此时发现文件也可以同步了，readme文件中只剩下了version 1， text文件也同步出现了
, 但是没有出现分支，不知道是什么原因

# 探索冲突

创造冲突:

```
git checkout -b dev3
echo 'version 2 and 3' >readme.md
git add .
git commit -m 'update'
git checkout master
echo "version 1 and version 3" >readme.md
git add .
git commit -m 'update'
git merge dev3
```

当我们在各自分支修改同一个文件的地方，并分别提交后，合并就会出现冲突问题.

```
<<<<<<< HEAD
version 2 and version 3
=======
version 2 and 3
>>>>>>> dev3
```

然后我们需要手动删除冲突的地方，然后提交即可
```
git add .
git commit -m 'update'
```
继续尝试修改，查看冲突形成的原因

```
同样先从master创建一个新的分支dev3
原来readme文件中的内容为version 2 and 3

在master中，将readme文件修改为version 2 and 3 and 4
在dev分支中，将readme文件修改为两行 version 2 and 3 version 4
分别commit后，发现会出现冲突
```

总结：git合并会检查两个相同文件中的同一行位置，如果同一行中，两个分支的内容不
一样，则会产生冲突，当merge的时候，git会自动将冲突的地方标记出来

# git分支文件同步

假如现在有两个分支，dev1和dev2，两个分支分别增加了文件，如何将dev2和dev1 的文
件进行同步呢？

```
分别创建两个新的分支dev1 and dev2
在dev1 中添加dev1.md文件，在dev2分支中添加dev2.md文件
分别commit后，然后使用
git merge --no-ff dev1
文件即可同步
```
