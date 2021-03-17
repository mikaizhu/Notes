git 查看所有文件

```
git ls-files
```

git 修改文件名

参考：https://medium.com/@qjli/daily-coding-tips-56-how-to-rename-folder-properly-in-git-96caceb70fc8

```
git mv old_file new_file
```

但是如果系统是区分大小写的，那么应该用下面命令。

```
git mv vim tmp && git mv tmp Vim
# tmp只是一个中间变量
```
