官网：https://github.com/huyng/bashmarks

参考：https://goddyzhaocn.wordpress.com/2011/08/21/bashmarks-introduction/

作为终端控，经常要使用命令来定位要访问的目录，于是，“cd path”这种命令每天不知到要打多少次。对于一些结构比较深的目录（尤其是要经常访问的），使用cd命令起来真的是噩梦（尽管有tab神器，依旧让我感觉非常不爽）。

使用：

```
s <bookmark_name> - Saves the current directory as "bookmark_name"
g <bookmark_name> - Goes (cd) to the directory associated with "bookmark_name"
p <bookmark_name> - Prints the directory associated with "bookmark_name"
d <bookmark_name> - Deletes the bookmark
l                 - Lists all available bookmarks
```

因为l在ubuntu下会冲突，所以进行修改，改成lb（list bookmarks）

```
进入到bashmarks.sh

找到function l

将function名字改成lb

make install

# 启动
source ~/.local/bin/bashmarks.sh
```

