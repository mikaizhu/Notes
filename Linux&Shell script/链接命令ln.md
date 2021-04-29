todo：查看链接文件相关内容

参考：https://www.cnblogs.com/peida/archive/2012/12/11/2812294.html

**如何使用vim修改链接文件**？

目前方法：删除软连接

假如现在有链接文件

```
file1 -> file2
```

删除链接，但是不删除数据

```
rm -rf ./file1
```

