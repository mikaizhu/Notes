# 使用场景

有时候我们文件消失或者没保存，可以通过git管理文件

比如我git一次性提交了很多文件，怎么讲文件的某一版本恢复到本地呢？

- 使用下面代码获得某个文件的hash值

```
git log filename
```

- 使用下面代码恢复到某个版本

```
git checkout hash
```

然后就会将这个文件恢复到本地

**比如我commit了两个文件file1，file2，然后想恢复一个到本地**

**刚才测试，使用checkout会将两个文件一起恢复，如果我只想恢复一个文件怎么办**

 **使用下面代码**

```
git checkout hash singal-filename
```

即可恢复某一单一文件

