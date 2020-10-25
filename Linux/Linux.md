# 每天一个Linux命令

# ls

1. 想查看文件大小怎么办？

```
ls -sh

28K Mine.ipynb       208K sample_submission.csv   24K torch.ipynb   16K 未命名.ipynb
                                                                                                      │ 64K Reference.ipynb   49M test.csv                74M train.csv
```

s表示size，h表示human readable。

2. 想查看文件的权限怎么办？

```
ls -l

-rw-rw-r-- 1 zwl zwl    28492 10月 21 21:02 Mine.ipynb
```

-表示是文件，后面是权限，分别对应三个权限，

3. 列出开头字母是t的所有文件

```
ls t*
```

```
ls -hl t*
```

4. 列出隐藏的文件

```
ls -a
```

# cd

1. 切换到根目录

```
cd /
```

2. 切换到用户目录

```
cd ~
```

3. 切换到上一次的目录

```
cd - # 这是减号
```

