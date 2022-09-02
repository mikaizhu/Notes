假如现在目录下有这些文件：

```
download.sh
mchar_data_list_0515.csv
mchar_sample_submit_A.csv
mchar_train.json
mchar_val.json
mode_expansion.md
README.md
```

要想保留其中的三个文件，download.sh,mchar_data_list_0515,README.md 三个文件。

使用模式拓展解决：`ls mchar_(sample|train|val)*`, 除了ls，还可以使用rm等命令。
可以自己尝试下。

- ?\(A\|B)表示匹配A or B,注意一定要加圆括号,格式为?\(), 问号表示匹配0个或者1个
  。
- \*号表示匹配0个或多个任意字符
