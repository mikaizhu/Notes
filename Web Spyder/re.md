- re.escape

因为正则表达中，有些字符是特殊的字符，在匹配的时候会出现问题，如果要匹配原始的字符，使用escape，不用自己一个个改。

```
# 比如我们想对下面字符进行匹配
'www.python.org'
# 但是.表示任意的意思，我们不想手动添加怎么办
re.escape('www.python.org')
# 输出
'www\\.python\\.org'

```

参考：https://www.kingname.info/2019/12/02/escape-in-python/



**相关教程**：

参考笔记：http://note.youdao.com/s/Ge0fneK0

参考官方教程：https://docs.python.org/zh-cn/3/library/re.html