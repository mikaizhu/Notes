常用的符号

```
.匹配任意字符，不包含空字符 
^匹配文本行开头
$匹配文本行结尾
grep -h 'zip$' file
```

方括号

```
# 匹配其中的一个
[bg]zip匹配 bzip or gzip
[^bg]zip 不匹配b和g
[b-g]匹配b-g之间的字母
```

选择

```
A｜B 表示匹配A或者B中的一个

echo "AA" | grep -E 'AA|BB'
```

量词操作符号

```
? 匹配前面的字符出现0 or 1 次
* 匹配....出现0或多次
+ 匹配....出现1或多次
....
```

参考：https://man.linuxde.net/docs/shell_regex.html